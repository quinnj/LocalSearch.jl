using Test
using LocalSearch
using Random
using SQLite

function fake_embed(texts::Vector{String})
    dims = 8
    m = zeros(Float32, dims, length(texts))
    for (i, t) in enumerate(texts)
        rng = MersenneTwister(hash(t))
        m[:, i] = randn(rng, Float32, dims)
        m[:, i] ./= sqrt(sum(abs2, m[:, i]))
    end
    return m
end

function fake_token_count(text::AbstractString)
    n = 0
    for _ in eachmatch(r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]", text)
        n += 1
    end
    return n
end

@testset "LocalSearch" begin
    @test :delete! in names(LocalSearch)

    @testset "BM25 search" begin
        store = Store(; embed=nothing)
        load!(store, "Julia is a high-performance programming language"; id="julia", title="Julia")
        load!(store, "Python is a popular general-purpose language"; id="python", title="Python")
        load!(store, "SQLite is a lightweight database engine"; id="sqlite", title="SQLite")

        results = search(store, "programming language")
        @test length(results) >= 1
        @test results[1].id in ("julia", "python")
        @test results[1].score > 0
        @test results[1].source == :fts

        results = search(store, "database")
        @test length(results) == 1
        @test results[1].id == "sqlite"
    end

    @testset "lexical query builder supports phrases and negation" begin
        @test LocalSearch.build_fts_query("\"machine learning\"") == "\"machine learning\""
        @test LocalSearch.build_fts_query("performance -sports") == "\"performance\"* NOT \"sports\"*"
        @test LocalSearch.build_fts_query("-sports") === nothing
        @test LocalSearch.build_fts_query("\"unterminated phrase") === nothing
    end

    @testset "BM25 phrase and negation search" begin
        store = Store(; embed=nothing)
        load!(store, "machine learning systems"; id="ml", title="ML")
        load!(store, "machine systems learning"; id="mixed", title="Mixed")
        load!(store, "performance review for sports team"; id="sports", title="Sports")
        load!(store, "performance review for database queries"; id="db", title="Database")

        phrase_results = search(store, "\"machine learning\"")
        @test length(phrase_results) == 1
        @test phrase_results[1].id == "ml"

        negated_results = search(store, "performance -sports")
        @test length(negated_results) == 1
        @test negated_results[1].id == "db"

        unmatched_quote_results = search(store, "\"unterminated phrase")
        @test isempty(unmatched_quote_results)

        all_negative_results = search(store, "-sports")
        @test isempty(all_negative_results)
    end

    @testset "BM25 score normalization preserves match strength ordering" begin
        store = Store(; embed=nothing)
        load!(store, "alpha beta gamma"; id="strong", title="Strong")
        load!(store, "alpha beta"; id="weak", title="Weak")

        fts_query = LocalSearch.build_fts_query("alpha beta")
        rank_by_id = Dict{String, Float64}()
        for row in SQLite.DBInterface.execute(store.db, """
            SELECT d.key, bm25(documents_fts, 1.0, 5.0, 10.0) as rank
            FROM documents_fts f
            JOIN documents d ON d.id = f.rowid
            WHERE documents_fts MATCH ?
            ORDER BY rank ASC
        """, (fts_query,))
            rank_by_id[String(row.key)] = Float64(row.rank)
        end
        @test length(rank_by_id) == 2

        score_by_id = Dict(result.id => result.score for result in search(store, "alpha beta"; limit=10))

        stronger_id, weaker_id = rank_by_id["strong"] < rank_by_id["weak"] ? ("strong", "weak") : ("weak", "strong")
        @test score_by_id[stronger_id] > score_by_id[weaker_id]
    end

    @testset "embedding formatting helpers" begin
        qwen_model = "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"

        @test LocalSearch.Embed.format_query_for_embedding("hello world") == "task: search result | query: hello world"
        @test LocalSearch.Embed.format_document_for_embedding("body text"; title="Doc Title") == "title: Doc Title | text: body text"
        @test LocalSearch.Embed.format_document_for_embedding("body text") == "title: none | text: body text"

        @test LocalSearch.Embed.format_query_for_embedding("hello world"; model=qwen_model) ==
            "Instruct: Retrieve relevant documents for the given query\nQuery: hello world"
        @test LocalSearch.Embed.format_document_for_embedding("body text"; title="Doc Title", model=qwen_model) ==
            "Doc Title\nbody text"
        @test LocalSearch.Embed.format_document_for_embedding("body text"; model=qwen_model) == "body text"
    end

    @testset "custom embeddings receive formatted query and document text" begin
        captured = Vector{Vector{String}}()

        function recording_embed(texts::Vector{String})
            push!(captured, copy(texts))
            return fill(0.25f0, 4, length(texts))
        end

        store = Store(; embed=recording_embed, token_count=fake_token_count, chunk_max_tokens=32, chunk_overlap_tokens=0)
        empty!(captured)  # Discard constructor probe used for dimension detection

        load!(store, "body text"; id="doc", title="Doc Title")
        @test captured == [["title: Doc Title | text: body text"]]

        empty!(captured)
        LocalSearch.search_vec(store, "body text"; limit=5)
        @test captured == [["task: search result | query: body text"]]
    end

    @testset "embedding metadata records active model label" begin
        store = Store(; embed=fake_embed, token_count=fake_token_count, chunk_max_tokens=32, chunk_overlap_tokens=0, embed_label="model-v1")
        load!(store, "body text"; id="doc", title="Doc Title")

        models = [String(row.model) for row in SQLite.DBInterface.execute(store.db,
            "SELECT DISTINCT model FROM chunks WHERE hash = (SELECT hash FROM documents WHERE key = ?)",
            ("doc",))]
        @test models == ["model-v1"]
    end

    @testset "model changes trigger re-embedding" begin
        path = tempname()

        store1 = Store(path; embed=fake_embed, token_count=fake_token_count, chunk_max_tokens=32, chunk_overlap_tokens=0, embed_label="model-v1")
        load!(store1, "body text"; id="doc", title="Doc Title")
        close(store1)

        captured = Vector{Vector{String}}()
        function recording_embed_with_model_change(texts::Vector{String})
            push!(captured, copy(texts))
            return fill(0.5f0, 4, length(texts))
        end

        store2 = Store(path; embed=recording_embed_with_model_change, token_count=fake_token_count, chunk_max_tokens=32, chunk_overlap_tokens=0, embed_label="model-v2")
        empty!(captured)  # constructor probe

        load!(store2, "body text"; id="doc", title="Doc Title")
        @test captured == [["title: Doc Title | text: body text"]]

        models = [String(row.model) for row in SQLite.DBInterface.execute(store2.db, "SELECT DISTINCT model FROM chunks")]
        @test models == ["model-v2"]
        close(store2)
    end

    @testset "title changes trigger re-embedding" begin
        captured = Vector{Vector{String}}()

        function recording_embed_with_title_updates(texts::Vector{String})
            push!(captured, copy(texts))
            return fill(0.5f0, 4, length(texts))
        end

        store = Store(; embed=recording_embed_with_title_updates, token_count=fake_token_count, chunk_max_tokens=32, chunk_overlap_tokens=0, embed_label="model-v1")
        empty!(captured)  # constructor probe

        load!(store, "body text"; id="doc", title="Old Title")
        empty!(captured)

        load!(store, "body text"; id="doc", title="New Title")
        @test captured == [["title: New Title | text: body text"]]
    end

    @testset "legacy chunks schema is migrated to include model metadata" begin
        path = tempname()
        db = SQLite.DB(path)
        SQLite.execute(db, """
            CREATE TABLE chunks (
                hash TEXT NOT NULL,
                seq INTEGER NOT NULL,
                pos INTEGER NOT NULL,
                embedded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
                PRIMARY KEY (hash, seq)
            )
        """)
        SQLite.close(db)

        store = Store(path; embed=nothing)
        columns = [String(row.name) for row in SQLite.DBInterface.execute(store.db, "PRAGMA table_info(chunks)")]
        @test "model" in columns
        close(store)
    end

    @testset "load! / update / delete / clear" begin
        store = Store(; embed=nothing)
        load!(store, "hello world"; id="a", tags=["greeting"])
        load!(store, "goodbye world"; id="b")

        # Update: same key, different content
        load!(store, "hello universe"; id="a")
        results = search(store, "universe")
        @test length(results) == 1
        @test results[1].id == "a"

        # No-op: same key, same content
        load!(store, "hello universe"; id="a")

        # Delete fully removes from search
        delete!(store, "a")
        results = search(store, "universe")
        @test isempty(results)

        # Clear
        clear!(store)
        results = search(store, "world")
        @test isempty(results)
    end

    @testset "delete cleans up orphaned data" begin
        store = Store(; embed=nothing)
        load!(store, "unique content here"; id="doc1")

        # Verify content exists
        row = first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM content"))
        @test row.n == 1

        delete!(store, "doc1")

        # Content should be cleaned up since no other doc references it
        row = first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM content"))
        @test row.n == 0
    end

    @testset "delete preserves shared content" begin
        store = Store(; embed=nothing)
        same_text = "shared content between documents"
        load!(store, same_text; id="doc1")
        load!(store, same_text; id="doc2")

        row = first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM content"))
        @test row.n == 1  # deduplicated

        delete!(store, "doc1")

        # Content should still exist (doc2 still references it)
        row = first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM content"))
        @test row.n == 1

        # doc2 still searchable
        results = search(store, "shared content")
        @test length(results) == 1
        @test results[1].id == "doc2"
    end

    @testset "delete removes vector/chunk rows for orphaned content" begin
        store = Store(; embed=fake_embed, token_count=fake_token_count, chunk_max_tokens=6, chunk_overlap_tokens=2)
        long_text = join(["token$(i)" for i in 1:80], " ")
        load!(store, long_text; id="vecdoc")

        hash_row = first(SQLite.DBInterface.execute(store.db, "SELECT hash FROM documents WHERE key = ?", ("vecdoc",)))
        hash = String(hash_row.hash)

        chunks_before = first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM chunks WHERE hash = ?", (hash,))).n
        vectors_before = first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM vectors")).n
        @test chunks_before > 1
        @test vectors_before == chunks_before

        delete!(store, "vecdoc")

        @test first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM documents WHERE key = 'vecdoc'")).n == 0
        @test first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM chunks WHERE hash = ?", (hash,))).n == 0
        @test first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM content WHERE hash = ?", (hash,))).n == 0
        @test first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM vectors")).n == 0
    end

    @testset "auto-generated IDs" begin
        store = Store(; embed=nothing)
        load!(store, "text without explicit id")
        load!(store, "another text")
        results = search(store, "text")
        @test length(results) == 2
        @test all(r -> !isempty(r.id), results)
    end

    @testset "batch load" begin
        store = Store(; embed=nothing)
        load!(store, ["one fish", "two fish", "red fish"]; ids=["1", "2", "3"], tags=[["a"], ["b"], ["b", "c"]])
        results = search(store, "two fish")
        @test length(results) >= 1
        @test results[1].id == "2"

        tagged = search(store, "fish"; tags=["c"])
        @test length(tagged) == 1
        @test tagged[1].id == "3"

        @test_throws ArgumentError load!(store, ["x", "y"]; ids=["only-one"])
        @test_throws ArgumentError load!(store, ["x", "y"]; tags=[["a"]])
    end

    @testset "token chunking only" begin
        store = Store(; embed=fake_embed, token_count=fake_token_count, chunk_max_tokens=8, chunk_overlap_tokens=2)
        short_text = "one two three"
        short_chunks = LocalSearch.chunk_text_by_tokens(store, short_text)
        @test length(short_chunks) == 1
        @test short_chunks[1].pos == 0

        long_text = join(["token$(i)" for i in 1:80], " ")
        coarse_chunks = LocalSearch.chunk_text_by_tokens(store, long_text; max_tokens=8, overlap_tokens=2)
        @test length(coarse_chunks) > 1
        @test all(fake_token_count(c.text) <= 8 for c in coarse_chunks)

        tight_chunks = LocalSearch.chunk_text_by_tokens(store, long_text; max_tokens=4, overlap_tokens=1)
        @test length(tight_chunks) > length(coarse_chunks)

        load!(store, long_text; id="coarse")
        coarse_hash = String(first(SQLite.DBInterface.execute(store.db, "SELECT hash FROM documents WHERE key = 'coarse'")).hash)
        db_coarse_chunks = first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM chunks WHERE hash = ?", (coarse_hash,))).n
        @test db_coarse_chunks == length(coarse_chunks)

        unique_tight_text = long_text * " unique-tail"
        load!(store, unique_tight_text; id="tight", chunk_max_tokens=4, chunk_overlap_tokens=1)
        tight_hash = String(first(SQLite.DBInterface.execute(store.db, "SELECT hash FROM documents WHERE key = 'tight'")).hash)
        db_tight_chunks = first(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM chunks WHERE hash = ?", (tight_hash,))).n
        expected_tight_chunks = length(LocalSearch.chunk_text_by_tokens(store, unique_tight_text; max_tokens=4, overlap_tokens=1))
        @test db_tight_chunks == expected_tight_chunks
    end

    @testset "token chunking prefers structural markdown boundaries" begin
        store = Store(; embed=fake_embed, token_count=fake_token_count, chunk_max_tokens=7, chunk_overlap_tokens=1)
        text = join([
            "alpha beta gamma delta epsilon zeta",
            "",
            "# Heading",
            "eta theta iota kappa lambda",
        ], "\n")

        chunks = LocalSearch.chunk_text_by_tokens(store, text; max_tokens=7, overlap_tokens=1)
        @test length(chunks) >= 2
        @test any(startswith(chunk.text, "# Heading") for chunk in chunks[2:end])
    end

    @testset "token chunking avoids splitting fenced code blocks when possible" begin
        store = Store(; embed=fake_embed, token_count=fake_token_count, chunk_max_tokens=10, chunk_overlap_tokens=1)
        text = join([
            "alpha beta gamma delta epsilon zeta",
            "```julia",
            "x = 1",
            "```",
            "eta theta iota",
        ], "\n")

        chunks = LocalSearch.chunk_text_by_tokens(store, text; max_tokens=10, overlap_tokens=1)
        @test length(chunks) >= 2
        for chunk in chunks
            if occursin("```julia", chunk.text)
                @test length(collect(eachmatch(r"```", chunk.text))) >= 2
            end
        end
    end

    @testset "tags metadata and tag-filtered search" begin
        store = Store(; embed=nothing)
        load!(store, "Julia language and compiler"; id="julia", tags=[" Lang ", "julia", "LANG"])
        load!(store, "Python language runtime"; id="python", tags=["lang", "python"])
        load!(store, "SQLite database engine"; id="sqlite", tags=["db"])

        julia_tags = [String(row.tag) for row in SQLite.DBInterface.execute(store.db,
            "SELECT tag FROM document_tags dt JOIN documents d ON d.id = dt.document_id WHERE d.key = ? ORDER BY tag",
            ("julia",))]
        @test julia_tags == ["julia", "lang"]

        results = search(store, "language"; tags=["julia"])
        @test length(results) == 1
        @test results[1].id == "julia"

        results = search(store, "language"; tags=["lang"])
        @test Set(r.id for r in results) == Set(["julia", "python"])

        results = search(store, "language"; tags=["python", "julia"])
        @test Set(r.id for r in results) == Set(["julia", "python"])

        load!(store, "Julia language and compiler"; id="julia", tags=["compiled"])
        @test isempty(search(store, "language"; tags=["julia"]))
        updated = search(store, "language"; tags=["compiled"])
        @test length(updated) == 1
        @test updated[1].id == "julia"
    end

    @testset "hybrid search with embeddings" begin
        store = Store(; embed=fake_embed, token_count=fake_token_count)
        load!(store, "Julia is great for scientific computing"; id="julia")
        load!(store, "SQLite is a fast embedded database"; id="sqlite")

        results = search(store, "computing")
        @test length(results) >= 1
        @test all(r -> r.source == :hybrid, results)
    end

    @testset "hybrid search respects tag filters" begin
        store = Store(; embed=fake_embed, token_count=fake_token_count)
        load!(store, "alpha context for simulations"; id="science", tags=["science"])
        load!(store, "alpha context for relational database"; id="db", tags=["db"])

        science_results = search(store, "alpha context"; tags=["science"])
        @test !isempty(science_results)
        @test all(r -> r.id == "science", science_results)

        db_results = search(store, "alpha context"; tags=["db"])
        @test !isempty(db_results)
        @test all(r -> r.id == "db", db_results)
    end

    @testset "Jaccard similarity" begin
        a = LocalSearch.tokenize_for_mmr("hello world foo")
        b = LocalSearch.tokenize_for_mmr("hello world bar")
        sim = LocalSearch.jaccard_similarity(a, b)
        @test 0.0 < sim < 1.0  # partial overlap

        # Identical texts
        c = LocalSearch.tokenize_for_mmr("same text")
        @test LocalSearch.jaccard_similarity(c, c) == 1.0

        # No overlap
        d = LocalSearch.tokenize_for_mmr("alpha beta")
        e = LocalSearch.tokenize_for_mmr("gamma delta")
        @test LocalSearch.jaccard_similarity(d, e) == 0.0

        # Both empty
        @test LocalSearch.jaccard_similarity(Set{String}(), Set{String}()) == 1.0
    end

    @testset "MMR re-ranking" begin
        store = Store(; embed=nothing)
        load!(store, "Julia is a programming language for scientific computing"; id="julia1")
        load!(store, "Julia is a programming language for data science"; id="julia2")
        load!(store, "SQLite is a lightweight database engine"; id="sqlite")

        # Without MMR — both Julia docs at top
        results_plain = search(store, "programming language"; limit=3)
        @test length(results_plain) >= 2

        # With MMR — should still return results, first is most relevant
        results_mmr = search(store, "programming language"; limit=3, mmr=true, mmr_lambda=0.5)
        @test length(results_mmr) >= 2
        @test results_mmr[1].id in ("julia1", "julia2")

        # MMR with single result
        single = search(store, "database"; limit=1, mmr=true)
        @test length(single) == 1
        @test single[1].id == "sqlite"

        # MMR with empty results
        empty_r = search(store, "nonexistent_xyz_term"; mmr=true)
        @test isempty(empty_r)
    end

    @testset "default store eagerly inits embeddings" begin
        store = Store()
        buf = IOBuffer()
        show(buf, store)
        s = String(take!(buf))
        @test occursin("vectors=", s)
        @test !occursin("embed=pending", s)
    end
end
