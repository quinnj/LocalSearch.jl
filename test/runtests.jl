using Test
using LocalSearch
using Random

@testset "LocalSearch" begin
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

    @testset "load! / update / delete / clear" begin
        store = Store(; embed=nothing)
        load!(store, "hello world"; id="a")
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
        using SQLite
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

        using SQLite
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
        load!(store, ["one fish", "two fish", "red fish"]; ids=["1", "2", "3"])
        results = search(store, "two fish")
        @test length(results) >= 1
        @test results[1].id == "2"
    end

    @testset "hybrid search with embeddings" begin
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

        store = Store(; embed=fake_embed)
        load!(store, "Julia is great for scientific computing"; id="julia")
        load!(store, "SQLite is a fast embedded database"; id="sqlite")

        results = search(store, "computing")
        @test length(results) >= 1
        @test all(r -> r.source == :hybrid, results)
    end

    @testset "chunking" begin
        # Large text should be chunked
        text = "word " ^ 1000  # ~5000 chars
        store = Store(; embed=nothing)
        load!(store, text; id="big")
        results = search(store, "word")
        @test length(results) == 1
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
