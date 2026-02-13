# Search: BM25 (FTS5), vector similarity, and hybrid RRF blending

const RRF_K = 60

struct RankedItem
    key::String
    title::String
    body::String
    score::Float64
end

# --- BM25 / FTS5 ---

function build_fts_query(query::AbstractString)
    terms = split(String(query))
    sanitized = String[]
    for term in terms
        clean = replace(term, r"[^\p{L}\p{N}']" => "")
        isempty(clean) && continue
        push!(sanitized, lowercase(clean))
    end
    isempty(sanitized) && return nothing
    length(sanitized) == 1 && return "\"$(sanitized[1])\"*"
    return join(["\"$t\"*" for t in sanitized], " AND ")
end

function search_fts(store::Store, query::AbstractString; limit::Int=20)
    fts_query = build_fts_query(query)
    fts_query === nothing && return SearchResult[]

    sql = """
    SELECT d.key, d.title, d.hash, c.body,
           bm25(documents_fts, 1.0, 5.0, 10.0) as rank
    FROM documents_fts f
    JOIN documents d ON d.id = f.rowid
    JOIN content c ON c.hash = d.hash
    WHERE documents_fts MATCH ? AND d.active = 1
    ORDER BY rank ASC
    LIMIT ?
    """
    rows = SQLite.DBInterface.execute(store.db, sql, (fts_query, limit))
    results = SearchResult[]
    for row in rows
        score = 1.0 / (1.0 + max(0.0, -Float64(row.rank)))
        push!(results, SearchResult(String(row.key), String(row.title), String(row.body), score, :fts))
    end
    return results
end

# --- Vector Search ---

function search_vec(store::Store, query::AbstractString; limit::Int=20)
    has_vectors(store) || return SearchResult[]

    query_embedding = store.embed([query])[:, 1]
    vec = Float32.(query_embedding)

    vec_rows = SQLite.DBInterface.execute(store.db,
        "SELECT hash_seq, distance FROM vectors WHERE embedding MATCH ? AND k = ?",
        (reinterpret(UInt8, vec), limit * 3))

    hash_seqs = String[]
    distances = Dict{String, Float64}()
    for row in vec_rows
        hs = String(row.hash_seq)
        push!(hash_seqs, hs)
        distances[hs] = Float64(row.distance)
    end
    isempty(hash_seqs) && return SearchResult[]

    placeholders = join(fill("?", length(hash_seqs)), ",")
    sql = """
    SELECT ch.hash || '_' || ch.seq as hash_seq,
           d.key, d.title, c.body
    FROM chunks ch
    JOIN documents d ON d.hash = ch.hash AND d.active = 1
    JOIN content c ON c.hash = d.hash
    WHERE ch.hash || '_' || ch.seq IN ($placeholders)
    """
    doc_rows = SQLite.DBInterface.execute(store.db, sql, Tuple(hash_seqs))

    # Keep best (closest) match per document
    seen = Dict{String, Tuple{SearchResult, Float64}}()
    for row in doc_rows
        hs = String(row.hash_seq)
        dist = get(distances, hs, 1.0)
        key = String(row.key)
        existing = get(seen, key, nothing)
        if existing === nothing || dist < existing[2]
            r = SearchResult(key, String(row.title), String(row.body), 1.0 - dist, :vec)
            seen[key] = (r, dist)
        end
    end

    pairs = sort!(collect(values(seen)); by=x -> x[2])
    return [r for (r, _) in pairs[1:min(limit, length(pairs))]]
end

# --- Reciprocal Rank Fusion ---

function reciprocal_rank_fusion(lists::Vector{Vector{RankedItem}}; weights::Vector{Float64}=Float64[], k::Int=RRF_K)
    scores = Dict{String, Tuple{RankedItem, Float64, Int}}()

    for (list_idx, list) in enumerate(lists)
        isempty(list) && continue
        w = list_idx <= length(weights) ? weights[list_idx] : 1.0
        for (rank, item) in enumerate(list)
            contribution = w / (k + rank)
            existing = get(scores, item.key, nothing)
            if existing === nothing
                scores[item.key] = (item, contribution, rank)
            else
                scores[item.key] = (existing[1], existing[2] + contribution, min(existing[3], rank))
            end
        end
    end

    fused = RankedItem[]
    for (item, score, top_rank) in values(scores)
        # Small bonus for top-ranked items
        adjusted = score
        top_rank == 1 && (adjusted += 0.05)
        top_rank <= 3 && top_rank > 1 && (adjusted += 0.02)
        push!(fused, RankedItem(item.key, item.title, item.body, adjusted))
    end
    sort!(fused; by=r -> r.score, rev=true)
    return fused
end

# --- Main Search ---

"""
    search(store, query; limit=10, min_score=0.0)

Search the store for documents matching `query`.

Returns a `Vector{SearchResult}` sorted by relevance. Uses BM25 full-text search
when no embedding function was provided, or hybrid BM25 + vector search with
Reciprocal Rank Fusion when embeddings are available.
"""
function search(store::Store, query::AbstractString; limit::Int=10, min_score::Float64=0.0)
    use_vectors = has_vectors(store)

    fts_results = search_fts(store, query; limit=20)

    if !use_vectors
        results = filter(r -> r.score >= min_score, fts_results)
        return results[1:min(limit, length(results))]
    end

    # Hybrid: BM25 + Vector with RRF blending
    vec_results = search_vec(store, query; limit=20)

    fts_ranked = [RankedItem(r.id, r.title, r.text, r.score) for r in fts_results]
    vec_ranked = [RankedItem(r.id, r.title, r.text, r.score) for r in vec_results]

    fused = reciprocal_rank_fusion([fts_ranked, vec_ranked]; weights=[2.0, 1.5])

    results = SearchResult[]
    for item in fused
        item.score < min_score && continue
        push!(results, SearchResult(item.key, item.title, item.body, item.score, :hybrid))
        length(results) >= limit && break
    end
    return results
end
