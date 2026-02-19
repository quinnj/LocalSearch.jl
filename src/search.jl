# Search: BM25 (FTS5), vector similarity, and hybrid RRF blending

const RRF_K = 60

struct RankedItem
    key::String
    title::String
    body::String
    score::Float64
end

# --- MMR (Maximal Marginal Relevance) ---

"""Tokenize text into a set of lowercase alphanumeric tokens for Jaccard similarity."""
function tokenize_for_mmr(text::AbstractString)
    tokens = Set{String}()
    for m in eachmatch(r"[a-z0-9_]+", lowercase(text))
        push!(tokens, m.match)
    end
    return tokens
end

"""Jaccard similarity between two token sets. Returns value in [0, 1]."""
function jaccard_similarity(a::Set{String}, b::Set{String})
    isempty(a) && isempty(b) && return 1.0
    (isempty(a) || isempty(b)) && return 0.0
    intersection_size = length(intersect(a, b))
    union_size = length(a) + length(b) - intersection_size
    return union_size == 0 ? 0.0 : intersection_size / union_size
end

"""
    mmr_rerank(results; lambda=0.7)

Re-rank search results using Maximal Marginal Relevance.
Iteratively selects items maximizing `λ * relevance - (1-λ) * max_similarity_to_selected`.
"""
function mmr_rerank(results::Vector{SearchResult}; lambda::Float64=0.7)
    n = length(results)
    n <= 1 && return copy(results)
    lambda = clamp(lambda, 0.0, 1.0)

    # Pre-tokenize
    tokens = [tokenize_for_mmr(r.text) for r in results]

    # Normalize scores to [0, 1]
    scores = [r.score for r in results]
    min_s, max_s = extrema(scores)
    range_s = max_s - min_s
    norm(s) = range_s == 0.0 ? 1.0 : (s - min_s) / range_s

    selected = Int[]
    remaining = Set(1:n)

    while !isempty(remaining)
        best_idx = 0
        best_mmr = -Inf
        best_orig = -Inf

        for i in remaining
            rel = norm(results[i].score)
            max_sim = isempty(selected) ? 0.0 : maximum(jaccard_similarity(tokens[i], tokens[j]) for j in selected)
            mmr_score = lambda * rel - (1 - lambda) * max_sim
            if mmr_score > best_mmr || (mmr_score == best_mmr && results[i].score > best_orig)
                best_mmr = mmr_score
                best_idx = i
                best_orig = results[i].score
            end
        end

        best_idx == 0 && break
        push!(selected, best_idx)
        delete!(remaining, best_idx)
    end

    return [results[i] for i in selected]
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

function build_tag_filter_clause(tags::Vector{String})
    isempty(tags) && return "", Any[]
    placeholders = join(fill("?", length(tags)), ",")
    clause = " AND EXISTS (SELECT 1 FROM document_tags dt WHERE dt.document_id = d.id AND dt.tag IN ($placeholders))"
    return clause, Any[tags...]
end

function search_fts(store::Store, query::AbstractString; limit::Int=20, tags::Vector{String}=String[])
    fts_query = build_fts_query(query)
    fts_query === nothing && return SearchResult[]
    normalized_tags = normalize_tags(tags)
    tag_clause, tag_params = build_tag_filter_clause(normalized_tags)

    sql = """
    SELECT d.key, d.title, d.hash, c.body,
           bm25(documents_fts, 1.0, 5.0, 10.0) as rank
    FROM documents_fts f
    JOIN documents d ON d.id = f.rowid
    JOIN content c ON c.hash = d.hash
    WHERE documents_fts MATCH ? AND d.active = 1$tag_clause
    ORDER BY rank ASC
    LIMIT ?
    """
    params = Any[fts_query]
    append!(params, tag_params)
    push!(params, limit)
    rows = SQLite.DBInterface.execute(store.db, sql, Tuple(params))
    results = SearchResult[]
    for row in rows
        score = 1.0 / (1.0 + max(0.0, -Float64(row.rank)))
        push!(results, SearchResult(String(row.key), String(row.title), String(row.body), score, :fts))
    end
    return results
end

# --- Vector Search ---

function search_vec(store::Store, query::AbstractString; limit::Int=20, tags::Vector{String}=String[])
    has_vectors(store) || return SearchResult[]
    normalized_tags = normalize_tags(tags)

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
    tag_clause, tag_params = build_tag_filter_clause(normalized_tags)
    sql = """
    SELECT ch.hash || '_' || ch.seq as hash_seq,
           d.key, d.title, c.body
    FROM chunks ch
    JOIN documents d ON d.hash = ch.hash AND d.active = 1
    JOIN content c ON c.hash = d.hash
    WHERE ch.hash || '_' || ch.seq IN ($placeholders)$tag_clause
    """
    params = Any[hash_seqs...]
    append!(params, tag_params)
    doc_rows = SQLite.DBInterface.execute(store.db, sql, Tuple(params))

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
    search(store, query; limit=10, min_score=0.0, mmr=false, mmr_lambda=0.7, tags=nothing)

Search the store for documents matching `query`.

Returns a `Vector{SearchResult}` sorted by relevance. Uses BM25 full-text search
when no embedding function was provided, or hybrid BM25 + vector search with
Reciprocal Rank Fusion when embeddings are available.

When `mmr=true`, applies Maximal Marginal Relevance re-ranking to promote diversity.
When `tags` is provided, only documents with at least one matching tag are considered.
"""
function search(store::Store, query::AbstractString;
                limit::Int=10,
                min_score::Float64=0.0,
                mmr::Bool=false,
                mmr_lambda::Float64=0.7,
                tags=nothing)
    use_vectors = has_vectors(store)
    normalized_tags = normalize_tags(tags)

    fts_results = search_fts(store, query; limit=20, tags=normalized_tags)

    if !use_vectors
        results = filter(r -> r.score >= min_score, fts_results)
        results = results[1:min(limit, length(results))]
        return mmr ? mmr_rerank(results; lambda=mmr_lambda) : results
    end

    # Hybrid: BM25 + Vector with RRF blending
    vec_results = search_vec(store, query; limit=20, tags=normalized_tags)

    fts_ranked = [RankedItem(r.id, r.title, r.text, r.score) for r in fts_results]
    vec_ranked = [RankedItem(r.id, r.title, r.text, r.score) for r in vec_results]

    fused = reciprocal_rank_fusion([fts_ranked, vec_ranked]; weights=[2.0, 1.5])

    results = SearchResult[]
    for item in fused
        item.score < min_score && continue
        push!(results, SearchResult(item.key, item.title, item.body, item.score, :hybrid))
        length(results) >= limit && break
    end
    return mmr ? mmr_rerank(results; lambda=mmr_lambda) : results
end
