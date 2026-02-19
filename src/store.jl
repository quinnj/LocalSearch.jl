# Store: SQLite-backed storage with optional vector embeddings

# Helper: get first row from a query result, or nothing
function _first_or_nothing(result)
    x = iterate(result)
    return x === nothing ? nothing : x[1]
end

const DEFAULT_CHUNK_MAX_TOKENS = 900
const DEFAULT_CHUNK_OVERLAP_TOKENS = 135

# Sentinel for "use default embedding"
const _USE_DEFAULT_EMBED = :__default__

function approximate_token_count(text::AbstractString)
    n = 0
    for _ in eachmatch(r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]", text)
        n += 1
    end
    return n
end

function normalize_tags(tags::Nothing)
    return String[]
end

function normalize_tags(tag::AbstractString)
    return normalize_tags([tag])
end

function normalize_tags(tags::AbstractVector)
    seen = Set{String}()
    normalized = String[]
    for raw in tags
        tag = lowercase(strip(String(raw)))
        isempty(tag) && continue
        tag in seen && continue
        push!(normalized, tag)
        push!(seen, tag)
    end
    sort!(normalized)
    return normalized
end

function validate_chunk_params(max_tokens::Int, overlap_tokens::Int)
    max_tokens > 0 || throw(ArgumentError("chunk_max_tokens must be > 0"))
    overlap_tokens >= 0 || throw(ArgumentError("chunk_overlap_tokens must be >= 0"))
    overlap_tokens < max_tokens || throw(ArgumentError("chunk_overlap_tokens must be < chunk_max_tokens"))
    return nothing
end

mutable struct Store
    db::SQLite.DB
    embed::Union{Nothing, Function}  # (Vector{String}) -> Matrix{Float32} (dims × n)
    dimensions::Int
    vec_initialized::Bool
    token_count::Union{Nothing, Function}  # (AbstractString) -> Int
    chunk_max_tokens::Int
    chunk_overlap_tokens::Int
end

"""
    Store(path=":memory:"; embed=:default, token_count=nothing, chunk_max_tokens=900, chunk_overlap_tokens=135)

Create a new search store backed by SQLite.

- `path`: SQLite database path, or `":memory:"` for in-memory (default)
- `embed`: embedding function, or a symbol/nothing to control behavior:
  - `:default` (default) — use built-in llama.cpp embeddings (downloads model on first call, ~300MB one-time)
  - a function `(Vector{String}) -> Matrix{Float32}` — custom embedding function
  - `nothing` — BM25-only mode, no vector search
- `token_count`: optional token counting function `(text) -> Int`; defaults to model tokenization for `:default` embeddings
- `chunk_max_tokens`: max tokens per embedded chunk
- `chunk_overlap_tokens`: overlap tokens between adjacent chunks

# Examples
```julia
# Default: hybrid BM25 + vector search with built-in embeddings
store = Store()

# BM25-only (no model download, no vector search)
store = Store(; embed=nothing)

# Custom embedding function
store = Store(; embed = texts -> my_model(texts))

# Persistent storage
store = Store("/path/to/index.sqlite")
```
"""
function Store(path::AbstractString=":memory:";
               embed=_USE_DEFAULT_EMBED,
               token_count::Union{Nothing,Function}=nothing,
               chunk_max_tokens::Int=DEFAULT_CHUNK_MAX_TOKENS,
               chunk_overlap_tokens::Int=DEFAULT_CHUNK_OVERLAP_TOKENS)
    db = SQLite.DB(path)
    configure_db!(db)
    return Store(db; embed, token_count, chunk_max_tokens, chunk_overlap_tokens)
end

function Store(db::SQLite.DB;
               embed=_USE_DEFAULT_EMBED,
               token_count::Union{Nothing,Function}=nothing,
               chunk_max_tokens::Int=DEFAULT_CHUNK_MAX_TOKENS,
               chunk_overlap_tokens::Int=DEFAULT_CHUNK_OVERLAP_TOKENS)
    validate_chunk_params(chunk_max_tokens, chunk_overlap_tokens)
    init_schema!(db)

    embed_fn = if embed === _USE_DEFAULT_EMBED
        Embed.default_embed
    elseif embed === nothing || embed === false
        nothing
    else
        embed
    end

    token_count_fn = if embed_fn === nothing
        nothing
    elseif token_count !== nothing
        token_count
    elseif embed_fn === Embed.default_embed
        Embed.default_token_count
    else
        approximate_token_count
    end

    store = Store(db, embed_fn, 0, false, token_count_fn, chunk_max_tokens, chunk_overlap_tokens)

    # Eagerly init: download model + create vectors table now
    if embed_fn !== nothing
        dims = if embed_fn === Embed.default_embed
            Embed.ensure_init!()
        else
            test = embed_fn(["test"])
            size(test, 1)
        end
        store.dimensions = dims
        init_vectors!(store)
    end

    return store
end

Base.close(store::Store) = SQLite.close(store.db)

function Base.show(io::IO, store::Store)
    row = _first_or_nothing(SQLite.DBInterface.execute(store.db, "SELECT COUNT(*) as n FROM documents WHERE active = 1"))
    n = row === nothing ? 0 : row.n
    vec_str = if store.vec_initialized
        ", vectors=$(store.dimensions)d"
    elseif store.embed !== nothing
        ", embed=pending"
    else
        ""
    end
    print(io, "LocalSearch.Store($(n) documents$(vec_str))")
end

# --- DB Setup ---

function configure_db!(db::SQLite.DB)
    for pragma in (
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA foreign_keys=ON",
        "PRAGMA busy_timeout=5000",
        "PRAGMA temp_store=MEMORY",
    )
        SQLite.execute(db, pragma)
    end
end

function init_schema!(db::SQLite.DB)
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS content (
            hash TEXT PRIMARY KEY,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
        )
    """)

    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            hash TEXT NOT NULL REFERENCES content(hash),
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            active INTEGER NOT NULL DEFAULT 1
        )
    """)
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)")
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS idx_documents_active ON documents(active)")
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS document_tags (
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            tag TEXT NOT NULL,
            PRIMARY KEY (document_id, tag)
        )
    """)
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag)")
    SQLite.execute(db, "CREATE INDEX IF NOT EXISTS idx_document_tags_document_id ON document_tags(document_id)")

    SQLite.execute(db, """
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            key, title, body,
            content='',
            tokenize='porter unicode61'
        )
    """)

    # Keep FTS in sync with documents table
    SQLite.execute(db, """
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, key, title, body)
            SELECT new.id, new.key, new.title, c.body
            FROM content c WHERE c.hash = new.hash;
        END
    """)
    SQLite.execute(db, """
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, key, title, body)
            SELECT 'delete', old.id, old.key, old.title, c.body
            FROM content c WHERE c.hash = old.hash;
        END
    """)
    SQLite.execute(db, """
        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE OF key, title, hash ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, key, title, body)
            SELECT 'delete', old.id, old.key, old.title, c.body
            FROM content c WHERE c.hash = old.hash;
            INSERT INTO documents_fts(rowid, key, title, body)
            SELECT new.id, new.key, new.title, c.body
            FROM content c WHERE c.hash = new.hash;
        END
    """)

    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS chunks (
            hash TEXT NOT NULL,
            seq INTEGER NOT NULL,
            pos INTEGER NOT NULL,
            embedded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
            PRIMARY KEY (hash, seq)
        )
    """)
end

function init_vectors!(store::Store)
    store.vec_initialized && return
    load_sqlite_vec!(store.db)
    dims = store.dimensions
    SQLite.execute(store.db, """
        CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
            hash_seq TEXT PRIMARY KEY,
            embedding float[$dims] distance_metric=cosine
        )
    """)
    store.vec_initialized = true
end

function load_sqlite_vec!(db::SQLite.DB)
    path = try
        Libdl.dlpath(sqlite_vec_jll.libvec0)
    catch
        string(sqlite_vec_jll.libvec0)
    end
    SQLite.enable_load_extension(db, true)
    errmsg = Ref{Ptr{Cchar}}(C_NULL)
    rc = SQLite.C.sqlite3_load_extension(db.handle, path, C_NULL, errmsg)
    SQLite.enable_load_extension(db, false)
    if rc != SQLite.C.SQLITE_OK
        err = errmsg[] == C_NULL ? "unknown error" : unsafe_string(errmsg[])
        errmsg[] != C_NULL && SQLite.C.sqlite3_free(errmsg[])
        error("Failed to load sqlite-vec extension: $err")
    end
end

function has_vectors(store::Store)
    store.vec_initialized || return false
    row = _first_or_nothing(SQLite.DBInterface.execute(store.db,
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'"))
    return row !== nothing
end

# --- load! ---

"""
    load!(store, text; id=nothing, title="")

Index a string for search. Returns the store for chaining.

- `id`: unique key for this document (auto-generated from content hash if omitted)
- `title`: optional title (boosts title-matching in search)
- `tags`: optional metadata tags to attach to this document
- `chunk_max_tokens`: optional per-call token chunk limit override
- `chunk_overlap_tokens`: optional per-call token overlap override

If a document with the same `id` already exists, it is replaced.
"""
function load!(store::Store, text::AbstractString;
               id::Union{Nothing,AbstractString}=nothing,
               title::AbstractString="",
               tags=nothing,
               chunk_max_tokens::Union{Nothing,Int}=nothing,
               chunk_overlap_tokens::Union{Nothing,Int}=nothing)
    hash = bytes2hex(sha256(text))
    key = id === nothing ? hash[1:16] : String(id)
    normalized_tags = normalize_tags(tags)
    max_tokens, overlap_tokens = resolve_chunk_params(store, chunk_max_tokens, chunk_overlap_tokens)

    # Dedup content by hash
    SQLite.execute(store.db, "INSERT OR IGNORE INTO content(hash, body) VALUES (?, ?)", (hash, String(text)))

    # Check for existing document with same key
    existing = _first_or_nothing(SQLite.DBInterface.execute(store.db,
        "SELECT id, hash, title FROM documents WHERE key = ?", (key,)))

    if existing !== nothing
        existing_hash = String(existing.hash)
        existing_id = Int(existing.id)
        same_hash = existing_hash == hash
        same_title = String(existing.title) == String(title)
        same_tags = document_tags(store.db, existing_id) == normalized_tags
        same_hash && same_title && same_tags && return store
        if same_hash
            same_title || SQLite.execute(store.db,
                "UPDATE documents SET title = ? WHERE key = ?",
                (String(title), key))
            replace_document_tags!(store.db, existing_id, normalized_tags)
            return store
        end
        Base.delete!(store, key)
    end

    SQLite.execute(store.db,
        "INSERT INTO documents(key, title, hash) VALUES (?, ?, ?)",
        (key, String(title), hash))
    inserted = _first_or_nothing(SQLite.DBInterface.execute(store.db,
        "SELECT id FROM documents WHERE key = ?", (key,)))
    inserted === nothing && error("Failed to insert document key=$key")
    replace_document_tags!(store.db, Int(inserted.id), normalized_tags)

    if store.embed !== nothing
        embed_content!(store, hash, text; max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    end

    return store
end

function resolve_chunk_params(store::Store, chunk_max_tokens::Union{Nothing,Int}, chunk_overlap_tokens::Union{Nothing,Int})
    max_tokens = chunk_max_tokens === nothing ? store.chunk_max_tokens : chunk_max_tokens
    overlap_tokens = chunk_overlap_tokens === nothing ? store.chunk_overlap_tokens : chunk_overlap_tokens
    validate_chunk_params(max_tokens, overlap_tokens)
    return max_tokens, overlap_tokens
end

function document_tags(db::SQLite.DB, document_id::Integer)
    rows = SQLite.DBInterface.execute(db,
        "SELECT tag FROM document_tags WHERE document_id = ? ORDER BY tag",
        (Int(document_id),))
    tags = String[]
    for row in rows
        push!(tags, String(row.tag))
    end
    return tags
end

function replace_document_tags!(db::SQLite.DB, document_id::Integer, tags::Vector{String})
    SQLite.execute(db, "DELETE FROM document_tags WHERE document_id = ?", (Int(document_id),))
    for tag in tags
        SQLite.execute(db, "INSERT INTO document_tags(document_id, tag) VALUES (?, ?)", (Int(document_id), tag))
    end
    return nothing
end

"""
    load!(store, texts; ids=nothing, titles=nothing, tags=nothing, chunk_max_tokens=nothing, chunk_overlap_tokens=nothing)

Batch-index multiple strings.
"""
function load!(store::Store, texts::AbstractVector{<:AbstractString};
               ids::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
               titles::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
               tags::Union{Nothing,AbstractVector}=nothing,
               chunk_max_tokens::Union{Nothing,Int}=nothing,
               chunk_overlap_tokens::Union{Nothing,Int}=nothing)
    ids !== nothing && length(ids) != length(texts) && throw(ArgumentError("ids length must match texts length"))
    titles !== nothing && length(titles) != length(texts) && throw(ArgumentError("titles length must match texts length"))
    tags !== nothing && length(tags) != length(texts) && throw(ArgumentError("tags length must match texts length"))
    for (i, text) in enumerate(texts)
        kw_id = ids !== nothing ? ids[i] : nothing
        kw_title = titles !== nothing ? titles[i] : ""
        kw_tags = if tags === nothing
            nothing
        else
            current = tags[i]
            current isa AbstractString ? [String(current)] : current
        end
        load!(store, text;
            id=kw_id,
            title=kw_title,
            tags=kw_tags,
            chunk_max_tokens=chunk_max_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens)
    end
    return store
end

function embed_content!(store::Store, hash::AbstractString, text::AbstractString; max_tokens::Int, overlap_tokens::Int)
    # Skip if already embedded (shared content across documents)
    existing = _first_or_nothing(SQLite.DBInterface.execute(store.db,
        "SELECT COUNT(*) as n FROM chunks WHERE hash = ?", (String(hash),)))
    existing !== nothing && existing.n > 0 && return

    chunks = chunk_text_by_tokens(store, String(text); max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    texts = [c.text for c in chunks]
    embeddings = store.embed(texts)  # dims × n_chunks

    for (i, chunk) in enumerate(chunks)
        seq = i - 1
        hash_seq = "$(hash)_$(seq)"
        vec = Float32.(embeddings[:, i])
        SQLite.execute(store.db,
            "INSERT OR REPLACE INTO chunks(hash, seq, pos) VALUES (?, ?, ?)",
            (String(hash), seq, chunk.pos))
        SQLite.execute(store.db,
            "INSERT OR REPLACE INTO vectors(hash_seq, embedding) VALUES (?, ?)",
            (hash_seq, reinterpret(UInt8, vec)))
    end
end

# --- delete! / clear! ---

"""
    delete!(store, id)

Remove a document by its key. Fully cleans up content, chunks, and vectors
if no other documents share the same content.
"""
function Base.delete!(store::Store, id::AbstractString)
    # Get the hash before deleting the document
    row = _first_or_nothing(SQLite.DBInterface.execute(store.db,
        "SELECT hash FROM documents WHERE key = ?", (String(id),)))
    row === nothing && return store

    hash = String(row.hash)

    # Delete document (triggers FTS cleanup)
    SQLite.execute(store.db, "DELETE FROM documents WHERE key = ?", (String(id),))

    # Clean up orphaned content/chunks/vectors if no other documents reference this hash
    remaining = _first_or_nothing(SQLite.DBInterface.execute(store.db,
        "SELECT COUNT(*) as n FROM documents WHERE hash = ?", (hash,)))

    if remaining !== nothing && remaining.n == 0
        # Remove vectors (must happen before chunks, since we need chunk seqs)
        if store.vec_initialized
            seqs = Int[]
            for crow in SQLite.DBInterface.execute(store.db,
                "SELECT seq FROM chunks WHERE hash = ?", (hash,))
                push!(seqs, Int(crow.seq))
            end
            for seq in seqs
                hash_seq = "$(hash)_$(seq)"
                SQLite.execute(store.db, "DELETE FROM vectors WHERE hash_seq = ?", (hash_seq,))
            end
        end
        SQLite.execute(store.db, "DELETE FROM chunks WHERE hash = ?", (hash,))
        SQLite.execute(store.db, "DELETE FROM content WHERE hash = ?", (hash,))
    end

    return store
end

"""
    clear!(store)

Remove all documents, content, and vectors from the store.
"""
function clear!(store::Store)
    SQLite.execute(store.db, "DELETE FROM documents")
    SQLite.execute(store.db, "DELETE FROM document_tags")
    SQLite.execute(store.db, "DELETE FROM content")
    SQLite.execute(store.db, "DELETE FROM chunks")
    if store.vec_initialized
        dims = store.dimensions
        SQLite.execute(store.db, "DROP TABLE IF EXISTS vectors")
        SQLite.execute(store.db, """
            CREATE VIRTUAL TABLE vectors USING vec0(
                hash_seq TEXT PRIMARY KEY,
                embedding float[$dims] distance_metric=cosine
            )
        """)
    end
    return store
end

# --- Chunking ---

function count_tokens(store::Store, text::AbstractString)
    store.token_count === nothing && throw(ArgumentError("Token counter is not configured for this store"))
    n = Int(store.token_count(text))
    return n < 0 ? 0 : n
end

function find_chunk_end_by_tokens(store::Store, chars::Vector{Char}, start_idx::Int, last_idx::Int, max_tokens::Int)
    lo = start_idx
    hi = last_idx
    best = start_idx

    while lo <= hi
        mid = (lo + hi) >>> 1
        current = count_tokens(store, String(chars[start_idx:mid]))
        if current <= max_tokens
            best = mid
            lo = mid + 1
        else
            hi = mid - 1
        end
    end

    return best
end

function find_overlap_start_by_tokens(store::Store, chars::Vector{Char}, chunk_start::Int, chunk_end::Int, overlap_tokens::Int)
    overlap_tokens == 0 && return chunk_end + 1

    lo = chunk_start
    hi = chunk_end
    best = chunk_end + 1

    while lo <= hi
        mid = (lo + hi) >>> 1
        current = count_tokens(store, String(chars[mid:chunk_end]))
        if current <= overlap_tokens
            best = mid
            hi = mid - 1
        else
            lo = mid + 1
        end
    end

    return best <= chunk_end ? best : (chunk_end + 1)
end

function chunk_text_by_tokens(store::Store, text::AbstractString;
                              max_tokens::Int=store.chunk_max_tokens,
                              overlap_tokens::Int=store.chunk_overlap_tokens)
    validate_chunk_params(max_tokens, overlap_tokens)
    body = String(text)
    isempty(body) && return [TextChunk("", 0)]
    total_tokens = count_tokens(store, body)
    total_tokens <= max_tokens && return [TextChunk(body, 0)]

    chars = collect(body)
    n = length(chars)
    chunks = TextChunk[]
    start_idx = 1

    while start_idx <= n
        end_idx = find_chunk_end_by_tokens(store, chars, start_idx, n, max_tokens)
        push!(chunks, TextChunk(String(chars[start_idx:end_idx]), start_idx - 1))
        end_idx >= n && break

        next_idx = overlap_tokens == 0 ? (end_idx + 1) :
            find_overlap_start_by_tokens(store, chars, start_idx, end_idx, overlap_tokens)
        next_idx <= start_idx && (next_idx = start_idx + 1)
        start_idx = next_idx
    end

    return chunks
end
