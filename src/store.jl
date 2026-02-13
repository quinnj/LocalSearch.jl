# Store: SQLite-backed storage with optional vector embeddings

# Helper: get first row from a query result, or nothing
function _first_or_nothing(result)
    x = iterate(result)
    return x === nothing ? nothing : x[1]
end

const CHUNK_MAX_CHARS = 3200
const CHUNK_OVERLAP_CHARS = 480

# Sentinel for "use default embedding"
const _USE_DEFAULT_EMBED = :__default__

mutable struct Store
    db::SQLite.DB
    embed::Union{Nothing, Function}  # (Vector{String}) -> Matrix{Float32} (dims × n)
    dimensions::Int
    vec_initialized::Bool
end

"""
    Store(path=":memory:"; embed=:default)

Create a new search store backed by SQLite.

- `path`: SQLite database path, or `":memory:"` for in-memory (default)
- `embed`: embedding function, or a symbol/nothing to control behavior:
  - `:default` (default) — use built-in llama.cpp embeddings (downloads model on first call, ~300MB one-time)
  - a function `(Vector{String}) -> Matrix{Float32}` — custom embedding function
  - `nothing` — BM25-only mode, no vector search

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
function Store(path::AbstractString=":memory:"; embed=_USE_DEFAULT_EMBED)
    db = SQLite.DB(path)
    configure_db!(db)
    init_schema!(db)

    embed_fn = if embed === _USE_DEFAULT_EMBED
        Embed.default_embed
    elseif embed === nothing || embed === false
        nothing
    else
        embed
    end

    store = Store(db, embed_fn, 0, false)

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

If a document with the same `id` already exists, it is replaced.
"""
function load!(store::Store, text::AbstractString; id::Union{Nothing,AbstractString}=nothing, title::AbstractString="")
    hash = bytes2hex(sha256(text))
    key = id === nothing ? hash[1:16] : String(id)

    # Dedup content by hash
    SQLite.execute(store.db, "INSERT OR IGNORE INTO content(hash, body) VALUES (?, ?)", (hash, String(text)))

    # Check for existing document with same key
    existing = _first_or_nothing(SQLite.DBInterface.execute(store.db,
        "SELECT id, hash FROM documents WHERE key = ?", (key,)))

    if existing !== nothing
        String(existing.hash) == hash && return store  # unchanged, no-op
        SQLite.execute(store.db, "DELETE FROM documents WHERE key = ?", (key,))
    end

    SQLite.execute(store.db,
        "INSERT INTO documents(key, title, hash) VALUES (?, ?, ?)",
        (key, String(title), hash))

    if store.embed !== nothing
        embed_content!(store, hash, text)
    end

    return store
end

"""
    load!(store, texts; ids=nothing, titles=nothing)

Batch-index multiple strings.
"""
function load!(store::Store, texts::AbstractVector{<:AbstractString};
               ids::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
               titles::Union{Nothing,AbstractVector{<:AbstractString}}=nothing)
    for (i, text) in enumerate(texts)
        kw_id = ids !== nothing ? ids[i] : nothing
        kw_title = titles !== nothing ? titles[i] : ""
        load!(store, text; id=kw_id, title=kw_title)
    end
    return store
end

function embed_content!(store::Store, hash::AbstractString, text::AbstractString)
    # Skip if already embedded (shared content across documents)
    existing = _first_or_nothing(SQLite.DBInterface.execute(store.db,
        "SELECT COUNT(*) as n FROM chunks WHERE hash = ?", (String(hash),)))
    existing !== nothing && existing.n > 0 && return

    chunks = chunk_text(String(text))
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
            chunk_rows = collect(SQLite.DBInterface.execute(store.db,
                "SELECT seq FROM chunks WHERE hash = ?", (hash,)))
            for crow in chunk_rows
                hash_seq = "$(hash)_$(crow.seq)"
                try
                    SQLite.execute(store.db, "DELETE FROM vectors WHERE hash_seq = ?", (hash_seq,))
                catch
                    # vec0 virtual table may not support all DELETE forms; safe to skip
                end
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

function chunk_text(text::AbstractString; max_chars::Int=CHUNK_MAX_CHARS, overlap_chars::Int=CHUNK_OVERLAP_CHARS)
    chars = collect(text)
    n = length(chars)
    n <= max_chars && return [TextChunk(String(text), 0)]

    chunks = TextChunk[]
    pos = 1
    while pos <= n
        end_pos = min(pos + max_chars - 1, n)

        if end_pos < n
            brk = find_natural_break(chars, pos, end_pos)
            brk > 0 && (end_pos = brk)
        end

        push!(chunks, TextChunk(String(chars[pos:end_pos]), pos - 1))
        end_pos >= n && break

        next = end_pos - overlap_chars + 1
        next <= pos && (next = end_pos + 1)
        pos = next
    end
    return chunks
end

function find_natural_break(chars::Vector{Char}, start::Int, stop::Int)
    search_from = max(start, start + floor(Int, (stop - start + 1) * 0.7))

    # Paragraph break
    for i in stop:-1:search_from
        i > start && chars[i] == '\n' && chars[i-1] == '\n' && return i
    end
    # Sentence end
    for i in stop:-1:search_from
        i < stop && chars[i] in ('.', '?', '!') && chars[i+1] in (' ', '\n') && return i + 1
    end
    # Line break
    for i in stop:-1:search_from
        chars[i] == '\n' && return i
    end
    # Word break
    for i in stop:-1:search_from
        chars[i] == ' ' && return i
    end

    return 0
end
