module LocalSearch

using SQLite
using SHA
using Dates
import sqlite_vec_jll
import llama_cpp_jll
import Libdl

export Store, load!, search, clear!

"""
    SearchResult

A single search result with the matched document's key, title, full text, relevance score, and source method.
"""
struct SearchResult
    id::String
    title::String
    text::String
    score::Float64
    source::Symbol  # :fts, :vec, :hybrid
end

function Base.show(io::IO, r::SearchResult)
    id_display = length(r.id) > 20 ? r.id[1:20] * "..." : r.id
    print(io, "SearchResult($(repr(id_display)), score=$(round(r.score; digits=4)), source=$(r.source))")
end

struct TextChunk
    text::String
    pos::Int  # character offset in original text
end

include("llama_cpp_native.jl")
include("llm.jl")
include("store.jl")
include("search.jl")

end
