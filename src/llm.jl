# Minimal embedding support via llama.cpp
# Ported from Qmd's LLM module - only the embedding path

module Embed

using Downloads
using ..LlamaCppNative

const DEFAULT_MODEL_URI = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
const EMBED_THREADS = 6

# --- Model download ---

function models_dir()
    dir = joinpath(first(Base.DEPOT_PATH), "localsearch", "models")
    mkpath(dir)
    return dir
end

function resolve_model(uri::AbstractString; verbose::Bool=true)
    if startswith(uri, "hf:")
        parts = split(uri[4:end], '/'; limit=3)
        length(parts) < 3 && error("Invalid hf URI: $uri")
        filename = String(parts[3])
        local_path = joinpath(models_dir(), filename)
        if !isfile(local_path)
            url = "https://huggingface.co/$(parts[1])/$(parts[2])/resolve/main/$(parts[3])"
            verbose && @info "Downloading embedding model (~300MB, one-time)..." url
            tmp = local_path * ".tmp"
            try
                Downloads.download(url, tmp)
                mv(tmp, local_path; force=true)
            catch e
                isfile(tmp) && rm(tmp; force=true)
                rethrow(e)
            end
        end
        return local_path
    end
    path = abspath(expanduser(String(uri)))
    isfile(path) || error("Model not found: $uri")
    return path
end

# --- Cached model state ---

const _state = Ref{Union{Nothing, @NamedTuple{
    model::Ptr{Cvoid}, ctx::Ptr{Cvoid}, dims::Int, n_batch::Int, uri::String
}}}(nothing)
const _state_lock = ReentrantLock()

function get_state(; model_uri::AbstractString=DEFAULT_MODEL_URI)
    lock(_state_lock) do
        s = _state[]
        s !== nothing && s.uri == model_uri && return s

        # Clean up old state
        if s !== nothing
            LlamaCppNative.free_context(s.ctx)
            LlamaCppNative.free_model(s.model)
            _state[] = nothing
        end

        model_path = resolve_model(model_uri)

        state = redirect_stderr(devnull) do
            LlamaCppNative.ensure_backend_init()

            base_mparams = LlamaCppNative.model_default_params()
            model = LlamaCppNative.load_model(model_path, base_mparams)
            model == C_NULL && error("Failed to load embedding model: $model_path")

            dims = Int(LlamaCppNative.n_embd(model))
            train_ctx = Int(LlamaCppNative.n_ctx_train(model))
            n_ctx = train_ctx > 0 ? train_ctx : 4096

            base_cparams = LlamaCppNative.context_default_params()
            cparams = LlamaCppNative.llama_context_params(
                UInt32(n_ctx),       # n_ctx
                UInt32(n_ctx),       # n_batch
                UInt32(n_ctx),       # n_ubatch
                UInt32(1),           # n_seq_max
                Int32(EMBED_THREADS),
                Int32(EMBED_THREADS),
                base_cparams.rope_scaling_type,
                LlamaCppNative.LLAMA_POOLING_TYPE_UNSPECIFIED,
                LlamaCppNative.LLAMA_ATTENTION_TYPE_UNSPECIFIED,
                base_cparams.flash_attn_type,
                base_cparams.rope_freq_base,
                base_cparams.rope_freq_scale,
                base_cparams.yarn_ext_factor,
                base_cparams.yarn_attn_factor,
                base_cparams.yarn_beta_fast,
                base_cparams.yarn_beta_slow,
                base_cparams.yarn_orig_ctx,
                base_cparams.defrag_thold,
                base_cparams.cb_eval,
                base_cparams.cb_eval_user_data,
                base_cparams.type_k,
                base_cparams.type_v,
                base_cparams.abort_callback,
                base_cparams.abort_callback_data,
                true,                # embeddings = true
                base_cparams.offload_kqv,
                base_cparams.no_perf,
                base_cparams.op_offload,
                false,               # swa_full
                base_cparams.kv_unified,
                base_cparams.samplers,
                base_cparams.n_samplers,
            )

            ctx = LlamaCppNative.new_context(model, cparams)
            if ctx == C_NULL
                LlamaCppNative.free_model(model)
                error("Failed to create embedding context")
            end

            (model=model, ctx=ctx, dims=dims, n_batch=n_ctx, uri=String(model_uri))
        end

        _state[] = state
        return state
    end
end

"""
    ensure_init!(; model_uri) -> Int

Download the model (if needed) and load it, returning the embedding dimensions.
Called by `Store()` to eagerly initialize the embedding backend.
"""
function ensure_init!(; model_uri::AbstractString=DEFAULT_MODEL_URI)
    state = get_state(; model_uri=model_uri)
    return state.dims
end

# --- Token preparation (matches Qmd's logic) ---

function begin_token_to_prepend(model::Ptr{Cvoid})
    vtype = LlamaCppNative.vocab_type(model)
    vtype == LlamaCppNative.LLAMA_VOCAB_TYPE_RWKV && return nothing
    if vtype == LlamaCppNative.LLAMA_VOCAB_TYPE_WPM
        token = LlamaCppNative.vocab_bos(model)
        return token < 0 ? nothing : token
    end
    vtype == LlamaCppNative.LLAMA_VOCAB_TYPE_UGM && return nothing
    LlamaCppNative.vocab_get_add_bos(model) || return nothing
    token = LlamaCppNative.vocab_bos(model)
    return token < 0 ? nothing : token
end

function end_token_to_append(model::Ptr{Cvoid})
    vtype = LlamaCppNative.vocab_type(model)
    vtype == LlamaCppNative.LLAMA_VOCAB_TYPE_RWKV && return nothing
    if vtype == LlamaCppNative.LLAMA_VOCAB_TYPE_WPM
        token = LlamaCppNative.vocab_sep(model)
        return token < 0 ? nothing : token
    end
    if vtype == LlamaCppNative.LLAMA_VOCAB_TYPE_UGM
        token = LlamaCppNative.vocab_eos(model)
        return token < 0 ? nothing : token
    end
    LlamaCppNative.vocab_get_add_eos(model) || return nothing
    token = LlamaCppNative.vocab_eos(model)
    return token < 0 ? nothing : token
end

function embed_tokens(model::Ptr{Cvoid}, text::AbstractString)
    tokens = LlamaCppNative.tokenize(model, text; add_special=false, parse_special=false)
    isempty(tokens) && return tokens
    bos = begin_token_to_prepend(model)
    if bos !== nothing
        tokens[1] == bos || pushfirst!(tokens, bos)
    end
    eos = end_token_to_append(model)
    if eos !== nothing
        tokens[end] == eos || push!(tokens, eos)
    end
    return tokens
end

# --- Core embedding function ---

"""
    default_embed(texts::Vector{String}) -> Matrix{Float32}

Embed texts using the default llama.cpp model (embeddinggemma-300M).
Returns a `dims × length(texts)` Float32 matrix.
Downloads the model on first use (~300MB, one-time).
"""
function default_embed(texts::Vector{String}; model::AbstractString=DEFAULT_MODEL_URI)
    state = get_state(; model_uri=model)
    dims = state.dims
    result = zeros(Float32, dims, length(texts))

    redirect_stderr(devnull) do
        for (i, text) in enumerate(texts)
            tokens = embed_tokens(state.model, text)
            isempty(tokens) && continue

            LlamaCppNative.kv_cache_clear(state.ctx)
            rc = LlamaCppNative.decode_batch(state.ctx, tokens; start_pos=0, seq_id=0, logits=false)
            rc < 0 && continue

            pooling = LlamaCppNative.pooling_type(state.ctx)
            embedding = if pooling == LlamaCppNative.LLAMA_POOLING_TYPE_NONE
                LlamaCppNative.embeddings_ith(state.ctx, length(tokens) - 1, dims)
            else
                LlamaCppNative.embeddings_seq(state.ctx, 0, dims)
            end

            embedding !== nothing && (result[:, i] = embedding)
        end
    end

    return result
end

function __init__()
    atexit() do
        s = _state[]
        if s !== nothing
            try
                redirect_stderr(devnull) do
                    LlamaCppNative.free_context(s.ctx)
                    LlamaCppNative.free_model(s.model)
                end
            catch
            end
            _state[] = nothing
        end
    end
end

end # module Embed
