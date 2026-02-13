module LlamaCppNative

using llama_cpp_jll

const libllama = llama_cpp_jll.libllama

const llama_token = Int32
const llama_pos = Int32
const llama_seq_id = Int32
const ggml_log_level = Cint

const LLAMA_POOLING_TYPE_UNSPECIFIED = Int32(-1)
const LLAMA_POOLING_TYPE_NONE = Int32(0)
const LLAMA_POOLING_TYPE_MEAN = Int32(1)
const LLAMA_POOLING_TYPE_CLS = Int32(2)
const LLAMA_POOLING_TYPE_LAST = Int32(3)
const LLAMA_POOLING_TYPE_RANK = Int32(4)

const LLAMA_ATTENTION_TYPE_UNSPECIFIED = Int32(-1)
const LLAMA_ATTENTION_TYPE_CAUSAL = Int32(0)
const LLAMA_ATTENTION_TYPE_NON_CAUSAL = Int32(1)

const LLAMA_FLASH_ATTN_TYPE_AUTO = Int32(-1)
const LLAMA_FLASH_ATTN_TYPE_DISABLED = Int32(0)
const LLAMA_FLASH_ATTN_TYPE_ENABLED = Int32(1)

const LLAMA_SPLIT_MODE_NONE = Int32(0)
const LLAMA_SPLIT_MODE_LAYER = Int32(1)
const LLAMA_SPLIT_MODE_ROW = Int32(2)

const LLAMA_VOCAB_TYPE_NONE = Int32(0)
const LLAMA_VOCAB_TYPE_SPM = Int32(1)
const LLAMA_VOCAB_TYPE_BPE = Int32(2)
const LLAMA_VOCAB_TYPE_WPM = Int32(3)
const LLAMA_VOCAB_TYPE_UGM = Int32(4)
const LLAMA_VOCAB_TYPE_RWKV = Int32(5)
const LLAMA_VOCAB_TYPE_PLAMO2 = Int32(6)

const LLAMA_KV_OVERRIDE_TYPE_INT = Int32(0)
const LLAMA_KV_OVERRIDE_TYPE_FLOAT = Int32(1)
const LLAMA_KV_OVERRIDE_TYPE_BOOL = Int32(2)
const LLAMA_KV_OVERRIDE_TYPE_STR = Int32(3)

const GGML_LOG_LEVEL_NONE = ggml_log_level(0)
const GGML_LOG_LEVEL_DEBUG = ggml_log_level(1)
const GGML_LOG_LEVEL_INFO = ggml_log_level(2)
const GGML_LOG_LEVEL_WARN = ggml_log_level(3)
const GGML_LOG_LEVEL_ERROR = ggml_log_level(4)
const GGML_LOG_LEVEL_CONT = ggml_log_level(5)

const LLAMA_PARAMS_FIT_STATUS_SUCCESS = Int32(0)
const LLAMA_PARAMS_FIT_STATUS_FAILURE = Int32(1)
const LLAMA_PARAMS_FIT_STATUS_ERROR = Int32(2)

const _default_log_callback_ptr = Ref{Ptr{Cvoid}}(C_NULL)

function default_log_callback()
    return _default_log_callback_ptr[]
end

struct llama_model_kv_override
    tag::Int32
    key::NTuple{128,UInt8}
    _pad::Int32
    value::NTuple{16,Int64}
end

struct llama_model_tensor_buft_override
    pattern::Ptr{Cchar}
    buft::Ptr{Cvoid}
end

struct llama_model_params
    devices::Ptr{Cvoid}
    tensor_buft_overrides::Ptr{llama_model_tensor_buft_override}
    n_gpu_layers::Int32
    split_mode::Int32
    main_gpu::Int32
    tensor_split::Ptr{Cfloat}
    progress_callback::Ptr{Cvoid}
    progress_callback_user_data::Ptr{Cvoid}
    kv_overrides::Ptr{llama_model_kv_override}
    vocab_only::Bool
    use_mmap::Bool
    use_direct_io::Bool
    use_mlock::Bool
    check_tensors::Bool
    use_extra_bufts::Bool
    no_host::Bool
    no_alloc::Bool
end

struct llama_context_params
    n_ctx::UInt32
    n_batch::UInt32
    n_ubatch::UInt32
    n_seq_max::UInt32
    n_threads::Int32
    n_threads_batch::Int32
    rope_scaling_type::Int32
    pooling_type::Int32
    attention_type::Int32
    flash_attn_type::Int32
    rope_freq_base::Cfloat
    rope_freq_scale::Cfloat
    yarn_ext_factor::Cfloat
    yarn_attn_factor::Cfloat
    yarn_beta_fast::Cfloat
    yarn_beta_slow::Cfloat
    yarn_orig_ctx::UInt32
    defrag_thold::Cfloat
    cb_eval::Ptr{Cvoid}
    cb_eval_user_data::Ptr{Cvoid}
    type_k::Int32
    type_v::Int32
    abort_callback::Ptr{Cvoid}
    abort_callback_data::Ptr{Cvoid}
    embeddings::Bool
    offload_kqv::Bool
    no_perf::Bool
    op_offload::Bool
    swa_full::Bool
    kv_unified::Bool
    samplers::Ptr{Cvoid}
    n_samplers::Csize_t
end

struct llama_sampler_chain_params
    no_perf::Bool
end

struct llama_batch
    n_tokens::Int32
    token::Ptr{llama_token}
    embd::Ptr{Cfloat}
    pos::Ptr{llama_pos}
    n_seq_id::Ptr{Int32}
    seq_id::Ptr{Ptr{llama_seq_id}}
    logits::Ptr{Int8}
end


const _backend_initialized = Ref(false)
const _backend_lock = ReentrantLock()

function ensure_backend_init()
    if !_backend_initialized[]
        lock(_backend_lock) do
            if !_backend_initialized[]
                ccall((:llama_backend_init, libllama), Cvoid, ())
                _backend_initialized[] = true
            end
        end
    end
    return nothing
end

function supports_gpu_offload()
    ensure_backend_init()
    return ccall((:llama_supports_gpu_offload, libllama), Bool, ())
end

function max_devices()
    ensure_backend_init()
    return ccall((:llama_max_devices, libllama), Csize_t, ())
end

function max_tensor_buft_overrides()
    ensure_backend_init()
    return ccall((:llama_max_tensor_buft_overrides, libllama), Csize_t, ())
end

function set_log_callback(callback::Ptr{Cvoid}, user_data::Ptr{Cvoid}=C_NULL)
    ensure_backend_init()
    ccall((:llama_log_set, libllama), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), callback, user_data)
    return nothing
end

function model_default_params()
    ensure_backend_init()
    return ccall((:llama_model_default_params, libllama), llama_model_params, ())
end

function context_default_params()
    ensure_backend_init()
    return ccall((:llama_context_default_params, libllama), llama_context_params, ())
end

function pooling_type(ctx::Ptr{Cvoid})
    ctx == C_NULL && return LLAMA_POOLING_TYPE_NONE
    return ccall((:llama_pooling_type, libllama), Int32, (Ptr{Cvoid},), ctx)
end

function sampler_chain_default_params()
    ensure_backend_init()
    return ccall((:llama_sampler_chain_default_params, libllama), llama_sampler_chain_params, ())
end

function load_model(path::AbstractString, params::llama_model_params)
    ensure_backend_init()
    return ccall((:llama_load_model_from_file, libllama), Ptr{Cvoid}, (Cstring, llama_model_params), path, params)
end

function params_fit(path::AbstractString, mparams::Ref{llama_model_params}, cparams::Ref{llama_context_params}; n_ctx_min::Integer=0, log_level::ggml_log_level=GGML_LOG_LEVEL_ERROR)
    ensure_backend_init()
    n_devices = max(Int(max_devices()), 1)
    tensor_split = fill(Cfloat(0), n_devices)
    margins = fill(Csize_t(0), n_devices)
    n_overrides = max(Int(max_tensor_buft_overrides()), 1)
    overrides = Vector{llama_model_tensor_buft_override}(undef, n_overrides)
    fill!(overrides, llama_model_tensor_buft_override(C_NULL, C_NULL))
    GC.@preserve tensor_split margins overrides mparams cparams begin
        return ccall((:llama_params_fit, libllama), Int32,
            (Cstring, Ref{llama_model_params}, Ref{llama_context_params}, Ptr{Cfloat}, Ptr{llama_model_tensor_buft_override}, Ptr{Csize_t}, UInt32, ggml_log_level),
            path, mparams, cparams, pointer(tensor_split), pointer(overrides), pointer(margins), UInt32(n_ctx_min), log_level)
    end
end

function free_model(model::Ptr{Cvoid})
    model == C_NULL && return nothing
    ccall((:llama_free_model, libllama), Cvoid, (Ptr{Cvoid},), model)
    return nothing
end

function new_context(model::Ptr{Cvoid}, params::llama_context_params)
    ensure_backend_init()
    return ccall((:llama_new_context_with_model, libllama), Ptr{Cvoid}, (Ptr{Cvoid}, llama_context_params), model, params)
end

function free_context(ctx::Ptr{Cvoid})
    ctx == C_NULL && return nothing
    ccall((:llama_free, libllama), Cvoid, (Ptr{Cvoid},), ctx)
    return nothing
end

function kv_cache_clear(ctx::Ptr{Cvoid})
    ctx == C_NULL && return nothing
    mem = ccall((:llama_get_memory, libllama), Ptr{Cvoid}, (Ptr{Cvoid},), ctx)
    mem == C_NULL && return nothing
    ccall((:llama_memory_clear, libllama), Cvoid, (Ptr{Cvoid}, Bool), mem, true)
    return nothing
end

function set_threads(ctx::Ptr{Cvoid}, n_threads::Integer, n_threads_batch::Integer)
    ctx == C_NULL && return nothing
    ccall((:llama_set_n_threads, libllama), Cvoid, (Ptr{Cvoid}, Int32, Int32), ctx, Int32(n_threads), Int32(n_threads_batch))
    return nothing
end

function model_has_encoder(model::Ptr{Cvoid})
    return ccall((:llama_model_has_encoder, libllama), Bool, (Ptr{Cvoid},), model)
end

function model_has_decoder(model::Ptr{Cvoid})
    return ccall((:llama_model_has_decoder, libllama), Bool, (Ptr{Cvoid},), model)
end

function n_embd(model::Ptr{Cvoid})
    return ccall((:llama_model_n_embd, libllama), Int32, (Ptr{Cvoid},), model)
end

function n_ctx_train(model::Ptr{Cvoid})
    return ccall((:llama_model_n_ctx_train, libllama), Int32, (Ptr{Cvoid},), model)
end

function model_vocab(model::Ptr{Cvoid})
    return ccall((:llama_model_get_vocab, libllama), Ptr{Cvoid}, (Ptr{Cvoid},), model)
end

function vocab_type(model::Ptr{Cvoid})
    vocab = model_vocab(model)
    vocab == C_NULL && return LLAMA_VOCAB_TYPE_NONE
    return ccall((:llama_vocab_type, libllama), Int32, (Ptr{Cvoid},), vocab)
end

function vocab_bos(model::Ptr{Cvoid})
    vocab = model_vocab(model)
    vocab == C_NULL && return llama_token(-1)
    return ccall((:llama_vocab_bos, libllama), llama_token, (Ptr{Cvoid},), vocab)
end

function vocab_eos(model::Ptr{Cvoid})
    vocab = model_vocab(model)
    vocab == C_NULL && return llama_token(-1)
    return ccall((:llama_vocab_eos, libllama), llama_token, (Ptr{Cvoid},), vocab)
end

function vocab_sep(model::Ptr{Cvoid})
    vocab = model_vocab(model)
    vocab == C_NULL && return llama_token(-1)
    return ccall((:llama_vocab_sep, libllama), llama_token, (Ptr{Cvoid},), vocab)
end

function vocab_get_add_bos(model::Ptr{Cvoid})
    vocab = model_vocab(model)
    vocab == C_NULL && return false
    return ccall((:llama_vocab_get_add_bos, libllama), Bool, (Ptr{Cvoid},), vocab)
end

function vocab_get_add_eos(model::Ptr{Cvoid})
    vocab = model_vocab(model)
    vocab == C_NULL && return false
    return ccall((:llama_vocab_get_add_eos, libllama), Bool, (Ptr{Cvoid},), vocab)
end

function token_is_eog(model::Ptr{Cvoid}, token::llama_token)
    vocab = model_vocab(model)
    vocab == C_NULL && return false
    return ccall((:llama_vocab_is_eog, libllama), Bool, (Ptr{Cvoid}, llama_token), vocab, token)
end

function tokenize(model::Ptr{Cvoid}, text::AbstractString; add_special::Bool=true, parse_special::Bool=false)
    input = String(text)
    text_len = Int32(sizeof(input))
    capacity = max(8, Int(text_len) + 8)
    tokens = Vector{llama_token}(undef, capacity)
    vocab = model_vocab(model)
    vocab == C_NULL && return llama_token[]
    count = ccall((:llama_tokenize, libllama), Int32, (Ptr{Cvoid}, Cstring, Int32, Ptr{llama_token}, Int32, Bool, Bool), vocab, input, text_len, tokens, Int32(capacity), add_special, parse_special)
    if count < 0
        capacity = Int(-count)
        resize!(tokens, capacity)
        count = ccall((:llama_tokenize, libllama), Int32, (Ptr{Cvoid}, Cstring, Int32, Ptr{llama_token}, Int32, Bool, Bool), vocab, input, text_len, tokens, Int32(capacity), add_special, parse_special)
    end
    count < 0 && return llama_token[]
    resize!(tokens, Int(count))
    return tokens
end

function detokenize(model::Ptr{Cvoid}, tokens::Vector{llama_token}; remove_special::Bool=true, unparse_special::Bool=false)
    isempty(tokens) && return ""
    buf_len = max(64, length(tokens) * 4)
    buffer = Vector{UInt8}(undef, buf_len)
    vocab = model_vocab(model)
    vocab == C_NULL && return ""
    count = ccall((:llama_detokenize, libllama), Int32, (Ptr{Cvoid}, Ptr{llama_token}, Int32, Ptr{UInt8}, Int32, Bool, Bool), vocab, tokens, Int32(length(tokens)), buffer, Int32(buf_len), remove_special, unparse_special)
    while count < 0
        buf_len = Int(-count)
        resize!(buffer, buf_len)
        count = ccall((:llama_detokenize, libllama), Int32, (Ptr{Cvoid}, Ptr{llama_token}, Int32, Ptr{UInt8}, Int32, Bool, Bool), vocab, tokens, Int32(length(tokens)), buffer, Int32(buf_len), remove_special, unparse_special)
    end
    count <= 0 && return ""
    return String(copy(buffer[1:Int(count)]))
end

function make_batch(tokens::Vector{llama_token}; logits::Bool=false)
    n_tokens = Int32(length(tokens))
    token_ptr = pointer(tokens)
    logits_buf = logits ? fill(Int8(1), length(tokens)) : Int8[]
    logits_ptr = logits ? pointer(logits_buf) : Ptr{Int8}(C_NULL)
    batch = llama_batch(n_tokens, token_ptr, C_NULL, C_NULL, C_NULL, C_NULL, logits_ptr)
    return batch, logits_buf
end

function encode(ctx::Ptr{Cvoid}, tokens::Vector{llama_token}; logits::Bool=false)
    batch, logits_buf = make_batch(tokens; logits=logits)
    GC.@preserve tokens logits_buf begin
        return ccall((:llama_encode, libllama), Int32, (Ptr{Cvoid}, llama_batch), ctx, batch)
    end
end

function decode(ctx::Ptr{Cvoid}, tokens::Vector{llama_token}; logits::Bool=false)
    batch, logits_buf = make_batch(tokens; logits=logits)
    GC.@preserve tokens logits_buf begin
        return ccall((:llama_decode, libllama), Int32, (Ptr{Cvoid}, llama_batch), ctx, batch)
    end
end

function decode_batch(ctx::Ptr{Cvoid}, tokens::Vector{llama_token}; start_pos::Integer=0, seq_id::Integer=0, logits::Bool=false)
    n_tokens = length(tokens)
    n_tokens == 0 && return Int32(0)
    positions = Vector{llama_pos}(undef, n_tokens)
    seq_counts = Vector{Int32}(undef, n_tokens)
    seq_ids = Vector{llama_seq_id}(undef, n_tokens)
    seq_ptrs = Vector{Ptr{llama_seq_id}}(undef, n_tokens)
    for i in 1:n_tokens
        positions[i] = llama_pos(start_pos + i - 1)
        seq_counts[i] = Int32(1)
        seq_ids[i] = llama_seq_id(seq_id)
        seq_ptrs[i] = pointer(seq_ids, i)
    end
    logits_buf = logits ? fill(Int8(1), n_tokens) : Int8[]
    logits_ptr = logits ? pointer(logits_buf) : Ptr{Int8}(C_NULL)
    batch = llama_batch(Int32(n_tokens), pointer(tokens), C_NULL, pointer(positions), pointer(seq_counts), pointer(seq_ptrs), logits_ptr)
    GC.@preserve tokens positions seq_counts seq_ids seq_ptrs logits_buf begin
        return ccall((:llama_decode, libllama), Int32, (Ptr{Cvoid}, llama_batch), ctx, batch)
    end
end

function encode_batch(ctx::Ptr{Cvoid}, tokens::Vector{llama_token}; start_pos::Integer=0, seq_id::Integer=0, logits::Bool=false)
    n_tokens = length(tokens)
    n_tokens == 0 && return Int32(0)
    positions = Vector{llama_pos}(undef, n_tokens)
    seq_counts = Vector{Int32}(undef, n_tokens)
    seq_ids = Vector{llama_seq_id}(undef, n_tokens)
    seq_ptrs = Vector{Ptr{llama_seq_id}}(undef, n_tokens)
    for i in 1:n_tokens
        positions[i] = llama_pos(start_pos + i - 1)
        seq_counts[i] = Int32(1)
        seq_ids[i] = llama_seq_id(seq_id)
        seq_ptrs[i] = pointer(seq_ids, i)
    end
    logits_buf = logits ? fill(Int8(1), n_tokens) : Int8[]
    logits_ptr = logits ? pointer(logits_buf) : Ptr{Int8}(C_NULL)
    batch = llama_batch(Int32(n_tokens), pointer(tokens), C_NULL, pointer(positions), pointer(seq_counts), pointer(seq_ptrs), logits_ptr)
    GC.@preserve tokens positions seq_counts seq_ids seq_ptrs logits_buf begin
        return ccall((:llama_encode, libllama), Int32, (Ptr{Cvoid}, llama_batch), ctx, batch)
    end
end

function decode_multi(ctx::Ptr{Cvoid}, tokens::Vector{llama_token}, positions::Vector{llama_pos}, seq_ids::Vector{llama_seq_id}; logits::Bool=false)
    n_tokens = length(tokens)
    n_tokens == 0 && return Int32(0)
    length(positions) == n_tokens || return Int32(-1)
    length(seq_ids) == n_tokens || return Int32(-1)
    seq_counts = fill(Int32(1), n_tokens)
    seq_ptrs = Vector{Ptr{llama_seq_id}}(undef, n_tokens)
    for i in 1:n_tokens
        seq_ptrs[i] = pointer(seq_ids, i)
    end
    logits_buf = logits ? fill(Int8(1), n_tokens) : Int8[]
    logits_ptr = logits ? pointer(logits_buf) : Ptr{Int8}(C_NULL)
    batch = llama_batch(Int32(n_tokens), pointer(tokens), C_NULL, pointer(positions), pointer(seq_counts), pointer(seq_ptrs), logits_ptr)
    GC.@preserve tokens positions seq_counts seq_ids seq_ptrs logits_buf begin
        return ccall((:llama_decode, libllama), Int32, (Ptr{Cvoid}, llama_batch), ctx, batch)
    end
end

function memory_seq_rm(ctx::Ptr{Cvoid}, seq_id::Integer, p0::Integer, p1::Integer)
    ctx == C_NULL && return false
    mem = ccall((:llama_get_memory, libllama), Ptr{Cvoid}, (Ptr{Cvoid},), ctx)
    mem == C_NULL && return false
    return ccall((:llama_memory_seq_rm, libllama), Bool, (Ptr{Cvoid}, llama_seq_id, llama_pos, llama_pos), mem, llama_seq_id(seq_id), llama_pos(p0), llama_pos(p1))
end

function embeddings_seq(ctx::Ptr{Cvoid}, seq_id::Integer, n_embd::Integer)
    ptr = ccall((:llama_get_embeddings_seq, libllama), Ptr{Cfloat}, (Ptr{Cvoid}, llama_seq_id), ctx, llama_seq_id(seq_id))
    ptr == C_NULL && return nothing
    data = unsafe_wrap(Vector{Float32}, Ptr{Float32}(ptr), Int(n_embd))
    return copy(data)
end

function embeddings_ith(ctx::Ptr{Cvoid}, idx::Integer, n_embd::Integer)
    ptr = ccall((:llama_get_embeddings_ith, libllama), Ptr{Cfloat}, (Ptr{Cvoid}, Int32), ctx, Int32(idx))
    ptr == C_NULL && return nothing
    data = unsafe_wrap(Vector{Float32}, Ptr{Float32}(ptr), Int(n_embd))
    return copy(data)
end

function sampler_chain_init(params::llama_sampler_chain_params)
    return ccall((:llama_sampler_chain_init, libllama), Ptr{Cvoid}, (llama_sampler_chain_params,), params)
end

function sampler_chain_add(chain::Ptr{Cvoid}, sampler::Ptr{Cvoid})
    ccall((:llama_sampler_chain_add, libllama), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), chain, sampler)
    return nothing
end

function sampler_init_greedy()
    return ccall((:llama_sampler_init_greedy, libllama), Ptr{Cvoid}, ())
end

function sampler_init_dist(seed::UInt32)
    return ccall((:llama_sampler_init_dist, libllama), Ptr{Cvoid}, (UInt32,), seed)
end

function sampler_init_top_k(k::Integer)
    return ccall((:llama_sampler_init_top_k, libllama), Ptr{Cvoid}, (Int32,), Int32(k))
end

function sampler_init_top_p(p::Real, min_keep::Integer)
    return ccall((:llama_sampler_init_top_p, libllama), Ptr{Cvoid}, (Cfloat, UInt64), Cfloat(p), UInt64(min_keep))
end

function sampler_init_min_p(p::Real, min_keep::Integer)
    return ccall((:llama_sampler_init_min_p, libllama), Ptr{Cvoid}, (Cfloat, UInt64), Cfloat(p), UInt64(min_keep))
end

function sampler_init_temp(temp::Real)
    return ccall((:llama_sampler_init_temp, libllama), Ptr{Cvoid}, (Cfloat,), Cfloat(temp))
end

function sampler_init_grammar(model::Ptr{Cvoid}, grammar::AbstractString, root::AbstractString)
    vocab = ccall((:llama_model_get_vocab, libllama), Ptr{Cvoid}, (Ptr{Cvoid},), model)
    vocab == C_NULL && return C_NULL
    return ccall((:llama_sampler_init_grammar, libllama), Ptr{Cvoid}, (Ptr{Cvoid}, Cstring, Cstring), vocab, grammar, root)
end

function sampler_sample(chain::Ptr{Cvoid}, ctx::Ptr{Cvoid}, idx::Integer)
    return ccall((:llama_sampler_sample, libllama), llama_token, (Ptr{Cvoid}, Ptr{Cvoid}, Int32), chain, ctx, Int32(idx))
end

function sampler_accept(chain::Ptr{Cvoid}, token::llama_token)
    ccall((:llama_sampler_accept, libllama), Cvoid, (Ptr{Cvoid}, llama_token), chain, token)
    return nothing
end

function sampler_free(chain::Ptr{Cvoid})
    chain == C_NULL && return nothing
    ccall((:llama_sampler_free, libllama), Cvoid, (Ptr{Cvoid},), chain)
    return nothing
end

function kv_override_str(key::AbstractString, value::AbstractString)
    key_buf = fill(UInt8(0), 128)
    key_bytes = collect(codeunits(String(key)))
    key_len = min(length(key_bytes), 127)
    copyto!(key_buf, 1, key_bytes, 1, key_len)
    value_buf = fill(UInt8(0), 128)
    value_bytes = collect(codeunits(String(value)))
    value_len = min(length(value_bytes), 127)
    copyto!(value_buf, 1, value_bytes, 1, value_len)
    return llama_model_kv_override(LLAMA_KV_OVERRIDE_TYPE_STR, Tuple(key_buf), 0, Tuple(reinterpret(Int64, value_buf)))
end

function kv_override_null()
    return llama_model_kv_override(LLAMA_KV_OVERRIDE_TYPE_INT, ntuple(_ -> UInt8(0), 128), 0, ntuple(_ -> Int64(0), 16))
end

end
