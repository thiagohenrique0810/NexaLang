# Maximum compression: INT4 weights + TurboQuant KV cache (3-bit)
model "gpt2" {
  weights quantized int4
  kv_cache turboquant bits=3
  attention flash streaming=true
  decode autoregressive
  scheduler continuous_batching max_batch=32 max_ctx=4096
  target auto
}
