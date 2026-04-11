# Quantized model with INT8 weights
model "gpt2" {
  weights quantized int8
  kv_cache standard
  attention flash streaming=true
  decode autoregressive
  scheduler continuous_batching max_batch=16 max_ctx=2048
  target auto
}
