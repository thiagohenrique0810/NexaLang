# Speculative decoding example
model "gpt2" {
  weights quantized int8
  kv_cache turboquant bits=4
  attention flash
  decode speculative drafter="gpt2" k=4
  scheduler continuous_batching max_batch=8 max_ctx=2048
  target auto
}
