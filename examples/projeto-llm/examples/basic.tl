# Basic TurboLang example — GPT-2 with default settings
model "gpt2" {
  weights full fp16
  kv_cache standard
  attention standard
  decode autoregressive
  scheduler static max_batch=1 max_ctx=1024
  target auto
}
