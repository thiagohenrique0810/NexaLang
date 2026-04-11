# TurboLang LLM Runtime — Roadmap

## V1.0 (Atual)
- [x] DSL TurboLang com parser completo
- [x] TurboIR e plano de execução
- [x] Carregamento de modelos HuggingFace
- [x] Quantização INT4/INT8
- [x] KV Cache com compressão TurboQuant
- [x] Geração de texto (prefill + decode)
- [x] Continuous batching
- [x] Speculative decoding básico
- [x] API HTTP (FastAPI)
- [x] Benchmark suite
- [x] Docker

## V1.1 (Próximo)
- [ ] Streaming SSE no endpoint /generate
- [ ] YAML config alternativo à DSL
- [ ] Endpoint de profiling
- [ ] Métricas Prometheus
- [ ] Tracing básico (OpenTelemetry)

## V2.0 (Futuro)
- [ ] Multi-GPU (tensor parallelism)
- [ ] Fine-tuning loop
- [ ] GGUF model format
- [ ] Flash Attention 2 via Triton
- [ ] Mixture of Experts (MoE)
- [ ] Knowledge distillation pipeline
- [ ] Custom backend NexaLang
