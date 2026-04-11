# TurboLang LLM Runtime — Arquitetura

## Visão Geral

```
┌─────────────┐     ┌──────────┐     ┌───────────┐     ┌──────────────┐
│  .tl file   │────>│ TurboLang│────>│  TurboIR  │────>│  Exec Plan   │
│  (DSL)      │     │  Parser  │     │  Compiler │     │  (Runtime)   │
└─────────────┘     └──────────┘     └───────────┘     └──────┬───────┘
                                                              │
                    ┌─────────────────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        TurboRuntime                                  │
│  ┌────────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │Model Loader│  │ Quantizer │  │ KV Cache │  │   Scheduler      │ │
│  │(HF/GGUF)  │  │(INT4/INT8)│  │+TurboQ   │  │(Continuous Batch)│ │
│  └─────┬──────┘  └─────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│        └───────────┬────┘             │                 │           │
│                    ▼                  ▼                 ▼           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Inference Engine (Prefill + Decode)             │    │
│  │         TurboKernels (Attention, MatMul, Fused Ops)         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         TurboServe (FastAPI)                         │
│  POST /generate  │  POST /chat  │  GET /metrics  │  POST /compile   │
└──────────────────────────────────────────────────────────────────────┘
```

## Fluxo de Compilação

1. **Lexer** tokeniza o arquivo `.tl`
2. **Parser** constrói a AST (Abstract Syntax Tree)
3. **Validator** verifica semântica (tipos, campos obrigatórios, constraints)
4. **IR Generator** converte AST → TurboIR (representação intermediária)
5. **Planner** transforma TurboIR → Execution Plan (configuração concreta do runtime)

## Fluxo de Inferência

1. **Model Loader** carrega pesos do modelo (HuggingFace Transformers ou safetensors)
2. **Quantizer** aplica quantização (INT4/INT8) nos pesos se configurado
3. **KV Cache** inicializa cache com compressão TurboQuant se habilitado
4. **Prefill** processa o prompt inteiro em uma passada
5. **Decode** gera tokens autoregressivamente
6. **Scheduler** gerencia batching contínuo de múltiplas requisições

## Decisões Técnicas

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Linguagem principal | Python | Integração direta com PyTorch/HF ecosystem |
| API framework | FastAPI | Async nativo, tipagem forte, OpenAPI auto |
| Model format | HuggingFace | Maior compatibilidade, fácil troca de modelo |
| Quantização | Abseil-free INT4/INT8 | Sem dependências externas pesadas |
| KV Cache compress | TurboQuant-style | Reduz VRAM 2-4x com perda mínima de qualidade |
| GPU kernels | PyTorch native + Triton opt | Funcional primeiro, otimizado depois |

## Riscos

- Modelos grandes podem não caber na VRAM local → fallback CPU
- Quantização INT4 pode degradar qualidade → benchmark comparativo
- Triton kernels requerem GPU NVIDIA → fallback PyTorch puro

## Próximos Passos (V2)

- Fine-tuning support
- Multi-GPU (tensor parallelism)
- Speculative decoding com draft model real  
- GGUF format support
- Prometheus metrics export
- MoE (Mixture of Experts) support
