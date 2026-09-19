# NexaLM: arquitetura, importação local e pacote de pesos

Este incremento implementa as definições estruturais do plano da primeira LLM e
a importação local de pesos Llama compatíveis. O pacote já fornece pesos ao
[forward Transformer CPU](NEXALM_EXECUCAO_CPU.md), validado em fixtures pequenas.
Treinamento e geração de texto com tokenizer permanecem no
[checklist](BLUEPRINT_512MB_CHECKLIST.md).

## Arquiteturas canônicas

[`architecture.nxl`](../models/nexalm512/architecture.nxl) é a fonte das definições
R0/v1. Os JSONs em `models/nexalm512/configs/` são fixtures verificadas contra essa
fonte. Os parâmetros compartilhados são contados uma única vez.

| Modelo | Camadas | Hidden | Heads Q/KV | FFN | Parâmetros físicos |
|---|---:|---:|---:|---:|---:|
| NexaLM512_R0 | 16 | 768 | 12/4 | 2048 | 125.854.464 |
| NexaLM512_v1 | 32 | 1024 | 16/4 | 2816 | 394.331.136 |

Ambos declaram vocabulário de 32.768, contexto de 2.048, head dimension de 64,
RMSNorm, RoPE, SwiGLU e embedding compartilhado com a projeção de saída.

```sh
python3 tools/nexa_model.py models/nexalm512/architecture.nxl --out artifacts/reports/nexalm-architecture.json
```

O frontend reutiliza o lexer Nexa, com parser estrutural separado do `nxc`. A saída
contém configuração, shapes, aliases, contagem e SHA-256 da fonte. A opção
`--sequence-length` acrescenta o grafo completo de prefill e lifetimes de ativações.
Integração ao compilador da linguagem e migração do TurboIR seguem pendentes em G0.

## Importação de um checkpoint local

A pasta de entrada deve conter `config.json` e uma destas formas:

- `model.safetensors`;
- `model.safetensors.index.json` e os shards referenciados pelo índice.

São aceitos F32, F16 e BF16 little-endian. O leitor valida o cabeçalho, cobertura
exata dos bytes, nomes, shapes e índice dos shards antes da conversão. Lê os valores
em chunks de até 64 KiB, com um shard aberto por vez. Não depende de PyTorch,
Transformers ou da biblioteca Safetensors, nem executa código do checkpoint.

O adaptador suporta a família Llama sem biases, com RMSNorm, GQA/MHA, RoPE de cabeça
inteira sem scaling e SwiGLU. Configurações desconhecidas, tensores ausentes ou
extras, shapes incompatíveis, pesos não finitos, biases e extensões arquiteturais
não suportadas são rejeitados. Defaults omitidos no config Hugging Face seguem
Llama, incluindo `tie_word_embeddings=false` e `rms_norm_eps=1e-6`.

```sh
# --out deve ser uma pasta nova.
python3 tools/nexa_convert.py --checkpoint /caminho/checkpoint --out artifacts/models/meu-modelo --group-size 32 --block-rows 64
python3 tools/nexa_inspect.py artifacts/models/meu-modelo
python3 tools/nexa_inspect.py artifacts/models/meu-modelo --verify
```

Matrizes são quantizadas em Q4_GROUPED; vetores de normalização permanecem F32.
Quando o config declara pesos compartilhados, `lm_head.weight` é um alias para
`model.embed_tokens.weight`. Se ambos vierem no checkpoint, seus valores decodificados
precisam ser iguais antes de eliminar a cópia. Sem tying, a projeção é física.

`config.json` e os arquivos opcionais `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json` e `tokenizer.model` são preservados com checksums. Os
assets são opacos: preservá-los ainda não implementa tokenização. A proveniência
registra hashes dos arquivos de origem completos; essa etapa exige uma leitura
sequencial adicional dos pesos. Alterações detectadas na fonte durante a operação
invalidam a importação, incluindo assets alterados entre hashing e cópia.

## Contrato do pacote

O diretório `NexaModelBundle`, versão 1, contém `manifest.json`, `tensors/` e
`assets/`. O manifesto registra configuração normalizada, nomes, shapes, codecs,
caminhos relativos, tamanhos, checksums, aliases e proveniência. Matrizes usam
arquivos [NexaPack V1](NEXAPACK_V1.md) inalterados; vetores usam RAW_F32 little-endian.
Esse diretório ainda não é o futuro executável `.nxb` com plano e kernels.

O writer monta uma pasta temporária e publica o resultado completo por rename
exclusivo. Um destino existente não é sobrescrito. O reader rejeita caminhos fora
da pasta e symlinks, valida configurações e cabeçalhos e resolve aliases explícitos.
Os payloads de pesos são verificados ao serem lidos. Assets são pequenos e seus
checksums são verificados na abertura.

Limites: manifesto de 1 MiB, proveniência de 64 KiB, até 4.096 tensores físicos,
vetores F32 de até 16 MiB, até 16 assets com 16 MiB por arquivo e 32 MiB no total.
Também se aplicam os limites de matriz/blocos do formato V1. O leitor Safetensors
limita cabeçalhos a 16 MiB, quantidade de tensores a 100.000 e shards a 4.096.

A inspeção padrão lê metadados e assets, sem carregar payloads de pesos. `--verify`
percorre todos os tensores físicos e valida checksums e valores do codec. O relatório
expõe shapes, codecs, bits, grupos e bytes por tensor. A taxa de compressão compara
FP32 com o payload packed, incluindo escalas; o tamanho físico com cabeçalhos é
informado separadamente. Essa validação não mede qualidade do modelo.

## Reprodução pequena, sem download

O fixture contém uma camada não treinada, 728 parâmetros, 11 tensores físicos e
12 nomes incluindo o alias da saída. Dois shards misturam F32/F16/BF16 e incluem
um tokenizer de teste. Use pastas novas para os dois primeiros comandos; em uma
retomada com artefatos existentes, execute apenas inspeção e benchmark.

```sh
python3 tests/model_checkpoint_fixture.py --out artifacts/checkpoints/nexalm-tiny
python3 tools/nexa_convert.py --checkpoint artifacts/checkpoints/nexalm-tiny --out artifacts/models/nexalm-tiny --group-size 4 --block-rows 3
python3 tools/nexa_inspect.py artifacts/models/nexalm-tiny --verify
python3 tools/nexa_bench.py --bundle artifacts/models/nexalm-tiny --tensor lm_head.weight --batch 2 --tile-rows 3 --memory-budget 96KiB --verify --report artifacts/reports/nexalm-bundle-q4.json
```

O benchmark usa entradas determinísticas e executa somente a matriz selecionada,
inclusive via alias. O limite cobre arena CPU, padding e scratch; não mede RSS ou
VRAM. A referência compara a aritmética dos pesos Q4. Ela não produz logits da LLM,
tokens/s, perplexidade ou uma prova de inferência em GPU de 512 MB.

O [guia de execução](NEXALM_EXECUCAO_CPU.md) mostra embedding, RMSNorm, RoPE,
atenção causal GQA, SwiGLU, residuais e logits já integrados e comparados com uma
referência. O benchmark de matriz acima continua sendo um diagnóstico separado.

## Referências do contrato

- [Formato oficial Safetensors](https://github.com/huggingface/safetensors/blob/main/README.md#format).
- [Configuração Llama no Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py).
- [Plano da primeira LLM](NexaLang_Plano_Implementacao_Primeira_LLM.pdf).
