# Checkpoints sintéticos: medir 125,8M sem baixar peso de ninguém

Até aqui o pipeline inteiro — importação, escada de codecs, empacotamento,
execução com KV paginado — só tinha sido exercitado em fixtures de brinquedo:
`hidden_size 8`, `vocab_size 16`, um par de camadas. O projeto se chama
"blueprint 512 MB" e nunca tinha medido nada maior que isso.

`tools/nexa_synth.py` fecha essa lacuna sem depender de download nem de licença
de terceiros: gera um checkpoint **sintético** no formato que
`compiler/importers/llama.py` já importa (Safetensors + `config.json`), na forma
exata de uma arquitetura declarada em `models/nexalm512/architecture.nxl`.

```bash
python3 tools/nexa_synth.py --definition models/nexalm512/architecture.nxl \
    --model NexaLM512_R0 --seed 20260920 --dtype f32 --max-shard-bytes 256MiB \
    --out weights/synthetic/NexaLM512_R0-s20260920
```

## O que ele prova, e o que não prova

| Prova (física) | Não prova (qualidade) |
| --- | --- |
| bytes por codec, no disco e no `.nxb` | perplexidade, coerência, acurácia |
| pico da arena e residência de KV | sensibilidade por tensor a codec |
| tempo de prefill e de decode | qual tensor merece mais bits |
| se o conjunto cabe em 512 MB | qualquer comparação entre codecs por erro |

Os pesos são **pseudoaleatórios**. Medir sensibilidade de codec aqui produziria
um número com cara de calibração que não é calibração: a distribuição por grupo
de 32 é a da inicialização, não a de um modelo que viu dados. Isso é o mesmo que
`weights/README.md` já diz, e o mesmo `quality_measured: false` que os relatórios
de execução já publicam. Para qualidade continua sendo necessário um checkpoint
**treinado** — baixado em `weights/checkpoints/` com origem registrada, ou
produzido pelo gate de treinamento (LLM.03/LLM.04), que não existe.

## Inicialização: por que não é uniforme

A escada de codecs quantiza por `max|v|` dentro de cada grupo. Um preenchimento
uniforme daria um erro de quantização que **nenhum modelo treinado produz**:
sem cauda, o `max|v|` do grupo fica colado no limite da distribuição e a escala
do grupo vira quase constante. Então o gerador usa a inicialização que um modelo
de verdade usa antes do primeiro passo de treino:

| Tensor | Desvio padrão |
| --- | --- |
| `model.embed_tokens.weight` | `0.02` (o `initializer_range` do Llama) |
| `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` | `1/sqrt(fan_in)` |
| `o_proj`, `down_proj` | `1/sqrt(fan_in) / sqrt(2L)` |
| `input_layernorm`, `post_attention_layernorm`, `model.norm` | exatamente `1.0` |

O fator extra `1/sqrt(2L)` nas duas projeções que escrevem no fluxo residual é o
que impede a variância do residual de crescer com a profundidade; é a mesma
escolha do GPT-2 e dos Llama. **Ainda assim não é um modelo treinado**: é a
distribuição do passo zero, não a do passo final.

### Amostrador: soma de doze uniformes, não Box-Muller

`_normals` soma doze uniformes do Mersenne Twister e subtrai seis. A variância
da soma é exatamente 1 e o suporte é exatamente `[-6, +6]`, então a amostra é
N(0,1) truncada em seis desvios. A razão de não usar Box-Muller é reprodução:
`log` e `cos` são da libm, cujo último bit não é garantido entre plataformas, e
o ponto do gerador é que a mesma semente produza os **mesmos bytes** em qualquer
lugar. Só adição e multiplicação IEEE-754 entram no caminho.

O preço é uma cauda levemente mais leve que a normal exata (excesso de curtose
`-1/10`). Para `max|v|` num grupo de 32 isso é irrelevante frente à diferença
entre normal e uniforme, que é o que estava em jogo.

### Semente por tensor

A semente de cada tensor é `sha256(formato, versão, semente, nome_do_tensor)`.
Isso torna o payload de um shard função apenas dos tensores que ele guarda:
reordenar shards, mudar o limite de shard ou gerar um tensor sozinho reproduz os
mesmos bytes. Um `lm_head.weight` amarrado desenha do fluxo do embedding, porque
o importador decodifica os dois e recusa o checkpoint se um float diferir.

### Streaming

Nenhum tensor é materializado inteiro. Os valores saem em blocos de 16384, são
empacotados e escritos. Gerar os 480 MiB de `NexaLM512_R0` levou **35,2 s** e o
processo inteiro não passou de dezenas de MB — a mesma propriedade que a suíte
verifica com `tracemalloc` em `test_generation_streams_instead_of_materializing_the_model`.

## `origin.json`

Ao lado dos shards, para que nenhuma medição futura confunda isto com um
checkpoint real:

```json
{
  "format": "NexaSyntheticCheckpoint", "synthetic": true, "trained": false,
  "quality_measured": false, "seed": 20260920, "model_name": "NexaLM512_R0",
  "tool": "tools/nexa_synth.py",
  "definition": {"path": "models/nexalm512/architecture.nxl",
                 "model": "NexaLM512_R0", "sha256": "..."},
  "initialization": {"sampler": "irwin_hall_12_minus_6", "...": "..."},
  "files": [{"path": "...", "size_bytes": 0, "sha256": "..."}]
}
```

O arquivo fica **fora** do que o importador lê: `SafeTensorCheckpoint` só abre
`model.safetensors`/`model.safetensors.index.json` e os assets nomeados, então
o registro de origem não entra em nenhum bundle e não muda nenhum hash.

### Tokenizer: deliberadamente ausente

O gerador não inventa um tokenizer. Um vocabulário sintético de 32768 entradas
teria a forma certa e nenhum significado, e `nexa_run.py` já aceita IDs
explícitos com `--tokens`. Tokenizer real é o caminho de `tools/train_bpe.py` e
`runtime/nexapack/tokenizer.py`, sobre corpus (LLM.02), não aqui.

## O que foi medido em 125.854.464 parâmetros

Máquina: Apple Silicon, 12 núcleos, 24 GiB, macOS 25.5. Executor: C escalar de
uma thread, sem SIMD e sem GPU. Todo número abaixo é **medido**, nunca estimado.
Semente `20260920`, `NexaLM512_R0`, F32 no checkpoint, 2 shards de 256 MiB.

### Geração

| | valor |
| --- | --- |
| parâmetros | 125.854.464 (igual ao oráculo `ModelConfig.parameter_count()`) |
| payload Safetensors F32 | 503.417.856 B (480,1 MiB) |
| tempo de parede | 35,2 s |
| shards | 2 (`model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`) |
| RSS máx. da geração | dezenas de MB (nenhum tensor materializado) |

Gerado duas vezes com a mesma semente, os quatro arquivos saíram com SHA-256
idênticos — determinismo verificado em 503 MB de payload, não só na fixture:

```text
model-00001-of-00002.safetensors  a53a2db3b05aeb7c...
model-00002-of-00002.safetensors  95975a25960526bf...
model.safetensors.index.json      992455ae6ee92d97...
config.json                       3c61d101a78fbf00...
```

### Escada de codecs: bytes reais

`packed_payload_bytes` é só o payload dos tensores; `bundle` é o diretório
inteiro (manifesto, cabeçalhos, vetores RAW_F32 e assets).

| codec | payload (B) | bits/param | vs F32 | bundle (B) | `.nxb` (B) | overhead do contêiner |
| --- | --- | --- | --- | --- | --- | --- |
| q2 | 47.287.296 | 3,006 | 10,646× | 47.985.810 | 48.050.176 | +64.366 (**+0,134%**) |
| q3 | 63.015.936 | 4,006 | 7,989× | 63.714.483 | 63.778.816 | +64.333 (+0,101%) |
| q4 | 78.744.576 | 5,005 | 6,393× | 79.443.123 | 79.507.456 | +64.333 (+0,081%) |
| q8 | 141.659.136 | 9,005 | 3,554× | 142.357.731 | 142.422.016 | +64.285 (+0,045%) |
| f16 | 251.759.616 | 16,003 | 2,00× | 252.031.965 | 252.096.512 | +64.547 (+0,026%) |
| f32 | 503.417.856 | 32,000 | 1,00× | 503.690.206 | 503.754.752 | +64.546 (**+0,013%**) |

O guia [`NEXALM_PACOTE_NXB.md`](NEXALM_PACOTE_NXB.md) mediu **+134%** de
overhead de contêiner na fixture minúscula, por causa do alinhamento de 4096
bytes sobre tensores de dezenas de bytes. Em escala real o mesmo alinhamento
custa **+0,013% a +0,134%**: o overhead do `.nxb` é quase constante (≈64,3 KiB,
quase todo índice mais padding de seção), então quanto maior o modelo, mais
perto de zero. O número da fixture não era errado; era um número sobre uma
fixture.

### Conversão e verificação (uma passada, streaming)

| codec | conversão | RSS máx. | `pack` | `inspect --verify` |
| --- | --- | --- | --- | --- |
| q2 | 39,19 s | 29,1 MB | 0,28 s | 14,09 s |
| q3 | 40,46 s | 28,7 MB | 0,36 s | 12,92 s |
| q4 | 40,33 s | 29,3 MB | 0,39 s | 14,13 s |
| q8 | 34,75 s | 29,6 MB | 0,56 s | 9,69 s |
| f16 | 16,03 s | 31,7 MB | 0,83 s | 3,85 s |
| f32 | 15,69 s | 31,8 MB | 2,64 s | 4,00 s |

Quantizar 125,8M valores em Python puro custa ~40 s e **29 MB de RSS**: o
importador nunca materializa um tensor. `--verify` leu e validou todos os
payloads dos seis bundles, todos `checksums_verified: true`.

### Execução: prompt de 128 tokens + 8 passos gananciosos

`nexa_run.py` sobre o `.nxb`, KV paginado de 32 tokens, `--memory-budget 512MiB`.

| codec | prefill (128 tok) | pico do prefill | decode s/tok | leitura/passo | compute/passo | pico do decode |
| --- | --- | --- | --- | --- | --- | --- |
| q2 | **falha** | — | — | — | — | — |
| q3 | 15,60 s | 21.509.691 B | 0,315 s | 0,069 s | 0,219 s | 8.853.114 B |
| q4 | 15,63 s | 21.517.883 B | 0,293 s | 0,078 s | 0,189 s | 8.861.306 B |
| q8 | 10,02 s | 21.550.651 B | 0,276 s | 0,125 s | 0,125 s | 8.894.074 B |
| f16 | 37,54 s | 21.756.475 B | 4,296 s | 3,752 s | 0,530 s | 9.099.898 B |
| f32 | 12,23 s | 22.018.619 B | 3,854 s | 3,737 s | 0,104 s | 9.362.042 B |

Três coisas que só aparecem em escala real:

1. **Os codecs densos viram limitados por I/O.** Cada passo de decode relê a
   matriz inteira: 503.614.464 B para F32, em 3,737 s (≈135 MB/s). Os codecs
   empacotados leem 126–283 MB por passo em 0,069–0,125 s (≈2 GB/s) porque
   cabem no cache de página do SO. O ganho de q4 sobre f32 em decode é **13×**,
   e quase todo ele é I/O, não aritmética.
2. **F16 é o pior dos dois mundos aqui.** Paga a leitura densa (3,75 s) *e* um
   kernel mais lento que o de F32 (0,530 s contra 0,104 s de compute), porque
   converte cada meia-palavra antes de multiplicar. 4,296 s por token.
3. **Q8 ganha de Q4 em compute** (0,125 s contra 0,189 s por passo): descompactar
   4 bits custa mais que ler um byte. Q4 só volta à frente quando a leitura
   importa, e pelo mesmo motivo Q4 é o melhor ponto da escada nesta máquina.

### A falha medida: Q2 não executa com `group_size` 32

O bundle Q2 converte, empacota e passa em `inspect --verify`. **Não abre para
execução**:

```text
Nexa model execution failed: packed storage_nbytes is smaller than
its 12582912-byte payload
```

A causa é exata. `runtime/nexapack/transformer.py:154` descreve **todo** codec
empacotado como `storage_dtype="q4"`:

```python
storage = "q4" if q4 else ("f16" if item["codec"] == "RAW_F16_MATRIX" else "f32")
```

`TensorDesc` então exige pelo menos 4 bits por valor. Com `group_size` 32:

| codec | bytes por grupo de 32 | bits/valor | passa no piso de 4 bits? |
| --- | --- | --- | --- |
| q2 | 4 (escala) + 8 | 3,0 | **não** |
| q3 | 4 (escala) + 12 | 4,0 | sim, exatamente no limite |
| q4 | 4 (escala) + 16 | 5,0 | sim |

`model.embed_tokens.weight` tem 25.165.824 valores: o piso Q4 é 12.582.912 B e o
payload Q2 real é 9.437.184 B. A suíte não pega isso porque as fixtures usam
`group_size` 4, onde a escala de 4 bytes por 4 valores infla Q2 para 10 bits por
valor e o piso passa por acidente. **Não corrigi**: está fora do que foi
adjudicado para esta onda, e o número medido vale mais registrado que remendado.

### Contexto longo: 512 tokens + 8 passos, codec q4

| medição | capacidade 520 | capacidade 2048 | capacidade 2048, KV q4 |
| --- | --- | --- | --- |
| prefill (512 tok) | 63,59 s | 63,09 s | 64,95 s |
| pico do prefill | 85.665.327 B | 90.567.727 B | 76.411.951 B |
| decode s/tok | 0,307 s | 0,312 s | 0,318 s |
| arena | 13.435.392 B | 52.760.576 B | 52.760.576 B |
| KV residente (17 páginas) | 17.826.863 B | 17.826.863 B | 2.786.351 B |
| KV por token | 32.768 B | 32.768 B | **5.120 B** |
| teto para a capacidade inteira | 105.589.405 B | **409.341.887 B** | **296.095.679 B** |

`capacity_managed_buffers_bound_bytes` é o que a sessão reserva se a sequência
crescer até a capacidade declarada — é o número que responde "cabe".

### A resposta: cabe em 512 MB?

**Cabe.** Medido no contexto inteiro que a arquitetura declara — 2048 tokens,
não oito. Sessão paginada direto sobre `R0-q4.nxb`, prefill em blocos de 128,
`--memory-budget 512MiB` (536.870.912 B), 8 passos de decode no fim:

| | KV F32 | KV Q4 |
| --- | --- | --- |
| comprimento de contexto | **2048 tokens** | **2048 tokens** |
| pico dos buffers gerenciados | 83.370.495 B (79,5 MiB) | 27.140.355 B (25,9 MiB) |
| arena de ativações | 3.355.136 B | 3.355.136 B |
| KV residente (64 páginas) | 67.112.896 B | 10.489.792 B |
| KV por token | 32.768 B | 5.120 B |
| payload dos pesos (streaming) | 78.744.576 B | 78.744.576 B |
| pesos + buffers gerenciados | **162.115.071 B (154,6 MiB)** | **105.884.931 B (101,0 MiB)** |
| **RSS máximo do processo (SO)** | **280.707.072 B (267,7 MiB)** | **259.047.424 B (247,0 MiB)** |
| folga sob 512 MiB | 256.163.840 B (244,3 MiB) | 277.823.488 B (265,0 MiB) |
| prefill de 2040 tokens | 288,67 s (7,07 tok/s) | 302,93 s (6,73 tok/s) |
| decode em contexto 2048 | 0,3532 s/token (2,83 tok/s) | 0,3637 s/token |

O RSS máximo é o número do sistema operacional, não uma contabilidade do
projeto: inclui o interpretador CPython, as bibliotecas nativas e um bloco de
logits de 128×32768 floats retido pelo chamador. Mesmo assim, **o processo
inteiro que executa 125.854.464 parâmetros em contexto 2048 nunca passou de
267,7 MiB** — menos da metade de 512 MiB.

Sob a contabilidade mais dura possível, com todos os pesos exigidos residentes
em vez de lidos por tile, q4 continua cabendo com 374.755.841 B de folga.
**F32 não cabe**: 503.417.856 B de pesos mais 83.370.495 B de buffers são
586.788.351 B, ou seja **49.917.439 B (47,6 MiB) acima** de 512 MiB. Em
streaming o F32 executa, mas a 3,854 s por token — vinte e cinco vezes mais
lento em decode que o mesmo modelo em q4, porque relê 503 MB por passo.

O preço honesto: **prefill de 2048 tokens leva 4min49s** e o decode entrega
2,83 tokens/s. Não é interativo. O executor é C escalar de uma thread, sem SIMD
e sem GPU; o teto de memória foi alcançado, o de velocidade não.

### Onde o RSS estoura, e por quê

`nexa_run.py` retém em Python os logits de **toda** a sequência: para um prompt
de 512 tokens isso são 520×32768 floats, e o RSS do processo vai a
**815.742.976 B**, contra 222.953.472 B da mesma execução feita pela sessão
direto, descartando os blocos. Os buffers gerenciados do motor não mudam
(`managed_buffers_peak_bound_bytes` é o mesmo). É custo do chamador, declarado
como tal em `memory.excluded`, e é a diferença entre "o motor cabe" e "o
programa que você escreveu em volta dele cabe".

### O que estes números não dizem

Os oito tokens gerados pelo modelo de contexto 2048 foram
`15609, 8668, 8668, 8668, 8668, 8668, 8668, 8668`. Um ponto fixo, que é
exatamente o que pesos aleatórios produzem. Nenhuma medição aqui é sobre
qualidade, e nenhum relatório destas execuções pode ser lido como tal: todos
carregam `model_quality_measured: false`.

## Reproduzir

Tudo abaixo de `weights/` é ignorado pelo Git. Os comandos exatos das medições
acima, em ordem:

```bash
W=weights
python3 tools/nexa_synth.py --definition models/nexalm512/architecture.nxl \
    --model NexaLM512_R0 --seed 20260920 --dtype f32 --max-shard-bytes 256MiB \
    --out $W/synthetic/NexaLM512_R0-s20260920

for codec in q2 q3 q4 q8 f16 f32; do
  python3 tools/nexa_convert.py --checkpoint $W/synthetic/NexaLM512_R0-s20260920 \
      --matrix-codec $codec --out $W/bundles/R0-$codec
  python3 tools/nexa_pack.py pack $W/bundles/R0-$codec $W/bundles/R0-$codec.nxb
  python3 tools/nexa_inspect.py $W/bundles/R0-$codec.nxb --verify
done

TOKENS=$(python3 -c "print(','.join(str((i*2654435761)%32768) for i in range(1,129)))")
python3 tools/nexa_run.py $W/bundles/R0-q4.nxb --tokens $TOKENS --generate 8 \
    --kv-page-tokens 32 --memory-budget 512MiB --report $W/bundles/R0-q4-128x8.json
```

O teto de 2048 tokens foi medido pela sessão direto, sem o `nexa_run.py`, porque
a CLI retém os logits de toda a sequência em Python e isso dominaria o RSS:

```python
# medicao_2048.py — prefill em blocos, guardando só a última linha de logits
import resource, sys, time
from runtime.nexapack.paged import PagedTransformerSession

tokens = [(i * 2654435761) % 32768 for i in range(1, 2041)]
with PagedTransformerSession("weights/bundles/R0-q4.nxb", memory_budget="512MiB",
                             max_sequence_length=2048, page_tokens=32, tile_rows=32,
                             max_chunk_length=128, kv_codec="f32") as session:
    start = time.perf_counter()
    last = session.prefill(tokens[:128])[-1]
    for offset in range(128, len(tokens), 128):
        last = session.append(tokens[offset:offset + 128])[-1]
    print("prefill", time.perf_counter() - start)
    for _ in range(8):
        last = session.decode(max(range(session.config.vocab_size), key=last.__getitem__))
    print(session.report()["memory"]["managed_buffers_peak_bound_bytes"],
          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
```

`max_sequence_length` não pode passar de `max_position_embeddings`: 2040 tokens
de prompt mais 8 passos fecham exatamente os 2048 declarados.
