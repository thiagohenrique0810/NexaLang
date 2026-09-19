# TurboQuant portátil no NexaPack

M1.06 implementa armazenamento, conversão e reconstrução CPU de vetores MSE.
O contêiner [NexaPack V1](NEXAPACK_V1.md) aceita o novo codec `TQ_MSE_SRHT`
versão 1. O layout Q4 permanece igual; leitores antigos rejeitam o codec novo.
Bundles, GEMV/GEMM e o Transformer ainda exigem pesos Q4 e rejeitam TQ antes de
ler payload ou carregar kernels para pesos TQ. O incremento seguinte acrescentou
[KV TQ paginado e atenção CPU](NEXALM_KV_TQ_CPU.md), mantendo pesos Q4.

## Contrato persistido

O prefixo, o índice limitado, offsets, alinhamento, SHA-256 e publicação atômica
continuam sendo os do contêiner V1. A shape é `[rows, dim]`, uma linha por vetor.
O JSON usa os campos comuns do Q4, substitui `storage_dtype` por `tq_mse`,
`codec_id` por `TQ_MSE_SRHT`, remove `group_size` e acrescenta:

| Campo | Contrato |
|---|---|
| `codec_version` | `1` |
| `bits` | Inteiro de 1 a 8, sem aceitar booleanos |
| `seed` | Inteiro com sinal de 32 bits |
| `transform_id` | `SRHT_XOSHIRO256SS_V1` |
| `codebook_f32le` | `2^bits` centroids F32 little-endian em hex minúsculo |

Dimensão deve ser potência de dois entre 1 e 1.048.576. O codebook é explícito,
finito e estritamente crescente; a soma F32 de centroids adjacentes não pode
estourar, pois os limites usam `0.5f * (left + right)`. São exatamente
`8 * 2^bits` caracteres hex, sem espaços. Campos extras, duplicados, valores
inválidos e versões desconhecidas são rejeitados antes da leitura de payload.

Cada linha contém:

```text
4 bytes: ASCII TQ02
4 bytes: norma IEEE binary32 little-endian
ceil(dim * bits / 8) bytes: índices sem sinal, LSB primeiro
row_bytes = 8 + ceil(dim * bits / 8)
```

O índice da coordenada `j` começa no bit `j*bits`, no bit baixo do byte
correspondente. Bits altos não usados no último byte são zero. A norma deve ser
finita e não negativa; `-0` é inválido. Vetor nulo usa norma `+0` e todos os
índices zero. Essa convenção independe do valor do centroid de índice zero.

`SRHT_XOSHIRO256SS_V1` preserva a transformação do runtime legado: seed com
sinal convertido para uint64 módulo `2^64`, estado inicial por SplitMix64 e
sequência xoshiro256**. Os primeiros `dim` resultados geram sinais: bit baixo
1 significa `+1`, zero significa `-1`. A transformação aplica esses sinais,
Walsh-Hadamard em ordem natural e normalização por `sqrt(dim)`. O inverso
desfaz a ordem. Implementação em `runtime/turboquant.c`; oracle independente
e vetores conhecidos em `tests/test_tq_portable_format_regressions.py`.

A quantização calcula a norma em double, normaliza as coordenadas em F32,
aplica SRHT e compara com os pontos médios do codebook. Empates escolhem o
índice inferior. A reconstrução consulta os centroids, aplica o inverso e a
norma persistida. Entradas não finitas e overflow numérico são erros.

Persistir os centroids evita que o leitor execute Lloyd-Max novamente: seed
sozinho não fixa resultados de libm em outra plataforma. O formato dos bytes
é independente do endian; o contrato não promete aritmética bit a bit idêntica
entre hardware, compiladores ou bibliotecas matemáticas diferentes.

## APIs e memória

`runtime/nexapack/format.py` oferece:

- `write_tq_matrix(path, rows, cols, bits, seed, row_source, *, block_rows=64,
  memory_budget=None, codebook_f32le=None)`: consome um vetor por vez e retorna
  o relatório de buffers do codec. Sem codebook, gera e persiste os centroids.
- `write_tq_records(path, rows, cols, bits, seed, codebook_f32le, row_source,
  *, block_rows=64)`: valida e grava linhas TQ02 já codificadas, sem código nativo.
- `NexaPackReader`: abertura lazy, `read_rows_into` e checksums mantidos;
  `validate_row` despacha validação Q4/TQ sem reconstruir floats.

Escrita TQ usa I/O sem buffering, checksum incremental e um índice limitado;
não acumula o bloco inteiro. O destino só é substituído após escrita e fsync
completos. Falhas removem o temporário e preservam o arquivo anterior.

`runtime/nexapack/tq.py` oferece `TQCodec(dim, bits=3, seed=42, *,
codebook_f32le=None, memory_budget=None)`, com context manager, `encode_row`,
`decode_row`, propriedade `codebook_f32le` e `memory_report()`.
Dimensão, bits e seed públicos são somente leitura. O wrapper serializa
operações concorrentes e `close`; rejeita reentrada no mesmo codec.
Buffers de trabalho são liberados inclusive quando o chamador retém uma exceção.

No C, `tq_create_mse_from_codebook` importa centroids sem Lloyd-Max;
`tq_export_mse_codebook` exporta os valores. `tq_mse_context_memory_size`
permite pré-admissão sem criar contexto. `tq_quantize_tq02` recebe scratch de
`dim` floats do chamador; `tq_dequantize_tq02` transforma na própria saída.
Os dois kernels não alocam heap. Capacidades, overflow e sobreposição de buffers
são verificados. Em erro, descartar a saída inteira; consulte retornos e ownership
em `runtime/turboquant.h`. As APIs TQ01/MSE/Prod anteriores permanecem compatíveis.

Para `D=dim`, `L=2^bits`, `R=row_bytes` e estado `C` informado pelo runtime:

```text
encode_peak = C + 8*D + 2*R     # entrada, scratch e packed + cópia de retorno
decode_peak = C + 4*D + R       # saída F32 e cópia packed
constructor_peak <= C + max(8*L, 4*(L+1))
codec_bound = max(encode_peak, decode_peak, constructor_peak)
CLI F32 bound = codec_bound + min(65536, 4*D)
CLI migração bound = 3*R + 8
```

O termo Lloyd-Max `4*(L+1)` só se aplica ao codebook gerado. Esses limites são
conservadores e incluem staging explícito. O orçamento é verificado antes de
alocar o contexto ou consumir vetores; carregamento/compilação da biblioteca
fica fora do escopo. Python, metadados, objetos, overhead de alocação, stack,
entrada do chamador, listas retornadas ao consumidor, bibliotecas e cache do SO
também ficam fora. O limite não mede RSS nem VRAM.

Leitura/inspeção e conversão têm buffers distintos: o reader usa scratch de
64 KiB, além do tile de destino. O orçamento da CLI de conversão não é aplicado
ao diagnóstico `nexa_inspect.py --verify` ou a listas retidas pelo consumidor.

## Conversão reproduzível

Na raiz do repositório, gere uma matriz sintética F32LE de 1 MiB:

```sh
python3 - <<'PY'
from pathlib import Path
import struct
path = Path('artifacts/models/tq-portable-demo.f32')
path.parent.mkdir(parents=True, exist_ok=True)
Path('artifacts/reports').mkdir(parents=True, exist_ok=True)
with path.open('wb') as stream:
    for row in range(4096):
        stream.write(struct.pack('<64f', *[
            ((row * 17 + col * 13) % 257 - 128) / 64 for col in range(64)
        ]))
PY
python3 tools/nexa_convert.py artifacts/models/tq-portable-demo.f32 --out artifacts/models/tq-portable-demo.nxp --rows 4096 --cols 64 --codec tq --bits 3 --seed 42 --block-rows 64 --memory-budget 96KiB > artifacts/reports/tq-portable-conversion.json
python3 tools/nexa_inspect.py artifacts/models/tq-portable-demo.nxp --verify > artifacts/reports/tq-portable-inspection.json
```

Resultados locais macOS ARM64/Python 3.14.5:

| Medida | Bytes |
|---|---:|
| Entrada F32 | 1.048.576 |
| Linha TQ02 | 32 |
| Payload TQ, 64 blocos | 131.072 |
| Arquivo completo | 143.360 |
| Estado do contexto | 404 |
| Scratch do quantizador | 256 |
| Limite de pico do codec | 980 |
| Limite de pico da conversão, incluindo leitura | 1.236 |

O arquivo packed também excede o orçamento de 96 KiB. A inspeção verificou
131.072 bytes de payload. O round-trip por linha teve erro absoluto máximo
0,7924648523, médio 0,1649009345 e RMSE 0,2061312307 contra os floats originais.
São erros de quantização de vetores sintéticos, sem avaliação de modelo ou
perplexidade. Relatório: `artifacts/reports/tq-portable-roundtrip.json`.

Exemplo de reconstrução limitada a uma linha, usando o codebook persistido:

```python
from runtime.nexapack import NexaPackReader
from runtime.nexapack.tq import TQCodec

with NexaPackReader('artifacts/models/tq-portable-demo.nxp') as pack:
    with TQCodec(pack.cols, pack.bits, pack.seed,
                 codebook_f32le=pack.codebook_f32le) as codec:
        for row in range(pack.rows):
            reconstructed = codec.decode_row(pack.read_rows(row, 1))
            # Consumir reconstructed antes de avançar, sem acumular a matriz.
```

Esse exemplo privilegia simplicidade; ler tiles coincidentes com os blocos
evita reler payload para checksums. A lista de floats pertence ao consumidor.

## Migração TQ01 explícita

TQ01 guarda a norma no endian do host e não contém dimensão, bits, seed ou
codebook. A migração exige esses metadados originais; não tenta deduzi-los nem
gerar centroids substitutos. Exporte-os do contexto que produziu o arquivo,
usando `tq_export_mse_codebook`. O JSON passado em `--codebook` deve ter
exatamente `dim`, `bits`, `seed`, `transform_id` e `codebook_f32le`, conforme
os contratos acima; tamanho máximo 8.192 bytes. Os parâmetros devem coincidir
com os informados na CLI. Esse arquivo também é aceito na conversão F32.

```sh
python3 tools/nexa_convert.py origem.tq01 --out migrado.nxp --rows 4096 --cols 64 --codec tq --bits 3 --seed 42 --legacy-tq01 --source-endianness big --codebook codebook-original.json --block-rows 64 --memory-budget 96KiB
python3 tools/nexa_inspect.py migrado.nxp --verify
```

A migração troca o magic e serializa a norma como little-endian, preservando
índices e centroids, sem requantização. Magic, tamanho, norma e padding inválidos
são rejeitados. Não aceita `-0` legado; o encoder legado não produz essa forma.
Input e output precisam ser arquivos distintos.

Na demo acima, uma origem big-endian simulada a partir dos registros TQ02
produziu arquivo migrado idêntico ao original, SHA-256
`69e09680f936241ccf1e3ba180d7d23798f2990c2fb90f99217bd50812576bfb`.
Seu limite de buffers foi 104 B, sem criar contexto ou quantizar. Trata-se de
simulação dos bytes de origem, não de execução em hardware big-endian.

## Validação e próximo passo

```sh
python3 -S -m unittest discover -s tests -p 'test_tq_portable*regressions.py' -v
python3 -m unittest discover -s tests -p 'test_*regressions.py' -v
python3 tests/run_tests.py
make -C runtime all test
```

Os 32 testes novos cobrem oracle independente, golden Q4 anterior à mudança,
interoperabilidade C/Python, endian, limites, corrupção, publicação atômica,
liberação com traceback retido, concorrência e orçamento. Os kernels passaram
ASan/UBSan, falhas de alocação instrumentadas e execução sem heap.
A matriz remota de plataformas permanece pendente.

O incremento seguinte integrou TQ02 às páginas KV CPU e implementou atenção com
estado, scratch e reconstrução por head contabilizados, conforme o
[guia KV TQ](NEXALM_KV_TQ_CPU.md). TQ em tiers permanece pendente; a
[política de idade CPU](NEXALM_KV_TIERS_CPU.md) usa F32/Q4/Q3. Estado do projeto e resultados
completos estão no [checklist de retomada](BLUEPRINT_512MB_CHECKLIST.md).
