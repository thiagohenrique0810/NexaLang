# KV TurboQuant paginado e atenção CPU

O décimo incremento aplica [TQ_MSE_SRHT V1/TQ02](NEXAPACK_TQ_V1.md) ao cache
usado pelo Transformer. `PagedTransformerSession` oferece `kv_codec="tq"` e a
CLI aceita `--kv-codec tq --kv-bits 3 --kv-seed 42`. K é quantizado após RoPE;
V, após projeção. Cada token/head possui um registro independente, inclusive
nas páginas parciais. Append não requantiza nem copia o prefixo confirmado.
Pesos do modelo continuam Q4.

## Layout e codebook

`PagedKVCachePlan` usa `layout="token_head_tq02"`, `codec_id="TQ_MSE_SRHT"`,
versão 1, identidade `SRHT_XOSHIRO256SS_V1`, bits, seed e codebook explícitos.
O codebook é compartilhado por K/V, heads e camadas da sessão; não é duplicado
em cada página ou token. Dimensão do head deve ser potência de dois, até
1.048.576; bits de 1 a 8, seed inteiro com sinal de 32 bits. `group_size` não se
aplica a TQ. Os layouts JSON F32/Q4/Q3 anteriores permanecem iguais.

```text
head_row_bytes = 8 + ceil(head_dim * bits / 8)
token_bytes = kv_heads * head_row_bytes
buffer_stride = align64(page_tokens * token_bytes)
page_payload_bytes = 2 * layers * page_tokens * token_bytes
page_allocation_bytes = 2 * layers * buffer_stride + 63
```

A norma F32LE, magic e índices TQ02 entram no tamanho físico. O padding de
alinhamento da página é contado separadamente. A última página pode conter
slots ainda não escritos; a atenção inspeciona somente os tokens visíveis.

Na admissão, o plano pode conter `codebook_f32le=null`, indicando que o contexto
ainda não foi criado. Depois de aprovado o orçamento e antes de alocar páginas
ou ler pesos, a primeira execução cria o contexto MSE e exporta os centroids.
O plano concreto e o relatório passam a conter o hex F32LE exato. Para reutilizar
um codebook, forneça `kv_codebook_f32le` à API Python com os mesmos bits/seed.
Não se deduz codebook a partir de registros packed nem se regenera um codebook
importado. Persistir os floats não promete aritmética idêntica em todo hardware.

## Atenção e temporários

`runtime/nexapack/tq_attention.c` implementa atenção causal MHA/GQA/MQA sobre
as tabelas de páginas. A biblioteca separada `nexa_tq_attention` reúne esse
kernel e o runtime TurboQuant. O caminho F32/Q4/Q3 mantém sua biblioteca/ABI.

O kernel reconstrói um vetor K ou V de cada vez. O mesmo buffer F32 de
`head_dim` elementos é usado pela quantização e pela desquantização em operações
distintas. Um acumulador double por head evita reconstruir V para cada canal.
Softmax usa máximo e somas em double, com scores exponenciados F32; K é
reconstruído duas vezes e V uma vez para cada query/head. Esse caminho não
materializa a página nem o prefixo inteiro em float e não aloca heap.

Buffers adicionais na arena planejada:

- `__tq_vector`: `4 * head_dim` bytes, compartilhados entre encode/decode.
- `__tq_accumulator`: `8 * head_dim` bytes.
- `__attention`: `4 * prefix_length` bytes, já usado pelos demais codecs.

O kernel verifica capacidades, overflow, alinhamento e sobreposição com
entradas, páginas, tabelas e estado do contexto. Também valida registros TQ02,
padding, normas e entradas finitas. Erros numéricos podem ocorrer depois de
escritas parciais; o executor descarta a saída e não confirma a transação.

## Memória, admissão e falhas

Para `D=head_dim`, `L=2^bits`, o estado MSE solicitado ao allocator é
`sizeof(tq_ctx) + 4*D + 4*(2*L-1)`. A pré-admissão independente do código nativo
usa reserva de 128 bytes para a struct, verificada por assert no C e pela API
de tamanho antes de construir o contexto. O relatório distingue reserva do
estado real. No macOS ARM64 usado na validação, a struct ocupa 88 bytes.

A reserva adicional inclui `max(8*L, 4*(L+1))` bytes para staging/construção;
esse staging não permanece alocado durante a atenção. Admitir contexto e
construção junto do pior workspace e da reserva de páginas é conservador.
A construção precede a alocação da arena e das páginas da primeira execução.

Os campos `kv_tq_context_bytes`, `kv_tq_context_reserved_bytes`,
`kv_tq_constructor_staging_bytes`, `kv_tq_vector_scratch_bytes` e
`kv_tq_accumulator_bytes` deixam esses custos explícitos.
`capacity_managed_buffers_bound_bytes` inclui a capacidade admitida;
`managed_buffers_peak_bound_bytes` usa páginas da transação e estado do contexto.
Nenhum desses campos mede RSS/VRAM. Python, metadados, allocator, bibliotecas,
cache do SO, logits retidos pelo consumidor e referência PyTorch ficam fora.

Prefill substitui o prompt de forma transacional; append preserva endereços e
bytes do prefixo. Falhas liberam páginas novas e workspace mesmo com traceback
retido. Se a primeira execução falha, o contexto recém-criado também é destruído
e o plano anterior é restaurado. Retry pode executar normalmente.
`reset()` libera páginas e mantém o codebook/contexto da sessão; `close()` libera
ambos. A sessão continua sendo de uma sequência, sem chamadas concorrentes,
evicção ou compartilhamento de prefixos. A
[política de idade CPU](NEXALM_KV_TIERS_CPU.md) é uma sessão própria F32/Q4/Q3;
TQ ainda não participa das páginas mistas dessa política.

## Execução reproduzível

Use o bundle sintético do [guia de importação](NEXALM_IMPORTACAO.md). Na raiz:

```sh
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 2 --kv-codec tq --kv-bits 3 --kv-seed 42 --prefill-chunk-size 2 --memory-budget 96KiB --tile-rows 3 --report artifacts/reports/nexalm-kv-tq.json

# Opcional: referência PyTorch independente do codec nativo.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 2 --kv-codec tq --prefill-chunk-size 2 --memory-budget 96KiB --tile-rows 3 --verify --reference-checkpoint artifacts/checkpoints/nexalm-tiny --report artifacts/reports/nexalm-kv-tq-verified.json
```

`--kv-bits`/`--kv-seed` exigem TQ; `--kv-group-size` exige Q4/Q3. Codec
comprimido exige páginas e `--kv-cache`. Os defaults TQ são três bits e seed 42.
A referência recebe os centroids exatos usados pela sessão. O relatório separa
erro de execução, erro dos pesos Q4, erro do KV e erro combinado. `verified`
aprova somente a equivalência ao oracle com a mesma quantização; não comprova
qualidade linguística ou perplexidade.

## Comparação local sob a mesma capacidade

Fixture sintética wide: head D64, uma camada, um head K/V, contexto 16, página
16, chunks de dois, IDs `[1,3,5,7,2,4,6,8]`, pesos Q4, grupos Q4/Q3 de 32,
TQ três bits/seed42. Todos os modos usam orçamento de 96 KiB e tile32.

| Medida | F32 | Q4 | Q3 | TQ |
|---|---:|---:|---:|---:|
| Bytes/token K+V | 512 | 80 | 64 | 64 |
| Payload de uma página | 8.192 | 1.280 | 1.024 | 1.024 |
| Alocação residente da página | 8.255 | 1.343 | 1.087 | 1.087 |
| Pico gerenciado entre chamadas | 79.614 | 72.702 | 72.446 | 73.618 |
| Capacidade admitida, incluindo substituição | 87.869 | 74.045 | 73.533 | 74.809 |
| Erro máximo de execução vs oracle do mesmo codec | 0 | 0 | 0 | 0 |
| Erro máximo do KV vs F32 nos logits | — | 0,2619032860 | 0,5301163346 | 0,5143706203 |

TQ e Q3 ocupam o mesmo payload nessa configuração. TQ acrescenta contexto de
404 B, scratch F32 de 256 B e acumulador de 512 B; alinhamento da arena também
entra na diferença. Seu pico total é maior que o Q3 nessa fixture. O erro TQ
ligeiramente menor nessa amostra não comprova ganho de qualidade geral.
Relatórios: `artifacts/reports/tq-kv-wide-{f32,q4,q3,tq}.json`.

Na fixture tiny com head D4 e IDs `[1,3,5,7]`, TQ usa 20 B/token K+V contra
32 B F32; o alinhamento ainda exige 191 B por página de dois tokens. O pico foi
67.233 B. Erro máximo de execução = 0; dos pesos = 0,4435420930; do KV =
1,1905089021; combinado = 1,1530171633. Relatório
`artifacts/reports/nexalm-kv-tq-verified.json`. Esses erros mostram que
quantização não é exata e heads pequenos não asseguram bom compromisso.

Para reproduzir a wide, gere uma pasta nova com a fixture determinística:

```sh
python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'tests')
from test_q4_paged_transformer_regressions import wide_bundle
path = Path('artifacts/models/nexalm-kv-wide')
if not path.exists():
    wide_bundle(path)
PY
python3 tools/nexa_run.py artifacts/models/nexalm-kv-wide --tokens 1,3,5,7,2,4 --decode-tokens 6,8 --kv-cache --kv-page-tokens 16 --kv-codec tq --kv-bits 3 --kv-seed 42 --prefill-chunk-size 2 --max-sequence-length 16 --tile-rows 32 --memory-budget 96KiB --verify --report artifacts/reports/tq-kv-wide-tq.json
```

Troque `--kv-codec` por `f32`, `q4` ou `q3`, remova `--kv-bits`/`--kv-seed` e,
nos dois codecs por grupo, acrescente `--kv-group-size 32`. Use relatórios
distintos. `--verify` requer Torch somente para o diagnóstico, fora do orçamento;
a execução nativa e o golden de regressão não dependem dele. Os valores acima
foram verificados em macOS ARM64/Python 3.14.5; não são prova de GPU.

## Validação

```sh
python3 -m unittest discover -s tests -p 'test_tq*regressions.py' -v
python3 -S -m unittest discover -s tests -p 'test_tq*regressions.py' -v
python3 -m unittest discover -s tests -p 'test_*regressions.py' -v
python3 tests/run_tests.py
make -C runtime all test
```

O oracle Python usa PRNG/SRHT/packing independentes. As provas incluem goldens
sem Torch, comparações de logits com Torch opcional, prefixo imutável, fronteiras
de página/chunk, codebook importado, orçamento exato, falhas, corrupção e
sanitizers nativos. Os resultados finais e a próxima tarefa ficam no
[checklist central](BLUEPRINT_512MB_CHECKLIST.md). TQ em tiers, evicção/recarga,
qualidade de modelo treinado, kernels de pesos TQ e backend GPU permanecem pendentes.
