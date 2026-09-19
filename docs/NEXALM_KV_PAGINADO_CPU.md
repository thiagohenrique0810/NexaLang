# KV paginado F32 em CPU

M4.01 acrescenta páginas alocadas conforme o contexto cresce e atenção nativa que
consulta essas páginas diretamente. Mantém a API incremental, os pesos Q4 por
tiles e o orçamento de buffers explícitos. O [cache contíguo de dois bancos](NEXALM_KV_CPU.md)
continua disponível para comparação. Este guia descreve os valores F32; o
[codec Q4 opcional](NEXALM_KV_Q4_CPU.md) e o [codec Q3](NEXALM_KV_Q3_CPU.md)
acrescentam compressão com o mesmo contrato de páginas e transações, selecionados
por `--kv-codec q4` ou `--kv-codec q3`.

## Uso

```sh
# Após gerar/importar a fixture tiny conforme o guia de importação.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 2 --prefill-chunk-size 2 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-paged.json
```

`--kv-page-tokens` exige `--kv-cache` e um inteiro positivo. Sem essa opção, o
modo incremental continua usando os dois bancos contíguos. `--prefill-chunk-size`
limita ativações e tamanho de cada transação; a capacidade total é controlada por
`--max-sequence-length`, cujo padrão na CLI é prompt + IDs solicitados.

Na API, `runtime.nexapack.paged.PagedTransformerSession` aceita `page_tokens=16`,
`max_chunk_length` e os mesmos argumentos de orçamento/contexto do executor CPU.
Oferece `prefill`, `append`, `decode`, `reset`, `close`, `report`, `token_ids`,
`cache_length`, `resident_page_count` e `resident_kv_bytes`. O último inclui padding
das alocações físicas. A sessão é síncrona e atende uma sequência.

## Layout e execução

Uma página física agrupa K/V de todas as camadas para `P` posições. Cada buffer
por camada é F32 `[P, kv_heads * head_dim]`, com início alinhado a 64 bytes. O
endereço de um head é o início desse buffer + `head * head_dim * 4`; o stride
entre tokens é `kv_heads * head_dim * 4`. Isso preserva GQA, MQA e MHA.

O plano `compiler/paged_kv_plan.py` distingue índices lógicos de páginas dos seus
endereços físicos. Cada CacheWrite contém segmentos explícitos com índice de
página, posição dentro da página, offset do chunk e quantidade de tokens. A
agenda inclui RoPE com offset, escrita, atenção e Commit, com lifetimes derivados
e serialização JSON validada. Um plano aceita até 65.536 segmentos por chamada.

O executor usa duas tabelas de ponteiros, K e V, dentro da arena planejada. Reutiliza
essas tabelas entre camadas e fornece ao kernel os buffers daquela camada. Os
ponteiros podem apontar para alocações fisicamente separadas ou fora da ordem
lógica; não existe concatenação temporária do prefixo. A atenção lê somente as
posições válidas, aplica causalidade e usa um float de scratch por posição.

## Transações e liberação

- A abertura da sessão valida a capacidade e não aloca páginas KV.
- `prefill` prepara páginas novas enquanto preserva o prompt anterior. Confirma
  o novo histórico somente após produzir logits e relatório; libera as páginas
  antigas depois do commit.
- `append`/`decode` preservam os endereços das páginas existentes. Escrevem apenas
  posições não confirmadas da última página e páginas novas quando necessário.
- Uma falha libera as páginas novas e preserva histórico, relatório e prefixo.
  Bytes sujos após o prefixo confirmado são ignorados e sobrescritos na retomada.
- `reset` prepara o relatório antes de invalidar o contexto e liberar todas as
  páginas. `close` também as libera. Exceções retidas podem manter o objeto de
  metadados de uma página abortada, mas sua alocação física é liberada, mesmo quando
  o consumidor tenta novamente dentro de `except`.

A atomicidade é por chamada. O prefill em vários chunks da CLI usa um prefill e
appends sucessivos; não constitui uma única transação para o prompt inteiro.

## Orçamento, reserva e residência

Com `L` camadas, largura KV `W`, página de `P` tokens, contexto máximo `C` e chunk
máximo `T`, os custos são:

```text
payload por página = 2 * L * P * W * 4
stride de buffer = align64(P * W * 4)
alocação por página = 2 * L * stride + 63
páginas residentes no contexto N = ceil(N / P)
páginas reservadas para admissão = ceil(C / P) + ceil(T / P)
```

A reserva cobre o pior caso: contexto anterior cheio mais um novo prefill de
tamanho máximo. Append precisa apenas das páginas do contexto resultante. Essa
reserva é uma restrição do planner; as páginas só recebem alocação quando usadas.
Não há evicção de posições válidas para caber em um orçamento insuficiente.

O preflight soma a reserva KV, a arena para o maior chunk, tabelas de ponteiros,
scratch do contexto máximo, reader de 64 KiB, alinhamento e reserva do usuário.
Offsets do preflight são preservados quando buffers diminuem, garantindo que uma
chamada válida caiba na capacidade aceita. Planos que excedem o limite falham antes
de alocar páginas ou carregar pesos.

O relatório separa:

| Campo em `memory` | Significado |
|---|---|
| `kv_resident_allocation_bytes` | Páginas do contexto confirmado após a chamada, incluindo padding. |
| `kv_transaction_peak_allocation_bytes` | Páginas simultâneas durante a chamada, incluindo o prompt anterior na substituição. |
| `kv_reserved_capacity_bytes` | Limite reservado para admissão; não é alocação residente. |
| `kv_page_table_bytes` | Tabelas K/V contidas na arena da chamada. |
| `persistent_kv_bytes` | Payload das páginas residentes, incluindo posições ainda não usadas na última página. |
| `kv_valid_prefix_bytes` | Payload das posições confirmadas. |
| `managed_buffers_peak_bound_bytes` | Arena, reader e páginas simultâneas da chamada, sem duplicar as tabelas. |
| `capacity_managed_buffers_bound_bytes` | Limite de buffers da capacidade máxima, excluindo somente a reserva adicional do usuário. |

`memory_plan.reserves` usa a reserva de admissão. Os limites gerenciados continuam
excluindo objetos Python, metadados, bibliotecas, cache do SO e logits retidos pelo
consumidor. Eles não medem RSS/VRAM. Páginas pequenas podem aumentar padding e
tabelas; reduzir payload residente não implica reduzir o pico total em todo caso.

## Evidência reproduzível

A fixture não treinada de 728 parâmetros, IDs `[1,3,5,7]`, páginas de dois tokens
e chunks de dois tokens produziu erro máximo zero contra a referência Q4 PyTorch.
O erro Q4 contra os pesos originais permanece 0,4435420930. Foram quatro posições
processadas, sem cópia do prefixo; o hash dos logits coincide com o modo contíguo.

Para reproduzir também a medição de erro, use a referência opcional PyTorch já
instalada e o checkpoint original da fixture. Suas alocações ficam fora do orçamento:

```sh
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 2 --prefill-chunk-size 2 --tile-rows 3 --memory-budget 96KiB --verify --reference-checkpoint artifacts/checkpoints/nexalm-tiny --report artifacts/reports/nexalm-kv-paged.json
```

Para comparar residência com a mesma capacidade de oito tokens:

```sh
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 4 --prefill-chunk-size 2 --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-paged-residency.json
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --prefill-chunk-size 2 --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-contiguous-residency.json
```

Nesse caso, o paginado manteve uma página de 191 B, contra 575 B dos dois bancos
contíguos. O pico gerenciado entre chamadas foi 66.814 B contra 67.070 B, com o
mesmo hash de logits. A reserva de admissão paginada foi 573 B, cobrindo até três
páginas; ela não foi apresentada como memória fisicamente alocada.

São provas sintéticas em CPU, sem conclusão sobre qualidade de linguagem ou ganho
de velocidade. O [checklist](BLUEPRINT_512MB_CHECKLIST.md) mantém TQ01 para KV,
tiers/evicção, múltiplas sequências, compartilhamento de prefixos e GPU como
trabalho futuro. [Q4](NEXALM_KV_Q4_CPU.md) e [Q3](NEXALM_KV_Q3_CPU.md) CPU já
medem separadamente erro de execução e erro introduzido na representação KV.
