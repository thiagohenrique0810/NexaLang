# NexaKV CPU: reuso de páginas cold com slots de recarga

O décimo terceiro incremento acrescenta `--kv-reload-slots N` ao modo
`--kv-policy age --kv-backing-store DIRETÓRIO`. O slot único do
[backing store](NEXALM_KV_BACKING_CPU.md) passa a ser um cache de `N` slots
admitidos que retêm páginas cold Q3 entre as duas passagens da atenção, entre
camadas e entre chamadas da mesma sessão.

O padrão continua `1`, que preserva o comportamento anterior. O cache troca I/O
por RAM: nada muda no formato, na quantização, na ordem de redução ou nos
logits. Uma página cold é imutável depois de publicada, então um slot carregado
entrega exatamente os mesmos bytes que uma nova leitura verificada entregaria.

## Política de admissão e substituição

`CPU_RELOAD_FIRST_TOUCH_MRU_V1`. Um slot é alocado no primeiro miss que precisa
dele — capacidade admitida não é página residente. Enquanto houver slot livre,
a página é admitida nele. Com o cache cheio, a substituição recai sobre o slot
**carregado por último**.

A ordem importa porque a atenção relê o mesmo prefixo em ordem crescente, duas
vezes por camada:

1. Passagem 0: máximo por consulta/head, página a página.
2. Passagem 1: recalcula scores, acumula denominador e V, divide.

Sob essa varredura cíclica, LRU descarta justamente a página que será pedida
primeiro na volta e erra em todas. A substituição do slot mais recente mantém
`N-1` páginas fixas e usa o último slot como passagem, o que dá acertos
determinísticos. Com `N` maior ou igual ao número de páginas cold do prefixo,
cada página é lida **uma vez por chamada** em vez de `2 × camadas` vezes.

Consequência direta: **`--kv-reload-slots 1` não retém nada** quando o prefixo
tem mais de uma página cold, porque a passagem 1 recomeça no início. O primeiro
slot só produz acertos com uma única página cold. O ganho é função do número de
slots e do tamanho do prefixo frio, não da existência do cache.

A passagem 0 poderia ser percorrida em qualquer ordem, mas a passagem 1 não: sua
ordem de soma define os bits do resultado. O contrato de ordem foi preservado,
e nenhuma leitura foi evitada às custas de mudar a redução.

## Ownership, transações e falhas

O cache é dono apenas dos seus slots. Arquivos, referências e páginas
confirmadas continuam do `KVPageStore` e da sessão. As regras:

- Uma entrada é válida enquanto sua referência estiver confirmada. Após o
  commit, o cache retém somente as referências das páginas finais; um prefill
  substituto ou uma página aposentada esvaziam suas entradas.
- Uma leitura que falha pode ter sobrescrito parte do slot. A entrada é
  descartada antes de carregar e só é publicada após a verificação completa do
  payload, então um erro nunca deixa bytes parciais visíveis à atenção.
- Qualquer falha ou cancelamento antes do commit libera **todos** os slots,
  inclusive os já carregados com sucesso. Uma transação descartada não deixa
  residência para trás; a chamada seguinte relê o que precisar.
- `reset` e `close` liberam os slots antes de publicar o estado vazio.
  Sessões distintas não compartilham slots, como não compartilham arquivos.
- Os slots sobrevivem à migração da mesma chamada. Suas páginas são cópias
  imutáveis, e a mesma admissão já reservou essa capacidade.

## Orçamento e relatórios

A reserva soma um termo novo, com `A_C` igual à alocação de uma página Q3:

```text
reload_cache_capacity = N * A_C
reserva = page_allocation_limit + reload_cache_capacity + migration_scratch + io_buffers
```

`reload_slots_used` de uma transação é `min(N, páginas cold da atenção)`: um slot
só pode conter uma página cold do próprio prefixo. O pico das fases passa a
incluir a residência do cache:

```text
attention_peak  = workspace + attention_pages + reload_bytes (+ io_buffers se houver recarga)
migration_peak  = migration_pages + reload_bytes + max(migration_scratch, io_buffers)
```

O relatório separa `kv_reload_slots`, `kv_reload_slots_used`,
`kv_reload_cache_capacity_bytes`, `kv_reload_cache_allocated_bytes` e
`kv_reload_cache_entries`, e acrescenta em `io` os campos
`kv_reload_cache_hits`, `kv_reload_cache_misses`, `kv_reload_bytes_avoided`,
`kv_reload_slot_admissions` e `kv_reload_slot_evictions`. `kv_page_reloads` e
`kv_backing_bytes_read` continuam contando apenas leituras efetivas.
`kv_reload_bytes_avoided` é o que essas leituras teriam custado, não uma medida
de tempo economizado. Tudo isso é contabilidade de buffers explícitos, **não
RSS, cache do SO, memória Python ou VRAM**.

## Evidências

Fixture D64/P2/H1/W1/G32, 512 tokens em chunks de 2, capacidade 512, tile 32,
orçamento 256KiB, macOS ARM64/Python 3.14.5. Uma camada, 253 páginas cold ao
final. Os logits foram **idênticos** nos quatro casos
(`bb25e54a8800b2700e0eb26cc4e089e98f53caab8867001d4968f7012037838f`) e iguais aos
dos tiers residentes:

| Slots | Recargas | Bytes lidos | Acertos | Cache residente | Pico gerenciado | Limite admitido | Amostra local |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64.260 | 43.625.778 | 2 | 191 B | 91.899 B | 93.136 B | ~4,2 s |
| 8 | 60.767 | 41.263.520 | 3.495 | 1.528 B | 93.236 B | 94.473 B | ~3,8 s |
| 64 | 36.351 | 24.705.826 | 27.911 | 12.224 B | 103.932 B | 105.169 B | ~3,3 s |
| 256 | 253 | 171.875 | 64.009 | 48.323 B | 140.031 B | 141.841 B | ~2,3 s |

As evicções para disco continuaram 254 em todos os casos: o cache muda a
releitura, não a política de idade nem a residência lógica. O KV residente em
RAM permaneceu 1.406 B; o que cresce é o cache, contabilizado à parte.

**Cenário sem ganho.** Com uma única página cold, `N` maior não evita leitura
alguma e apenas aumenta a capacidade admitida — o caso está coberto por
regressão. No extremo oposto, 256 slots elevam o pico a 140.031 B, acima dos
130.431 B dos tiers **residentes** medidos no incremento anterior: reter tudo em
slots reproduz o custo de RAM do modo residente e ainda paga metadata e I/O.
Entre os dois extremos, o número de slots é um controle explícito de RAM contra
I/O, e não uma melhoria automática. Os tempos são amostras locais únicas, não
benchmark repetido nem evidência sobre NVMe, disco de rede ou GPU.

## Execução e reprodução

```bash
python3 tools/nexa_run.py artifacts/models/nexalm-tiny \
  --tokens 1,3,5 --decode-tokens 7,2 --kv-cache --kv-page-tokens 1 \
  --kv-policy age --kv-hot-pages 1 --kv-warm-pages 1 --kv-group-size 3 \
  --kv-backing-store artifacts/kv-store/reuso --kv-reload-slots 4 \
  --prefill-chunk-size 2 --memory-budget 96KiB --tile-rows 3 \
  --report artifacts/reports/kv-reload-tiny.json
```

Para a fixture de 512 tokens, use `python3 tests/kv_backing_fixture.py --out
artifacts/models/nexalm-kv-offload-512`, os IDs `[(i*7+1)%13 for i in range(512)]`
e `--kv-reload-slots` em 1, 8, 64 e 256; os relatórios locais ficam em
`artifacts/reports/kv-reload-slots-*.json`. Regressões:

```sh
python3 -m unittest discover -s tests -p 'test_reload_cache_regressions.py' -v
python3 -m unittest discover -s tests -p 'test_offloaded*regressions.py' -v
```

## Limites

Prefetch, leitura assíncrona, slots compartilhados entre sessões e promoção de
**precisão** continuam pendentes. Recarregar uma página Q3 em F32 não recupera o
original: promoção de residência e promoção de precisão são coisas diferentes, e
apenas a primeira foi implementada. Importância por página, budgets por camada e
critérios de qualidade para transições permanecem em M4.05d, dependentes da
calibração de M6.01. O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a
suíte, os comandos e a próxima tarefa.
