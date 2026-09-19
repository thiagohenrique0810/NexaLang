# Política hot/warm/cold do KV no CPU

`TieredTransformerSession` acrescenta uma política de idade de páginas:
**hot F32, warm Q4_GROUPED V1 e cold Q3_GROUPED V1**. A atenção lê páginas de
formatos diferentes diretamente, preservando o prefixo causal completo.
Neste modo os tiers continuam na RAM do host. O incremento seguinte acrescenta
[backing store CPU opcional](NEXALM_KV_BACKING_CPU.md), com evicção e recarga de
cold Q3. Promoção, TQ em páginas mistas e integração GPU permanecem pendentes.

O modo homogêneo F32/Q4/Q3/TQ continua disponível. Os pesos do modelo seguem Q4.

## Política e momento da transição

`TieredKVPolicy(hot_pages=1, warm_pages=1, group_size=32)` é imutável.
`hot_pages` deve ser positivo; `warm_pages` pode ser zero. Grupo tem os mesmos
limites dos codecs Q4/Q3. A política V1 é `CPU_PAGE_AGE_F32_Q4_Q3_V1`.

Para página lógica `i`, após uma chamada que termina com `N` páginas:

```text
age_rank = N - 1 - i
hot:  age_rank < hot_pages
warm: hot_pages <= age_rank < hot_pages + warm_pages
cold: age_rank >= hot_pages + warm_pages
```

A última página parcial conta como a mais recente e permanece F32. Somente
páginas completas migram. Idade usa posição lógica, não relógio ou frequência
de acesso. Nenhum token ainda necessário à atenção é descartado.

Cada chamada executa nesta ordem:

1. Valida a transição e o orçamento antes de ler pesos ou alocar páginas novas.
2. Grava o novo KV em F32; append pode completar o sufixo da última página hot.
3. Executa todas as camadas e produz logits com os codecs confirmados anteriores
   e as páginas F32 novas. Atenção usa posições e tabelas de cada página.
4. Recodifica páginas completas para o tier definido pela nova idade, em novas
   alocações. Mantém todas as páginas de origem até a publicação.
5. Prepara relatório e resultados, confirma páginas/tokens juntos e libera origens
   substituídas. Em falha, descarta os destinos e preserva o estado anterior.

Uma chamada longa pode saltar de F32 diretamente para Q3. Q4→Q3 reconstrói o
Q4 atual em F32 e quantiza essa reconstrução. Não há cópia oculta do KV original
para recuperar precisão. Páginas já cold não são recodificadas novamente.

**Fronteiras de chunk fazem parte do contrato numérico.** Uma página migrada
entre chamadas afeta a atenção das chamadas seguintes. Prefill longo e prefill
dividido podem produzir logits diferentes; repetir a mesma sequência de chunks
preserva o caminho de quantização. O relatório da CLI registra `token_chunks`,
e o oracle reproduz esse histórico em vez de aplicar um único codec ao final.

## Plano e execução

`compiler/tiered_kv_plan.py` define política, layouts, descritores imutáveis e
transições. Cada página registra índice, início lógico, tokens válidos, idade,
tier, codec/versão, grupo, identidade SHA-256 do layout e bytes da alocação.
O hash identifica o contrato do layout, não é checksum dos valores KV.

`TieredKVTransition` distingue páginas usadas pela atenção, destinos após a
transição, páginas F32 novas, migrações e residência/pico. JSON é validado contra
o plano canônico; metadados limitados a 65.536 páginas. Os formatos de plano
homogêneos anteriores não mudam. F32 nas páginas é nativo do host; o executor
Transformer atual exige host little-endian.

`nexa_causal_gqa_attention_paged_mixed` recebe tabelas de ponteiros, codec e
capacidade por página. F32, Q4 e Q3 são lidos por valor, acumulando em double.
O kernel suporta MHA/GQA/MQA, valida somente o prefixo visível e não aloca heap
nem expande um head ou página. Scores F32 continuam na arena planejada.

`nexa_kv_reencode_rows` transforma uma sequência de heads usando um scratch
F32 de `4 * head_dim` bytes e três doubles (24 B) de estatísticas. Esses buffers
pertencem ao chamador. A API nativa aceita todos os pares F32/Q4/Q3; a política
usa somente F32→Q4, F32→Q3 e Q4→Q3. Erro numérico pode deixar saída parcial;
o executor só publica destinos inteiramente válidos.

`kv_migration` registra bytes lógicos de origem/destino, páginas, contagem de valores,
erro máximo, soma quadrática e RMSE da transição. O erro compara a reconstrução
da origem com a do destino em double, incluindo o arredondamento da ponte F32.
Ele não mede o erro acumulado desde o KV original nem a qualidade dos logits.
Os bytes contam registros recodificados, não tráfego físico de memória; validação
e estatísticas podem reler valores da mesma origem.
`migration_wall_seconds` inclui alocação, recodificação e preparação de métricas
da fase de migração; a CLI também soma esses custos em `run_totals`.

## Orçamento e rollback

Cada codec reutiliza o tamanho físico do layout paginado, incluindo escalas,
padding e até 63 B para alinhar a base da página. A reserva de páginas é:

```text
reserva = residência canônica da capacidade total
        + ceil(max_chunk_length / page_tokens) * alocação_página_F32
        + max_pages * max(alocação_página_Q4, alocação_página_Q3)
```

O último termo é conservador: nem todas as páginas migram em uma chamada.
Tamanhos de grupo que fazem Q4/Q3 superar F32 também são admitidos com seus bytes
reais; escolher um tier de menor precisão não garante menor alocação.
Workspace, tabelas, leitura, reserva do usuário e `4D+24` de scratch de migração
entram no mesmo limite antes da execução.

O relatório distingue:

- Residência final de páginas por tier.
- Fase de atenção: páginas antigas/novas F32 + arena + scratch do reader.
- Fase de migração: páginas de origem + substitutas + scratch de recodificação.
- Pico gerenciado: máximo das duas fases; a arena de atenção é liberada antes
  da migração, sem somar buffers que não coexistem.

Na substituição de prompt, o KV antigo inteiro fica residente até o commit.
Falha de leitura, alocação, kernel, migração ou relatório mantém tokens, páginas
e relatório confirmados. Donos de buffers são explicitamente liberados mesmo
com exceções retidas. Append só altera o sufixo ainda não confirmado da página
hot; bytes válidos anteriores permanecem intactos em rollback.
`reset()` e `close()` liberam todas as páginas. A sessão não aceita concorrência.

Os limites são de buffers gerenciados CPU, não RSS ou VRAM. Excluem objetos
Python/metadados, overhead de alocação, bibliotecas, cache do SO, logits retidos
pelo consumidor e PyTorch usado opcionalmente como referência.

## Uso e reprodução

Com o bundle sintético do [guia de importação](NEXALM_IMPORTACAO.md):

```sh
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --kv-cache --kv-page-tokens 1 --kv-policy age --kv-hot-pages 1 --kv-warm-pages 1 --kv-group-size 3 --prefill-chunk-size 2 --memory-budget 96KiB --tile-rows 3 --report artifacts/reports/kv-tiers-tiny.json

# Referência opcional, preservando os mesmos chunks e transições.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --kv-cache --kv-page-tokens 1 --kv-policy age --kv-hot-pages 1 --kv-warm-pages 1 --kv-group-size 3 --prefill-chunk-size 2 --memory-budget 96KiB --tile-rows 3 --verify --reference-checkpoint artifacts/checkpoints/nexalm-tiny --report artifacts/reports/kv-tiers-tiny-verified.json
```

`--kv-policy age` exige cache e páginas. Não combinar com `--kv-codec q4/q3/tq`
ou opções TQ: a política fixa os codecs por tier. Sem `--kv-policy age`, o modo
homogêneo anterior é mantido. A API Python equivalente está em
`runtime.nexapack.tiered.TieredTransformerSession`.

Na tiny D4, chunks `[[1,3],[5],[7]]`, a política terminou em Q3/Q3/Q4/F32.
O kernel coincidiu com o oracle; erro máximo do KV nos logits = 0,1466720104,
dos pesos = 0,4435420930 e combinado = 0,4435420930. As quatro páginas ocupam
764 B, pois o alinhamento elimina o ganho físico nessa fixture pequena.

## Comparação wide local

Fixture [wide D64](NEXALM_KV_TQ_CPU.md), uma camada/head K/V, contexto 16,
página de dois tokens, chunks `[[1,3],[5,7],[2,4],[6],[8]]`, grupo 32 e orçamento
de 96 KiB. Todos os modos usam a mesma capacidade e os mesmos IDs.

| Medida | F32 | Q4 | Q3 | Idade H1/W1 |
|---|---:|---:|---:|---:|
| Payload KV final | 4.096 B | 640 B | 512 B | 1.440 B |
| Alocações de páginas finais | 4.348 B | 1.276 B | 764 B | 1.788 B |
| Pico gerenciado entre chamadas | 75.323 B | 72.316 B | 71.932 B | 73.980 B |
| Capacidade admitida | 81.142 B | 74.230 B | 73.078 B | 77.958 B |
| Erro máximo de execução contra seu oracle | 0 | 0 | 0 | 0 |
| Erro máximo de KV contra F32 nos logits | — | 0,2619032860 | 0,5301163346 | 0,3118940145 |

Idade mantém 1.024 B de payload hot, 160 B warm e 256 B cold nesse caso. Cinco
páginas foram recodificadas ao longo da execução, com 3.392 B lógicos de origem
e 736 B de destino. A fase de migração somou aproximadamente 0,138 ms nesta execução local;
essa amostra isolada não é um benchmark de velocidade. A política troca
memória, precisão e custo de migração; não há superioridade universal.

Relatórios: `artifacts/reports/kv-tiers-wide-{f32,q4,q3,age}.json`.
Para reproduzir a política, após gerar a fixture wide pelo guia citado:

```sh
python3 tools/nexa_run.py artifacts/models/nexalm-kv-wide --tokens 1,3,5,7,2,4 --decode-tokens 6,8 --kv-cache --kv-page-tokens 2 --kv-policy age --kv-hot-pages 1 --kv-warm-pages 1 --kv-group-size 32 --prefill-chunk-size 2 --max-sequence-length 16 --tile-rows 32 --memory-budget 96KiB --verify --report artifacts/reports/kv-tiers-wide-age.json
```

Para os baselines, remova as três opções de política e escolha `--kv-codec`;
F32 também exige remover `--kv-group-size`. Use saídas distintas. A referência
avalia equivalência numérica com pesos sintéticos, sem avaliar perplexidade.

## Validação e continuação

```sh
python3 -m unittest discover -s tests -p 'test_tiered*regressions.py' -v
python3 -S -m unittest discover -s tests -p 'test_tiered*regressions.py' -v
python3 -m unittest discover -s tests -p 'test_*regressions.py' -v
python3 tests/run_tests.py
make -C runtime all test
```

Planner, kernels mistos/recodificação, oracle independente, golden sem Torch,
lifetimes, rollback e CLI integram a regressão. A evidência local cobre macOS
ARM64; execução remota da matriz e GPU continuam pendentes. O
[checklist central](BLUEPRINT_512MB_CHECKLIST.md) registra totais e retomada.
