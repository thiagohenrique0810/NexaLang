# NexaKV CPU: evicção e recarga com backing store

O décimo segundo incremento acrescenta `--kv-backing-store DIRETÓRIO` ao modo
`--kv-policy age`. Páginas hot F32 e warm Q4 ficam em RAM; páginas cold Q3
completas são gravadas em arquivos privados. Todos os tokens continuam visíveis
à atenção causal. A recarga preserva bytes, sem nova quantização.

Este modo é CPU, síncrono e de uma sequência por sessão. O diretório contém um
cache temporário, não um checkpoint de inferência retomável após encerrar o
processo. Os modos anteriores permanecem disponíveis.

## Contrato de residência e execução

A política e a ordem de migração seguem o [contrato de tiers](NEXALM_KV_TIERS_CPU.md):
novos tokens entram em F32; depois de todas as camadas/logits do chunk, páginas
completas podem passar a Q4 ou Q3. O histórico de chunks continua afetando a
quantização. Uma página já cold nunca é recodificada durante a recarga.

`TieredKVTransition` descreve a história lógica de codecs. `OffloadedKVPlan`
contabiliza separadamente as páginas físicas em RAM. No relatório, os bytes da
transição lógica são explicitamente identificados como o cenário hipotético
todo residente; `physical_memory` contém a residência realmente admitida.

A atenção usa dois percursos em ordem crescente de páginas por camada:

1. Calcula o máximo por posição de consulta/head.
2. Recalcula scores, acumula denominador e V e divide para produzir a saída F32.

O estado usa `8 × chunk × query_heads × (head_dim + 2)` bytes: máximos,
denominadores e acumuladores double. Os pesos exponenciais são arredondados
para F32 como no kernel misto residente, mantendo a mesma ordem de redução
por consulta/lane. Não há vetor de scores do prefixo nem prefixo descompactado.

Cada página em disco é verificada e carregada em um slot Q3. O slot inclui K/V
de todas as camadas. Com o padrão de **um slot**, a ordem de execução exige nova
leitura em cada passagem de cada camada, e os números desta página descrevem
esse caso. O incremento seguinte admite `--kv-reload-slots N` e retém páginas
entre passagens, camadas e chamadas; veja o [reuso de páginas cold](NEXALM_KV_RELOAD_CPU.md).
Prefetch e promoção de precisão continuam pendentes. Os kernels não alocam heap.

## Arquivos e integridade

`KVPageStore` cria `nexa-kv-*` dentro do diretório escolhido. Sessões compartilham
o diretório pai, mas possuem subdiretórios e referências exclusivos — com uma
exceção explícita: uma [sequência derivada](NEXALM_KV_SEQUENCIAS_CPU.md) toma um
hold sobre os arquivos do pai e sobre o store, e o último dono é quem remove. Apenas arquivos
criados pela própria instância são removidos. Reset elimina referências antigas;
close remove os arquivos próprios e o subdiretório se estiver vazio. Arquivos
alheios e o diretório pai são preservados.

O formato `NEXAKV01`, versão 1, usa header little-endian `<8sHHIQ32s>`:

| Campo | Tamanho | Significado |
| --- | ---: | --- |
| magic | 8 B | `NEXAKV01` |
| versão/flags | 2 B + 2 B | versão 1; flags zero |
| metadata_bytes | 4 B | JSON canônico, máximo 8.192 B |
| extent_bytes | 8 B | extensão física do payload |
| metadata_sha256 | 32 B | checksum da metadata |

A metadata inclui identidade do manifesto do modelo, descritor lógico/codec/layout,
extensão e SHA-256 do payload. O reader compara a metadata canônica com a referência
emitida pela própria sessão. Não usa um caminho fornecido pelo arquivo.

O payload guarda exatamente `page_extent_bytes`: K/V de todas as camadas e seu
padding de alinhamento, excluindo os 63 bytes extras da alocação que alinham a base.
Checksums cobrem esses bytes, inclusive padding. Header inválido, troca entre
páginas/sessões, corrupção, truncamento e bytes excedentes são rejeitados antes
de consumir a página no kernel.

Escrita e leitura usam views emprestadas, chunks de até 65.536 B e arquivos sem
buffer Python de payload. A escrita publica um temporário com `fsync` e rename
atômico. Não há expansão/cópia integral adicional. A reserva de buffers de I/O é
16.896 B para metadata/header/digest/EOF; objetos Python e estado interno do hash
continuam fora do escopo de buffers explícitos.

Em sistemas POSIX, os caminhos são relativos a um descritor do diretório e a abertura
usa `O_NOFOLLOW` quando disponível. O fallback de plataforma opera sobre o diretório
privado. Não há promessa de recuperação após queda de energia/processo: arquivos
órfãos de uma sessão encerrada abruptamente não são reaproveitados automaticamente.

## Transações e falhas

Admissão de orçamento precede criação de diretório, slot, páginas, arena e biblioteca
nativa. Atenção, recodificação, persistência e preparação do relatório precedem a
publicação de tokens/páginas. Todas as origens confirmadas sobrevivem até o commit.

Falha de leitura/escrita, checksum, falta de espaço ou cancelamento antes do commit
preserva prefixo, tokens e relatório anteriores. Destinos e slot são descartados,
inclusive com traceback retido. Arquivos novos são removidos; falha de remoção
mantém ownership para nova tentativa ou close. Retry pode reutilizar a sessão após
resolver a causa, por exemplo restaurar bytes corrompidos ou liberar espaço.

Após o commit, falha ou cancelamento durante a coleta de arquivos antigos adia
somente essa coleta: a chamada já confirmada retorna normalmente. Antes da próxima
transação, cancelamento na coleta propaga sem executar novos tokens. Essa distinção
evita anunciar falha de decode depois de já publicar o token. Reset e close também
liberam buffers; close com erro de remoção pode ser repetido.

## Orçamento e relatórios

Se `A_F`, `A_W`, `A_C` são alocações de páginas F32/Q4/Q3, `H/W` as contagens da
política, `N` a capacidade em páginas e `E=ceil(max_chunk/page_tokens)`:

```text
resident_capacity = min(N,H)*A_F + min(max(N-H,0),W)*A_W
page_reserve = resident_capacity + E*A_F + min(N,H+W+E)*max(A_W,A_C)
```

A reserva também inclui slot `A_C`, scratch de migração `4*head_dim+24`, I/O,
arena/ativações, alinhamento e reader de pesos. Não pressupõe que Q3/Q4 sempre
ocupam menos que F32: grupos grandes e heads pequenos podem inverter o ganho.
Todos os destinos podem coexistir até o commit, e um prefill substituto mantém
as origens antigas durante sua preparação.

O pico de buffers gerenciados é o máximo entre atenção e migração/persistência.
O relatório separa payload lógico, payload em disco, residência RAM, slot usado,
reserva de slot, estado da atenção e limite da capacidade. É contabilidade dos
buffers explícitos, **não RSS, cache do SO, memória Python ou VRAM**. Os campos de
RSS/VRAM continuam `null`.

`kv_backing_bytes_read/written` contam header+metadata+payload das operações bem
sucedidas; `kv_page_reloads` inclui as duas passagens por camada. Chamadas que
falham preservam o relatório confirmado anterior, sem publicar métricas parciais.
Tempos de leitura, escrita, migração e execução são separados. Read está contido
no tempo de execução; eviction acrescenta sua etapa após a execução nativa.

## Execução e evidências

```bash
python3 tools/nexa_run.py artifacts/models/nexalm-tiny \
  --tokens 1,3 --decode-tokens 5,7 --kv-cache --kv-page-tokens 1 \
  --kv-policy age --kv-hot-pages 1 --kv-warm-pages 1 --kv-group-size 3 \
  --kv-backing-store artifacts/kv-store/demo --prefill-chunk-size 2 \
  --memory-budget 96KiB --tile-rows 3 --verify \
  --report artifacts/reports/kv-offloaded-tiny.json
```

`--verify` é opcional e usa Torch como oracle; a execução CPU e a fixture golden
não dependem dele. O oracle repete os mesmos chunks e mede quantização separadamente.

Na fixture sintética D64/P2/H1/W1/G32, 512 tokens em chunks de 2 produziram logits
**exatamente iguais** nos modos residente e offloaded, no mesmo host macOS ARM64:

| Métrica | Tiers residentes | Cold em disco |
| --- | ---: | ---: |
| Páginas residentes ao final | 256 | 2 |
| Alocação KV residente | 49.920 B | 1.406 B |
| Pico dos buffers gerenciados | 130.431 B | 91.899 B |
| Limite admitido para capacidade | 212.566 B | 93.136 B |
| Evicções / recargas | 0 / 0 | 254 / 64.262 |
| Bytes lidos / gravados no backing store | 0 / 0 | 43.627.130 / 172.555 |

Uma execução local observou aproximadamente 0,66 s residente e 2,76 s offloaded
com um slot;
não é benchmark repetido nem prova de velocidade de disco/NVMe. A redução de RAM
tem custo de I/O. Em contextos curtos, slot/metadata podem aumentar o pico.
Modelo treinado, perplexidade, promoção de precisão e GPU continuam pendentes.

Para reproduzir a fixture, execute `python3 tests/kv_backing_fixture.py --out
artifacts/models/nexalm-kv-offload-512`, use os 512 IDs `[(i*7+1)%13 for i in range(512)]`
e os dois modos com capacidade 512, chunk 2, página 2, H1/W1/G32, tile 32 e orçamento
256KiB. Os relatórios locais estão em `artifacts/reports/kv-offloaded-512-*.json`.
O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a próxima tarefa.
