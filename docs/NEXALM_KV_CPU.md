# KV F32 incremental e prefill em chunks

O executor incremental reutiliza as chaves e valores já calculados. Cada decode
executa embedding e projeções somente para o novo ID; atenção consulta o prefixo
armazenado. Ele mantém os kernels C, pesos Q4 por tiles e normas F32 do
[forward CPU](NEXALM_EXECUCAO_CPU.md), sem PyTorch no runtime.

## API e CLI

`runtime.nexapack.incremental.IncrementalTransformerSession` fornece:

- `prefill(ids)`: substitui o prompt e retorna os logits fornecidos nessa chamada.
- `append(ids)`: acrescenta um chunk ao contexto, retornando somente seus logits.
- `decode(id)`: acrescenta um ID, retornando uma linha de logits.
- `reset()` e `close()`: invalidam o contexto e encerram a sessão, respectivamente.
- `token_ids`, `cache_length`, `active_bank` e `report()`: estado confirmado.

`append`/`decode` exigem um prefill bem-sucedido. A sessão é síncrona e atende uma
sequência. A CLI preserva a execução anterior por padrão e ativa o cache com
`--kv-cache`:

```sh
# Use a fixture já gerada/importada conforme o guia de importação.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --kv-cache --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-f32.json

# Divide o prompt, limitando ativações e logits temporários a dois tokens.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --prefill-chunk-size 2 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-chunks.json
```

Na API, `max_sequence_length` é a capacidade do contexto/cache;
`max_chunk_length` limita o tamanho de cada prefill/append. O segundo assume o
primeiro quando omitido. A CLI converte `--prefill-chunk-size` nesse limite e
divide o prompt entre um prefill e chamadas append. A atomicidade é por chamada;
um consumidor da API que quiser substituir um prompt inteiro em uma só transação
deve permitir esse tamanho de chunk.

IDs fora do vocabulário, chunks vazios, limites inválidos e contexto excedido são
rejeitados. Não há tokenização ou detokenização de texto neste comando.

## Plano explícito e transações

`compiler/kv_plan.py` define `KVCachePlan` e `KVStepPlan`, com JSON versionado e
validação estrita. O primeiro aloca dois bancos F32, cada um com K/V por camada,
no layout `[capacity, kv_heads * head_dim]`, alinhados a 64 bytes. O segundo
descreve a ordem de Compute, RoPE com offset, CacheWrite, CachedAttention e Commit.
Os lifetimes de ativações são derivados dessa agenda; o executor consome seus
nomes, offsets, tamanhos e dependências.

Em um prefill, o executor escreve no banco inativo. Em append/decode, escreve
somente o sufixo ainda não confirmado do banco ativo. A atenção lê o prefixo
válido e as novas posições da chamada, respeitando a máscara causal. A posição
inicial de RoPE é o comprimento anterior do contexto.

O commit de banco/histórico ocorre após todas as camadas, projeção de saída e
produção de resultados/relatório. Uma falha pode deixar bytes no banco inativo ou
no sufixo não confirmado, mas não altera o prefixo anterior. A tentativa seguinte
sobrescreve a faixa necessária. Não existe cópia do cache antigo nem buffer de
rollback do prefixo. Reset apenas invalida posições; não zera toda a alocação.

Exceções guardadas pelo consumidor não devem manter arenas transitórias ou scratch
de checksum vivos. O executor e o reader liberam esses proprietários em `finally`,
inclusive quando uma nova tentativa ocorre dentro do bloco `except`.

## Orçamento

Todos os bancos permanecem reservados pela sessão. Com `L` camadas, capacidade
`C`, `Hkv` heads KV e dimensão `D`, o payload dos dois bancos ocupa:

```text
2 bancos * 2 (K,V) * L * C * Hkv * D * 4 bytes
```

O relatório separa payload, alinhamento por buffer e padding de até 63 bytes da
base da arena KV. Soma essa alocação inteira à arena de ativações/staging, ao
scratch de checksum de 64 KiB e à reserva do usuário. O scratch da atenção tem
um float por posição visível; não há matriz de scores quadrática.

Com chunks limitados, o preflight combina ativações do maior chunk permitido com
scratch do contexto máximo e os dois bancos completos. Essa validação acontece
antes da alocação KV, leitura de pesos ou carregamento da biblioteca nativa. Cada
chamada também valida o plano específico antes de carregar payloads.

As chamadas reutilizam os offsets do preflight, reduzindo somente os tamanhos dos
buffers. Isso garante que qualquer chunk dentro da capacidade aceita caiba no
orçamento: recalcular first-fit para tamanhos menores poderia aumentar fragmentação.
Os lifetimes e a ausência de sobreposição são validados novamente em cada plano.

Em contexto 2.048, o layout atual reserva 128 MiB de payload KV para R0 e 256 MiB
para v1, antes de ativações, staging, alinhamento e scratch. São dois bancos F32
para transações; não são cache comprimido ou medição de VRAM. O modo paginado
opcional é descrito em outro guia; [Q4 paginado](NEXALM_KV_Q4_CPU.md) oferece
quantização do KV sem alterar os dois bancos F32 descritos aqui.

O teto cobre buffers CPU gerenciados. Objetos Python, metadados, assets, bibliotecas,
cache do SO e logits retidos pelo consumidor estão fora dele. RSS/VRAM continuam
sem medição. Os pesos Q4 de projeção ainda são lidos a cada chamada; o cache evita
recalcular posições antigas, sem manter os pesos residentes.

## Evidência e interpretação dos relatórios

A fixture de 728 parâmetros, com prefill `[1,3]` e decode `5,7`, executou quatro
posições no total, contra nove no baseline por recomputação. O último decode leu
uma linha de embedding, usou GEMMs com batch 1 e escreveu 32 bytes novos de KV,
sem copiar o prefixo. Os dois bancos somaram 256 B de payload e 319 B de alocação
com padding. O pico contabilizado entre as três chamadas foi 67.070 B sob 96 KiB.
Limitando chunks a dois tokens, o segundo comando acima atingiu 66.814 B e produziu
o mesmo hash dos logits do contexto completo.

Os logits coincidiram com a referência Q4 PyTorch nesse caso. O erro máximo de
quantização contra os pesos originais permaneceu 0,4435420930. A fixture não foi
treinada e esses números não demonstram qualidade de linguagem ou desempenho GPU.

`report()` da sessão descreve o chunk executado: `processed_tokens`, posição
inicial, contexto confirmado, grafo do chunk, plano KV e I/O. O hash de logits da
sessão corresponde a esse chunk. A CLI reúne as linhas e expõe um hash do contexto
completo, preservando também `last_chunk_logits_sha256`; `run_totals` agrega as
chamadas, inclusive chunks de prefill.

Há provas de equivalência para diferentes limites de chunk, múltiplas camadas,
MHA/MQA/GQA, embeddings tied/untied, golden sem Torch, oracle com KV independente,
orçamento exato, corrupção após escrita da primeira camada e falhas tardias.
ASan/UBSan e testes sem heap cobrem os kernels com offset/cache.

O marco seguinte acrescentou [KV paginado F32](NEXALM_KV_PAGINADO_CPU.md), com
alocação sob demanda e atenção consumindo as páginas diretamente. Use
`--kv-cache --kv-page-tokens 16` para esse modo; os dois bancos descritos neste guia
continuam disponíveis. O [checklist](BLUEPRINT_512MB_CHECKLIST.md) mantém demais codecs KV,
tiers, múltiplas sequências e backend GPU como gates futuros.
