# NexaKV CPU: critérios de qualidade para transições de idade

O vigésimo sétimo incremento acrescenta um **teto de erro por página** à política
de idade. Antes, envelhecer era incondicional: ao sair do tier quente a página era
recodificada e a nova versão substituía a original, qualquer que fosse o erro.
Agora o erro é medido **antes** de qualquer publicação, e uma página danificada
demais simplesmente não é adotada — ela permanece no codec em que já estava.

```bash
python3 tools/nexa_run.py MODELO --tokens 1,3 --decode-tokens 5 \
  --kv-policy age --kv-page-tokens 1 --kv-group-size 3 \
  --kv-quality-max-rmse 0.05 --kv-retain-pages 2 \
  --max-sequence-length 8 --tile-rows 3 --memory-budget 1MiB
```

## Retenção não é promoção

O item M4.05d pede "promoção de precisão". Promoção real — converter Q3 de volta
para F32 — **não existe** e o blueprint nunca poderá entregá-la: os bits
descartados na quantização não estão em lugar nenhum, e decodificar Q3 em um
buffer F32 devolve exatamente os mesmos valores, com mais bytes.

O que é possível é decidir **antes** de descartar. A transação de idade já mantinha
origem e destino residentes até o commit, justamente para poder desfazer um
re-encode que falhasse. O teto usa essa mesma janela para uma decisão de
qualidade em vez de uma de erro: o destino é escrito, medido e então descartado,
e o `final_pages` continua apontando para a página de origem. Por isso uma página
retida é **byte a byte** a que o prefill escreveu, não uma reconstrução que por
acaso arredonda de volta.

A consequência é que a retenção é pegajosa: os bytes não mudam, então medir de
novo a mesma migração só gastaria o mesmo trabalho para chegar ao mesmo veredito.
Uma página retida nunca mais é oferecida à migração.

## Orçamento explícito

Reter é gastar RAM. `--kv-retain-pages N` é obrigatório junto do teto e limita
quantas páginas podem ficar acima da própria idade; o plano soma
`N × (A_f32 − A_q3)` à reserva **antes** da primeira alocação, de modo que a
admissão nunca descobre a conta no meio de uma transação.

Numa configuração D64 (16 camadas, 4 heads KV, head_dim 64, páginas de 16 tokens,
grupo 32):

| | Bytes |
| --- | ---: |
| Página F32 | 524.351 |
| Página Q4 | 81.983 |
| Página Q3 | 65.599 |
| Reserva de retenção com `N=4` | 1.835.008 |
| Reserva total sem retenção | 7.295.164 |
| Reserva total com `N=4` | 9.130.172 |

Esgotado o orçamento, a página envelhece **mesmo falhando no teto**, e o relatório
diz isso em `retentions_declined`. O limite de residência vence o critério de
qualidade: o contrário transformaria um teto apertado em estouro de memória.

## O que foi medido

Fixture tiny, páginas de 1 token, grupo 3, prefill `[1,3]` e três decodes. O erro
por página aparece em `kv_migration.page_rmse`:

| Migração | RMSE por página |
| --- | ---: |
| F32 → Q4 | 0,030 – 0,044 |
| Q4 → Q3 | 0,077 – 0,097 |

A ponte para Q3 erra cerca de 2,3× mais que a primeira queda, o que dá sentido a
um teto entre as duas faixas. Comparando os logits finais contra um oráculo sem
envelhecimento (`--kv-hot-pages 8`, tudo F32):

| Configuração | Codecs finais | Erro máx. nos logits |
| --- | --- | ---: |
| Idade pura, sem teto | q3,q3,q3,q4,f32 | 0,05480 |
| Teto 0,05 · 2 páginas | q4,q4,q3,q4,f32 | 0,05474 |
| Teto 0 · 2 páginas | f32,f32,q3,q4,f32 | 0,01114 |
| Teto 0 · 4 páginas | f32,f32,f32,f32,f32 | 0,00000 |

O teto zero com orçamento total recupera o resultado exato — de novo porque a
página retida é a original, não uma reconstrução.

O resultado interessante é o do teto 0,05: ele mantém duas páginas em Q4 em vez de
Q3 e melhora os logits em 0,00006. **Um teto de RMSE local não é garantia de
qualidade global.** Ele limita o dano que uma transição individual pode causar; o
erro que dominava esse prefixo estava nas páginas que passaram no teto. Escolher
*quais* páginas merecem o orçamento exige importância por página, que depende da
calibração de M6.01 e ficou em M4.05e.

## Contrato

`TieredKVPolicy` ganhou `quality_max_rmse` e `retain_pages`; ambos entram no
`policy_id` serializado e na identidade de layout, então uma sequência derivada
não pode herdar um prefixo decidido sob outro teto. `TieredKVTransition` carrega
`retained_pages` (o que entra) e `final_retained_pages` (o que sobrevive), e os
descritores canônicos aceitam uma página **apenas acima** da precisão que a idade
lhe daria — envelhecer nunca anda para trás.

Um prefixo substituído não compartilha página nenhuma com o anterior, então
`prefill` e `reset` devolvem o orçamento inteiro. Uma sequência derivada herda as
páginas *e* a conta: se o pai gastou tudo, a filha não retém mais nada.

## Limites

O teto **não** é aceito junto de `--kv-backing-store`, e a sessão offloaded recusa
a opção em vez de ignorá-la: uma página retida deixa de ser cold, o que a tira do
backing store e muda o plano físico, os slots de recarga e o contrato de evicção.
Isso é M4.05e junto de prefetch, leitura assíncrona e importância por página.

O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
próxima tarefa.
