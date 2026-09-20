# PrecisionMap: seleção por custo físico medido

O vigésimo nono incremento troca o número que o planejador de precisão otimiza.
Até aqui ele escolhia codecs contra o **payload** — os bytes que o codec
codifica. Agora também sabe escolher contra o **custo físico**: os bytes que o
arquivo do tensor realmente ocupa, incluindo cabeçalho do container, metadados
por bloco e checksums.

```bash
python3 tools/nexa_precision.py plan --calibration report.json \
  --max-rmse 0.05 --cost physical --out precision.json
```

## Por que os dois números discordam

O container NexaPack cobra um **overhead fixo por tensor empacotado** —
cabeçalho e metadados de bloco — que não escala com o tamanho da matriz. Medido
diretamente sobre o escritor, com grupo 32 e oito linhas:

| Valores | Q4 no disco | F16 no disco | F32 no disco | Menor |
| ---: | ---: | ---: | ---: | --- |
| 512 | 4.416 | 1.024 | 2.048 | F16 |
| 2.048 | 5.376 | 4.096 | 8.192 | F16 |
| 8.192 | 9.216 | 16.384 | 32.768 | **Q4** |
| 32.768 | 24.576 | 65.536 | 131.072 | **Q4** |
| 524.288 | 331.776 | 1.048.576 | 2.097.152 | **Q4** |

O ponto de virada fica entre 2.048 e 8.192 valores. **Abaixo dele, empacotar
aumenta o arquivo**: o Q4 de uma matriz de 512 valores codifica 192 bytes e
ocupa 4.416. Acima, o overhead vira ruído e a compressão manda — que é o caso
de qualquer modelo real.

O overhead é o mesmo em números absolutos nos três tamanhos; o que muda é a
proporção. Por isso ele decide a escolha só nos tensores pequenos, e por isso
não aparecia enquanto o planejador contava payload.

## O que isso muda na prática

Na fixture tiny (oito matrizes de 512 bytes lógicos cada), com teto de erro
frouxo para deixar o planejador ir ao mais barato que ele enxerga:

| Custo otimizado | Codecs escolhidos | Bytes planejados | Tensores no disco | Bundle |
| --- | --- | ---: | ---: | ---: |
| `payload` | 8× Q2 | 880 | 33.744 | 39.073 |
| `physical` | 8× F16 | 1.408 | **1.504** | **9.555** |

O plano que se diz mais barato — 880 contra 1.408 bytes — produz um bundle
**22 vezes maior**. Não é um erro do planejador: ele otimizava exatamente o que
lhe foi dado. O erro era o número.

Nenhum codec empacotado sobrevive à seleção física nessa fixture, e isso sai de
graça: a fronteira de dominância que já existia descarta uma opção que custa
mais e não é mais precisa. Q4 custa 4.288 bytes físicos e erra mais que F16, que
custa 256 — não há orçamento que o torne a resposta.

## Contrato

O custo faz parte da política, porque as mesmas sensibilidades ordenam
diferente contra cada um:

| Custo | `policy_id` |
| --- | --- |
| `payload` | `GREEDY_SENSITIVITY_PER_BYTE_V2` |
| `physical` | `GREEDY_SENSITIVITY_PER_PHYSICAL_BYTE_V3` |

O `policy_id` deixou de ser constante do módulo e passou a ser campo do mapa, e
`PrecisionMap.cost_basis` diz contra o que aquele mapa foi planejado. Um mapa
antigo continua sendo lido; um mapa com política desconhecida é recusado.

A calibração mede os dois números na **mesma variante real** que já construía
para medir sensibilidade, então `physical_bytes` não é estimativa: é o
`physical_file_bytes` do bundle. Cada medição traz também
`container_overhead_bytes` e `physical_saved_bytes` — este último **negativo**
quando empacotar não compensa, que é como o relatório mostra o problema sem
precisar do planejador.

Planejar por custo físico exige um relatório que os contenha; um relatório antigo
é recusado nomeando o campo que falta, e continua planejando por payload.

## Limites

Falta **misturar codecs dentro de um tensor** (M6.02d). Não é uma extensão do
que existe: hoje um tensor empacotado tem um codec, um `group_size` e um
`row_bytes` uniformes, e o leitor calcula o offset de um bloco multiplicando.
Codec por bloco exige formato versionado com codec, offset e identidade por
bloco, e um despacho que troque de kernel dentro do mesmo matmul. O ganho também
precisa ser demonstrado: o overhead fixo medido aqui é **por arquivo**, e
metadados por bloco heterogêneo tendem a aumentá-lo.

O custo físico cobre o que está no disco. Não é a residência em RAM durante a
execução, que os relatórios de memória já reportam separadamente.

A suspeita que esta seção registrava — "um codec menor pode decodificar mais
devagar" — deixou de ser suspeita em M6.03a: **Q2 guarda 40% menos bytes que Q4
e leva 12,6% mais tempo para decodificar**, medido no kernel que o executor
roda. A escada inteira, as duas ordenações e o teto por tempo estão em
[compressão](NEXALM_COMPRESSAO.md).

O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
próxima tarefa.
