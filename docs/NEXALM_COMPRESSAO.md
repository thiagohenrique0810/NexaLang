# CompressionPlanner: os eixos que existem e o tempo que foi medido

Este incremento faz duas coisas que não se parecem. A primeira é um
planejador que **não decide nada de novo**: ele encaminha a escolha de codec
para o `select_precision` que já existia e produz o mesmo JSON, byte a byte. A
segunda é a única que acrescenta conhecimento: o **tempo de decodificação de
cada codec foi medido**, e com ele o planejador ganhou um teto por tempo ao
lado do teto por bytes.

A primeira parte existe por causa da segunda. Uma fachada sobre código que
funciona só se justifica se carregar alguma coisa que o código embaixo não
carrega — aqui, o registro explícito dos eixos que o planejador **não** sabe
decidir, e o eixo de velocidade que deixou de ser suposição.

```bash
python3 tools/nexa_calibrate.py --checkpoint ./tinyllama --tokens 1,3 \
  --report calibracao.json                      # mede bytes, erro e tempo
python3 tools/nexa_precision.py plan --calibration calibracao.json \
  --max-rmse 0.05 --max-decode-ns 300000 \
  --out precisao.json --axes eixos.json
```

## O número: bytes não predizem tempo

A escada de codecs foi cronometrada no kernel que o executor realmente roda —
`nexa_<codec>_matmul` sobre um bloco de 32×1024 com um vetor de ativação, o
mesmo caminho que `TransformerSession` toma para cada matriz de pesos. Medir
`decode_row` linha a linha mediria sobretudo o despacho de ctypes do Python,
que o executor não paga nos pesos.

macOS 26.5.2, ARM64, clang `-O2`, uma thread, Python 3.14.5. Mediana de 21
trials, repetições por trial escolhidas para cada trial durar ≥ 10 ms:

| Codec | Bytes/valor | ns por byte | ns por valor | Pior spread num trial | Entre 3 execuções |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2 | 0,375 | 4,5273 | **1,6977** | 7,8% | 1,3% |
| Q3 | 0,500 | 3,4468 | **1,7234** | 52,2% | 0,7% |
| Q4 | 0,625 | 2,4127 | **1,5079** | 5,9% | 0,8% |
| Q8 | 1,125 | 0,8881 | **0,9992** | 8,2% | 1,2% |
| F16 | 2,000 | 1,9761 | **3,9523** | 9,4% | 1,2% |
| F32 | 4,000 | 0,2035 | **0,8139** | 8,3% | 0,9% |

As duas ordenações:

```text
por bytes:  Q2 < Q3 < Q4 < Q8 < F16 < F32
por tempo:  F32 < Q8 < Q4 < Q2 < Q3 < F16
```

São quase o inverso uma da outra. Três leituras diretas:

- **Q2 guarda 40% menos bytes que Q4 e demora 12,6% mais para decodificar.**
  Era exatamente a pergunta em aberto, e a resposta é sim: o codec menor é mais
  lento. Desempacotar códigos de dois bits custa mais aritmética por byte do
  que desempacotar quatro.
- **F16 é o pior dos dois mundos nesta máquina**: ocupa o dobro de Q8 (1,78×
  os bytes) e leva **3,96×** o tempo. Contra F32 ocupa metade e leva **4,86×**
  — a conversão half→float é escalar, e copiar float é quase de graça.
- **F32 é o mais rápido de todos.** Otimizar só velocidade diz "não comprima
  nada". É por isso que o tempo entra como **teto**, não como custo otimizado:
  minimizar tempo sozinho não é um plano de compressão.

O spread de 52,2% é honesto e vale explicar: numa das três execuções, um trial
de Q3 pegou o escalonador do sistema. A mediana dos 21 trials mal se moveu
(0,7% entre execuções), que é a razão de a mediana ser publicada e a média não.
O relatório traz `ns_per_block_min`, `ns_per_block_max` e `relative_spread`
justamente para que esse caso seja visível em vez de absorvido no número.

## O que isso muda num plano

Três tensores de 128×512 no grupo 32 — os tamanhos são os reais do formato:
24.576 bytes em Q2, 32.768 em Q3, 40.960 em Q4, 73.728 em Q8, 131.072 em F16 e
262.144 densos.

Com um teto de erro frouxo, o plano por bytes escolhe **3× Q2**: 73.728 bytes e
≈345.000 ns para decodificar. Baixando o teto de tempo em 10% (execução de
2026-09-20):

| Limite | Codecs escolhidos | Bytes | ns para decodificar |
| --- | --- | ---: | ---: |
| Só bytes | Q2, Q2, Q2 | 73.728 | 345.330 |
| Bytes + `max_decode_ns` 310.797 | Q4, Q4, Q2 | 106.496 | **309.752** |

Tirar **10,3%** do tempo custou **+44,4%** de bytes. Em outras execuções o
planejador chegou ao mesmo teto por caminhos diferentes — `Q8, Q2, Q2` com
122.880 bytes, ou `Q4, Q4, Q4` — porque Q2 e Q3 ficam a 2% um do outro em tempo
por valor e a taxa medida decide o desempate. O tamanho do efeito é estável; a
identidade do tensor promovido não é, e isso está dito aqui em vez de escondido
numa média.

O caso mais instrutivo aparece quando há **teto de erro e teto de tempo juntos**
(`--max-rmse 0.05 --max-decode-ns` a 50% do tempo do plano só-erro):

| Limite | Codecs | Bytes | ns |
| --- | --- | ---: | ---: |
| Só erro | F16, Q8, Q8 | 278.528 | 387.024 |
| Erro + tempo | **F32**, Q8, Q8 | 409.600 | **181.504** |

O tensor mais sensível sai de F16 para F32: **1,47× os bytes, 2,1× mais rápido
e um erro estimado menor**. F16 estava ali por ser o menor codec preciso o
bastante; o relógio diz que era também o mais lento da escada. Nenhum plano
antes deste incremento tinha como enxergar essa troca.

Os dois planos de cada tabela são respostas certas para perguntas diferentes, e
antes deste incremento só existia a pergunta de cima.

A ordem é deliberada. O teto de tempo é resolvido **primeiro**, comprando
tempo com os menos bytes extras por nanossegundo economizado; só depois roda a
subida por erro-evitado-por-byte, que recusa qualquer degrau que devolva o
plano para cima do teto. Um plano que não cabe no tempo não é um plano, então
a viabilidade não negocia com a qualidade.

A fronteira de dominância por tensor também ficou bidimensional: uma opção
sobrevive a não ser que alguma mais barata seja ao mesmo tempo **tão precisa
quanto e tão rápida quanto** ela. Sem taxas medidas o tempo de toda opção é
zero, a segunda metade do teste é vazia, e a fronteira é exatamente a de antes
— que é como a identidade byte a byte se sustenta.

## A previsão é conferida contra o relógio

O plano publica `provenance.decode_time.planned_decode_ns`. Esse número é
taxa medida × bytes do tensor, e duas regressões o conferem contra o relógio —
`test_the_predicted_decode_time_of_a_plan_matches_a_measurement_of_that_plan` e
`test_the_measured_rate_predicts_the_relative_cost_of_every_codec_pair`. Elas
codificam as matrizes que o plano descreve, rodam os kernels do executor sobre
elas com um laço de cronometragem escrito **dentro do teste** e comparam:

| Plano | Previsto | Relógio | Razão |
| --- | ---: | ---: | ---: |
| Limitado por bytes (3× Q2) | 346.743 ns | 322.836 ns | 0,931 |
| Limitado por tempo (Q4, Q4, Q2) | 295.785 ns | 283.453 ns | 0,958 |

A taxa foi medida num bloco 32×1024 e aplicada aqui a um 128×512: não são a
mesma medição, e a diferença de forma desloca o número de forma sistemática —
relógio/previsto ficou entre **0,856 e 1,035** por codec em três execuções. Por
isso a banda absoluta aceita é larga (0,5×–2,0×) e vem acompanhada de uma
segunda checagem, mais afiada, **entre pares de codecs**: ali o deslocamento
comum se cancela, o pior par ficou 18% fora em três execuções, e a tolerância é
30%. Uma taxa ligada ao codec errado move um par por 3× a 20× e não passa em
nenhuma das duas.

## Os eixos que o planejador recusa

M6.03 pede codec, sparsity e low-rank. Dois dos três não existem neste
repositório, e o `CompressionPlanner` **levanta erro nomeando o eixo** em vez de
aceitar o argumento e planejar outra coisa:

| Eixo | Situação |
| --- | --- |
| `codec` | planejado, contra bytes medidos e sensibilidade medida |
| `decode_time` | medido; vira restrição quando `max_decode_ns` é declarado |
| `sparsity` | **recusado**: não há codec em `PACKED_CODECS`, não há kernel em `_PACKED_KERNELS`, não há campo de manifesto (M6.06) |
| `low_rank` | **recusado**: não há fatoração no importador, no manifesto ou no grafo, nem verificação de equivalência (M6.07) |
| `peak_vram` | **não mensurável**: não há caminho de execução em GPU aqui |
| `transfer_bytes` | **não mensurável**: não há transferência host→device para cronometrar |
| `energy` | **não mensurável**: nenhum contador de energia é lido nesta máquina |

Passar `sparsity=None` também é recusado: nomear o eixo já é a afirmação de que
ele foi planejado. As razões acima não são texto decorativo — a regressão
`test_the_refusal_reasons_still_describe_the_runtime` lê `PACKED_CODECS` e
`_PACKED_KERNELS` e falha no dia em que um codec esparso entrar, obrigando o
texto a ser reescrito em vez de continuar mentindo.

## Contrato

- `CompressionPlan.to_json()` **é** o mapa de precisão, e nada mais. Ele
  alimenta `nexa_convert.py --precision-map` exatamente como antes; um arquivo
  que um conversor lê não pode crescer campos que o conversor não entende. A
  proveniência de eixos sai por `--axes` ou por `to_report_json()`.
- Os dois `policy_id` não mudaram: `GREEDY_SENSITIVITY_PER_BYTE_V2` e
  `GREEDY_SENSITIVITY_PER_PHYSICAL_BYTE_V3`. O teto de tempo é uma restrição
  adicional dentro da mesma política, como já era o teto de erro, e aparece em
  `provenance.decode_time`.
- Um plano **sem** `max_decode_ns` não ganha campo nenhum de proveniência. O
  silêncio é o registro honesto de que o tempo não foi consultado.
- A taxa tem política própria, `BLOCK_MATMUL_NS_PER_STORED_BYTE_V1`, gravada no
  relatório de calibração. Pedir um teto de tempo contra um relatório sem o
  bloco `decode_time` é recusado nomeando o bloco.

## Limites

O teto de tempo conta **uma decodificação de cada peso**. Não é latência ponta
a ponta: leitura de disco, atenção, ativações e a orquestração em Python ficam
de fora, e nenhum desses é proporcional aos bytes do peso.

A taxa é desta máquina, deste compilador, desta thread e destes kernels
escalares. Ela não viaja para outro host, e não diz nada sobre GPU — por isso o
relatório carrega o host junto do número, e por isso o planejador se recusa a
planejar `peak_vram`.

O número inclui o multiply-add por valor, porque é o que o executor paga. Isso
favorece levemente os codecs de menos bytes numa comparação por byte, e é a
razão de a tabela publicar **ns por valor** ao lado de ns por byte: a decisão
entre dois codecs para o mesmo tensor é por valor.

Sparsity e low-rank continuam sem existir. Um planejador que os aceitasse seria
pior que nenhum, porque quem chamasse acreditaria que o eixo foi considerado.

O [custo físico](NEXALM_CUSTO_FISICO.md) explica por que os bytes do arquivo
não são os bytes do codec; a [calibração](NEXALM_CALIBRACAO.md) explica de onde
vêm as sensibilidades. O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a
suíte, os comandos e a próxima tarefa.
