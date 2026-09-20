# Álgebra de grafo: avaliador independente, passes e verificador

Este incremento entrega M6.08a. A manchete honesta é um zero: **os três passes
casam ZERO sítios no grafo Llama real.** Medido em `lower_model` de duas
camadas, S=4 — 33 operações, 15 MatMul, **nenhum par MatMul→MatMul** — e também
nas duas arquiteturas de `models/nexalm512/architecture.nxl` (243 e 483
operações). O valor da entrega é o avaliador e o verificador, não o ganho.

```bash
python3 tools/nexa_graph.py rewrite \
  --definition models/nexalm512/architecture.nxl --model NexaLM512_R0
```

## Por que existe um segundo oráculo

`tests/transformer_reference.py` é um oráculo Torch preso à estrutura Llama:
ele sabe o que é uma camada, não o que é um operador. Um verificador de
reescrita precisa do contrário — avaliar qualquer `ModelGraph`, inclusive um
que nenhum importador produziria.

`compiler/graph_eval.py` é esse avaliador: Python puro, float64, sem kernels C
e sem PyTorch. Acumula em double e arredonda cada elemento produzido para
float32, porque toda ativação do IR é um tensor F32.

### Oráculo contra oráculo

A circularidade se mata comparando duas implementações que não compartilham
aritmética. O executor C escalar (`runtime/nexapack/transformer.py`) e o
avaliador Python rodam sobre **os mesmos pesos decodificados e os mesmos
tokens**, e os logits são comparados um a um:

| Camadas | Heads | KV heads | Tied | Erro abs. máx. | Erro rel. máx. |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 2 | 1 | sim | 2,384e-07 | 3,389e-06 |
| 2 | 4 | 2 | não | 4,768e-07 | 6,169e-06 |
| 2 | 2 | 2 | sim | 2,384e-07 | 1,807e-06 |
| 2 | 4 | 1 | não | 2,980e-07 | 9,640e-06 |

Tolerância declarada **antes** da medição: `atol = 1e-5`, `rtol = 1e-4`. O erro
não é zero de propósito: o kernel C guarda os exponenciais do softmax em float32
e o avaliador os mantém em double. Um zero aqui significaria que as duas
implementações compartilham código que não deveriam. Se o avaliador errasse a
RoPE meia-rotação, o GQA causal ou o silu, o teste falharia — e falha: os bugs
correspondentes foram injetados e capturados.

## Contrato de reescrita

`compiler/graph_algebra.py` define `GraphRewrite`: casar sítios, devolver um
`ModelGraph` novo e validado, e **declarar o que promete**.

| Exatidão | O que o pass manager exige |
| --- | --- |
| `exact` | bytes float32 das saídas **idênticos** |
| `tolerance` | erro absoluto medido ≤ tolerância publicada |

`run_passes(graph, passes, verifier=...)` mede a promessa antes de aceitar o
grafo. Um pass que se diz exato e mexe num bit é recusado com `RewriteError`,
e o chamador fica com o grafo anterior. Um pass que casa zero sítios não é
verificado — não há o que verificar — e o relatório registra o zero.

### Os três passes

| Pass | Exatidão | O que casa |
| --- | --- | --- |
| `DeadOpElimination` | `exact` | operação de que nenhuma saída declarada depende |
| `CommonSubexpressionElimination` | `exact` | mesma operação, mesmas entradas, mesmos atributos |
| `MatMulProjectionFold` | `tolerance` 1e-4 | `x@A^T@B^T` → `x@(B·A)^T` |

O fold é o único inexato: somar sobre a dimensão compartilhada em outra ordem
move os últimos bits do float32. Ele só casa quando a matriz dobrada **não é
maior** que as duas que substitui, e recusa quando o intermediário tem outro
consumidor ou quando algum dos operandos não é constante.

### Constante que nenhum checkpoint tem

`B·A` não existe em bundle nenhum. O fold não inventa os números: publica um
`ConstantDerivation` no relatório — nome, receita, fontes e forma — e quem for
executar o grafo reescrito materializa aquele tensor. O verificador usa a mesma
receita, então a medição de erro cobre a constante derivada de verdade.

`RewriteReport` é versionado (`schema_version` 1) e serializável, com um
registro por pass: sítios, contagens de operações e tensores antes/depois, a
tolerância publicada e a verificação medida.

## O zero, com todos os números

| Grafo | Operações | MatMul | DCE | CSE | Fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny, 2 camadas, S=4 | 33 | 15 | 0 | 0 | 0 |
| NexaLM512_R0, S=4 | 243 | 113 | 0 | 0 | 0 |
| NexaLM512_v1, S=4 | 483 | 225 | 0 | 0 | 0 |

A razão é estrutural, não acidental: `lower_model` não emite operação morta nem
subexpressão repetida, e entre duas projeções sempre há RoPE, atenção ou um
`Add` residual. **Não existe par MatMul→MatMul num Transformer Llama.** O fold
serve a grafos que uma reescrita anterior produza, não ao grafo lowering.

## Limites

O fold não foi exercitado numa execução real: o grafo reescrito exige que o
executor materialize a constante derivada, e `runtime/nexapack` ainda não lê
`ConstantDerivation`. Enquanto isso, `nexa_graph.py rewrite --out` produz um
grafo que só o avaliador Python consegue rodar.

Faltam os passes que dariam ganho num Llama — fusão de RMSNorm com a projeção
seguinte, fusão de RoPE na projeção Q/K, `Add` residual em cima do matmul.
Todos dependem de kernels fundidos que não existem, então seriam reescritas sem
executor. Ficam preservados em M6.08b.

O verificador mede em grafos pequenos, por amostragem de duas sementes. Ele
prova que um pass está errado; não prova que está certo para toda entrada.

O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte e a próxima tarefa.
