---
adr: ADR-0010
title: Esquema serializado do ModelIR — o grafo é derivado, não escrito à mão
status: accepted
identifiers:
  - name: compiler.model_ir:SCHEMA_VERSION
    value: 1
prior_art: []
---

# ADR-0010 — Esquema serializado do ModelIR

## Contexto

O runtime executa um grafo de operações com offsets de arena resolvidos antes de
qualquer alocação. Esse grafo precisa existir como dado, não como código, para
que o planejamento de memória seja verificável sem rodar o modelo.

## Problema técnico

Um grafo escrito à mão por arquitetura não escala e, pior, mente com facilidade:
nada garante que o grafo serializado corresponda à arquitetura que o manifesto
declara. Um grafo com uma operação a menos planeja uma arena menor do que a
execução vai precisar.

## Decisão

`ModelGraph` é **derivado** da `ModelConfig` por `lower_model`, e serializado com
`SCHEMA_VERSION` 1. Um documento cuja versão de esquema não seja exatamente 1 é
recusado na leitura, antes de qualquer interpretação.

A estrutura do grafo é uma função da contagem de camadas, e isso é o que torna a
derivação auditável: `15 * camadas + 3` operações. Quinze operações por camada,
mais três fora delas (embedding, norma final, projeção de saída).

Os lifetimes são seriais e derivados do grafo, não anotados. O runtime usa os
offsets que o plano resolveu e preserva residuais; constantes são streamadas em
buffers separados.

## Medições próprias

A contagem de operações, obtida rebaixando um modelo de verdade:

```adr-measurement
name: graph_ops_one_layer
value: 18
unit: operações
source: graph_ops_one_layer
```

```adr-measurement
name: graph_ops_sixteen_layers
value: 243
unit: operações
source: graph_ops_sixteen_layers
```

```adr-measurement
name: graph_ops_thirty_two_layers
value: 483
unit: operações
source: graph_ops_thirty_two_layers
```

Os dois últimos são exatamente os números publicados para R0 e v1 no registro do
terceiro incremento — 243 e 483 operações — recalculados aqui a partir de
`lower_model` com 16 e 32 camadas. Três pontos determinam a relação afim:
(243 − 18) / (16 − 1) = 15 operações por camada, e 18 − 15 = 3 fora delas.

## Alternativas descartadas

**Grafo escrito à mão por arquitetura.** Recusado pelo modo de falha acima: nada
liga o grafo à configuração, e a divergência aparece como arena insuficiente em
tempo de execução.

**Lifetimes anotados no grafo serializado.** Seriam um segundo lugar onde a
mesma verdade é escrita. Derivá-los mantém uma fonte só.

**Deixar as constantes na arena principal.** Recusado: elas têm tempo de vida
diferente das ativações e são streamadas de buffers separados, o que é o que
permite o pico ser dominado pelos logits e não pelos pesos.

## Limites declarados

`SCHEMA_VERSION` 1 é a única versão. Recusar outra não é compatibilidade
comprovada, é ausência de alternativa.

A relação `15 * camadas + 3` vale para a arquitetura Transformer que
`lower_model` rebaixa hoje. Ela não é uma propriedade de toda arquitetura, e uma
variante com atenção diferente mudaria a constante.

`MatMulProjectionFold` — uma das reescritas algébricas — **nunca foi executada
fora do avaliador**, porque o runtime não lê `ConstantDerivation`. E
`MAX_EVAL_ELEMENTS` impede verificar as arquiteturas reais, de modo que o zero de
M6.08 prova que nada casa, não que nada quebraria se casasse.
