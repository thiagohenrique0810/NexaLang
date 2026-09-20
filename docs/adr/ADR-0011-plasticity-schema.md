---
adr: ADR-0011
title: Esquema de plasticidade — regiões de parâmetro como alvo físico nomeado
status: accepted
identifiers:
  - name: compiler.model_plasticity:SCHEMA_VERSION
    value: 1
prior_art:
  - id: P31
    role: solution-category
    note: A trilha M8 cita P04/P20/P31 para treino, e o PDF de Plastic Learning descreve a família. A citação nomeia a família de soluções; o esquema e as contagens abaixo são deste repositório.
---

# ADR-0011 — Esquema de plasticidade e regiões de parâmetro

## Contexto

Aprendizado plástico exige dizer **o que pode mudar**. Sem isso, um adapter pode
escrever sobre qualquer tensor, e a distinção entre o núcleo estável do modelo e
a memória de trabalho não existe como dado.

## Problema técnico

"Treinável" não é uma propriedade de um tensor inteiro. Um adapter de baixo
posto toca um subconjunto de linhas; uma região de memória de trabalho pode ser
uma fatia. Um esquema que só marque tensores inteiros obriga a promover o tensor
todo a plástico para tornar uma fatia plástica.

## Decisão

`ModelPlasticityConfig` com `SCHEMA_VERSION` 1, resolvido contra uma
`ModelConfig` em um `PlasticityMap`. A unidade é a **região**, e uma região é um
conjunto de `TensorSpan` — tensor mais limites de linha — e não um tensor.

Cinco classes de região, e elas são uma escada de permanência: `stable_core`,
`mature_expert`, `plastic_expert`, `adapter_delta`, `working_memory`. A ordem
vai do que nunca muda ao que é descartável.

`protected_tensors` e `validate_learning_target` fazem a verificação ser ativa:
um alvo de aprendizado fora de uma região plástica é **recusado**, não ignorado.

O adapter nulo é byte-idêntico ao baseline. O adapter não nulo bate com um
oráculo float64 independente **e difere** do baseline — as três asserções
juntas, porque sem a terceira um delta que nunca fosse somado passaria nas duas
primeiras.

## Medições próprias

As classes de região e sua ordem, que é a escada de permanência:

```adr-measurement
name: plasticity_region_classes
value: ('stable_core', 'mature_expert', 'plastic_expert', 'adapter_delta', 'working_memory')
unit: classes de região
source: plasticity_region_classes
```

```adr-measurement
name: plasticity_region_class_count
value: 5
unit: classes
source: plasticity_region_class_count
```

## Alternativas descartadas

**Marcar tensores inteiros como treináveis.** Recusado pelo problema acima: uma
fatia plástica obrigaria a promover o tensor todo.

**Só duas classes, congelado e treinável.** Não distingue um expert maduro de um
delta de adapter, que têm políticas de persistência diferentes — um sobrevive ao
descarte da memória de trabalho, o outro não.

**Deixar a verificação do alvo para o laço de treino.** Recusado: a recusa é
barata na resolução do mapa e cara depois que gradientes já escreveram.

## Limites declarados

**Isto é contrato e estrutura, não aprendizado.** Nenhum treino real rodou contra
este esquema: PL.04 e as demais tarefas de valor dependem de checkpoint treinado.

A defesa do adapter foi verificada, não afirmada — substituir o corpo de
`_apply_adapters` por `return` derruba 10 testes. Isso prova que o caminho é
exercitado, não que o resultado é bom para um modelo.

`SCHEMA_VERSION` 1 é a única versão.
