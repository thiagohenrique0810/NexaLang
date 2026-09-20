# ADRs do NexaLang — formato e registro verificado por máquina

Um ADR aqui documenta um **contrato de armazenamento que já existe no código e
já foi medido**. Ele não propõe, não planeja e não antecipa: descreve uma
decisão tomada, os identificadores que ela fixa e os números que a sustentam.

O que separa estes arquivos de Markdown decorativo é que **o código é o
oráculo**. `tests/test_adr_regressions.py` caminha por `compiler` e
`runtime.nexapack`, colhe todo identificador de contrato que encontra e exige
uma bijeção com o que os ADRs reivindicam. Acrescentar `Q5_GROUPED`, subir um
`FORMAT_VERSION` ou renomear um `POLICY_ID` sem escrever o ADR derruba a suíte.

## Identificador de contrato

Um nome de módulo é identificador de contrato quando, ignorando underscores à
esquerda, ele é todo maiúsculo e:

- é exatamente `MAGIC`, `FORMAT`, `FORMAT_VERSION` ou `SCHEMA_VERSION`; ou
- termina em `CODEC_ID`, `CODEC_IDS`, `CODEC_VERSION`, `CODECS`,
  `TRANSFORM_ID`, `POLICY_ID` ou `POLICY_IDS`.

Nomes privados contam. `_GROUPED_CODECS` decide qual dtype cada codec grava e é
tão contrato quanto `CODEC_ID`; escondê-lo do censo por causa do underscore
seria escolher não enxergar.

O valor é **achatado**: dicionários contribuem chaves e valores, tuplas e listas
contribuem elementos, tudo na ordem de iteração. É isso que faz um codec novo
dentro de `PACKED_CODECS` invalidar o ADR que declarou o conjunto antigo, em vez
de passar despercebido por o nome não ter mudado.

Um identificador é reivindicado por **exatamente um** ADR. Um re-export conta
como identificador próprio: `runtime.nexapack:CODEC_ID` é a superfície que os
consumidores importam, e `runtime.nexapack.format:CODEC_ID` é onde ela nasce.
Os dois aparecem, normalmente no mesmo ADR.

## Template

~~~markdown
---
adr: ADR-0000
title: Título curto, em uma linha
status: accepted
identifiers:
  - name: pacote.modulo:NOME
    value: 'VALOR_LITERAL'
prior_art:
  - id: P07
    role: problem
    note: Por que a categoria descreve o problema, e não uma verificação.
---

# ADR-0000 — Título

## Contexto

O que existia antes, e por que a decisão precisou ser tomada.

## Problema técnico

O problema concreto, em termos do formato, do kernel ou do orçamento.

## Decisão

O que foi decidido, com os identificadores que a decisão fixa.

## Medições próprias

Cada número num bloco `adr-measurement`, com a fonte que o recalcula:

```adr-measurement
name: nome_da_medicao
value: 4096
unit: bytes
source: nome_em_SOURCES
```

## Alternativas descartadas

O que foi considerado e por que perdeu — de preferência com o número que
decidiu.

## Limites declarados

O que a decisão **não** prova. Esta seção é obrigatória.
~~~

## Medições: recalculadas contra citadas

Todo bloco `adr-measurement` declara um `source`, e há só dois tipos.

**Recalculado.** `source` nomeia uma chave de `SOURCES` em
`tests/adr_registry.py`. O teste executa a função e compara com `value`. A
função obtém o número do código real — escrevendo um arquivo de verdade,
perguntando a largura ao escritor de verdade, construindo um plano de verdade.
Um número que o ADR afirma e o código contradiz derruba a suíte.

Uma fonte que não alcance `compiler/` nem `runtime/` é **recusada** pelo próprio
teste: uma função que devolvesse um literal concordaria com o ADR fizesse o
código o que fizesse, e isso é a tautologia que este registro existe para
impedir.

**Citado.** `source: CITED` e um `cited_from` obrigatório apontando o registro
de incremento. Serve para número que não é recalculável — tempo de parede é o
caso típico: uma mediana de nanossegundos por valor depende da máquina, do
compilador e da carga, e um teste que a reafirmasse seria intermitente ou
frouxo demais para significar algo.

**Um número citado não é um número verificado.** O teste conhece a lista exata
de medições citadas; converter uma medição recalculada em citada exige editar o
teste, então a distinção não se perde em silêncio.

## Prior art (P01–P31)

A bibliografia P01–P31 vem dos PDFs de planejamento. Ela entra num ADR de duas
formas, e o parser recusa qualquer outra:

- `role: problem` — a categoria enuncia o problema que o contrato resolve.
- `role: solution-category` — o contrato pertence a uma família conhecida de
  soluções.

**Nunca como verificação independente.** Nenhuma alegação de um PDF confirma
uma medição deste repositório. As medições próprias, e só elas, sustentam os
números; o campo `note` de cada referência tem de deixar isso explícito.

## O bloco atual

| ADR | Contrato |
| --- | --- |
| [ADR-0001](ADR-0001-nexapack-v1-container.md) | Contêiner NexaPack v1 (`.nxp`) |
| [ADR-0002](ADR-0002-grouped-codec-ladder.md) | Escada de codecs agrupados Q2/Q3/Q4/Q8 |
| [ADR-0003](ADR-0003-dense-codecs.md) | Codecs densos RAW_F16 / RAW_F32 |
| [ADR-0004](ADR-0004-model-bundle-manifest.md) | Manifesto do bundle de modelo |
| [ADR-0005](ADR-0005-turboquant-storage.md) | Armazenamento TurboQuant TQ_MSE_SRHT |
| [ADR-0006](ADR-0006-nxb-container.md) | Contêiner de arquivo único `.nxb` |
| [ADR-0007](ADR-0007-mixed-codecs-per-block.md) | Codecs mistos por bloco de linhas |
| [ADR-0008](ADR-0008-precision-and-compression-policy.md) | Políticas de precisão e compressão |
| [ADR-0009](ADR-0009-kv-plans-and-tiers.md) | Planos de KV, tiers e recarga |
| [ADR-0010](ADR-0010-model-ir-schema.md) | Esquema serializado do ModelIR |
| [ADR-0011](ADR-0011-plasticity-schema.md) | Esquema de plasticidade e regiões |
| [ADR-0012](ADR-0012-joint-admission.md) | Admissão conjunta de memória |
