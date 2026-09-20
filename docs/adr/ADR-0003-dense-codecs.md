---
adr: ADR-0003
title: Codecs densos RAW_F16 e RAW_F32 — quando não empacotar é a resposta certa
status: accepted
identifiers:
  - name: runtime.nexapack.bundle:DENSE_CODECS
    value: ('f16', 'RAW_F16_MATRIX', 'f32', 'RAW_F32_MATRIX')
  - name: compiler.calibration:DENSE_CODECS
    value: ('f16', 'RAW_F16_MATRIX', 'f32', 'RAW_F32_MATRIX')
prior_art: []
---

# ADR-0003 — Codecs densos RAW_F16 e RAW_F32

## Contexto

A escada de ADR-0002 assume que empacotar ajuda. Em M6.02c isso foi medido e a
premissa quebrou para matrizes pequenas: o overhead do contêiner NexaPack é
**fixo por arquivo**, não escala com o tamanho da matriz, e abaixo de alguns
milhares de valores empacotar **aumenta** o arquivo.

## Problema técnico

Um planejador que só conhece codecs empacotados não tem como expressar "esta
matriz não deve ser empacotada". Ele escolheria o codec mais estreito, prometeria
o menor payload e entregaria o maior arquivo — que foi exatamente o que
aconteceu: na fixture tiny, o plano por payload escolheu 8× Q2, prometeu 880
bytes de payload e produziu um bundle 22 vezes maior que o plano por custo
físico.

## Decisão

Dois codecs densos sem escala: `RAW_F16_MATRIX` com 2 bytes por valor e
`RAW_F32_MATRIX` com 4. Eles não têm grupo, não têm escala e não têm tabela de
níveis — **a largura armazenada é o contrato inteiro**. Isso é o que os torna
utilizáveis como piso do planejador: o custo é previsível sem medir nada.

`DENSE_CODECS` aparece em dois módulos, com o mesmo conteúdo, porque a
calibração precisa das variantes densas para medir sensibilidade e o bundle
precisa delas para gravar. Os dois são reivindicados aqui.

## Medições próprias

A largura por valor, lida da tabela que o escritor de matriz densa usa:

```adr-measurement
name: f16_bytes_per_value
value: 2.0
unit: bytes por valor
source: f16_bytes_per_value
```

```adr-measurement
name: f32_bytes_per_value
value: 4.0
unit: bytes por valor
source: f32_bytes_per_value
```

```adr-measurement
name: f16_bits
value: 16
unit: bits por valor
source: f16_bits
```

```adr-measurement
name: f32_bits
value: 32
unit: bits por valor
source: f32_bits
```

Ao contrário dos codecs agrupados, aqui bits/8 **é** a largura: não há escala
por grupo a somar. Comparado com o Q4 de ADR-0002 (0,625 B/valor), o F16 custa
3,2× e o F32 custa 6,4× por valor — e ainda assim o F16 venceu a seleção física
na fixture tiny, porque o que decide arquivo pequeno não é o payload.

O tempo de decodificação denso é citado, não recalculado, e contém a surpresa do
incremento: o F32 é o **mais rápido** de todos os seis codecs, e o F16 é o mais
lento:

```adr-measurement
name: f32_decode_ns_per_value
value: 0.8139
unit: ns por valor (macOS ARM64, clang -O2, escalar, mediana de 21 trials)
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro da onda 4a (M6.03a)
```

```adr-measurement
name: f16_decode_ns_per_value
value: 3.9523
unit: ns por valor (macOS ARM64, clang -O2, escalar, mediana de 21 trials)
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro da onda 4a (M6.03a)
```

## Alternativas descartadas

**BF16.** Tem a faixa dinâmica do F32 com metade dos bytes, o que é melhor que
F16 para pesos. Não foi implementado como codec de armazenamento: o importador
já lê BF16 de Safetensors e converte. Fica registrado como não avaliado.

**Nenhum codec denso, compensando com um grupo enorme no Q8.** Um grupo do
tamanho da linha faria a escala custar quase nada por valor, mas uma escala por
linha destrói grupos com outliers — e continuaria pagando o overhead fixo do
contêiner empacotado que era o problema original.

## Limites declarados

O F32 denso é formato de referência e calibração, **não de embarque**: custa oito
vezes o Q4. O próprio código diz isso, e é só o limite de bloco que protege uma
leitura.

Estes números são larguras de armazenamento. Eles não dizem nada sobre erro:
o F32 é exato por construção e o F16 não, mas quanto o F16 erra depende do
tensor.
