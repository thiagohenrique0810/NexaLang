---
adr: ADR-0002
title: Escada de codecs agrupados Q2/Q3/Q4/Q8 — um contêiner, quatro larguras
status: accepted
identifiers:
  - name: runtime.nexapack.format:CODEC_ID
    value: 'Q4_GROUPED'
  - name: runtime.nexapack.format:CODEC_VERSION
    value: 1
  - name: runtime.nexapack.format:Q2_CODEC_ID
    value: 'Q2_GROUPED'
  - name: runtime.nexapack.format:Q2_CODEC_VERSION
    value: 1
  - name: runtime.nexapack.format:Q3_CODEC_ID
    value: 'Q3_GROUPED'
  - name: runtime.nexapack.format:Q3_CODEC_VERSION
    value: 1
  - name: runtime.nexapack.format:Q8_CODEC_ID
    value: 'Q8_GROUPED'
  - name: runtime.nexapack.format:Q8_CODEC_VERSION
    value: 1
  - name: runtime.nexapack.format:_GROUPED_CODECS
    value: ('Q4_GROUPED', 'q4', 'Q8_GROUPED', 'q8', 'Q3_GROUPED', 'q3', 'Q2_GROUPED', 'q2')
  - name: runtime.nexapack:CODEC_ID
    value: 'Q4_GROUPED'
  - name: runtime.nexapack:CODEC_VERSION
    value: 1
  - name: runtime.nexapack.executor:CODEC_ID
    value: 'Q4_GROUPED'
  - name: runtime.nexapack.executor:CODEC_VERSION
    value: 1
prior_art:
  - id: P02
    role: solution-category
    note: Quantização agrupada com escala por grupo é uma família conhecida, citada na trilha M6 (P01-15). A citação nomeia a família; não verifica as larguras medidas abaixo, que vêm do escritor deste repositório.
---

# ADR-0002 — Escada de codecs agrupados Q2/Q3/Q4/Q8

## Contexto

Um modelo dentro de 512 MB não cabe com uma largura só. Tensores diferentes
toleram erros diferentes, e a escolha de codec por tensor (ADR-0008) só tem
sentido se houver uma escada de codecs para escolher.

## Problema técnico

Quatro codecs poderiam significar quatro formatos, quatro leitores e quatro
modos de falha. Cada formato novo é um lugar onde offset, checksum e cobertura
podem divergir, e divergem silenciosamente: um leitor que erra a largura da
linha lê o payload do tensor vizinho e ainda bate com o próprio SHA, porque os
offsets deslizam juntos.

## Decisão

Um contêiner só, quatro larguras. Todo codec agrupado grava uma escala float32
little-endian por grupo, seguida dos códigos empacotados a partir do bit menos
significativo. O que muda entre eles é a contagem de níveis e o número de bits
por valor — não o layout do arquivo, não o índice, não a checagem.

`_GROUPED_CODECS` mapeia o id de codec para o dtype de armazenamento e é o
único lugar onde essa correspondência existe. Um normalizador (`_block_codec`)
aceita as duas grafias — `'q3'` e `'Q3_GROUPED'` — para que um erro de digitação
vire uma recusa em vez de uma terceira grafia de codec.

Q4 é o padrão: `CODEC_ID` sem qualificador é `Q4_GROUPED`, no pacote, no módulo
de formato e no executor.

## Medições próprias

A largura por valor em grupo 32, perguntada ao escritor real sobre uma linha de
1024 colunas. São exatamente os quatro valores da coluna "B/valor" da tabela de
velocidade de decodificação do registro da onda 4a, recalculados aqui:

```adr-measurement
name: q2_bytes_per_value_group32
value: 0.375
unit: bytes por valor
source: q2_bytes_per_value_group32
```

```adr-measurement
name: q3_bytes_per_value_group32
value: 0.5
unit: bytes por valor
source: q3_bytes_per_value_group32
```

```adr-measurement
name: q4_bytes_per_value_group32
value: 0.625
unit: bytes por valor
source: q4_bytes_per_value_group32
```

```adr-measurement
name: q8_bytes_per_value_group32
value: 1.125
unit: bytes por valor
source: q8_bytes_per_value_group32
```

Nenhuma dessas larguras é o número de bits dividido por oito: a escala float32
por grupo custa 4 bytes, e é ela que põe 0,125 B/valor em cima de cada codec em
grupo 32.

```adr-measurement
name: grouped_scale_bytes
value: 4
unit: bytes por grupo
source: grouped_scale_bytes
```

Os níveis por codec, de onde sai a escala `max|v| / levels`:

```adr-measurement
name: q2_levels
value: 1
unit: níveis
source: q2_levels
```

```adr-measurement
name: q3_levels
value: 3
unit: níveis
source: q3_levels
```

```adr-measurement
name: q4_levels
value: 7
unit: níveis
source: q4_levels
```

```adr-measurement
name: q8_levels
value: 127
unit: níveis
source: q8_levels
```

O tempo de decodificação por valor **não** é recalculado aqui. Ele foi medido no
kernel que o executor roda, e depende de máquina, compilador e carga:

```adr-measurement
name: q4_decode_ns_per_value
value: 1.5079
unit: ns por valor (macOS ARM64, clang -O2, escalar, mediana de 21 trials)
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro da onda 4a (M6.03a)
```

```adr-measurement
name: q2_decode_ns_per_value
value: 1.6977
unit: ns por valor (macOS ARM64, clang -O2, escalar, mediana de 21 trials)
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro da onda 4a (M6.03a)
```

## Alternativas descartadas

**Um formato por codec.** Recusado pelo modo de falha descrito acima: janelas de
offset aplicadas num lugar e esquecidas noutro produzem leitura errada que passa
no próprio checksum.

**Codec 2:4 esparso em grupo 32**, avaliado no reconhecimento da onda 2: custa
16 B por grupo, idêntico ao Q3, e erra mais. Um codec que empata em bytes e
perde em erro nunca seria escolhido pelo planejador, então não foi implementado.

**Escala em float16.** Economizaria 2 B por grupo — 0,0625 B/valor em grupo 32,
10% do Q4 — mas introduz uma segunda precisão de escala no mesmo arquivo e a
faixa dinâmica do f16 recorta grupos com outliers. Não foi medido; fica
registrado como não avaliado, não como descartado por número.

## Limites declarados

As larguras acima são exatas e verificáveis. O **erro** de cada codec não está
nesta seção porque depende do tensor: o RMSE de Q4→Q3 medido na fixture tiny
(0,077–0,097) não se transfere para pesos treinados.

A ordenação por bytes e a ordenação por tempo discordam, quase invertidas
(ADR-0008). Esta escada ordena bytes, e só bytes.

`qint<N>` e `PackedVector<N>`, os tipos de armazenamento do lado da linguagem em
`bootstrap/`, empacotam byte a byte igual a estes codecs para N em {2,3,4,8}.
Eles **não** aparecem no censo de identificadores deste registro: a regra
percorre `compiler` e `runtime.nexapack`, e a implementação daqueles tipos está
fora dos dois. Um renomeio lá não derruba nenhum teste deste ADR.
