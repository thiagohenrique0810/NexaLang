---
adr: ADR-0005
title: TurboQuant TQ_MSE_SRHT — transformada nomeada como parte do formato
status: accepted
identifiers:
  - name: runtime.nexapack.format:TQ_CODEC_ID
    value: 'TQ_MSE_SRHT'
  - name: runtime.nexapack.format:TQ_CODEC_VERSION
    value: 1
  - name: runtime.nexapack.format:TQ_TRANSFORM_ID
    value: 'SRHT_XOSHIRO256SS_V1'
  - name: runtime.nexapack.tq:TQ_CODEC_ID
    value: 'TQ_MSE_SRHT'
  - name: runtime.nexapack.tq:TQ_CODEC_VERSION
    value: 1
  - name: runtime.nexapack.tq:TQ_TRANSFORM_ID
    value: 'SRHT_XOSHIRO256SS_V1'
  - name: runtime.nexapack:TQ_CODEC_ID
    value: 'TQ_MSE_SRHT'
  - name: runtime.nexapack:TQ_CODEC_VERSION
    value: 1
  - name: runtime.nexapack:TQ_TRANSFORM_ID
    value: 'SRHT_XOSHIRO256SS_V1'
  - name: compiler.paged_kv_plan:TQ_CODEC_ID
    value: 'TQ_MSE_SRHT'
  - name: compiler.paged_kv_plan:TQ_CODEC_VERSION
    value: 1
  - name: compiler.paged_kv_plan:TQ_TRANSFORM_ID
    value: 'SRHT_XOSHIRO256SS_V1'
prior_art:
  - id: P16
    role: solution-category
    note: A trilha M4 cita P16-20 para cache KV. A categoria situa transformadas aleatórias estruturadas como família; não confirma nenhuma largura nem nenhum erro medido aqui.
---

# ADR-0005 — Armazenamento TurboQuant TQ_MSE_SRHT

## Contexto

Os codecs agrupados de ADR-0002 quantizam no espaço original. Para o cache KV,
onde os vetores são ativações e não pesos, uma transformada aleatória
estruturada antes de quantizar distribui a energia entre coordenadas e reduz o
custo dos outliers.

## Problema técnico

Uma transformada aleatória só é reversível se o leitor puder reconstruí-la
exatamente. Guardar a matriz seria absurdo — é o tamanho do que se quer
comprimir. Gerá-la de uma semente resolve o tamanho, mas cria um contrato novo
e frágil: **o gerador pseudoaleatório vira parte do formato**. Um arquivo
gravado com um PRNG e lido com outro decodifica ruído sem disparar nenhum
checksum, porque os bytes armazenados estão íntegros.

## Decisão

O codec é `TQ_MSE_SRHT`, versão 1, e a transformada tem **identidade própria**:
`SRHT_XOSHIRO256SS_V1`. Nomear a transformada separadamente do codec é a decisão
central deste ADR. Ela permite que a transformada mude de versão sem renomear o
codec, e — mais importante — faz um leitor recusar um arquivo cuja transformada
ele não implementa, em vez de decodificar ruído silenciosamente.

O índice guarda `bits`, `seed`, `transform_id` e o `codebook_f32le` explícito. O
codebook é serializado no índice, e não regenerado na leitura: regenerá-lo
faria a fidelidade depender de o leitor reproduzir bit a bit a mesma otimização
MSE que o escritor rodou.

O contexto é exclusivamente MSE, com estado linear. Não há matriz QJL.

Os três identificadores aparecem em quatro módulos — formato, implementação TQ,
superfície do pacote e plano de KV paginado — e todos são reivindicados aqui,
porque um renomeio parcial é precisamente o modo de falha descrito acima.

## Medições próprias

A largura armazenada por vetor de 64 dimensões, perguntada ao próprio
quantizador:

```adr-measurement
name: tq_row_bytes_64_bits3
value: 32
unit: bytes por vetor de 64 dimensões
source: tq_row_bytes_64_bits3
```

```adr-measurement
name: tq_row_bytes_64_bits4
value: 40
unit: bytes por vetor de 64 dimensões
source: tq_row_bytes_64_bits4
```

Em 3 bits, 32 bytes para 64 valores é 0,5 B/valor — a mesma largura do
`Q3_GROUPED` de ADR-0002 sobre o mesmo vetor. TQ não é escolhido por ocupar
menos, e sim por onde coloca o erro.

O codebook reservado no índice, em entradas float32:

```adr-measurement
name: tq_codebook_entries_bits3
value: 8
unit: entradas float32
source: tq_codebook_entries_bits3
```

Oito entradas são `2**3`: o codebook é denso sobre os códigos de 3 bits, e sua
largura serializada é o que o índice reserva antes de qualquer alocação.

## Alternativas descartadas

**Regenerar o codebook na leitura, a partir da semente.** Economizaria 64 bytes
de índice por matriz em 3 bits. Recusado porque amarra a fidelidade à
reprodução exata de uma otimização de ponto flutuante entre escritor e leitor,
possivelmente compilados de formas diferentes.

**Reusar `CODEC_VERSION` para versionar a transformada.** Recusado: um leitor
que conhece o codec mas não a transformada precisa saber disso, e um só número
não distingue os dois casos.

**TQ01 como formato corrente.** Foi preservado apenas por migração, com endian e
codebook de origem explícitos. O formato corrente é TQ02, little-endian.

## Limites declarados

O erro de TQ contra os codecs agrupados não está medido aqui e não se deduz da
largura. Os dois ocupam 0,5 B/valor em 64 dimensões; qual erra menos depende da
distribuição do vetor.

`TQ_MSE_SRHT` é armazenamento e atenção de KV. Ele **não** é um codec de pesos:
nenhum tensor de peso deste repositório é gravado em TQ.

A escolha de 3 bits como padrão de KV vem das fixtures CPU. Não há evidência de
qualidade de modelo treinado por trás dela.
