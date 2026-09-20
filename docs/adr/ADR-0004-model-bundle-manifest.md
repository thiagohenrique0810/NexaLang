---
adr: ADR-0004
title: Manifesto do bundle de modelo — um diretório que se descreve inteiro
status: accepted
identifiers:
  - name: runtime.nexapack.bundle:FORMAT
    value: 'NexaModelBundle'
  - name: runtime.nexapack.bundle:FORMAT_VERSION
    value: 1
  - name: runtime.nexapack.bundle:MATRIX_CODECS
    value: ('q2', 'q3', 'q4', 'q8', 'f16', 'f32')
  - name: runtime.nexapack.bundle:PACKED_CODECS
    value: ('q2', 'Q2_GROUPED', 'q3', 'Q3_GROUPED', 'q4', 'Q4_GROUPED', 'q8', 'Q8_GROUPED')
  - name: compiler.calibration:PACKED_CODECS
    value: ('q2', 'Q2_GROUPED', 'q3', 'Q3_GROUPED', 'q4', 'Q4_GROUPED', 'q8', 'Q8_GROUPED')
prior_art: []
---

# ADR-0004 — Manifesto do bundle de modelo

## Contexto

Um modelo não é uma matriz. São dezenas de tensores, cada um com sua forma, seu
codec e seu arquivo, mais o tokenizer, mais a configuração da arquitetura. O
contêiner de ADR-0001 descreve uma matriz; falta o que descreve o conjunto.

## Problema técnico

Se cada tensor souber apenas de si, o carregamento vira descoberta: listar o
diretório, adivinhar pelo sufixo, abrir cada arquivo para saber o que é. Isso
torna impossível recusar um bundle **antes** de ler o payload, e é justamente
antes de ler o payload que a recusa é barata.

## Decisão

Um `manifest.json` na raiz do diretório declara formato, versão, arquitetura,
cada tensor com forma/codec/caminho/`file_bytes`/blocos, os assets e a
proveniência. `FORMAT` é a string `NexaModelBundle` e `FORMAT_VERSION` é 1.

`MATRIX_CODECS` é a união das duas famílias — os quatro empacotados de ADR-0002
e os dois densos de ADR-0003 — e é a lista contra a qual um codec declarado é
validado. A ordem importa: empacotados primeiro, densos depois, do mais estreito
ao mais largo.

`PACKED_CODECS` aparece tanto no bundle quanto na calibração, porque a
calibração constrói variantes reais de cada codec para medir sensibilidade e
precisa da mesma correspondência dtype→id que o escritor usa.

Os tetos são parte do contrato, não detalhe de implementação: um manifesto é
recusado antes de ser interpretado se passar de 1 MiB.

## Medições próprias

A escada completa que um bundle aceita tem seis degraus:

```adr-measurement
name: bundle_matrix_codec_count
value: 6
unit: codecs
source: bundle_matrix_codec_count
```

A mesma escada, do lado da calibração, empacotados seguidos de densos:

```adr-measurement
name: calibration_codec_ladder
value: ('q2', 'q3', 'q4', 'q8', 'f16', 'f32')
unit: ordem de codecs
source: calibration_codec_ladder
```

Os tetos que protegem a leitura de um bundle não confiável:

```adr-measurement
name: bundle_max_tensors
value: 4096
unit: tensores
source: bundle_max_tensors
```

```adr-measurement
name: bundle_max_manifest_bytes
value: 1048576
unit: bytes
source: bundle_max_manifest_bytes
```

```adr-measurement
name: bundle_max_assets
value: 16
unit: assets
source: bundle_max_assets
```

O teto do manifesto é o mesmo 1 MiB do índice NexaPack de ADR-0001. Os dois
limites são independentes no código e coincidem em valor; um teste que
comparasse os dois estaria afirmando uma relação que ninguém decidiu.

## Alternativas descartadas

**Descoberta por sufixo de arquivo.** Um `.nxp` no diretório seria um tensor,
um `.f16` seria denso. Recusado: o nome do arquivo passa a ser o contrato, e um
diretório com um arquivo a mais vira um bundle com um tensor a mais.

**Manifesto embutido no primeiro tensor.** Evitaria um arquivo, mas amarraria a
descrição do conjunto à existência e à integridade de um membro arbitrário.

**Publicação in-place.** O bundle é escrito numa área temporária, sincronizado e
então movido por `os.replace`. A alternativa — escrever direto no destino — foi
recusada porque uma falha no meio deixa um bundle parcial que o manifesto
descreve como completo.

## Limites declarados

O manifesto descreve um **diretório**. Empacotá-lo num arquivo só é outro
contrato, em ADR-0006, e o bundle não sabe se está dentro de um.

`FORMAT_VERSION` 1 não prova compatibilidade: é a única versão que existe.

A proveniência que o manifesto carrega é declaração do produtor. Ela não é
verificada contra nada — o bundle não sabe se o checkpoint que diz ter usado é
mesmo aquele.
