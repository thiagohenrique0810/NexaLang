---
adr: ADR-0001
title: Contêiner NexaPack v1 — prefixo fixo, índice limitado, blocos verificados
status: accepted
identifiers:
  - name: runtime.nexapack.format:MAGIC
    value: b'NEXAPACK'
  - name: runtime.nexapack.format:FORMAT
    value: 'NexaPack'
  - name: runtime.nexapack.format:FORMAT_VERSION
    value: 1
  - name: runtime.nexapack:FORMAT_VERSION
    value: 1
prior_art:
  - id: P01
    role: problem
    note: A trilha M6 cita P01-15 ao enunciar precisão e compressão; entra aqui como enunciado do problema, não como confirmação de nenhum número medido abaixo.
---

# ADR-0001 — Contêiner NexaPack v1

## Contexto

O pipeline de modelos precisa ler uma matriz de pesos sem materializá-la e sem
confiar no produtor. Um formato que exigisse carregar o arquivo inteiro para
descobrir onde as linhas começam empurraria o pico de memória para além do teto
de 512 MB antes de qualquer multiplicação acontecer.

## Problema técnico

Três exigências colidem. O leitor precisa saber o layout **antes** de alocar,
o que pede um índice na frente. O índice não pode crescer sem limite, porque um
índice arbitrariamente grande é um vetor de exaustão de memória tão bom quanto
o payload. E a corrupção precisa ser detectável por bloco, não só por arquivo,
senão qualquer verificação obriga a ler tudo.

## Decisão

Um prefixo little-endian de 64 bytes com layout fixo `<8sHHIQQ32s`: magic,
versão de formato, flags, comprimento do índice JSON, offset do payload,
tamanho físico exato e o SHA-256 dos bytes do índice. O índice JSON vem em
seguida, seguido de zeros até a fronteira de 4096. Cada entrada carrega limites
de linha, limites absolutos de byte e SHA-256 próprio.

`MAGIC` é `b'NEXAPACK'` e `FORMAT_VERSION` é 1. Todo formato deste repositório
está na versão 1: não há migração v1→v2 porque não há v2.

Os checksums detectam corrupção. Eles **não** autenticam um publicador — não há
assinatura, e um arquivo adulterado por quem saiba recomputar SHA-256 passa.

## Medições próprias

O prefixo tem largura fixa, e é a largura que o `struct` real produz:

```adr-measurement
name: nexapack_header_bytes
value: 64
unit: bytes
source: nexapack_header_bytes
```

```adr-measurement
name: nexapack_header_struct
value: '<8sHHIQQ32s'
unit: struct format
source: nexapack_header_struct
```

O alinhamento e os dois tetos que protegem o leitor:

```adr-measurement
name: nexapack_alignment_bytes
value: 4096
unit: bytes
source: nexapack_alignment_bytes
```

```adr-measurement
name: nexapack_max_metadata_bytes
value: 1048576
unit: bytes
source: nexapack_max_metadata_bytes
```

```adr-measurement
name: nexapack_max_blocks
value: 8192
unit: blocos
source: nexapack_max_blocks
```

O índice se identifica pelo nome do formato, gravado em todo arquivo:

```adr-measurement
name: nexapack_index_format_name
value: 'NexaPack'
unit: nome de formato no índice
source: nexapack_index_format_name
```

## Alternativas descartadas

**Índice no fim do arquivo**, como um ZIP. Permitiria escrever em uma passagem
sem resolver o layout antes, mas obriga a fazer um `seek` para o fim antes de
qualquer leitura — o que inviabiliza ler de um stream e, mais grave, impede que
o payload comece numa fronteira de página conhecida.

**Índice de tamanho livre.** Recusado porque o comprimento do índice é lido do
prefixo antes de qualquer validação: sem o teto de 1 MiB, um prefixo mentiroso
faria o leitor alocar o que quisesse.

**Um único checksum de arquivo.** Mais barato de escrever, mas transforma
verificação em leitura completa, e é exatamente isso que o formato existe para
evitar.

## Limites declarados

Os checksums são de integridade, não de autenticidade.

O alinhamento de 4096 é uma preferência declarada por um `mmap` que **ainda não
existe**: nenhum leitor deste repositório mapeia o arquivo. Ele é custo pago
hoje contra um ganho futuro, e ADR-0006 mede quanto esse custo é.

A versão 1 não prova compatibilidade com nada, porque não há outra versão de
onde ou para onde migrar.
