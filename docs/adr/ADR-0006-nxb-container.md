---
adr: ADR-0006
title: Contêiner .nxb — um arquivo só, e o que o alinhamento cobra por isso
status: accepted
identifiers:
  - name: runtime.nexapack.container:MAGIC
    value: b'NEXABNDL'
  - name: runtime.nexapack.container:FORMAT
    value: 'NexaBundleContainer'
  - name: runtime.nexapack.container:FORMAT_VERSION
    value: 1
prior_art: []
---

# ADR-0006 — Contêiner de arquivo único `.nxb`

## Contexto

O bundle de ADR-0004 é um diretório: 12 arquivos na fixture tiny, 39 no modelo
pequeno de 4 camadas. Distribuir um modelo como diretório significa distribuir
uma árvore, e uma árvore perde integridade de formas que um arquivo não perde —
um membro a mais, um membro a menos, permissões diferentes.

## Problema técnico

O modo de falha desta mudança é **silencioso**. Cada arquivo do bundle passa a
ser uma janela dentro de um arquivo maior, e uma janela aplicada num lugar e
esquecida noutro lê o payload do tensor vizinho — e ainda passa no próprio
SHA-256, porque os offsets deslizam juntos e o checksum é calculado sobre a
janela errada de forma consistente.

## Decisão

Um prefixo de 64 bytes que **reusa o layout `HEADER` do NexaPack**: magic
`NEXABNDL`, versão, flags, comprimento do índice JSON, offset do payload,
tamanho físico exato e SHA-256 do índice. Cada arquivo do bundle vira uma
seção; as seções são ordenadas e cada uma ocupa um slot alinhado em 4096. Uma
seção começa onde o slot anterior terminou, de modo que o índice cobre todo o
payload sem buraco e sem sobreposição, e o preenchimento dentro do slot é
verificado como zero.

O `kind` da seção é o gancho versionado. A V1 aceita `manifest`, `tensor` e
`asset` — exatamente o que um diretório de bundle contém hoje. Cada `kind` está
amarrado ao formato de caminho que pode carregar, para que o rótulo seja uma
checagem e não uma etiqueta. Um `kind` desconhecido é **recusado**, nunca
ignorado: um leitor V1 jamais entende pela metade um arquivo escrito por uma
versão posterior.

## Medições próprias

O alinhamento é o mesmo do `.nxp`, e é ele que domina o custo:

```adr-measurement
name: nxb_alignment_bytes
value: 4096
unit: bytes
source: nxb_alignment_bytes
```

```adr-measurement
name: nxb_max_sections
value: 8192
unit: seções
source: nxb_max_sections
```

```adr-measurement
name: nxb_section_kinds
value: ('asset', 'manifest', 'tensor')
unit: kinds aceitos na V1
source: nxb_section_kinds
```

O preço do alinhamento, perguntado ao layout real. Um manifesto de 64 bytes mais
um tensor de 20 bytes — 84 bytes de dados:

```adr-measurement
name: nxb_bytes_for_one_small_tensor
value: 12288
unit: bytes
source: nxb_bytes_for_one_small_tensor
```

São três páginas de 4096 para 84 bytes de conteúdo: uma para o prefixo e o
índice, uma para o manifesto, uma para o tensor. Com oito tensores, 224 bytes de
dados:

```adr-measurement
name: nxb_bytes_for_eight_small_tensors
value: 40960
unit: bytes
source: nxb_bytes_for_eight_small_tensors
```

Dez páginas. **O arquivo cresce por seção, não por byte de dado** — cada seção
custa uma página inteira, por menor que seja. Essa é a aritmética por trás dos
números publicados contra bundles reais, que são citados:

```adr-measurement
name: nxb_overhead_tiny_fixture_percent
value: 134.17
unit: % sobre a soma dos file_bytes do diretório (fixture tiny)
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro da onda 3 (M1.10b1)
```

```adr-measurement
name: nxb_overhead_small_model_percent
value: 1.60
unit: % sobre a soma dos file_bytes do diretório (modelo pequeno, 4 camadas)
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro da onda 3 (M1.10b1)
```

```adr-measurement
name: nxb_real_cost_tiny_fixture_bytes
value: 4096
unit: bytes contra st_blocks * 512 (fixture tiny)
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro da onda 3 (M1.10b1)
```

Os dois percentuais discordam por mais de duas ordens de grandeza pelo mesmo
motivo que 84 bytes viram 12.288: o custo é por seção. Um modelo com tensores
grandes dilui; a fixture tiny, não.

## Alternativas descartadas

**Sem alinhamento, seções concatenadas.** Eliminaria todo o overhead medido
acima — o arquivo seria a soma exata dos membros mais o índice. Foi recusado
para que um `.nxp` aninhado comece numa fronteira de página e preserve o
alinhamento que tinha como arquivo isolado, permitindo a um leitor futuro
mapeá-lo sem copiar. **Isso é preferência declarada, não otimização
comprovada**: nenhum leitor deste repositório mapeia o arquivo hoje.

**TAR ou ZIP.** Resolveriam o empacotamento com ferramentas existentes. Recusados
porque o índice ficaria no fim (ZIP) ou não existiria (TAR), e porque nenhum dos
dois garante alinhamento de seção.

**Payloads de pacote executável** — plano, kernels, variantes, fallback, que o
item M1.10b nomeia. Deliberadamente ausentes da V1: seriam `kind`s aceitos sem
nada que os escrevesse ou lesse.

## Limites declarados

**O `.nxb` não economiza bytes.** Ele custa. O ganho é 12 e 39 arquivos virando
**1**, e a integridade de um arquivo só.

O oráculo de execução entregue com este contrato **não é identidade de bits
entre a referência Python e a sessão nativa**. É identidade de bits entre três
execuções nativas (diretório, `.nxb`, diretório restaurado) e entre
Python-sobre-diretório e Python-sobre-`.nxb`, mantendo nativo × Python dentro da
tolerância da casa. A razão é que o oráculo é float64 e o kernel é C escalar.

Uma cláusula de validação em `format.py` (`offset + block_size > total_size`) é
redundante: removê-la não derruba nenhum teste, porque a checagem de cobertura
duas linhas adiante recusa o mesmo arquivo. Fica registrada como redundância
declarada.
