---
adr: ADR-0009
title: Planos de KV — esquema, codecs por página, tiers por idade e recarga
status: accepted
identifiers:
  - name: compiler.kv_plan:SCHEMA_VERSION
    value: 1
  - name: compiler.paged_kv_plan:SCHEMA_VERSION
    value: 1
  - name: compiler.tiered_kv_plan:SCHEMA_VERSION
    value: 1
  - name: compiler.tiered_kv_plan:POLICY_ID
    value: 'CPU_PAGE_AGE_F32_Q4_Q3_V1'
  - name: compiler.tiered_kv_plan:_CODECS
    value: ('hot', 'f32', 'warm', 'q4', 'cold', 'q3')
  - name: compiler.tiered_kv_plan:_CODEC_IDS
    value: ('f32', 'F32_NATIVE', 'q4', 'Q4_GROUPED', 'q3', 'Q3_GROUPED')
  - name: compiler.offloaded_kv_plan:RELOAD_POLICY_ID
    value: 'CPU_RELOAD_FIRST_TOUCH_MRU_V1'
  - name: runtime.nexapack.offloaded:RELOAD_POLICY_ID
    value: 'CPU_RELOAD_FIRST_TOUCH_MRU_V1'
  - name: runtime.nexapack.offloaded:_CODEC_IDS
    value: ('f32', 0, 'q4', 4, 'q3', 3)
  - name: runtime.nexapack.paged:_KV_CODEC_IDS
    value: ('f32', 0, 'q3', 3, 'q4', 4, 'q8', 8)
  - name: runtime.nexapack.tiered:_CODEC_IDS
    value: ('f32', 0, 'q4', 4, 'q3', 3)
prior_art:
  - id: P18
    role: problem
    note: A trilha M4 cita P16-20 ao enunciar o cache KV. A categoria descreve o problema de crescimento do KV com o contexto; as larguras medidas abaixo vêm do plano deste repositório.
---

# ADR-0009 — Planos de KV, tiers e recarga

## Contexto

O cache KV cresce linearmente com o contexto e não para de crescer. Num teto de
512 MB ele disputa memória com os pesos, e a disputa é decidida por tokens de
contexto, não por parâmetros.

## Problema técnico

Uma página de KV não tem uma largura: ela tem a largura do codec em que foi
escrita. Se o plano e o runtime discordarem sobre qual codec é qual, a atenção
lê a fatia errada de uma página — e, diferente de um arquivo, aqui não há
checksum por bloco para pegar isso.

Além disso, as páginas não são iguais entre si. As recentes são lidas a cada
passo; as antigas, raramente. Tratá-las com a mesma precisão paga precisão onde
ela não é usada.

## Decisão

Três planos, três esquemas na versão 1: `kv_plan` para o cache contíguo,
`paged_kv_plan` para páginas e `tiered_kv_plan` para tiers por idade.

O plano paginado nomeia o codec por string (`f32`, `q4`, `q3`, `q8`, `tq`) e o
runtime tem um mapa numérico próprio, `_KV_CODEC_IDS`, cujo valor é o número de
bits — `f32` é 0, `q3` é 3, `q4` é 4, `q8` é 8. O zero do f32 não é um bug: é a
marca de "não empacotado".

A política de tiers é `CPU_PAGE_AGE_F32_Q4_Q3_V1`, e ela **diz a escada no
próprio nome**: hot em F32, warm em Q4, cold em Q3. Um mapa de tiers produzido
sob outra escada não é confundível com este.

A recarga de páginas cold tem política separada,
`CPU_RELOAD_FIRST_TOUCH_MRU_V1`: admissão por primeiro toque, substituição do
slot mais recente.

`_CODEC_IDS` aparece em três módulos com conteúdos **diferentes**: no plano de
tiers mapeia dtype para id de codec NexaPack (`'q4'` → `'Q4_GROUPED'`), e nos
dois runtimes mapeia dtype para largura em bits. São contratos distintos com o
mesmo nome, e por isso cada um é reivindicado separadamente.

## Medições próprias

A largura por token na fixture D64/P16/G32 — um head de 64 dimensões, páginas de
16 tokens, grupo 32 — perguntada ao plano paginado real, K mais V:

```adr-measurement
name: kv_bytes_per_token_f32_d64
value: 512
unit: bytes por token (K+V)
source: kv_bytes_per_token_f32_d64
```

```adr-measurement
name: kv_bytes_per_token_q4_d64
value: 80
unit: bytes por token (K+V)
source: kv_bytes_per_token_q4_d64
```

```adr-measurement
name: kv_bytes_per_token_q3_d64
value: 64
unit: bytes por token (K+V)
source: kv_bytes_per_token_q3_d64
```

São os três números publicados no registro do sétimo incremento, recalculados
aqui a partir do plano. O F32 custa **8 vezes** o Q3 por token: é essa razão que
torna o tier cold uma decisão de capacidade de contexto, não de arredondamento.

A escada de tiers, na ordem hot → warm → cold:

```adr-measurement
name: tier_codec_ladder
value: ('f32', 'q4', 'q3')
unit: codec por tier
source: tier_codec_ladder
```

A troca de I/O por RAM que a recarga faz é citada. Na fixture D64 de 512 tokens,
256 slots reduziram 64.260 recargas e 43,6 MB lidos para 253 recargas e 171.875
bytes, elevando o pico gerenciado de 91.899 para 140.031 bytes:

```adr-measurement
name: reload_cache_peak_bytes_256_slots
value: 140031
unit: bytes de pico gerenciado (fixture D64, 512 tokens, 256 slots)
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro do décimo terceiro incremento
```

```adr-measurement
name: reload_cache_peak_bytes_one_slot
value: 91899
unit: bytes de pico gerenciado (fixture D64, 512 tokens, um slot)
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro do décimo terceiro incremento
```

## Alternativas descartadas

**Um único plano com o codec como parâmetro.** Recusado porque páginas, tiers e
offload têm identidades de layout diferentes: uma sequência derivada não pode
herdar um prefixo decidido sob outro teto de qualidade, e isso só é verificável
se a política fizer parte da identidade.

**Promoção real de precisão, Q3 → F32.** Declarada **impossível** e registrada
assim: a informação descartada na quantização não volta. O que existe é retenção
— a página fica no codec em que já estava quando o re-encode erraria demais.

**Reduzir a reserva de uma sequência derivada por ela compartilhar prefixo.**
Recusado em ADR-0012: a derivada pode dar `append`, e aí as páginas deixam de
ser compartilhadas.

## Limites declarados

Todos os tiers ficam em RAM, exceto as páginas cold enviadas ao backing store.
Não há tier em GPU, e nada aqui foi medido em GPU.

Os erros de re-encode medidos na fixture tiny — F32→Q4 em 0,030–0,044 de RMSE,
Q4→Q3 em 0,077–0,097 — são **daquela fixture**. Não são qualidade de modelo.

A recarga é **troca explícita de I/O por RAM, não ganho automático**: os dois
números citados acima mostram o pico gerenciado subindo 52%. Com um slot e mais
de uma página cold não há retenção, e esse caso continua sendo o padrão.

`_KV_CODEC_IDS` inclui `q8`, que o plano paginado aceita, mas a escada de tiers
não usa: o tier warm é Q4.
