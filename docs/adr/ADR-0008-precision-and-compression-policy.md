---
adr: ADR-0008
title: Políticas de precisão e compressão — o custo otimizado entra no policy_id
status: accepted
identifiers:
  - name: compiler.precision_map:SCHEMA_VERSION
    value: 1
  - name: compiler.precision_map:POLICY_ID
    value: 'GREEDY_SENSITIVITY_PER_BYTE_V2'
  - name: compiler.precision_map:POLICY_IDS
    value: ('payload', 'GREEDY_SENSITIVITY_PER_BYTE_V2', 'physical', 'GREEDY_SENSITIVITY_PER_PHYSICAL_BYTE_V3')
  - name: compiler.precision_map:CODECS
    value: ('q2', 'q3', 'q4', 'q8', 'f16', 'f32')
  - name: compiler.precision_map:DECODE_RATE_POLICY_ID
    value: 'BLOCK_MATMUL_NS_PER_STORED_BYTE_V1'
  - name: compiler.codec_speed:DECODE_RATE_POLICY_ID
    value: 'BLOCK_MATMUL_NS_PER_STORED_BYTE_V1'
  - name: compiler.codec_speed:SPEED_CODECS
    value: ('q2', 'q3', 'q4', 'q8', 'f16', 'f32')
  - name: compiler.planner.compression:DECODE_RATE_POLICY_ID
    value: 'BLOCK_MATMUL_NS_PER_STORED_BYTE_V1'
  - name: compiler.planner.compression:PLANNER_POLICY_ID
    value: 'COMPRESSION_CODEC_ONLY_WITH_MEASURED_DECODE_TIME_V1'
prior_art:
  - id: P30
    role: problem
    note: A trilha M6 cita P01-15/P28/P30 para precisão e compressão. A categoria enuncia o problema de alocar largura por tensor; as políticas e os números abaixo são deste repositório e não têm confirmação externa.
---

# ADR-0008 — Políticas de precisão e compressão

## Contexto

Com a escada de seis codecs (ADR-0002, ADR-0003), falta decidir qual tensor
recebe qual. A calibração mede sensibilidade por tensor; o planejador ordena.

## Problema técnico

Três descobertas sucessivas quebraram a formulação ingênua.

**O custo otimizado não é único.** O planejador otimizava o *payload* — os bytes
que o codec codifica. O contêiner cobra overhead fixo por tensor empacotado, que
não escala com a matriz, e abaixo de ~2.000–8.000 valores em grupo 32 empacotar
**aumenta o arquivo**. Na fixture tiny o plano por payload escolhe 8× Q2, promete
880 bytes e entrega um bundle 22 vezes maior que o plano por custo físico.

**A velocidade não segue os bytes.** Um codec menor pode decodificar mais devagar.
As duas ordenações discordam, quase invertidas.

**Otimizar velocidade responde "não comprima nada".** O F32 é o codec mais rápido
de todos. Por isso o tempo entra como **teto**, nunca como custo otimizado.

## Decisão

O custo otimizado é **parte da política**, não um parâmetro dela.
`select_precision(..., cost="payload"|"physical")` e `POLICY_IDS` liga cada base
de custo a um `policy_id` distinto: `GREEDY_SENSITIVITY_PER_BYTE_V2` e
`GREEDY_SENSITIVITY_PER_PHYSICAL_BYTE_V3`. O `policy_id` deixou de ser constante
do módulo e virou campo do mapa, com `cost_basis` para lê-lo de volta. Mapas
antigos continuam válidos; política desconhecida é recusada.

O tempo de decodificação tem política própria,
`BLOCK_MATMUL_NS_PER_STORED_BYTE_V1`, medida no kernel que o executor roda —
não estimada de bits por valor. Ela é referenciada por três módulos, e os três
nomes são reivindicados aqui.

O planejador de compressão declara o que faz e o que não faz no próprio nome:
`COMPRESSION_CODEC_ONLY_WITH_MEASURED_DECODE_TIME_V1`. "Codec only" é uma
limitação escrita no identificador.

## Medições próprias

As três escadas — seleção de precisão, medição de velocidade, aceitação do
bundle — são a mesma, na mesma ordem. Se uma divergir, o plano nomeia um codec
que outra etapa não conhece:

```adr-measurement
name: precision_codec_ladder
value: ('q2', 'q3', 'q4', 'q8', 'f16', 'f32')
unit: ordem de codecs
source: precision_codec_ladder
```

```adr-measurement
name: speed_codec_ladder
value: ('q2', 'q3', 'q4', 'q8', 'f16', 'f32')
unit: ordem de codecs
source: speed_codec_ladder
```

As duas bases de custo e as políticas que cada uma nomeia:

```adr-measurement
name: precision_cost_bases
value: ('payload', 'physical')
unit: bases de custo
source: precision_cost_bases
```

```adr-measurement
name: precision_payload_policy_id
value: 'GREEDY_SENSITIVITY_PER_BYTE_V2'
unit: policy_id
source: precision_payload_policy_id
```

```adr-measurement
name: precision_physical_policy_id
value: 'GREEDY_SENSITIVITY_PER_PHYSICAL_BYTE_V3'
unit: policy_id
source: precision_physical_policy_id
```

```adr-measurement
name: decode_rate_policy_id
value: 'BLOCK_MATMUL_NS_PER_STORED_BYTE_V1'
unit: policy_id
source: decode_rate_policy_id
```

```adr-measurement
name: compression_planner_policy_id
value: 'COMPRESSION_CODEC_ONLY_WITH_MEASURED_DECODE_TIME_V1'
unit: policy_id
source: compression_planner_policy_id
```

A discordância entre as duas ordenações é tempo de parede e é citada. Por bytes,
Q2 < Q3 < Q4 < Q8 < F16 < F32; por tempo, F32 < Q8 < Q4 < Q2 < Q3 < F16:

```adr-measurement
name: q2_slower_than_q4_percent
value: 12.6
unit: % a mais de tempo por valor, Q2 sobre Q4, guardando 40% menos bytes
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro da onda 4a (M6.03a)
```

```adr-measurement
name: f16_slower_than_f32_factor
value: 4.86
unit: × o tempo por valor do F32, ocupando metade dos bytes
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro da onda 4a (M6.03a)
```

```adr-measurement
name: payload_plan_bundle_size_factor_tiny
value: 22
unit: × o bundle do plano por custo físico (fixture tiny)
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro do vigésimo nono incremento (M6.02c)
```

As larguras por valor que ancoram a ordenação por bytes **são** recalculadas —
em ADR-0002 e ADR-0003. O que é citado aqui é só o tempo.

## Alternativas descartadas

**Otimizar tempo de decodificação.** Responde "não comprima nada", porque o F32
é o mais rápido. Por isso o tempo é teto (`--max-decode-ns`), ortogonal aos
tetos de bytes e de erro.

**Um `policy_id` constante do módulo com o custo como parâmetro.** Recusado: dois
mapas produzidos por regras diferentes teriam a mesma identidade de política, e
aplicar um no lugar do outro não seria detectável.

**Somar sensibilidades para estimar a qualidade de um mapa.** Explicitamente
recusado no próprio módulo: sensibilidades individuais não se somam, e o custo
estimado de um mapa é auxílio de ordenação, nunca alegação de qualidade.

## Limites declarados

**Um mapa de precisão é um plano, não uma medição.** Ele carrega a identidade do
checkpoint e os tokens de calibração, e aplicá-lo noutro modelo é recusado.

O custo estimado de um mapa não é uma alegação de qualidade, pelo motivo acima.

Na fixture tiny **nenhum codec empacotado sobrevive à seleção física**. Isso diz
respeito àquela fixture, de matrizes pequenas; não é resultado sobre modelos.

Os tempos citados são de uma máquina, um compilador e um kernel escalar. Eles
não são recalculados por nenhum teste deste registro, e não deveriam ser: uma
asserção sobre tempo de parede seria intermitente ou frouxa demais para
significar algo.
