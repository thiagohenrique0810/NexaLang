---
adr: ADR-0012
title: Admissão conjunta de memória — o teto é do processo, não da sessão
status: accepted
identifiers: []
prior_art: []
---

# ADR-0012 — Admissão conjunta de memória

## Contexto

Cada sessão já admitia o próprio orçamento contra o próprio limite. O teto real,
porém, é do **processo**: duas sessões de 300 MiB passam nas duas admissões
individuais e juntas estouram um processo de 512 MiB.

## Problema técnico

Falta o termo compartilhado. Uma admissão por sessão é uma verificação local de
uma restrição global, e verificações locais de restrições globais sempre
permitem a soma proibida.

O segundo problema é **quando** admitir. Uma reserva que variasse por chamada
deixaria de bater com o que o pool guarda; uma reserva feita depois de alocar o
primeiro peso não protege nada.

## Decisão

`SessionMemoryPool` com teto compartilhado, `admit`/`release` por handle opaco,
contadores (reservado, pico, membros, admissões, recusas) e `PoolAdmissionError`
carregando os quatro números do caso.

A sessão reserva `capacity_managed_buffers_bound_bytes` — o limite superior que
já publicava — **fixado na construção**, depois do plano de capacidade e antes
de qualquer peso, arena ou página. Uma sessão recusada fecha o bundle e não
deixa reserva.

Compartilhar prefixo **não** reduz a reserva. A sequência derivada admite o
próprio limite contra o mesmo teto, porque pode dar `append` e então as páginas
deixam de ser compartilhadas; admitir o número menor seria admitir um estado que
as duas sequências são livres de abandonar.

`cancel()` é o único método seguro de outra thread. `_call_guard` envolve a
transação inteira; `_check_cancelled` fica no laço de operadores, dentro de
`call()` — por onde passa todo kernel nativo, o que dá granularidade por tile de
matmul e por linha de embedding — e nos laços de migração e de evicção.

A política se chama `PROCESS_JOINT_ADMISSION_UPPER_BOUND_V1`, e o nome declara a
limitação: é um **limite superior declarado**, não residência medida.

## Medições próprias

Este ADR não reivindica nenhum identificador de contrato, e o motivo é uma
inconsistência real do código, registrada aqui em vez de corrigida.
`PROCESS_JOINT_ADMISSION_UPPER_BOUND_V1` é um **literal embutido** no corpo de
`SessionMemoryPool.to_dict()`, não uma constante de módulo. A regra de censo
deste registro enxerga nomes de módulo, e não há nome a enxergar — ao contrário
de `compiler.tiered_kv_plan:POLICY_ID` ou
`compiler.offloaded_kv_plan:RELOAD_POLICY_ID`, que são constantes.

A consequência é honesta e precisa ser dita: **renomear essa string não derruba
a bijeção**. O que a prende é a medição abaixo, que pergunta a um pool real o
que ele publica:

```adr-measurement
name: admission_policy_id
value: 'PROCESS_JOINT_ADMISSION_UPPER_BOUND_V1'
unit: policy_id publicado por SessionMemoryPool.to_dict()
source: admission_policy_id
```

```adr-measurement
name: admission_schema_version
value: 1
unit: schema_version publicado por SessionMemoryPool.to_dict()
source: admission_schema_version
```

O teto compartilhado recusa de verdade. Dez pedidos de 300 bytes contra um teto
de 1024 admitem três e recusam sete:

```adr-measurement
name: admission_refuses_over_limit
value: 3
unit: admissões concedidas em dez pedidos de 300 contra teto de 1024
source: admission_refuses_over_limit
```

Três é o piso de 1024/300. A recusa é aritmética do pool, não amostragem.

A disputa concorrente é citada, porque reproduzi-la num teste de ADR seria
duplicar um teste de concorrência que já existe:

```adr-measurement
name: threads_admitted_against_ten_slots
value: 10
unit: admissões de dezesseis threads disputando dez vagas atrás de uma barreira
source: CITED
cited_from: docs/BLUEPRINT_512MB_CHECKLIST.md, registro do vigésimo oitavo incremento (M4.06d)
```

## Alternativas descartadas

**Admissão por sessão contra o limite da sessão.** É o que existia, e é o bug:
permite a soma proibida.

**Reserva variável por chamada.** Recusada porque deixaria de bater com o que o
pool guarda, tornando os contadores não auditáveis.

**Reduzir a reserva da sequência derivada por compartilhamento de prefixo.**
Recusada pelo argumento do `append` acima. Compartilhar prefixo reduz
residência, não reserva.

**Cancelamento por flag checada entre chamadas.** Recusado: a granularidade
seria a chamada inteira, e uma chamada é exatamente o que se quer cancelar.

## Limites declarados

**O escopo é limite superior declarado, não residência medida.** O pool não sabe
quanta memória o processo realmente usa; ele soma limites que as sessões
declararam. A própria string de política diz isso, e o `scope` publicado no
dicionário repete.

Chamada concorrente de verdade é recusada com `ValueError`. Concorrência
suportada é `cancel()`, e só.

`PROCESS_JOINT_ADMISSION_UPPER_BOUND_V1` não está sob a bijeção de
identificadores, pelo motivo declarado na seção de medições. Corrigir isso
exigiria editar `runtime/nexapack/admission.py`, fora do escopo deste item.
