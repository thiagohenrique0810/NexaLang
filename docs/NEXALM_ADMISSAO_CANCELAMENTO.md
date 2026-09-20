# NexaLM CPU: admissão conjunta e cancelamento

O vigésimo oitavo incremento fecha duas lacunas do contrato de execução: várias
sessões passavam cada uma na própria admissão e juntas estouravam o processo, e
uma chamada em andamento não podia ser interrompida — só havia rollback de falha.

## Admissão conjunta

Cada sessão já admitia o próprio orçamento: o plano se recusa a existir quando a
arena mais as reservas passam do que o chamador declarou. Essa checagem é
**local**. Duas sessões de 300 MiB passam cada uma na sua admissão e juntas
ocupam 600 MiB num processo de 512 MiB.

`SessionMemoryPool` é o termo compartilhado que faltava:

```python
from runtime.nexapack.admission import SessionMemoryPool

pool = SessionMemoryPool("512MiB")
a = TieredTransformerSession(path, memory_pool=pool, memory_budget="300MiB")
b = TieredTransformerSession(path, memory_pool=pool, memory_budget="300MiB")
# PoolAdmissionError: ... requires 314572800 bytes, 314572800 of 536870912 are
# already reserved by open sessions, 222298112 available
```

Na CLI, `--memory-pool 512MiB` cria o teto e o compartilha com a sequência
derivada de `--fork-tokens`.

O que a sessão reserva é `capacity_managed_buffers_bound_bytes` — o limite
superior que ela já publicava no relatório, fixado na construção e não o que
mede depois. Uma reserva que variasse por chamada deixaria de bater com o que o
pool guarda.

A admissão acontece **depois** do plano de capacidade e **antes** de qualquer
peso, arena ou página KV. Uma sessão recusada fecha o próprio bundle na saída e
não deixa reserva: o pool volta exatamente ao estado anterior.

### Compartilhar não reduz a reserva

Uma sequência derivada compartilha as páginas completas do prefixo, então a
residência real cai. A reserva **não** cai: a derivada pode dar `append`, e aí
as páginas deixam de ser compartilhadas. Admitir o número menor seria admitir um
estado que as duas sequências são livres de abandonar. O pool reserva um limite
superior, nunca residência medida — é a mesma disciplina do resto do blueprint.

### Concorrência

`admit`/`release` são a única operação que várias threads podem executar ao mesmo
tempo sobre o mesmo objeto. Uma regressão coloca dezesseis threads disputando dez
vagas atrás de uma barreira: exatamente dez entram. Liberar duas vezes não credita
duas vezes.

## Cancelamento

`session.cancel()` pede que a chamada em andamento pare. É o **único** método que
outra thread pode chamar enquanto a sessão executa; tudo que ele toca está sob um
lock próprio. Retorna se havia chamada rodando — um `cancel()` fora de chamada é
no-op e não arma nada para a próxima.

```python
thread = threading.Thread(target=lambda: session.prefill(tokens))
thread.start()
...
session.cancel()          # de outra thread
thread.join()             # a chamada levanta SessionCancelled
```

A verificação fica em três pontos, escolhidos para cobrir tudo sem espalhar
checagens: no laço de operadores, dentro de `call()` — por onde passa **todo**
kernel nativo, o que dá granularidade por tile de matmul e por linha de
embedding — e no laço de migração de páginas.

O guard envolve a transação inteira, não só a fase numérica. Cancelar depois que
as páginas novas já foram escritas, durante a migração ou a evicção, descarta
tudo do mesmo jeito: `SessionCancelled` sai pelo mesmo `finally` que trata uma
falha, então o prefixo comprometido, suas páginas e seu relatório são os de
antes da chamada. Uma regressão cancela dentro de `_migrate_pages` e compara os
bytes do prefixo byte a byte com os de antes.

Uma chamada concorrente de verdade — duas threads executando a mesma sessão — é
**recusada** com `ValueError` em vez de corromper o estado. A sessão continua
sendo de uma thread só; o cancelamento é a exceção explícita a essa regra.

Depois de cancelada, a sessão continua utilizável: cancelar é rollback, não
falta.

## Limites

Falta **reuso de prefixo entre sessões que não derivam uma da outra** (M4.06e).
Não é um `fork` generalizado: `fork` adota o prefixo **inteiro**, e o interessante
aqui é adotar um prefixo **parcial**, as primeiras K páginas em comum. No
executor homogêneo isso é direto, porque todas as páginas têm o mesmo codec. Sob
a política de idade não é: o codec de uma página vem da distância até o fim, e
truncar o prefixo muda todas as idades — a página adotada ficaria acima da
precisão que a nova idade lhe dá, que é exatamente o mecanismo de retenção de
[M4.05d](NEXALM_KV_QUALIDADE.md), com orçamento limitado. Reusar exigiria
re-envelhecer na adoção, pagando re-encode, ou recusar prefixo parcial sob idade.

O pool não mede RSS, pesos mapeados nem alocação do intérprete Python; ele soma
os limites superiores declarados dos buffers gerenciados, que é o mesmo escopo do
resto dos relatórios.

O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
próxima tarefa.
