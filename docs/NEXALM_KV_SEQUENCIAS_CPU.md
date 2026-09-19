# NexaKV CPU: sequências derivadas e prefixo compartilhado

O décimo quarto incremento acrescenta `fork()` ao executor paginado homogêneo
(`--kv-cache --kv-page-tokens`, codecs F32/Q4/Q3/TQ) e a flag de evidência
`--fork-tokens`. Uma sequência derivada continua um prefixo já calculado sem
recomputá-lo e sem copiar suas páginas completas.

Isso atende a parte de múltiplas sequências e prefixos de M4.06. Tiers por idade
e backing store recusam `fork` explicitamente: a recodificação por idade
substitui páginas do prefixo, e uma página compartilhada migraria para uma
sequência enquanto outra ainda a lê. Esse caso ficou em M4.06b.

## Por que o compartilhamento é seguro

Uma página completa é imutável. `append` só escreve na página parcial mais nova
ou em páginas novas, e `prefill` substitui o conjunto inteiro em vez de
sobrescrever o anterior. A adoção então:

1. Retém as páginas completas do pai — a mesma alocação, sem cópia.
2. Copia **apenas** a página parcial, se existir, para que cada sequência
   mantenha escrita exclusiva sobre a própria cauda.

As páginas ganham contagem de referências. `release` devolve uma referência e só
libera a alocação com o último dono, então `reset`, `prefill` substituto e
`close` do pai não invalidam a derivada — em qualquer ordem. Como defesa em
profundidade, uma escrita que alcance uma página compartilhada é rejeitada antes
de tocar bytes, com o prefixo e o relatório preservados.

A adoção exige o mesmo manifesto de modelo e um layout de página idêntico
(codec, tamanho, grupo, bits/seed e codebook TQ). O contexto herdado precisa
caber na capacidade da sequência derivada. Uma falha durante a adoção devolve as
referências retidas, libera as cópias e deixa o pai intacto.

## Custo, orçamento e relatórios

A derivada é uma sessão completa: admite o próprio orçamento antes de reter ou
copiar qualquer página, e tem a própria arena de ativações. A reserva continua
conservadora — o compartilhamento reduz a residência real, não o teto admitido
para cada sequência.

O relatório separa `kv_shared_page_count`, `kv_shared_allocation_bytes` e
`kv_owned_allocation_bytes`; a soma dos dois últimos é a residência da
sequência. Logo após a adoção, `kv_prefix_adoption` informa tokens herdados,
páginas compartilhadas, páginas copiadas e bytes copiados. Bytes compartilhados
aparecem no relatório de cada sequência, mas existem uma vez no processo: somar
sequências conta a mesma página mais de uma vez.

## Evidências

Fixture D64, prompt de 256 tokens em chunks de 32, páginas de 16 tokens, KV Q4
G32, capacidade 512, dois ramos de 4 tokens cada, macOS ARM64/Python 3.14.5. Os
logits dos ramos foram **idênticos** aos de sequências independentes que
recomputaram o mesmo prompt:

| Métrica | Duas sequências independentes | Prefixo compartilhado |
| --- | ---: | ---: |
| KV residente somado | 45.662 B | 24.174 B |
| Prefixo | 2 × 21.488 B | 21.488 B, uma vez |
| Páginas próprias por ramo | 1.343 B | 1.343 B |
| Páginas copiadas na adoção | — | 0 |
| Amostra local de execução | ~0,63 s | ~0,04 s |

Com `R` ramos, a residência passa de `R × (prefixo + cauda)` para
`prefixo + R × cauda`, e o prompt é executado uma vez em vez de `R`. O tempo é
amostra local única, não benchmark repetido: o ganho vem de não recomputar o
prefixo, e cresce com o tamanho do prompt.

Quando o fork cai no meio de uma página, a adoção copia essa página — no máximo
uma por sequência derivada, com bytes reportados em `copied_bytes`.

## Uso

```python
with PagedTransformerSession(bundle, page_tokens=16, kv_codec="q4",
                             kv_group_size=32, max_sequence_length=512) as parent:
    parent.prefill(prompt)
    branch = parent.fork()          # herda o prefixo, sem recomputar
    try:
        branch.append([2, 4])       # diverge do pai a partir daqui
    finally:
        branch.close()
```

Evidência por linha de comando, com o relatório em `derived_sequence`:

```bash
python3 tools/nexa_run.py artifacts/models/nexalm-tiny \
  --tokens 1,3,5,7 --kv-cache --kv-page-tokens 2 --kv-codec q4 --kv-group-size 4 \
  --max-sequence-length 8 --tile-rows 3 --memory-budget 1MiB --fork-tokens 2,4 \
  --report artifacts/reports/kv-sequencias-tiny.json
```

Regressões: `python3 -m unittest discover -s tests -p 'test_paged_sequences_regressions.py' -v`.

## Limites

Uma sessão continua **uma sequência**, síncrona, sem uso concorrente; várias
sequências são várias sessões que compartilham páginas, cada uma com sua arena e
seu orçamento. Não há escalonador, batch, admissão conjunta nem deduplicação
automática de prefixos entre sessões independentes: o compartilhamento é
explícito, por `fork`. Tiers por idade, backing store, cancelamento assíncrono de
uma chamada em andamento e limites globais por processo permanecem em M4.06b.
O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
próxima tarefa.
