# NexaKV CPU: sequências derivadas e prefixo compartilhado

O décimo quarto incremento acrescenta `fork()` ao executor paginado homogêneo
(`--kv-cache --kv-page-tokens`, codecs F32/Q4/Q3/TQ) e a flag de evidência
`--fork-tokens`. Uma sequência derivada continua um prefixo já calculado sem
recomputá-lo e sem copiar suas páginas completas. O décimo quinto estende isso
à política de idade (`--kv-policy age`), com migração privada.

O vigésimo sexto estende ao backing store: páginas cold são arquivos, e duas
sequências passam a **compartilhar o arquivo** em vez de copiá-lo.

O trigésimo tira o parentesco da conta: `adopt_prefix()` e
`--reuse-prefix-tokens` deixam duas sessões construídas separadamente
compartilharem as páginas em que seus prompts coincidem.

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

## Tiers por idade: migração privada

Sob `--kv-policy age`, a derivada herda também os descritores do prefixo: os
mesmos codecs, idades e identidades de layout. A recodificação por idade **cria
uma página nova** e libera a de origem, então migrar é privado à sequência que
migra: a página compartilhada continua válida e inalterada para as demais, que
seguem com seus próprios codecs.

O preço é que cada sequência pode pagar a mesma recodificação separadamente, e
duas sequências que migram a mesma página lógica passam a ocupar duas páginas
físicas. O ganho de compartilhamento cai conforme o prefixo envelhece de formas
diferentes em cada ramo.

A adoção exige o mesmo manifesto de modelo e um layout de página idêntico
(codec, tamanho, grupo, bits/seed e codebook TQ). Sob tiers, a comparação inclui
a política inteira: hot/warm e `group_size` não mudam o layout F32, mas decidem
como uma página Q4/Q3 herdada é lida. O contexto herdado precisa
caber na capacidade da sequência derivada. Uma falha durante a adoção devolve as
referências retidas, libera as cópias e deixa o pai intacto.

## Backing store: o arquivo é que é compartilhado

Sob `--kv-backing-store`, uma página cold não ocupa RAM — ela é um arquivo
publicado. A adoção então não retém um buffer, e sim toma **mais um hold** sobre
o arquivo no store do pai, e a sequência derivada guarda um hold sobre o próprio
store. Daí as duas propriedades que importam:

- O pai pode fechar **primeiro**. Seus arquivos sobrevivem enquanto a derivada
  os lê, e o último dono é quem remove o arquivo e o diretório.
- Um `prefill` substituto ou um `reset` no pai aposenta as páginas dele sem
  tocar no que a derivada ainda lê: `remove` decrementa, e só apaga em zero.

Cada sequência escreve suas próprias páginas novas no seu próprio store, então
nada é escrito no store de outra. `copied_bytes` permanece zero na adoção: o
prefixo inteiro é compartilhado, inclusive a parte em disco.

## Reuso sem parentesco: duas sessões que só combinam nos tokens

`fork` exige parentesco: a derivada nasce do pai. O reuso de prefixo não exige
nada disso. Os bytes de uma página completa dependem **apenas dos tokens nas
posições absolutas que ela cobre** — não de quem os calculou, nem de como a
sequência chegou até lá, nem do tamanho dos chunks. Então duas sessões
construídas separadamente podem dividir uma página sempre que seus prompts
concordam em todos os tokens daquela página.

A regra vive sozinha em `runtime/nexapack/prefix_reuse.py`, sem sessão, sem
alocação e sem I/O:

```python
common_page_prefix(tokens_a, tokens_b, page_tokens)  # = len(prefixo comum) // page_tokens
```

O piso é deliberado. A página parcial mais nova continua sendo escrita pelo seu
dono, então ela nunca é imutável e nunca é adotada — mesmo quando os dois
prompts concordam nos tokens que ela já contém.

O adotante declara o prompt que **pretende** rodar e recebe exatamente o
prefixo de páginas que os dois prompts têm em comum:

```python
with PagedTransformerSession(bundle, page_tokens=16, kv_codec="q4",
                             kv_group_size=32, max_sequence_length=512) as reuse:
    adoption = reuse.adopt_prefix(source, prompt)       # nenhuma relação com source
    reuse.append(prompt[adoption["inherited_tokens"]:])  # só o que falta é executado
```

`kv_prefix_adoption` ganha, nesse caminho, `source: "shared_page_prefix"`,
`common_tokens` (onde os prompts deixaram de concordar), `requested_tokens` e
`source_tokens`. `copied_bytes` é zero: a adoção para numa fronteira de página,
então não existe página parcial para copiar. As demais propriedades são as de
`fork` — contagem de referências, imutabilidade da página compartilhada,
orçamento próprio e independência das duas sessões em qualquer ordem.

Pela linha de comando, `--reuse-prefix-tokens` roda um segundo prompt numa
sessão **independente** e publica o bloco `reused_prefix`:

```bash
python3 tools/nexa_run.py artifacts/models/nexalm-tiny \
  --tokens 1,3,5,7 --kv-cache --kv-page-tokens 2 --kv-codec q4 --kv-group-size 4 \
  --max-sequence-length 8 --tile-rows 3 --memory-budget 1MiB \
  --reuse-prefix-tokens 1,3,5,2,4
```

### O que é recusado

Manifesto diferente, layout de página diferente (codec, `page_tokens`, grupo,
bits/seed/codebook TQ), prefixo comum menor que uma página inteira, sessão
adotante que já tem prefixo próprio, sessão de origem fechada e prompt maior
que a capacidade do adotante.

A recusa que precisa de explicação é a **política de idade**. A intuição diz que
truncar um prefixo o deixa mais velho; é o contrário. Em
`compiler/tiered_kv_plan.py::desired_pages` a idade conta a partir da página mais
nova — `age = count - 1 - page_index` —, então adotar K das N páginas da origem
**reduz** a idade de cada página herdada em N-K e faz a política exigir *mais*
precisão dela. Uma página que a origem já envelheceu para Q4 ou Q3 teria de ser
promovida de volta a F32, e nada promove: o F32 que a recodificação destruiu não
volta dos bytes empacotados.

A recusa é mais estreita do que parece, e o código mede a diferença em vez de
supor. Quando toda página herdada já tem pelo menos a precisão que a nova idade
exige, a adoção é aceita — na prática, um prefixo que a origem nunca envelheceu,
já que a página K-1 vira idade zero, sempre hot F32. O caso restante — páginas
herdadas *acima* da nova idade, que é o que o teto de qualidade produz — é
representável apenas como retenção contra `retain_pages`, um orçamento que esta
adoção não admite; fica recusado de propósito. Sob backing store a recusa é
total, inclusive no caso hot: uma página cold é um arquivo no store da origem, e
a ordem de aposentadoria de holds parciais não foi provada aqui.

### Evidências do reuso

Mesma fixture D64 do `fork`: prompt de 256 tokens em chunks de 32, páginas de
16 tokens, KV Q4 G32, capacidade 512, cauda de 4 tokens por sequência, macOS
ARM64/Python 3.14.5. A segunda sessão foi construída sozinha e nunca derivou da
primeira; seus logits foram idênticos aos de uma sessão independente que
recomputou os 260 tokens.

| Métrica | Duas sessões sem parentesco | Com prefixo reusado |
| --- | ---: | ---: |
| KV residente somado | 45.662 B | 24.174 B |
| Prefixo | 2 × 21.488 B | 21.488 B, uma vez |
| Páginas próprias por sessão | 1.343 B | 1.343 B |
| Páginas copiadas na adoção | — | 0 (`copied_bytes` = 0) |
| Segunda sessão, já aquecido | ~0,051 s | ~0,007 s |

São os mesmos 45.662 B → 24.174 B do `fork`, como tinha de ser: o que muda é a
origem do direito de compartilhar, não a contabilidade. O tempo é amostra local
única depois do primeiro uso (a primeira execução do processo paga a compilação
da biblioteca nativa, ~0,63 s), não benchmark repetido.

A prova central da suíte não é o número, é a **identidade byte a byte**: as
páginas adotadas são comparadas com `ctypes.string_at(page.address,
page_extent_bytes)` contra as de um controle que só executou os primeiros
K × `page_tokens` tokens. Se qualquer byte de página dependesse de algo além da
posição absoluta, a comparação falharia — nos quatro codecs, e tanto quando os
prompts divergem numa fronteira de página quanto no meio de uma.

## Custo, orçamento e relatórios

A derivada é uma sessão completa: admite o próprio orçamento antes de reter ou
copiar qualquer página, e tem a própria arena de ativações. A reserva continua
conservadora — o compartilhamento reduz a residência real, não o teto admitido
para cada sequência.

O relatório separa `kv_shared_page_count`, `kv_shared_allocation_bytes` e
`kv_owned_allocation_bytes`; a soma dos dois últimos é a residência da
sequência. Sob backing store, `kv_shared_backing_pages` conta as páginas
compartilhadas que vivem em disco — elas aparecem em `kv_shared_page_count` com
zero byte residente — e `kv_inherited_stores` diz de quantos stores de outra
sequência esta lê. Logo após a adoção, `kv_prefix_adoption` informa tokens herdados,
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

Sob tiers, troque o codec pela política, e o relatório mostra os codecs herdados:

```bash
python3 tools/nexa_run.py artifacts/models/nexalm-tiny \
  --tokens 1,3,5 --kv-cache --kv-page-tokens 1 --kv-policy age \
  --kv-hot-pages 1 --kv-warm-pages 1 --kv-group-size 3 \
  --max-sequence-length 8 --tile-rows 3 --memory-budget 1MiB --fork-tokens 7,2
```

Regressões:

```sh
python3 -m unittest discover -s tests -p 'test_paged_sequences_regressions.py' -v
python3 -m unittest discover -s tests -p 'test_tiered_sequences_regressions.py' -v
python3 -m unittest discover -s tests -p 'test_prefix_reuse_regressions.py' -v
```

## Limites

Cancelamento assíncrono de uma chamada em andamento e admissão conjunta por
processo permanecem em M4.06d. Cancelar uma chamada em curso exige executá-la
fora da thread que cancela, o que esta sessão síncrona não faz; o que existe é
rollback de falhas e de interrupções.

Uma sessão continua **uma sequência**, síncrona, sem uso concorrente; várias
sequências são várias sessões que compartilham páginas, cada uma com sua arena e
seu orçamento. Não há escalonador, batch nem deduplicação **automática** de
prefixos: não existe índice de páginas por conteúdo, e ninguém procura uma
origem candidata. O compartilhamento é sempre explícito — `fork` entre
parentes, `adopt_prefix` entre sessões que só combinam nos tokens, e em ambos
os casos quem chama escolhe a origem. O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
próxima tarefa.
