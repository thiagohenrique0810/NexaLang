# Pacote `.nxb` V1: um arquivo, os mesmos bytes

O trigésimo incremento troca o **número de arquivos**, não o conteúdo deles. Um
bundle de modelo deixa de ser um diretório com N+2 arquivos e passa a caber em
um `.nxb` único, com execução direta a partir dele. Nenhum tensor é
reconvertido: o `.nxb` guarda cada `.nxp`, cada `.f32` e cada asset **byte a
byte**, e `unpack` reproduz o diretório original com os mesmos caminhos e os
mesmos conteúdos.

```bash
python3 tools/nexa_pack.py pack modelo/ modelo.nxb
python3 tools/nexa_pack.py verify modelo.nxb
python3 tools/nexa_pack.py unpack modelo.nxb restaurado/
python3 tools/nexa_inspect.py modelo.nxb --verify
python3 tools/nexa_run.py modelo.nxb --tokens 1,5,2 --report fora/run.json
```

## O que este item NÃO entrega

O título original de M1.10b nomeia quatro cargas: **plano, kernels, variantes e
fallback**. Nenhuma das quatro entra aqui, e a razão é medida, não estética.

Existe exatamente **uma implementação por kernel**: `runtime/build_runtime.py`
compila com o clang do host e `_load_kernels` carrega um único `.dylib`. Não há
segundo backend, e M2.01/M2.02/M2.03 continuam pendentes. Uma tabela de
variantes com uma entrada, que escolhe sempre a si mesma, é afirmação
infalsificável: ela passaria em qualquer teste porque não existe caso em que
possa escolher errado.

O "plano" também não tem consumidor. `lower_model(config, sequence_length)`
recebe o comprimento como argumento de execução, então um plano persistido
valeria para um único comprimento de sequência e seria recalculado em todos os
outros.

Os dois ficam para **M1.10c** (plano/variantes, quando houver um segundo
backend para escolher entre) e **M1.10d** (kernels e fallback embutidos). O
gancho já está no formato: `kind` de seção. Um `kind` desconhecido é
**recusado**, não ignorado, então um `.nxb` V2 com seções de plano não pode ser
lido pela metade por um leitor V1.

## Contêiner

Todos os inteiros são little-endian. O prefixo de 64 bytes é o mesmo
`struct.Struct('<8sHHIQQ32s')` do [NexaPack V1](NEXAPACK_V1.md):

| Campo | Tamanho | Valor/contrato |
|---|---:|---|
| magic | 8 | bytes ASCII `NEXABNDL` |
| versão | 2 | 1 |
| flags | 2 | 0; flags desconhecidas são rejeitadas |
| tamanho JSON | 4 | 1 a 1.048.576 bytes |
| início payload | 8 | cabeçalho + JSON arredondados para 4096 bytes |
| tamanho arquivo | 8 | tamanho físico exato, sem dados extras |
| SHA-256 JSON | 32 | checksum dos bytes UTF-8 do índice |

O índice JSON tem `format: NexaBundleContainer`, `format_version: 1`,
`alignment: 4096` e `sections`. Cada seção tem `kind`, `path`, `offset`
absoluto, `bytes` e `sha256`.

`kind` é `manifest`, `tensor` ou `asset`, e o `kind` decide que caminho a seção
pode carregar: `manifest` só `manifest.json`, `tensor` só `tensors/...`,
`asset` só `assets/...`. Um `kind` que pudesse nomear qualquer caminho seria um
rótulo, não uma checagem. A seção do manifesto é sempre a primeira.

### Layout: um slot alinhado por seção

Cada seção ocupa um slot de `align(bytes, 4096)`. A seção seguinte começa onde o
slot anterior termina, então o índice cobre todo o payload **sem buraco e sem
sobreposição**, e o arquivo termina em múltiplo de 4096. O padding dentro de um
slot é verificado como zero na abertura; sem isso, a cauda de um slot seria o
lugar para esconder bytes que nada no arquivo confere.

O alinhamento não é enfeite: um `.nxp` aninhado começa em fronteira de página,
então o payload interno dele mantém o alinhamento que tinha como arquivo
avulso, e um leitor futuro pode mapeá-lo sem copiar. Ele **custa**, e o custo
está medido abaixo.

Caminhos usam a mesma regra do manifesto — a validação é literalmente a mesma
função (`container.relative_parts`), porque um caminho que um aceitasse e o
outro recusasse deixaria o pacote diferente do diretório que ele afirma
reproduzir. `../`, absoluto, `C:/`, barra invertida, `//`, `./`, vazio e
duplicado são recusados.

## A janela é a unidade de verificação

`NexaPackReader(path, *, window_offset=0, window_bytes=None)` lê uma matriz que
vive dentro de um arquivo maior. Antes deste incremento, `_load_metadata`
comparava o `total_size` do cabeçalho com `os.fstat(...).st_size` e
`read_rows_into` fazia seek **absoluto** — um `.nxp` embutido era ilegível.

Com a janela:

- `total_size` do cabeçalho é comparado com `window_bytes`, não com o tamanho
  do arquivo;
- todo seek é `window_offset + block.offset`;
- uma janela que escapa do arquivo é recusada **na abertura**, antes de
  qualquer leitura de payload.

Esse é o modo de falha silenciosa deste formato, e a razão de a janela ser um
conceito único em vez de aritmética espalhada: **uma janela aplicada num lugar e
esquecida noutro lê o payload da seção vizinha e ainda passa no próprio SHA**,
porque os offsets do índice e o payload deslizam juntos. Quatro recusas no
leitor, cada uma com teste próprio em `tests/test_container_regressions.py`:

| Recusa | O que acontece sem ela |
| --- | --- |
| janela um byte curta | a matriz é lida além do que a seção possui |
| janela um byte longa (invade o padding seguinte) | idem, na outra direção |
| `total_size` do cabeçalho ≠ `window_bytes` | uma seção é lida com o tamanho de outra |
| janela cujo fim escapa do arquivo | o erro só aparece no meio de uma leitura |

A recusa por `offset + size` de bloco além de `total_size` continua no código,
mas é **redundante** dada a terceira linha acima: blocos são contíguos e têm de
terminar exatamente em `total_size`, então apagar só aquela cláusula deixa o
arquivo recusado pela cobertura. Isso está medido, não suposto — a injeção de
bug correspondente passou nos testes.

Um `SectionStream` faz o mesmo para quem não é matriz: offsets relativos à
seção e leitura sempre limitada a ela, de forma que um `seek` ou `read` além do
fim devolve dado curto em vez dos bytes do vizinho.

## API

`runtime/nexapack/container.py`:

- `pack_bundle(directory, destination)`: abre o diretório com o **mesmo**
  `ModelBundleReader` que o runtime usa, então um bundle inválido é recusado
  antes de qualquer escrita. Grava em temporário, `fsync`, `os.replace` — o
  mesmo contrato de `write_model_bundle`. Retorna os números medidos.
- `unpack_bundle(source, destination)`: reconstrói o diretório verificando o
  SHA-256 de cada seção, roda o preflight do `ModelBundleReader` sobre o
  staging e publica com o rename exclusivo; um destino existente sobrevive.
- `NexaContainerReader(path)`: abre lendo só prefixo, índice e padding.
  `open_section`, `window`, `verify`, `measurements`.
- `is_container(path)`: reconhece pelo magic, nunca pelo sufixo.

`runtime/nexapack/bundle.py`: `ModelBundleReader` aceita diretório **ou**
`.nxb`. A API pública não mudou — por isso `transformer.py`, `paged.py`,
`tiered.py`, `offloaded.py` e os arquivos de teste que importam
`ModelBundleReader` continuam intocados. Internamente há dois backends
(`_DirectoryBackend` e `_ContainerBackend`) que entregam tamanho, identidade e
um stream por payload; toda checagem acima deles — tamanho, checksum, codec,
confinamento — é o mesmo código nos dois casos.

`ModelBundleReader.source_kind` diz `"directory"` ou `"container"`, e
`.container` devolve o índice validado (ou `None`).

## Números medidos

`inspect` publica `storage`, com `container_overhead_bytes` (arquivo menos a
soma das seções), o padding de alinhamento por seção, a contagem de arquivos e
`nxb_vs_directory_bytes` — **assinado**, como o `physical_saved_bytes` de
M6.02c.

Medido diretamente sobre os arquivos escritos, com grupo 32 e blocos de 64
linhas (a fixture tiny usa grupo 4 e blocos de 3):

| | fixture tiny | modelo pequeno (4 camadas, vocab 4096, hidden 256) |
| --- | ---: | ---: |
| arquivos no diretório | 12 | 39 |
| arquivos no `.nxb` | **1** | **1** |
| soma dos `file_bytes` | 36.732 | 2.438.960 |
| `.nxb` no disco | 86.016 | 2.478.080 |
| `nxb_vs_directory_bytes` | **+49.284** (+134,17%) | **+39.120** (+1,60%) |
| cabeçalho + índice + padding do índice | 4.096 | 8.192 |
| padding de alinhamento | 45.188 (média 3.766 B/seção) | 30.928 (média 793 B/seção) |
| blocos alocados, diretório | 81.920 | 2.469.888 |
| blocos alocados, `.nxb` | 86.016 | 2.478.080 |
| delta de blocos alocados | +4.096 | +8.192 |

**O `.nxb` nunca saiu menor.** M6.02c já tinha medido que o contêiner NexaPack
cobra overhead fixo por tensor empacotado; aninhar os `.nxp` verbatim preserva
todos eles e ainda acrescenta uma fronteira de 4096 por seção. Na fixture tiny o
arquivo mais do que dobra, porque cada tensor tem alguns KiB e paga quase uma
página inteira de padding. No modelo pequeno o mesmo overhead vira 1,6%, e a
tendência continua: o padding é **por seção**, não por byte de modelo.

Contra os blocos que o sistema de arquivos realmente aloca — que é o que o disco
cobra — a diferença é de uma a duas páginas nos dois casos, porque o diretório
também arredonda cada arquivo para 4 KiB. A linha `nxb_vs_directory_bytes`
compara com a soma dos `file_bytes` porque é essa a soma que o manifesto
declara; ela é pessimista com o `.nxb` de propósito.

O que o `.nxb` compra, então, é **contagem de arquivos e uma unidade de
publicação**, não espaço. Quem quiser espaço continua olhando para o codec,
não para o empacotamento.

## Limites

- Plano, kernels, variantes e fallback ficam para M1.10c/M1.10d, pelas razões
  acima. `kind` desconhecido recusado é o gancho, sob bump de versão.
- O SHA-256 de seção detecta corrupção; não autentica o publicador. Ele é
  verificado em `unpack`, em `nexa_pack.py verify` e em `nexa_inspect --verify`.
  Durante a execução ele é redundante com as checagens que o próprio bundle já
  faz por tensor (blocos do `.nxp`, `sha256` do vetor, `sha256` do asset), e por
  isso a abertura continua barata.
- Não há compressão, deduplicação nem leitura por `mmap`. O alinhamento existe
  para tornar o `mmap` possível depois; ninguém o usa ainda.
- Não há escrita incremental: republicar um tensor reescreve o arquivo inteiro.
- O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
  próxima tarefa.
