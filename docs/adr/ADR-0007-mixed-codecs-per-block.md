---
adr: ADR-0007
title: MIXED_GROUPED — um codec por bloco de linhas, e o preço disso
status: accepted
identifiers:
  - name: runtime.nexapack.format:MIXED_CODEC_ID
    value: 'MIXED_GROUPED'
  - name: runtime.nexapack.format:MIXED_CODEC_VERSION
    value: 1
prior_art:
  - id: P28
    role: problem
    note: A trilha M6 cita P28 entre as referências de compressão. Entra como enunciado do problema de heterogeneidade intratensor; a tabela de custo abaixo é medida neste repositório e não é confirmada por nenhuma referência.
---

# ADR-0007 — Codecs mistos por bloco de linhas

## Contexto

ADR-0008 escolhe um codec **por tensor**. A pergunta seguinte é se vale escolher
por região dentro do tensor: linhas com estatísticas diferentes poderiam tolerar
larguras diferentes.

## Problema técnico

Era **estruturalmente impossível**, não caro. O leitor re-derivava o offset de
todo bloco multiplicando uma largura de linha única, válida para o arquivo
inteiro, pela contagem de linhas. Um passo heterogêneo não tinha como ser
expresso: não havia onde escrever que este bloco é mais estreito que o próximo.

## Decisão

Um codec de composição, `MIXED_GROUPED` versão 1, com **conjunto próprio de
chaves**. Numa matriz mista, `row_bytes` sai do nível do arquivo e entra em cada
bloco, ao lado do `codec_id` daquele bloco. O leitor **recomputa** a largura a
partir do codec declarado e recusa o valor declarado que discorde — o número no
índice é verificado, não obedecido.

Os quatro codecs uniformes mantêm o conjunto de chaves que já tinham. Essa
separação é o que preserva os arquivos homogêneos **byte a byte idênticos**: o
metadado deles não ganha nada e não perde nada.

A escrita usa a mesma passagem única de streaming e os mesmos codificadores. O
codec é lido uma vez por bloco em vez de uma vez por arquivo. Um segundo
codificador seria uma segunda grafia de codec, e o payload de um arquivo misto
deixaria de ser comparável byte a byte com o uniforme.

**Este ADR remove uma impossibilidade estrutural e põe um preço na mesa. Ele não
demonstra ganho.**

## Medições próprias

Fixture: 1.024 linhas, 64 colunas, grupo 32, metade dos blocos rebaixada de q4
para q2. Uma linha q4 ocupa 40 bytes e uma q2 ocupa 24, então rebaixar metade
das linhas economiza **exatamente 8.192 bytes de payload qualquer que seja a
contagem de blocos**. É isso que deixa o número de blocos ser a única variável,
e nenhum ganho vir da escolha de codec:

```adr-measurement
name: mixed_payload_saved_bytes
value: 8192
unit: bytes de payload economizados
source: mixed_payload_saved_bytes
```

Com 64 blocos, a mistura paga:

```adr-measurement
name: mixed_index_delta_64_blocks
value: 2487
unit: bytes de índice
source: mixed_index_delta_64_blocks
```

```adr-measurement
name: mixed_file_delta_64_blocks
value: -8192
unit: bytes de arquivo
source: mixed_file_delta_64_blocks
```

Com 256 blocos, empata:

```adr-measurement
name: mixed_index_delta_256_blocks
value: 9847
unit: bytes de índice
source: mixed_index_delta_256_blocks
```

```adr-measurement
name: mixed_file_delta_256_blocks
value: 0
unit: bytes de arquivo
source: mixed_file_delta_256_blocks
```

Com 512, perde:

```adr-measurement
name: mixed_index_delta_512_blocks
value: 20133
unit: bytes de índice
source: mixed_index_delta_512_blocks
```

```adr-measurement
name: mixed_file_delta_512_blocks
value: 12288
unit: bytes de arquivo
source: mixed_file_delta_512_blocks
```

Com 1.024, perde quatro vezes o que economizou:

```adr-measurement
name: mixed_index_delta_1024_blocks
value: 39927
unit: bytes de índice
source: mixed_index_delta_1024_blocks
```

```adr-measurement
name: mixed_file_delta_1024_blocks
value: 32768
unit: bytes de arquivo
source: mixed_file_delta_1024_blocks
```

**Segunda perda, sem contrapartida nenhuma.** A tampa de 1 MiB de metadados de
ADR-0001 fecha antes de `MAX_BLOCKS`, e fecha mais cedo para o índice mais
largo:

```adr-measurement
name: uniform_max_blocks_under_metadata_cap
value: 7716
unit: blocos
source: uniform_max_blocks_under_metadata_cap
```

```adr-measurement
name: mixed_max_blocks_under_metadata_cap
value: 5996
unit: blocos
source: mixed_max_blocks_under_metadata_cap
```

Misturar custa **22% da capacidade de índice**: uma matriz que o formato uniforme
descreve é recusada como mista.

A promessa de que os arquivos homogêneos não mudaram é verificável no disco. Um
bloco uniforme grava cinco chaves, e não as sete que o leitor apresenta — o
`codec_id` e o `row_bytes` por bloco são sintetizados na leitura:

```adr-measurement
name: uniform_block_keys_on_disk
value: ('offset', 'row_count', 'sha256', 'size', 'start_row')
unit: chaves por bloco no índice gravado
source: uniform_block_keys_on_disk
```

## Alternativas descartadas

**O atalho, recusado por escrito.** Construir uma matriz com as primeiras linhas
quase-zero e o resto outlier mostraria q2+q8 vencendo qualquer codec único por
uma margem confortável. A fixture foi deliberadamente construída **sem estrutura
por bloco**, com todas as linhas nas mesmas estatísticas, porque uma fixture
desenhada para ganhar não demonstra nada sobre um modelo.

**Confiar no `row_bytes` declarado por bloco.** Seria mais rápido. Recusado: um
valor declarado que o leitor não recomputa é um offset que um arquivo malformado
escolhe.

**Estender o conjunto de chaves uniforme para sempre carregar `codec_id` por
bloco**, unificando os dois formatos. Recusado porque mudaria todo arquivo
homogêneo existente, trocando a compatibilidade byte a byte por simetria de
código.

## Limites declarados

**Nada executa um tensor misto.** `bundle.py`, `executor.py` e `transformer.py`
leem `reader.row_bytes`, que é `None` num arquivo misto. O despacho de kernel
por bloco dentro do mesmo matmul não foi feito. Hoje só a API Python escreve um
arquivo misto, e nada o lê para calcular.

**A pergunta que decide o item continua sem resposta: qual bloco recebe qual
codec.** Ela depende de sensibilidade por bloco em pesos reais, que depende de
checkpoint treinado. Sem isso, toda atribuição de codec a bloco é arbitrária.

Regra de break-even que sai da tabela: um bloco só paga o próprio índice se o
rebaixamento economizar mais de ~35 bytes.

O `mixed_index_delta_512_blocks` acima é **20.133**, medido com a metade inicial
dos blocos rebaixada. O registro do incremento publica **20.215** para a mesma
fixture. A diferença é real e é de padrão: o delta de índice depende de quais
blocos são rebaixados, porque isso muda a largura decimal dos offsets e tamanhos
no JSON. Com rebaixamento alternado o mesmo código dá 20.184. Os outros três
deltas de índice (2.487, 9.847, 39.927) e os quatro deltas de arquivo
reproduzem o registro exatamente. O número declarado aqui é o que esta fixture,
totalmente especificada, produz.
