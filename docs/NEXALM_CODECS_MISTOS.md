# Codecs mistos por bloco: o que o formato passa a permitir

Este incremento (M6.02d) troca **uma restrição do formato**, não a qualidade de
nenhum modelo. Até aqui, um tensor empacotado tinha um `codec_id`, um
`group_size` e um `row_bytes` **por arquivo**, e o leitor derivava o offset de
cada bloco **multiplicando** `row_count × row_bytes`. Com isso, um passo
heterogêneo era estruturalmente impossível: não havia onde escrever que o bloco
3 é Q8 e o bloco 4 é Q2.

Agora existe `MIXED_GROUPED`, com `codec_id` e `row_bytes` **por bloco**.

```python
from runtime.nexapack.format import write_mixed_matrix

write_mixed_matrix(path, rows, cols, group_size, row_source,
                   block_codecs=['q2', 'q3', 'q4', 'q8'], block_rows=64)
```

```bash
python3 tools/nexa_inspect.py misto.nxp --verify
```

## O que este item NÃO prova

Nada aqui demonstra que misturar codecs melhora um modelo. O formato passou a
**permitir** a mistura e passou a **cobrar um preço medido** por ela. A pergunta
difícil — *qual bloco recebe qual codec* — depende de sensibilidade por bloco
medida em pesos reais, que é **M6.01d** e continua bloqueada em checkpoint
treinado. Sem isso, qualquer atribuição de codec a bloco é arbitrária.

E há uma armadilha específica que este guia se recusa a cair: seria fácil
construir uma matriz cujas primeiras 64 linhas são quase-zero e as seguintes são
outliers, mostrar que q2+q8 vence qualquer codec único, e chamar isso de ganho.
**Isso não demonstraria nada sobre um modelo** — demonstraria que a matriz foi
desenhada para o resultado. Nenhuma matriz deste incremento tem estrutura por
bloco (veja [Fixtures](#fixtures)), e nenhum número abaixo é um número de
qualidade.

`nexa_inspect` continua publicando `model_quality_measured: false`, pelo mesmo
motivo que [custo físico](NEXALM_CUSTO_FISICO.md) publica `quality_measured`.

## O número fácil e o número certo

O custo de metadados por bloco é pequeno: medido, **+39 bytes por bloco** no
índice JSON (matriz de 64 colunas, grupo 32, Q3 — o valor varia entre ~34 e ~39
conforme o comprimento do nome do codec e a largura em dígitos de `row_bytes`).

E como `payload_offset` é alinhado a 4096, esses bytes **somem no padding** que
o arquivo uniforme já pagava. É tentador liderar com esse zero. Seria desonesto
por duas razões.

**Primeira: o zero não é um ganho, é a remoção de uma objeção.** Ele diz que o
formato heterogêneo não custa caro, não que misturar compense.

**Segunda: o zero acaba antes do que parece.** O delta em bytes de arquivo não é
uma escada que sobe uma vez — é um dente de serra, porque cada índice cruza suas
próprias fronteiras de página. Medido com blocos de 1 linha, 64 colunas, grupo
32, todos Q3:

| Blocos | Índice uniforme | Índice misto | Δ índice | Arquivo uniforme | Arquivo misto | Δ arquivo |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 22 | 3.112 | 3.961 | +849 | 4.800 | 4.800 | 0 |
| **23** | 3.243 | 4.131 | +888 | 4.832 | **8.928** | **+4.096** |
| 61 | 8.282 | 10.652 | +2.370 | 14.240 | 14.240 | 0 |
| 128 | 17.155 | 22.138 | +4.983 | 24.576 | 28.672 | +4.096 |
| 1.024 | 137.372 | 177.299 | +39.927 | 172.032 | 212.992 | +40.960 |

**A partir de 23 blocos o arquivo já cresce**, uma página inteira de uma vez, e
volta a zero em 61 quando o arquivo uniforme também cruza a fronteira. Dizer
"zero até 1.024 blocos" seria estar medindo o alinhamento, não o codec.

## Onde a mistura para de pagar

Este é o número que vale o incremento. Uma matriz só, 1.024 linhas, 64 colunas,
grupo 32. Metade dos blocos fica em Q4 e metade cai para Q2 — então **o payload
economizado é sempre os mesmos 8.192 bytes**, em todas as linhas da tabela. A
única coisa que muda é em quantos blocos a matriz é cortada:

| Blocos | Linhas/bloco | Payload economizado | Δ índice | Δ arquivo | Veredito |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 512 | 8.192 | +69 | −8.192 | paga |
| 16 | 64 | 8.192 | +615 | −8.192 | paga |
| 64 | 16 | 8.192 | +2.487 | −8.192 | paga |
| 128 | 8 | 8.192 | +4.983 | −4.096 | paga menos |
| 256 | 4 | 8.192 | +9.847 | **0** | empata |
| 512 | 2 | 8.192 | +20.215 | **+12.288** | **perde** |
| 1.024 | 1 | 8.192 | +39.927 | **+32.768** | **perde** |

**A mistura para de pagar em 256 blocos e passa a perder em 512.** Em 1.024
blocos ela economiza 8 KiB de payload e custa 32 KiB de arquivo — quatro vezes
mais do que economizou. Confirmado sobre arquivos que o escritor realmente
produziu, não só sobre o layout calculado
(`test_the_same_curve_on_files_the_writer_actually_produced`).

A regra por trás disso é simples e vale para qualquer forma: **um bloco só paga
o próprio índice se o rebaixamento dele economizar mais do que ~35 bytes.**

| Colunas | Grupo | Linha Q4 | Linha Q2 | Economia/linha | Índice/bloco | Linhas/bloco necessárias |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 8 | 6 | 2 | 34,0 | **17** |
| 16 | 16 | 12 | 8 | 4 | 33,5 | 9 |
| 32 | 32 | 20 | 12 | 8 | 34,5 | 5 |
| 64 | 32 | 40 | 24 | 16 | 34,5 | 3 |
| 256 | 32 | 160 | 96 | 64 | 34,0 | 1 |
| 4.096 | 32 | 2.560 | 1.536 | 1.024 | 35,5 | 1 |

Em matriz larga, um bloco de uma linha já se paga. Em matriz estreita, são
necessárias 17 linhas por bloco antes de a troca valer o índice que ela exige.

## O outro caso em que misturar perde: o teto de blocos

`MAX_BLOCKS` é 8.192 e `MAX_METADATA_BYTES` é 1 MiB. Com blocos de uma linha, a
tampa de metadados fecha antes do teto de blocos — e fecha mais cedo para o
índice mais largo:

| Forma | Blocos de 1 linha que o índice descreve |
| --- | ---: |
| uniforme | **7.716** |
| misto | **5.996** |

Uma matriz que o formato uniforme indexa é **recusada** como mista: misturar
custa 22% da capacidade de índice. Não é uma perda de bytes, é uma perda de
alcance, e é a única das medições aqui em que a mistura não tem contrapartida
nenhuma.

## Formato

`MIXED_GROUPED` tem **conjunto próprio de chaves de metadados**. Não é enfeite:
é o que mantém os quatro conjuntos homogêneos e os arquivos deles **byte a byte
inalterados**. O índice misto não ganha nem perde nada em relação ao que existia
— ele é outro conjunto.

| Chave | Uniforme | Misto |
| --- | --- | --- |
| `codec_id` (arquivo) | `Q4/Q8/Q3/Q2_GROUPED` | `MIXED_GROUPED` |
| `storage_dtype` | `q4`/`q8`/`q3`/`q2` | `mixed` |
| `group_size` | por arquivo | por arquivo (igual) |
| `row_bytes` | **por arquivo** | **ausente** |
| bloco: `codec_id` | ausente | **presente** |
| bloco: `row_bytes` | ausente | **presente** |

`group_size` continua por arquivo de propósito: ele não muda a largura relativa
dos codecs, e mantê-lo único deixa um único número para a validação por bloco
conferir.

### O leitor deixa de multiplicar

Antes, `_load_metadata` derivava `size` de cada bloco de `row_count × row_bytes`
do arquivo. Agora, para cada bloco:

1. `codec_id` do bloco tem de ser um dos quatro codecs agrupados;
2. `row_bytes` declarado é comparado com `_row_bytes(cols, group_size, codec)` —
   recomputado, nunca aceito;
3. `size` tem de ser `row_count ×` essa largura;
4. `offset` tem de ser contíguo ao bloco anterior, e o último tem de terminar
   exatamente no `total_size` do cabeçalho.

Sem (2) e (3), uma largura errada por um grupo ainda ladrilharia o payload, e
**todo bloco seguinte seria decodificado a partir de bytes do anterior**.

### Duas recusas redundantes, medidas

Como o `.nxb` já registrou para a checagem de `offset + size`, aqui também há
cláusulas que o resto da validação já cobre. Isso está **medido por injeção de
bug**, não suposto:

| Cláusula | Sem ela | Por que fica |
| --- | --- | --- |
| `codec_id` do bloco pertence aos quatro | derivar a largura de um codec desconhecido já falha | a **mensagem**: sem ela o leitor acusa `group_size`, e o operador procura no campo errado |
| `offset` contíguo | a cobertura exata do payload já falha | aponta o bloco onde a cadeia quebrou |

O `row_bytes` por bloco, ao contrário, **não** é redundante: remover a
comparação com a largura recomputada faz duas mutações de índice passarem.

## Identidade byte a byte

`write_mixed_matrix` usa a **mesma passagem única de streaming** e os **mesmos
codificadores** de `write_grouped_matrix` — o codec é lido uma vez por bloco em
vez de uma vez por arquivo. O teste correspondente escreve a mesma fonte duas
vezes, com `codec=Q3_GROUPED` e com `block_codecs=['q3'] * 4`, e exige que a
**região de payload seja idêntica**. Os cabeçalhos diferem em 147 bytes, e esse
é o ponto.

Esse teste mede **layout**, não codificação: um escritor misto com codificador
próprio seria uma segunda grafia do mesmo codec, e o projeto já recusa isso. A
independência do codificador vem de outro lugar — o bloco Q3 do arquivo misto é
cruzado com `tests/q3_reference.py`, a implementação independente que existe
exatamente para impedir duas grafias de um codec.

Os quatro arquivos homogêneos continuam byte a byte o que eram: o teste fixa o
SHA-256 de cada um, capturado do escritor **antes** de `MIXED_GROUPED` existir.

## Fixtures

Todas as matrizes deste incremento são **sintéticas** e geradas por
`((linha × 37 + coluna × 11) mod 23) − 11 + 0,5 × ((linha + coluna) mod 3)`.
Ela é determinística e tem **as mesmas estatísticas em todas as linhas**: nenhum
bloco de linhas é mais fácil ou mais difícil que outro. Foi escolhida assim de
propósito — uma fixture com estrutura por bloco faria a mistura parecer boa sem
que nada de modelo tivesse sido medido.

**Nenhuma afirmação de qualidade é feita neste incremento.** Todos os números
acima são bytes de arquivo e bytes de índice, contados. Erro de quantização,
perplexidade e sensibilidade por bloco não foram medidos aqui.

## Limites

- **Nenhum planejador escolhe os codecs.** `block_codecs` é um argumento; quem
  o preenche é o chamador. A escolha informada é M6.01d.
- **Nada acima da camada de formato mudou.** Sem bundle, sem transformer, sem
  `.nxb`, sem kernel: `bundle.py`, `executor.py` e `transformer.py` continuam
  lendo `reader.row_bytes`, que é `None` num arquivo misto. Um tensor misto
  ainda não pode ser executado, e isso é deliberado — despachar kernel por
  bloco dentro do mesmo matmul é trabalho próprio.
- **`group_size` continua uniforme.** Variá-lo por bloco multiplicaria as
  combinações sem que exista ainda alguém para escolher entre elas.
- O SHA-256 por bloco detecta corrupção; não autentica publicador.
- O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
  próxima tarefa.
