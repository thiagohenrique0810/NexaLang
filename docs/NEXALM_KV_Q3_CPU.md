# KV Q3 paginado em CPU

O modo Q3 comprime K/V por token e head, incluindo páginas parciais. A atenção
consome escalas e códigos diretamente, preservando as
[transações e o orçamento das páginas](NEXALM_KV_PAGINADO_CPU.md).
Os pesos do modelo continuam Q4; este incremento não implementa pesos ou GEMM Q3.

## Uso

```sh
# Depois de gerar/importar a fixture tiny conforme o guia de importação.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 8 --kv-codec q3 --kv-group-size 4 --prefill-chunk-size 2 --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-q3.json

# Opcional: PyTorch e checkpoint original permitem separar os erros.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 8 --kv-codec q3 --kv-group-size 4 --prefill-chunk-size 2 --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB --verify --reference-checkpoint artifacts/checkpoints/nexalm-tiny --report artifacts/reports/nexalm-kv-q3.json
```

Na API, use `PagedTransformerSession(..., kv_codec="q3", kv_group_size=32)`.
O grupo padrão é 32; valores inteiros de 1 a 1.048.576 são aceitos, incluindo
grupos maiores que o head. Na CLI, Q3 exige `--kv-cache` e `--kv-page-tokens`.
`--kv-group-size` é válido para Q4 e Q3. O padrão continua F32.
Planos JSON F32/Q4 preservam seus contratos; Q3 identifica seu codec e versão.

## Formato Q3_GROUPED V1

Cada buffer K ou V usa a ordem `[token, head KV, grupo dentro do head]`:

- Grupo de `G` coordenadas: escala F32 little-endian e `ceil(3*G/8)` bytes de códigos.
- Escala `float32(max(abs(grupo))/3)`. Divisão pela escala armazenada, arredondamento
  para o inteiro mais próximo com empates para longe de zero e saturação em `[-3,3]`.
- Códigos de três bits em complemento de dois: `q & 7`. O código `100` (-4) é
  reservado e inválido. O valor da coordenada `i` começa no bit `3*i`, LSB primeiro;
  códigos podem atravessar dois bytes. Não há alinhamento entre coordenadas.
- Coordenadas ausentes na cauda e bits altos sem uso no último byte ficam zero.
  Grupo com escala zero contém somente códigos zero.
- Entradas não finitas são inválidas. Underflow da escala de um grupo não zero
  produz erro numérico; a atenção rejeita escalas, códigos e padding inválidos.

K é quantizado após RoPE e V após projeção. Só os tokens novos são codificados;
suas escalas não dependem de futuros tokens. Completar uma página não recodifica o
prefixo. O quantizador escreve diretamente nos segmentos das páginas.

O kernel lê `escala * código` em escalares double durante a atenção, sem criar um
buffer F32 de head, página ou cache e sem arredondamento intermediário para F32.
As exponenciais do softmax usam o scratch F32 planejado; reduções usam double e a
saída final é F32. `nexa_q3_row_size`, `nexa_q3_quantize` e
`nexa_causal_gqa_attention_paged_q3` compõem a implementação C.

## Memória e atomicidade

Com dimensão de head `D`, `Hkv` heads, `L` camadas e `P` tokens por página:

```text
bytes por head = ceil(D/G) * (4 + ceil(3*G/8))
bytes por token de um buffer K ou V = Hkv * bytes por head
payload por página = 2 * L * P * Hkv * bytes por head
alocação por página = 2 * L * align64(P * Hkv * bytes por head) + 63
```

A reserva de admissão continua cobrindo `ceil(C/P) + ceil(T/P)` páginas para
contexto máximo `C` e chunk máximo `T`. Residência real, staging e reserva são
reportados separadamente, junto de escalas, padding, tabelas de ponteiros e scratch.
O orçamento cobre buffers CPU gerenciados, não RSS, objetos Python, bibliotecas,
cache do SO, logits retidos pelo consumidor ou PyTorch de verificação. VRAM não
foi medida e permanece nula nos relatórios.

Falhas preservam tokens, relatório e bytes do prefixo confirmado. Sufixos escritos
antes de uma falha ficam invisíveis e são sobrescritos no retry. Páginas novas e
workspace são liberados mesmo quando o consumidor retém a exceção. Substituir o
prompt prepara páginas novas antes do commit; `reset`/`close` liberam a residência.

## Comparação reproduzível F32/Q4/Q3

A fixture abaixo usa pesos sintéticos determinísticos, `D=64`, um head/camada,
`P=16`, grupos de 32, contexto máximo 16 e chunks de dois tokens. Os três modos
processam os mesmos oito IDs, sem recomputar o prefixo. Execute na raiz do projeto:

```sh
# Apenas na primeira geração: a pasta de saída deve ser nova.
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'tests')
from test_q4_paged_transformer_regressions import wide_bundle
wide_bundle(Path('artifacts/models/nexalm-kv-wide'))
PY

python3 tools/nexa_run.py artifacts/models/nexalm-kv-wide --tokens 1,3,5,7,2,4 --decode-tokens 6,8 --kv-cache --kv-page-tokens 16 --kv-codec f32 --prefill-chunk-size 2 --max-sequence-length 16 --tile-rows 32 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-wide-f32.json
python3 tools/nexa_run.py artifacts/models/nexalm-kv-wide --tokens 1,3,5,7,2,4 --decode-tokens 6,8 --kv-cache --kv-page-tokens 16 --kv-codec q4 --kv-group-size 32 --prefill-chunk-size 2 --max-sequence-length 16 --tile-rows 32 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-wide-q4.json
python3 tools/nexa_run.py artifacts/models/nexalm-kv-wide --tokens 1,3,5,7,2,4 --decode-tokens 6,8 --kv-cache --kv-page-tokens 16 --kv-codec q3 --kv-group-size 32 --prefill-chunk-size 2 --max-sequence-length 16 --tile-rows 32 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-wide-q3.json
```

Acrescente `--verify` às três execuções para reproduzir os erros abaixo com PyTorch
local. A referência usa equações PyTorch e um codec Python independente do C.

| Medida | F32 KV | Q4 KV | Q3 KV |
|---|---:|---:|---:|
| Bytes/token, K/V de todas as camadas | 512 | 80 | 64 |
| Payload de uma página | 8.192 B | 1.280 B | 1.024 B |
| Alocação física da página residente | 8.255 B | 1.343 B | 1.087 B |
| Reserva KV de admissão | 16.510 B | 2.686 B | 2.174 B |
| Limite gerenciado para a capacidade | 87.869 B | 74.045 B | 73.533 B |
| Maior pico gerenciado entre chamadas | 79.614 B | 72.702 B | 72.446 B |
| Erro máximo de execução vs oracle do mesmo codec | 0 | 0 | 0 |
| Erro máximo do KV vs KV F32, mantendo os pesos Q4 | 0 | 0,2619032860 | 0,5301163346 |

Neste caso, Q3 reduziu o payload KV em 20% e a alocação física da página em
aproximadamente 19,1% contra Q4, com maior erro numérico do KV. Escalas e alinhamento
podem anular a economia em heads pequenos: na fixture tiny com `D=4`, `G=4`, `P=8`,
Q3 e Q4 ocupam os mesmos 12 B/token e 191 B por página.

Na tiny, Q3 teve erro de execução zero, erro dos pesos de 0,4435420930, erro do KV
de 0,6113268733 e erro combinado de 0,9679856896. Esses erros não são aditivos.
Na fixture wide não há checkpoint original para medir o erro dos pesos:
`quantization_error` fica nulo. `verified` valida somente a execução com a mesma
representação; não aprova qualidade/perplexidade. Os pesos são sintéticos e não
há conclusão sobre qualidade de linguagem, ganho de velocidade ou GPU.

## Validação e próximos passos

```sh
python3 -m unittest discover -s tests -p 'test_q3*regressions.py' -v
python3 -S -m unittest discover -s tests -p 'test_q3_paged_transformer_regressions.py' -v
```

Os testes incluem referência independente dos bytes, atenção MHA/MQA/GQA,
caudas/códigos entre bytes, grupos ímpares e maiores que o head, corrupção,
overflow, aliases, sanitizers e kernels sem heap. Sessões exercitam invariância
entre chunks/páginas, golden sem Torch, prefixos imutáveis, orçamento e rollback.
O [checklist](BLUEPRINT_512MB_CHECKLIST.md) mantém TQ01, tiers, evicção, múltiplas
sequências, qualidade de checkpoint treinado e GPU como trabalho futuro.
