# Codecs de peso por tensor: Q4, Q8 e matrizes RAW_F32

O décimo sétimo incremento acrescenta o **segundo codec de peso** e o despacho
por tensor no executor. Até aqui toda matriz era Q4 e o executor só sabia
`nexa_q4_matmul`; agora um tensor pode ser guardado denso em F32, e o mesmo
bundle pode misturar os dois.

Isso não é um formato de distribuição: uma matriz F32 custa oito vezes a forma
Q4. É a referência exata que faltava — sem erro de quantização — e o
pré-requisito de qualquer decisão de precisão por tensor (M6.01/M6.02), além de
ser o lugar onde Q2/Q3/Q8/F16 entram depois (M1.05b).

## Q8_GROUPED

O vigésimo incremento acrescenta o terceiro codec: um byte com sinal por valor,
depois da mesma escala float32 por grupo, com `escala = max|v| / 127` e o código
`-128` reservado. O contêiner é o mesmo NexaPack — blocos, índice e checksums
idênticos —, então só mudam o `codec_id`, o tamanho do grupo em bytes e o kernel.

Custo e erro ficam entre Q4 e denso. Numa linha de exemplo com grupo 4, o erro
máximo do round-trip caiu de 0,179 (Q4) para 0,0098 (Q8), com o grupo passando
de 6 para 8 bytes. Num modelo sintético de duas camadas com grupo 8, o erro nos
logits contra a referência densa caiu de 0,319 para 0,011, e o payload ficou
entre o Q4 e o denso — que é exatamente o ponto: o precision map agora tem um
degrau intermediário real para escolher.

Os três codecs convivem no mesmo bundle. Uma regressão constrói um modelo com
embedding denso, um tensor Q8 e outro Q4, usando pesos que **todos** os codecs
guardam sem erro (zero e ±máximo do grupo, já que Q4 escala por `max/7` e Q8 por
`max/127`), e verifica que os três caminhos produzem os mesmos logits.

## Formato

`RAW_F32_MATRIX`, row-major, little-endian, dividido em **blocos de linhas**:

```json
{"shape": [rows, cols], "codec": "RAW_F32_MATRIX", "codec_version": 1,
 "path": "tensors/0003.f32", "file_bytes": 12288,
 "blocks": [{"start_row": 0, "row_count": 64, "sha256": "..."}]}
```

O bloco é a unidade de verificação **e** de leitura: uma leitura parcial não
poderia conferir os bytes que consumiu, então o tamanho de bloco escolhido na
escrita (`--block-rows`) é o tile que o executor lê. A abertura do bundle exige
que os blocos cubram cada linha exatamente uma vez, em ordem, e rejeita tamanho
de arquivo, checksum, contagem ou cobertura divergentes.

Vetores rank-1 (normas) continuam `RAW_F32` como antes; só matrizes escolhem
codec.

## Despacho no executor

`nexa_f32_matmul` usa a mesma ordem de redução e o mesmo acumulador double do
kernel Q4, então um tensor mantido em F32 difere da sua forma empacotada
**apenas** pela quantização. A prova disso é uma regressão: com pesos que Q4
representa exatamente — múltiplos de 0,5 num grupo cujo máximo é 3,5, de modo
que a escala vira 3,5/7 = 0,5 — os dois caminhos produzem logits idênticos, com
o mesmo SHA-256.

O embedding denso lê apenas os blocos que contêm os tokens pedidos, e o
relatório separa `q4_payload_bytes_read` de `raw_payload_bytes_read`, de forma
que um bundle misto mostra os dois caminhos ativos.

## Uso

```bash
# Um codec para todas as matrizes.
python3 tools/nexa_convert.py --checkpoint CHECKPOINT --out artifacts/models/q8 --matrix-codec q8 --group-size 32

# Codec por tensor, misturando livremente.
python3 tools/nexa_convert.py --checkpoint CHECKPOINT --out artifacts/models/misto2 \
  --tensor-codec model.embed_tokens.weight=q8 \
  --tensor-codec model.layers.0.mlp.down_proj.weight=f32

# Todo o modelo denso: referência exata para comparar quantizações.
python3 tools/nexa_convert.py --checkpoint CHECKPOINT --out artifacts/models/ref --dense-all --block-rows 64

# Um tensor denso e o resto Q4, para isolar o efeito daquele tensor.
python3 tools/nexa_convert.py --checkpoint CHECKPOINT --out artifacts/models/misto \
  --dense-tensor model.layers.0.mlp.down_proj.weight

python3 tools/nexa_inspect.py artifacts/models/ref --verify   # verifica bloco a bloco
python3 tools/nexa_run.py artifacts/models/ref --tokens 1,3 --decode-tokens 5 --tile-rows 3 --memory-budget 4MiB
```

Todos os modos de KV continuam disponíveis nesse caminho: o codec de peso é
independente do codec de cache.

## Custo e limites

Uma matriz densa ocupa `rows × cols × 4` bytes no disco e um tile de
`block_rows × cols × 4` na leitura, que entra no orçamento como qualquer outro
buffer. Blocos grandes demais são recusados na abertura com mensagem explícita:
converta com `--block-rows` menor. O teto de arquivo é 16 GiB e o de blocos
65.536 por tensor.

O embedding denso lê um bloco inteiro por token alcançado, então um bloco grande
custa I/O mesmo para um único token. Para prompts longos com vocabulário grande,
prefira Q4 no embedding ou blocos menores.

Q2, Q3 e F16 de **pesos** continuam pendentes em M1.05c, assim como o despacho
de atenção para esses layouts (M4.03d). A calibração e o PrecisionMap ainda
comparam apenas Q4 contra denso: medir e planejar com Q8 no espaço de escolha
é M6.01c/M6.02b, e o contrato do mapa não muda por isso — apenas ganha mais um
valor possível por tensor. O [checklist](BLUEPRINT_512MB_CHECKLIST.md)
registra a suíte, os comandos e a próxima tarefa.
