# NexaPack V1 e primeiro executor CPU

Status: formato experimental para **uma matriz Q4 ou TQ MSE**. Este guia descreve
o contêiner comum e a execução Q4; o [contrato TQ portátil](NEXAPACK_TQ_V1.md)
documenta armazenamento/conversão TQ02, sem kernel de matriz TQ. A
[atenção KV TQ](NEXALM_KV_TQ_CPU.md) usa registros desse codec em páginas próprias. O
[manifesto de modelo](NEXALM_IMPORTACAO.md) reúne vários arquivos V1 sem alterar
este contrato. O contêiner `.nxp` é independente do formato TQ01. O futuro `.nxb`
deve agregar plano, tensores e kernels e ainda não está implementado.

## Contêiner

Todos os inteiros e floats persistidos são little-endian. Prefixo de 64 bytes,
equivalente a `struct.Struct('<8sHHIQQ32s')`:

| Campo | Tamanho | Valor/contrato |
|---|---:|---|
| magic | 8 | bytes ASCII `NEXAPACK` |
| versão | 2 | 1 |
| flags | 2 | 0; flags desconhecidas são rejeitadas |
| tamanho JSON | 4 | 1 a 1.048.576 bytes |
| início payload | 8 | cabeçalho + JSON arredondados para 4096 bytes |
| tamanho arquivo | 8 | tamanho físico exato, sem dados extras |
| SHA-256 JSON | 32 | checksum dos bytes UTF-8 do JSON |

Para Q4, o JSON contém `format`, `format_version`, `shape: [rows, cols]`,
`logical_dtype: f32`, `storage_dtype: q4`, `codec_id: Q4_GROUPED`,
`codec_version: 1`, `group_size`, `endianness: little`, `checksum: sha256`,
`row_bytes`, `block_rows` e `blocks`.

Cada bloco tem `start_row`, `row_count`, `offset` absoluto, `size` e `sha256`
hexadecimal. Os blocos são contíguos e cobrem todas as linhas e todo o payload,
sem sobreposição ou buracos. O padding após o JSON é zero. Campos duplicados,
desconhecidos e versões/formatos incompatíveis são rejeitados.

Limites V1: 8192 blocos, grupo até 1.048.576 valores, linha codificada até 64 MiB,
uma chamada de leitura até 64 MiB, inteiros até `2**63-1`. Matrizes maiores devem
ser consumidas em chamadas menores. Aumentar `block_rows` reduz o índice, mas
amplia o trabalho de checksum para leituras parciais.

SHA-256 detecta corrupção; não autentica o publicador. O reader verifica o
cabeçalho na abertura e cada bloco solicitado antes de retornar os dados. Arquivo
aberto não representa uma autorização para ignorar checksums em leituras futuras.

## Codec Q4_GROUPED versão 1

Pesos são armazenados por linha. Cada grupo ocupa:

```text
4 bytes: scale float32 little-endian
ceil(group_size / 2) bytes: dois valores Q4 por byte, primeiro no nibble baixo
```

Os valores usam complemento de dois em `[-7, 7]`; o código `-8` é inválido.
Valores inexistentes no último grupo e o nibble alto extra em grupos ímpares
devem ser zero. `scale = float32(max(abs(group)) / 7)`. A quantização divide pela
escala persistida, arredonda empates afastando de zero e limita a `[-7, 7]`.

Grupo inteiramente zero usa escala zero e payload zero. Escalas negativas/não
finitas, entradas não finitas e escala de grupo não nulo que arredonda para zero
são rejeitadas. Não há conversão silenciosa de NaN ou overflow para valores finitos.

```text
group_bytes = 4 + ceil(group_size / 2)
row_bytes = ceil(cols / group_size) * group_bytes
payload_bytes = rows * row_bytes
```

Exemplo: `cols=128`, `group_size=32` usa 80 bytes por linha, incluindo escalas,
contra 512 bytes FP32. Os cabeçalhos do arquivo são contabilizados separadamente.

## APIs e execução

Python em `runtime/nexapack/format.py`:

- `write_q4_matrix(path, rows, cols, group_size, row_source, block_rows=64)`:
  consome um iterador por grupos, escreve arquivo temporário, sincroniza e substitui
  o destino somente após sucesso.
- `NexaPackReader(path)`: context manager; abrir carrega somente índice limitado.
- `read_rows_into(start, count, destination)`: preenche um buffer exato fornecido
  pelo chamador. Usa 64 KiB de scratch fixo para ler/verificar blocos intersectados.
- `read_rows(start, count)`: conveniência que aloca o resultado; o executor usa
  `read_rows_into` para preencher a arena planejada.
- `payload_bytes_read`: bytes solicitados ao arquivo, incluindo checksum fora das
  linhas retornadas. Isso não mede tráfego físico de disco nem transferência GPU.
- `validate_q4_row(data, cols, group_size)`: valida escala, códigos e padding com
  memória auxiliar constante, sem lista de valores float. `nexa_inspect.py --verify`
  combina essa validação com os checksums do reader; a inspeção padrão segue lazy.

C em `runtime/nexapack/q4.h`:

- `nexa_q4_row_size` / `nexa_q4_size`: tamanhos verificados, zero para argumentos
  inválidos ou overflow.
- `nexa_q4_quantize`: conversão float32 → Q4 com contagens/capacidades explícitas.
- `nexa_q4_matmul`: `input[batch, cols] @ weights[rows, cols]^T`, produzindo
  `output[batch, rows]`. Acumula em double e retorna float32.
- `nexa_q4_decode_row`: decodifica uma linha packed selecionada no buffer F32 do
  chamador, validando escala/códigos/padding e overflow. É usado pelo embedding
  no [executor Transformer CPU](NEXALM_EXECUCAO_CPU.md), sem expandir a matriz inteira.

O kernel não aloca heap/workspace e desquantiza cada valor durante o produto.
Não há uma matriz completa de pesos float na execução. Retornos negativos
identificam argumento inválido, capacidade insuficiente, overflow, dado inválido
ou faixa numérica inválida. O chamador descarta a saída inteira em caso de erro.

`runtime/nexapack/executor.py` valida o plano antes de ler payload ou carregar o
kernel, aloca uma arena com base alinhada a 64 bytes e usa seus offsets reais.
Entrada, tile packed e tile de saída têm lifetimes simultâneos quando necessário.
O orçamento inclui scratch de leitura e até 63 bytes de padding da base. Saídas
são consumidas por tile; reter dados em callbacks é responsabilidade do chamador.
O preflight limita o tile a 64 MiB de payload para respeitar o contrato do reader;
o relatório registra a quantidade efetiva de linhas por tile.

`tools/nexa_convert.py` lê float32 LE em chunks de no máximo 64 KiB, inclusive
quando uma linha é maior que esse limite. O writer conserva somente o grupo de
quantização corrente e o índice limitado, além dos buffers de I/O.

## Interpretação do benchmark

`tools/nexa_bench.py` usa entradas determinísticas e gera JSON/CSV. O SHA-256 dos
metadados identifica o índice e os checksums do payload. Digests de saída por batch
não dependem do número de linhas por tile. A referência opcional Python reconstrói
um valor por vez e verifica aritmética Q4; não valida perplexidade ou qualidade de
um modelo. Tempo dessa referência aparece separado do tempo do kernel.

`managed_buffers_peak_bound_bytes` é um limite contabilizado dos buffers explícitos,
nunca uma amostra do driver ou do RSS. Python, objetos de metadados, bibliotecas,
cache do SO e resultados retidos pelo consumidor estão fora desse escopo.
`peak_vram_bytes` e `peak_rss_bytes` ficam nulos até haver instrumentação real.

O teste com arquivo packed maior que a arena prova leitura por blocos no CPU.
Ele não prova streaming PCIe, overlap, modelo completo ou uso de uma GPU de 512 MB.
