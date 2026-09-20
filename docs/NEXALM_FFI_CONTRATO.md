# Contrato de FFI: `qint<N>` e `PackedVector<N>`

Este guia descreve **somente o que o compilador implementa hoje**. Tudo que
aparece aqui foi exercitado por `tests/test_qint_regressions.py` nos dois modos
de execução (nativo e `--jit`). O que não está aqui não existe na linguagem.

## O que os dois tipos são

`qint<N>` é **um código de nível armazenado**. `PackedVector<N>` é **um buffer
de códigos empacotados**. Nenhum dos dois é um número de ponto flutuante nem um
tensor: são tipos de armazenamento/ABI, e a única coisa que garantem é o
*layout* dos bytes.

`N` só pode ser `2`, `3`, `4` ou `8`. Essa lista não é uma escolha de gosto: é
exatamente o conjunto de larguras que têm um codec agrupado escrito em
`runtime/nexapack/format.py` (`Q2_GROUPED`, `Q3_GROUPED`, `Q4_GROUPED`,
`Q8_GROUPED`, todos versão 1). Uma largura sem codec atrás seria uma promessa
que o compilador não pode cumprir, então ela é **recusada com nome**:

```text
Error [E0008]: qint<5> has no packed codec; supported widths are 2, 3, 4, 8
Error [E0008]: qint<0> has no packed codec; supported widths are 2, 3, 4, 8
Error [E0008]: PackedVector<16> has no packed codec; supported widths are 2, 3, 4, 8
```

## ABI

| Tipo | Representação LLVM | Significado |
| --- | --- | --- |
| `qint<N>` | `i8` | um código em complemento de dois, sinalizado |
| `PackedVector<N>` | `i8*` | ponteiro para o primeiro byte do payload empacotado |

Todas as larguras suportadas cabem num byte, então a forma **desempacotada** de
um código é um byte com sinal qualquer que seja `N`. A forma de `N` bits só
existe dentro de um `PackedVector<N>`.

`qint<N>` carrega o intervalo do codec, não o do byte. O código reservado
`-2**(N-1)` nunca é um valor válido:

| `N` | códigos válidos | reservado | codec |
| ---: | :---: | :---: | --- |
| 2 | -1 … 1 | -2 | `Q2_GROUPED` |
| 3 | -3 … 3 | -4 | `Q3_GROUPED` |
| 4 | -7 … 7 | -8 | `Q4_GROUPED` |
| 8 | -127 … 127 | -128 | `Q8_GROUPED` |

Um literal inteiro pode nomear um código diretamente; fora do intervalo é erro
de compilação:

```nexa
let baixo: qint<4> = -7;   // ok
let alto:  qint<4> = 8;    // Error [E0008]: 8 is not a qint<4> code; the range is [-7, 7] and -8 is reserved
let reserv: qint<4> = -8;  // Error [E0008]: -8 is not a qint<4> code; ... and -8 is reserved
```

Não há coerção implícita. Para ler um código como inteiro use `cast::<i32>`, que
faz **extensão de sinal** (diferente de `u8`, que é estendido com zero). O
caminho inverso, `cast::<qint<N>>(algum_i32)`, **trunca e não verifica o
intervalo** — é um cast explícito, com a mesma semântica de qualquer
estreitamento de inteiro no bootstrap.

## Layout de `PackedVector<N>`

Um vetor empacotado é uma sequência de registros de grupo. Cada registro é:

```text
4 bytes : escala float32 little-endian
ceil(N * group_size / 8) bytes : group_size códigos de N bits,
                                 empacotados a partir do bit menos significativo
```

- `escala = float32(max(abs(grupo)) / (2**(N-1) - 1))`.
- Quantizar divide pela escala persistida, arredonda empates **afastando de
  zero** e limita a `[-L, L]`.
- Grupo inteiramente zero usa escala zero e payload zero.
- O último grupo codifica só os valores que existem: as posições de código
  restantes e os bits altos sobrando do último byte ficam **zero**.
- `bytes = ceil(count / group_size) * (4 + ceil(N * group_size / 8))`.

Isto é byte a byte o que `runtime/nexapack/format.py` escreve. Não é um formato
paralelo: é o mesmo, com um kernel só, parametrizado por `N`.

## Superfície da linguagem

Seis intrínsecos, no namespace `qpack`. `N` **nunca** vem de um valor: vem do
tipo `PackedVector<N>` do primeiro argumento, ou de um turbofish. Um buffer não
pode ser medido numa largura e lido em outra.

| Chamada | Retorno | O que faz |
| --- | --- | --- |
| `qpack::size::<N>(count, group_size)` | `i64` | bytes exigidos pelo layout; `0` se os argumentos forem inválidos |
| `qpack::groups::<N>(count, group_size)` | `i64` | número de registros de grupo; `0` se inválido |
| `qpack::pack(packed, values, count, group_size)` | `i32` | quantiza `count` floats em `packed`; `0` = ok |
| `qpack::unpack(packed, output, count, group_size)` | `i32` | decodifica para `count` floats; `0` = ok |
| `qpack::code(packed, count, group_size, index)` | `qint<N>` | um código armazenado, sem decodificar o resto |
| `qpack::scale(packed, count, group_size, group)` | `f32` | a escala de um grupo |

`packed` é sempre `PackedVector<N>`; qualquer outro tipo é recusado:

```text
Error [E0002]: qpack::pack expects a PackedVector<N>, got '*u8'
Error [E0008]: qpack::size needs an explicit width, as qpack::size::<4>(...)
```

Status negativos vêm de `enum nexa_q4_status` (`-1` argumento inválido, `-2`
buffer pequeno, `-3` overflow, `-4` dado inválido, `-5` faixa numérica).

`qpack::code` e `qpack::scale` não têm canal de status separado. Uma leitura que
falha devolve um valor que o próprio contrato de armazenamento chama de
impossível: o **código reservado** `-2**(N-1)` para `code`, e uma **escala
negativa** (`-1.0`) para `scale`. Isso é detectável no chamador e não inventa um
terceiro mecanismo de erro.

### Exemplo completo

```nexa
extern "C" {
    fn malloc(size: i32) -> *u8;
    fn free(ptr: *u8);
}

fn main() -> i32 {
    let count = 33;
    let group = 16;
    let bytes = cast::<i32>(qpack::size::<4>(count, group));
    let packed = cast::<PackedVector<4>>(malloc(bytes));
    let values = cast::<*f32>(malloc(count * 4));
    let back = cast::<*f32>(malloc(count * 4));
    for i in 0..count { values[i] = cast::<f32>(i - 16) * 0.25; }
    assert!(qpack::pack(packed, values, count, group) == 0, "pack");
    assert!(qpack::unpack(packed, back, count, group) == 0, "unpack");
    let primeiro = cast::<i32>(qpack::code(packed, count, group, 0));
    free(cast::<*u8>(packed)); free(cast::<*u8>(values)); free(cast::<*u8>(back));
    return primeiro;
}
```

Um `PackedVector<N>` nasce de um `cast` sobre um ponteiro que o chamador
alocou. **O compilador não conhece a capacidade desse ponteiro.** Ele passa aos
kernels exatamente o tamanho que `qpack::size::<N>` exige, então a verificação
de buffer pequeno do kernel nunca é o que pega um buffer curto vindo da
linguagem: alocar menos que `qpack::size::<N>(count, group_size)` é corrupção de
memória, como qualquer ponteiro cru neste bootstrap.

## Símbolos C

Os intrínsecos descem para `runtime/nexapack/q4.c`, declarados em `q4.h`:

```c
size_t nexa_qpack_size  (size_t bits, size_t count, size_t group_size);
size_t nexa_qpack_groups(size_t bits, size_t count, size_t group_size);
int nexa_qpack_pack  (size_t bits, const float *values, size_t value_count,
                      size_t count, size_t group_size,
                      uint8_t *packed, size_t packed_bytes);
int nexa_qpack_unpack(size_t bits, const uint8_t *packed, size_t packed_bytes,
                      size_t count, size_t group_size,
                      float *output, size_t output_count);
int nexa_qpack_code  (size_t bits, const uint8_t *packed, size_t packed_bytes,
                      size_t count, size_t group_size, size_t index, int8_t *code);
int nexa_qpack_scale (size_t bits, const uint8_t *packed, size_t packed_bytes,
                      size_t count, size_t group_size, size_t group, float *scale);
```

`nexa_qpack_unpack` despacha para os decodificadores por codec que já existiam
(`nexa_q2_decode_row`, `nexa_q3_decode_row`, `nexa_q4_decode_row`,
`nexa_q8_decode_row`), para que um `unpack` da linguagem e um decode do runtime
não possam divergir em regra de validação. `nexa_qpack_pack` é um empacotador
genérico: uma regressão verifica que, com `bits = 4`, ele produz exatamente os
mesmos bytes que o `nexa_q4_quantize` que já existia.

Ligação: `nx.py` compila `runtime/nexapack/q4.c` junto quando o IR chama
qualquer `nexa_qpack_*`; em `--jit`, `bootstrap/jit.py` carrega a biblioteca
`nexa_q4` construída por `runtime/build_runtime.py`. Nenhum binário pré-compilado
entra nessa cadeia.

## `extern` e ABI

O backend emite **uma** convenção de chamada: a ABI C da plataforma. Um bloco
`extern` só declara assinaturas e nunca emite corpo, então uma ABI que o backend
não implementa produzia, até este incremento, um programa que compilava e
executava com status 0 contra a convenção errada — o `visit_ExternBlock` era um
`pass` tanto no analisador semântico quanto no gerador de código. Agora:

```text
Error [E0009]: Unsupported extern ABI: "Fortran-77"
  = the backend emits "C" only
```

A comparação é exata e sensível a maiúsculas: `extern "c"` também é recusado.
`tests/test_qint_regressions.py::test_extern_abi_the_backend_cannot_emit_is_refused`
fixa esse comportamento; era um teste vermelho antes da correção.

## Custo medido

Bits por valor **lidos do buffer que o empacotador da linguagem realmente
escreveu** (`tests/test_qint_regressions.py::test_bits_per_value_measured_from_the_emitted_buffer`),
para contagens que preenchem grupos inteiros:

| `group_size` | `qint<2>` | `qint<3>` | `qint<4>` | `qint<8>` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 40,0000 | 40,0000 | 40,0000 | 40,0000 |
| 7 | 6,8571 | 8,0000 | 9,1429 | 12,5714 |
| 8 | 6,0000 | 7,0000 | 8,0000 | 12,0000 |
| 16 | 4,0000 | 5,0000 | 6,0000 | 10,0000 |
| 32 | 3,0000 | 4,0000 | 5,0000 | 9,0000 |
| 33 | 3,1515 | 4,1212 | 5,0909 | 8,9697 |

A escala de quatro bytes é todo o overhead, e ele é **dividido pelo grupo**:
`32 / group_size` bits por valor sempre que os códigos preenchem bytes inteiros.
Em grupo 32 isso é **exatamente 1 bit extra** por valor (`N + 1`: 3, 4, 5 e 9) e
em grupo 8 são **exatamente 4 bits extras** (`N + 4`: 6, 7, 8 e 12).

Quando `N * group_size` não é múltiplo de 8, o arredondamento do payload para
bytes inteiros cobra mais que isso: em `group_size = 7`, `qint<2>` custa 6,8571
bits por valor, não 2 + 32/7 = 6,5714. Em `group_size = 1`, todas as larguras
custam os mesmos 40 bits por valor — quatro bytes de escala mais um byte de
payload, qualquer que seja `N`. `qint<2>` com grupo 1 é **vinte vezes** pior que
`f32`; a largura só significa alguma coisa com grupos grandes.

**Caudas não são grátis.** Um grupo parcial ocupa um registro inteiro:

| Caso | bytes | bits/valor |
| --- | ---: | ---: |
| `qint<4>`, grupo 32, 33 valores | 40 | 9,6970 |
| `qint<4>`, grupo 32, 32 valores | 20 | 5,0000 |
| `qint<2>`, grupo 33, 64 valores | 26 | 3,2500 |
| `qint<8>`, grupo 17, 33 valores | 42 | 10,1818 |

Um valor a mais que o grupo quase dobra o custo por valor. Quem dimensiona um
tensor com esses tipos escolhe `count` múltiplo de `group_size` ou paga por isso.

## Limites declarados

- **Só armazenamento.** Não há aritmética sobre `qint<N>`, nem matmul, nem
  conversão de/para tensores ou bundles NexaPack pela linguagem. `pack` e
  `unpack` operam num vetor de floats do chamador, não num arquivo `.nxp`.
- **Sem contêiner.** A linguagem escreve/lê o *payload* de linha. O cabeçalho de
  64 bytes, o índice JSON, os blocos e os checksums SHA-256 do contêiner
  NexaPack continuam só em `runtime/nexapack/format.py`.
- **Uma linha.** `count` é uma contagem de valores, não uma matriz. Não existe
  `rows` na superfície da linguagem.
- **Sem verificação de capacidade.** Ver acima: o tipo não carrega o tamanho do
  buffer.
- **`cast::<qint<N>>` não valida o código.** Só literais são verificados.
- **Sem `qint<N>` em struct, array ou retorno agregado** além do que a
  representação `i8`/`i8*` dá de graça: nada foi testado nessa direção.
- **`size_t` é `i64`.** As declarações LLVM assumem host de 64 bits, que é o que
  este bootstrap constrói.
- **Larguras inválidas param na semântica**, não no parser: `qint<5>` já fazia
  parse antes deste incremento e morria mais tarde como erro de tipo genérico.
- O limite de `group_size` do codec (1.048.576) vive nos kernels, não no tipo.
