# Contexto TurboQuant exclusivo para MSE

M1.07 acrescenta `tq_create_mse(dim, bits, seed)`, que preserva a quantização MSE
existente sem reservar a matriz QJL quadrática, os codebooks Prod ou o buffer
persistente sem uso do contexto legado. O estado cresce linearmente com a dimensão
para uma precisão fixa. `tq_create` continua disponível com suporte MSE/Prod e a
mesma sequência aleatória, sem troca implícita de modo durante a execução.

Este incremento preparou a integração [TQ portátil no NexaPack](NEXAPACK_TQ_V1.md),
concluída em M1.06 com registros TQ02 e centroids persistidos. Um incremento
posterior acrescentou [TQ ao KV paginado](NEXALM_KV_TQ_CPU.md), incluindo contexto
e scratch no orçamento da sessão. O KV genérico da biblioteca padrão é outro componente.

## API, compatibilidade e ownership

```c
#include "turboquant.h"

tq_ctx *ctx = tq_create_mse(64, 3, 42);
if (!ctx) { /* parâmetros inválidos, overflow ou falta de memória */ }
/* Em caso de sucesso: */
size_t resident = tq_context_memory_bytes(ctx);
size_t packed_bytes = tq_packed_size(ctx, 1);
/* O chamador aloca entrada, saída packed e reconstrução com os tamanhos exigidos. */
/* tq_quantize_packed(ctx, input, packed, 1); */
/* tq_dequantize_packed(ctx, packed, reconstructed, 1); */
tq_destroy(ctx);
```

Dimensão deve ser potência de dois positiva; bits de 1 a 8. O construtor verifica
overflow da soma de alocações antes de reservar memória. O modo MSE considera
tamanhos lineares, sem rejeitar dimensões por causa de uma matriz QJL inexistente.
Isso não garante que uma dimensão válida caiba na memória disponível.

O chamador possui o contexto e deve destruí-lo uma vez, após todas as operações.
`tq_destroy(NULL)` é seguro; falhas de construção liberam todo estado parcial.
Não destruir o mesmo ponteiro duas vezes nem durante outra operação. Execução
usa o contexto como estado de leitura e temporários próprios; chamadas concorrentes
precisam de entradas/saídas válidas e saídas independentes.

No contexto MSE, `tq_prod_idx_packed_size` e `tq_prod_qjl_packed_size` retornam
zero. Quantização/desquantização Prod retornam `-2` antes de alocar ou escrever
saídas. Para Prod, crie um contexto legado com `tq_create` e bits de 2 a 8.
Não existe promoção automática para um contexto maior.

Packed MSE mantém os bytes TQ01: magic de quatro bytes, norma F32 no endian do
host e códigos compactados. Dimensão, bits e seed continuam sendo parâmetros
externos. M1.06 acrescenta APIs TQ02 com norma little-endian, scratch do chamador
e migração explícita; as APIs deste guia não reinterpretam registros existentes.

## Estado persistente e temporários

`tq_context_memory_bytes(ctx)` informa a soma dos bytes solicitados das alocações
persistentes pertencentes ao contexto, incluindo sua struct. Para `NULL`, retorna
zero. Não é RSS nem medição de VRAM; exclui overhead do allocator, stack, threads,
bibliotecas, entrada/saída do chamador e temporários das operações.

Com dimensão `D`, bits `b`, `L=2^b` e `S=sizeof(tq_ctx)`:

```text
MSE:    S + 4*D + 4*(2*L - 1)
legado: MSE + 4*D + 4*D*D + codebook Prod
codebook Prod: 4*(2*(L/2) - 1) para b>=2; zero para b=1
```

O campo privado usado para registrar o total faz parte de `S`; o tamanho da
struct depende da plataforma. Compare os modos na mesma compilação.

| Operação | Heap temporário adicional, além do contexto e buffers do chamador |
|---|---|
| Construção MSE | `4*(L+1)` durante Lloyd-Max; não permanece no contexto |
| Quantização serial, raw ou packed | `4*D` |
| Desquantização serial, raw ou packed | zero; usa o buffer de saída |
| Quantização paralela | até oito buffers de `4*D`; threads/stack à parte |
| Desquantização paralela | zero de heap próprio do codec; threads/stack à parte |
| Diagnóstico `tq_mse`, N vetores | `N*(8+ceil(D*b/8)) + 4*N*D + 4*D` no pico |
| Quantização Prod, contexto legado | `12*D` |
| Desquantização Prod, contexto legado | `8*D` |

Na construção, o temporário Lloyd-Max existe antes de todos os buffers persistentes
estarem prontos; não basta somá-lo ao total final para obter o pico real. Os testes
interceptam alocações para medir simultaneidade. `tq_mse` materializa o batch packed
e a reconstrução para diagnóstico; não é uma operação de streaming com scratch
constante. A quantização packed legada ainda aloca seu vetor de trabalho a cada
chamada; a nova API `tq_quantize_tq02` recebe scratch do chamador, sem heap interno.

## Uso pela linguagem e consumidores

- `compress::create` e `Quantizer::new`/`with_seed` preservam o modo legado.
  `compress::create_mse`, `Quantizer::new_mse`/`with_seed_mse` optam pelo estado menor.
- `CompressedBuffer::new` preserva o contrato anterior, incluindo acesso ao seu
  quantizador; `CompressedBuffer::new_mse` torna a opção explícita.
- `quick_compress`/`quick_decompress`, `std::kv_cache_quant::new_cache` e o wrapper
  TurboQuant do script TinyLlama usam apenas MSE e passam a criar contextos menores.
- O JIT resolve o novo símbolo na biblioteca compilada a partir do fonte.
  Quantização automática de GPU continua desabilitada.

Exemplo de criação explícita na linguagem:

```nxl
fn main() -> i32 {
    let ctx = compress::create_mse(64, 3, 42);
    if (cast::<i64>(ctx) == 0) { return 1; }
    compress::destroy(ctx);
    return 0;
}
```

## Comparação no macOS ARM64

Na mesma compilação, `sizeof(tq_ctx)=88`, bits=3 e seed=42:

| Dimensão | Estado legado | Estado MSE | Scratch de quantização serial |
|---|---:|---:|---:|
| 64 | 17.072 B | 404 B | 256 B |
| 1.024 | 4.202.672 B | 4.244 B | 4.096 B |
| 4.096 | 67.141.808 B | 16.532 B | 16.384 B |

São bytes persistentes retornados pela API. A instrumentação de alocações dos
testes confere os valores de D64/D1024; D4096 foi comparado pela API e pelos
resultados numéricos. Eles não representam toda a memória do processo. Em D4096, a redução do estado
é de aproximadamente 64 MiB para 16,1 KiB. Os buffers packed dos três vetores de
teste permaneceram iguais em tamanho e conteúdo; a reconstrução também coincidiu
byte a byte entre os dois modos. Isso comprova compatibilidade MSE, sem afirmar
que a quantização recupera os dados originais sem perda.

Relatório da comparação: `artifacts/reports/turboquant-mse-contexts.json`.
O relatório instrumentado `artifacts/reports/turboquant-mse-memory.json` separa
estado, construção e scratch. Para D64/D1024, os picos de construção MSE foram
412/4.252 B, contra os estados persistentes de 404/4.244 B. A diferença de oito
bytes decorre da duração do buffer temporário Lloyd-Max.
Para reproduzir a contabilidade pública após compilar o runtime:

```sh
python3 - <<'PY'
import ctypes as c
from runtime.build_runtime import build_runtime
lib = c.CDLL(str(build_runtime('turboquant')))
lib.tq_context_memory_bytes.argtypes = [c.c_void_p]
lib.tq_context_memory_bytes.restype = c.c_size_t
lib.tq_destroy.argtypes = [c.c_void_p]
lib.tq_destroy.restype = None
for dim in (64, 1024, 4096):
    for name in ('tq_create', 'tq_create_mse'):
        create = getattr(lib, name)
        create.argtypes = [c.c_int, c.c_int, c.c_int]
        create.restype = c.c_void_p
        ctx = create(dim, 3, 42)
        if not ctx:
            raise MemoryError(name)
        try:
            print(dim, name, lib.tq_context_memory_bytes(ctx))
        finally:
            lib.tq_destroy(ctx)
PY
```

## Validação e retomada

```sh
python3 -m unittest discover -s tests -p 'test_turboquant_mse_regressions.py' -v
python3 -m unittest discover -s tests -p 'test_mse_integration_regressions.py' -v
make -C runtime all test
artifacts/build/runtime/test_tq_mse_regressions memory > artifacts/reports/turboquant-mse-memory.json
python3 -S -m unittest discover -s tests -p 'test_turboquant_mse_regressions.py' -v
```

Os testes de memória distinguem residência, scratch e pico de construção, exercitam
falhas de alocação e verificam liberação antes de retry. Testes numéricos comparam
modos e preservam vetores Prod anteriores à mudança. A próxima etapa e a evidência
completa ficam no [checklist central](BLUEPRINT_512MB_CHECKLIST.md).
Validação local: 364 regressões e 110 testes bootstrap passaram; oito testes
nativos MSE também passaram sem site-packages. Cinco cenários novos de integração
foram executados tanto como binário nativo quanto pelo JIT. A matriz remota de
plataformas continua pendente.
