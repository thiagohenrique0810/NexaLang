# Execução Transformer CPU sobre pesos Q4

O pipeline já executa o forward Llama completo de uma sequência, com uma ou várias
camadas, a partir do [pacote importado](NEXALM_IMPORTACAO.md). O runtime usa kernels
C e não importa PyTorch. A prova atual usa modelos sintéticos pequenos; não há
checkpoint R0/v1 treinado, tokenizer executável ou backend GPU neste pipeline.

O runtime é compilado a partir do fonte e salvo em `artifacts/build/runtime/`.
No Windows, se uma DLL estiver em uso e sua substituição for negada, o builder
publica a nova versão com nome único e retorna esse caminho, preservando a sessão
anterior. Esse tratamento foi testado com bloqueio simulado; a matriz remota de
plataformas permanece pendente no checklist.

## Grafo e operações

`compiler/model_lowering.py::lower_model(config, sequence_length, weight_storage=...)`
produz um `ModelGraph` validado. A entrada é `tokens: U32[S]` e a saída é
`logits: F32[S,vocab]`. Pesos são constantes externas com nomes Hugging Face;
o head tied usa diretamente a constante do embedding, sem duplicá-la.

Cada camada executa RMSNorm, projeções Q/K/V, RoPE, atenção causal GQA, projeção O,
residual, RMSNorm, gate/up, SwiGLU, down e residual. Embedding, normalização final e
head completam o grafo: `15 * num_hidden_layers + 3` operações. MHA é o caso em que
o número de heads KV e query coincide; MQA usa um head KV.

```sh
python3 tools/nexa_model.py models/nexalm512/architecture.nxl --sequence-length 4 --out artifacts/reports/transformer-graphs.json
```

O comando acrescenta `model_ir` e `activation_requests` às definições estruturais.
Sem `--sequence-length`, a saída anterior é preservada. Compilar o grafo de R0/v1
não cria seus pesos nem executa esses modelos. O frontend continua separado de
`nxc`; migração do TurboIR legado e integração com a linguagem seguem pendentes.

## Execução e estado

`runtime.nexapack.transformer.TransformerSession` consome os operadores do grafo
em ordem, usando os offsets reais do plano. Projeções usam o GEMM Q4 existente,
com pesos lidos em tiles; embedding decodifica somente as linhas solicitadas.
Normas F32 são carregadas diretamente em um buffer planejado. Não existe uma
matriz completa de pesos desquantizados no executor.

- `prefill(ids)` substitui o histórico e retorna logits para todos os IDs.
- `decode(id)` acrescenta um ID e retorna seus logits, **recomputando o prefixo**.
- `reset()` limpa o histórico; `close()` encerra a sessão.
- IDs, contexto e orçamento são validados. Uma falha de leitura, checksum ou
  cálculo preserva o histórico e o relatório da última execução bem-sucedida.

Esse modo por recomputação permanece como baseline e é o padrão da CLI. A sessão
é síncrona, para uma sequência, e não deve ser compartilhada entre chamadas
concorrentes.

O modo opcional `--kv-cache` usa
`runtime.nexapack.incremental.IncrementalTransformerSession`: prefill substitui o
prompt, `append(ids)` acrescenta um chunk e decode calcula somente o novo ID,
consultando o KV anterior. Dois bancos F32 permitem substituir o prompt sem
copiar o cache antigo; append/decode escrevem apenas o sufixo ainda não confirmado.
O commit é por chamada, após sucesso de todas as camadas e produção do relatório.
Veja o [guia de KV incremental](NEXALM_KV_CPU.md) para o plano, as transações e a API.
O [modo paginado](NEXALM_KV_PAGINADO_CPU.md), selecionado por
`--kv-cache --kv-page-tokens N`, aloca páginas conforme o contexto cresce e preserva
o mesmo contrato de transações. A atenção lê diretamente as páginas F32.
Com `--kv-codec q4`, o [KV é comprimido por token/head](NEXALM_KV_Q4_CPU.md) e
a atenção lê os blocos Q4 diretamente, com erro de quantização medido separadamente.

Os kernels usam ativações F32 e acumulação double onde necessário. RoPE usa a
rotação entre as duas metades da cabeça e `rope_theta` do config. As posições são
`0..S-1` no baseline; o modo incremental acrescenta o offset do contexto anterior.
A atenção mapeia grupos query para heads KV e aplica máscara causal. Seu softmax
é estável, usa um float de scratch por posição visível do contexto e não
materializa scores `S*S`. SwiGLU evita
overflow da exponencial para entradas muito negativas. Os kernels verificam
capacidades, dimensões, aliasing de saídas e valores não finitos; não alocam heap.

## Memória e relatórios

Os lifetimes são derivados da ordem do grafo: input existe no início; um resultado
nasce no evento do produtor e permanece até sua última leitura. Operandos e
resultado coexistem durante uma operação. Residuais ficam vivos até o Add, saídas
até o consumidor final; buffers mortos podem ser reutilizados.

No baseline, o plano inclui ativações, IDs de entrada U32, tile packed, tile de
resultado, norma F32 e scratch de atenção. Acrescenta até 63 bytes de alinhamento da arena,
64 KiB de scratch do reader e a reserva opcional do usuário. A capacidade máxima
de contexto é verificada antes de carregar pesos, biblioteca nativa ou arena; cada
chamada também valida seu próprio plano antes da carga de payloads.

Com `--kv-cache`, o orçamento também inclui a alocação completa dos dois bancos
KV F32 e seu alinhamento. `--prefill-chunk-size` limita ativações e logits
temporários por chunk, enquanto o cache reserva a capacidade total de contexto.
O preflight combina o maior chunk permitido, scratch do contexto máximo e ambos
os bancos antes de alocar KV ou carregar pesos. A divisão do prompt na CLI usa
um prefill seguido de chamadas append; cada chamada é uma transação separada.

Esse teto é de **buffers CPU gerenciados**, não de RSS ou VRAM. Objetos Python,
metadados, assets, bibliotecas, cache do SO, listas de logits retornadas e a
referência opcional ficam fora dele. `peak_rss_bytes` e `peak_vram_bytes` continuam
nulos. O grafo, plano, bytes lidos, tiles e tempos de cada chamada constam no JSON.
`run_totals` resume as chamadas da CLI; os tempos excluem planejamento, compilação
da biblioteca e referência e não devem ser interpretados como TTFT completo.

## Reprodução offline

Se a fixture da etapa de importação já existir, comece pelo terceiro comando.
Os dois primeiros exigem pastas de destino novas.

```sh
python3 tests/model_checkpoint_fixture.py --out artifacts/checkpoints/nexalm-tiny
python3 tools/nexa_convert.py --checkpoint artifacts/checkpoints/nexalm-tiny --out artifacts/models/nexalm-tiny --group-size 4 --block-rows 3
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-forward.json

# Geração determinística de IDs; não faz tokenização/detokenização.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --generate 2 --memory-budget 96KiB

# KV incremental opcional, com prefill dividido em chunks de até dois IDs.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --prefill-chunk-size 2 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-chunks.json
```

`--eos-token ID` permite parar a geração quando esse ID for escolhido.
`--include-logits` inclui os valores no relatório. A CLI dimensiona o contexto para
prompt + IDs solicitados; `--max-sequence-length` permite declarar uma capacidade
maior. Relatórios devem ficar fora das pastas de modelo e checkpoint.
`--prefill-chunk-size` exige `--kv-cache` e um tamanho positivo. As métricas do
modo incremental e os campos de cache estão no [guia correspondente](NEXALM_KV_CPU.md).

## Validação numérica

O oracle em `tests/transformer_reference.py` implementa as equações Llama em PyTorch,
independentemente dos kernels C. Sua versão incremental mantém KV real e compara
tanto o decode nativo por recomputação quanto o executor com cache. Somente essa
verificação opcional requer PyTorch; o limite do diagnóstico é de 2 milhões de
parâmetros e 256 posições.
Ela materializa os pesos e usa memória fora do orçamento nativo.

```sh
# Requer PyTorch já disponível para o diagnóstico opcional.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --tile-rows 3 --memory-budget 96KiB --verify --reference-checkpoint artifacts/checkpoints/nexalm-tiny --report artifacts/reports/nexalm-forward.json
```

O relatório separa erro de execução (C vs referência com os mesmos pesos Q4) de
erro de quantização (referência Q4 vs pesos Safetensors originais). A tolerância
de execução é `abs(error) <= 1e-5 + 1e-4 * abs(reference)`; qualidade/perplexidade
exigem outro gate e não são deduzidas desse teste.

No **baseline por recomputação**, a fixture não treinada de 728 parâmetros,
IDs `[1,3,5,7]`, teve os seguintes resultados locais no macOS ARM64:

| Medida | Resultado |
|---|---:|
| Operações por forward | 18 |
| Erro absoluto máximo C vs referência Q4 | 0 |
| Erro absoluto máximo Q4 vs pesos originais | 0,4435420930 |
| Arena, incluindo padding | 1.343 B |
| Arena + scratch do reader | 66.879 B |
| Orçamento declarado | 98.304 B (96 KiB) |
| Scratch de atenção no prefixo de 4 IDs | 16 B |

Esses valores descrevem a fixture, sem concluir qualidade de um modelo treinado.
Há também testes de duas camadas, MHA/GQA/MQA, tied/untied, causalidade, posições
RoPE maiores que zero, corrupção e retomada, ASan/UBSan e proibição de heap nos
kernels. `tests/fixtures/transformer_tiny_golden.json` conserva logits produzidos
exclusivamente pelo oracle PyTorch, hashes e tolerâncias para validar a CI sem Torch.

Os resultados do KV F32 incremental ficam no [guia específico](NEXALM_KV_CPU.md),
separados dessa medição histórica. Demais codecs/tiers KV, GPU e qualidade de um
checkpoint treinado permanecem nos gates do [checklist](BLUEPRINT_512MB_CHECKLIST.md).

O [plano Nexa Omni](NexaLang_Plano_Implementacao_Nexa_Omni.pdf) propõe uma camada
futura de orquestração multi-modelo. Seus [ajustes de integração](NEXA_OMNI_AJUSTES.md)
definem dependências e critérios próprios; executar o cache de uma sessão não
conclui os gates de Packs, roteamento ou compartilhamento de KV entre modelos.
