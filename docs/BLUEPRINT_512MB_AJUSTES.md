# Ajustes de engenharia ao Blueprint 512 MB

Data: 2026-09-19. Referência: `NexaLang_Blueprint_Tecnico_512MB.pdf`, versão 0.1.
Este complemento registra as decisões de implementação sobre o código atual.
O PDF permanece como fonte original; o checklist é o registro vivo de execução.
O [índice de implementação](PLANOS_IMPLEMENTACAO_INDICE.md) mapeia os seis PDFs
locais para módulos, páginas/seções, dependências e guias de ajustes; deve ser
consultado ao iniciar cada item do checklist.

## Estado real da base

- `bootstrap/` compila a linguagem; a MIR emitida ainda não é a entrada do codegen
  LLVM de produção. Passes novos precisam de uma rota explícita até a execução.
- TurboIR em `examples/projeto-llm` representa configurações. ModelIR com operações,
  tensores físicos e dependências é um componente novo, em `compiler/`.
- O protótipo PyTorch reconstrói pesos INT4/INT8 em float e usa o KV interno do
  Hugging Face. Ele não demonstra armazenamento packed nem atenção NexaKV.
- O runtime legado deve rejeitar um orçamento que não pode impor antes de carregar
  pesos. Preservar `memory_limit` no JSON não equivale a implementar o limite.
- SPIR-V/OpenCL é experimental. A quantização automática de GPU está desabilitada
  até preservar escalas e resultados. A referência histórica à RX 580 não valida
  os novos kernels ou backends.

## Ordem dos primeiros marcos

1. Baseline, contabilidade explícita e planos rejeitados quando excedem o orçamento.
2. ModelIR/Tensor mínimos, MemoryPlan e formato packed especificado em conjunto.
3. Q4 físico e GEMV/GEMM CPU que consomem grupos compactados diretamente.
4. Uma camada Transformer e um backend GPU com memória medida e testes numéricos.
5. Streaming, KV integrado e modelo fixo 250–500M; depois o marco ~1B/512 MB.
6. Demais codecs/backends, precisão adaptativa, fusão, autotuning e modelos próprios.

O MemoryPlan mínimo é antecipado da fase 4 do PDF. O objetivo de 1B é um marco de
integração. Uma biblioteca de descritores ou um microbenchmark não conclui esse marco.

## Orçamento e medições

- `MB` significa 1.000.000 bytes; `MiB`, 1.048.576 bytes. A CLI deve exigir unidades
  explícitas em texto e guardar bytes inteiros nos relatórios.
- A tabela da página 15 soma 284 MB usando mínimos e 544 MB usando máximos com
  reserva de 64 MB. As escolhas devem satisfazer uma única restrição conjunta.
- Separar payload de pesos, entrada/saída, KV, staging, workspace, alinhamento,
  reserva, RAM e memória de dispositivo. Metadados e overhead do host são reais.
- Um orçamento de buffers gerenciados pela aplicação não limita o RSS do processo,
  o cache do SO ou o consumo do driver. Relatórios devem identificar o escopo.
- Picos do allocator, estimativas do planner, RSS e telemetria do driver devem ter
  campos distintos. Campo desconhecido fica nulo, nunca recebe uma estimativa
  apresentada como medição.
- 1B pesos ideais Q4 são 500 MB (476,84 MiB), antes de escalas, cabeçalhos e KV.
  Quantização não remove o custo de transferência quando os pesos são streamados.
- Limitar buffers em uma GPU maior não comprova o desempenho de uma placa física
  de 512 MB. Cada backend precisa registrar dispositivo, driver e capacidades.

## Formatos, importação e compatibilidade

- NexaPack é o contêiner versionado de tensores. O primeiro arquivo `.nxp` contém
  uma matriz Q4. O pacote executável `.nxb` com plano/kernels fica para outro marco.
- Q4 V1 usa grupos por linha, escala float32 little-endian e nibbles assinados
  de -7 a 7. Grupos finais têm padding zero. Isso é diferente do codec SRHT TQ01.
- TQ01 preserva a norma e custa `8 + ceil(dim * bits / 8)` bytes por vetor, com
  norma host-endian. [M1.06 integra TQ portátil](NEXAPACK_TQ_V1.md) como codec
  TQ_MSE_SRHT V1 do contêiner: registros TQ02 com norma F32 little-endian,
  dimensão/bits/seed/SRHT explícitos e centroids F32LE persistidos. Migração exige
  endian e codebook originais; não infere nem requantiza. APIs TQ01 permanecem iguais.
- [M1.07 acrescenta contexto exclusivo MSE](TURBOQUANT_MSE_CPU.md), sem matriz
  quadrática QJL, codebook Prod ou buffer persistente sem uso. `tq_create` continua
  legado; `tq_create_mse` é explícito e mantém os resultados MSE. Prod em contexto
  MSE falha sem promover/alocar estado. Bytes persistentes e scratch são separados.
- Safetensors armazena tensores; o importador também precisa da configuração e do
  mapeamento da arquitetura. GGUF/ONNX exigem adaptadores próprios e validação de ops.
  Fonte: https://huggingface.co/docs/safetensors/index
- Começar com uma família de arquitetura e um backend comprovados. Não declarar
  suporte aos demais apenas porque existe um campo enum ou um diretório.

## Complemento do plano da primeira LLM

O [novo plano](NexaLang_Plano_Implementacao_Primeira_LLM.pdf) fixa uma família
concreta para a importação inicial: NexaLM R0/v1, compatível com Llama sem biases,
RMSNorm, GQA, RoPE sem scaling e SwiGLU. R0 tem 125.854.464 parâmetros e v1 tem
394.331.136; pesos compartilhados são contados uma vez. Esses alvos complementam
as provas de escala do blueprint, sem antecipar sua validação em GPU.

- `models/nexalm512/architecture.nxl` define os dois contratos. O frontend separado
  reutiliza o lexer, valida configuração/shapes e gera operadores Transformer no
  ModelIR com `--sequence-length`. Migração do TurboIR e integração ao `nxc`
  permanecem abertas em G0.
- `NexaModelBundle` V1 reúne matrizes `.nxp` e normas F32, com aliases, assets e
  proveniência. A leitura V1 anterior continua válida. O executável `.nxb` é futuro.
- O importador local aceita Safetensors F32/F16/BF16, arquivo único ou shards,
  valida a família declarada e preserva tokenizer como asset. Isso não implementa
  tokenização nem inferência completa; detalhes em [NexaLM/importação](NEXALM_IMPORTACAO.md).
- As primeiras provas usam pesos sintéticos pequenos. Não se baixa nem treina
  R0/v1 antes de validar o forward e os logits contra uma referência reproduzível.
  O treinamento de referência poderá usar PyTorch; o runtime final segue independente.
- O terceiro incremento executa o grafo CPU com kernels C, pesos Q4 por tiles e
  normas F32. Lifetimes seriais vêm dos usos do grafo; offsets do plano são usados
  pelo executor. A prova numérica cobre fixtures pequenas e não encerra G1 de
  qualidade/corpus de um checkpoint treinado.
- O baseline de [execução CPU](NEXALM_EXECUCAO_CPU.md) recomputa o prefixo.
  O quarto incremento acrescenta [KV F32 incremental](NEXALM_KV_CPU.md) opcional,
  com RoPE posicional, prefill em chunks e plano explícito de escrita/atenção/commit.
  Dois bancos permitem substituir o prompt sem copiar o prefixo anterior; ambos
  entram integralmente no orçamento. Append escreve apenas o sufixo não confirmado.
  Reset e falhas tardias preservam atomicidade; exceções retidas não acumulam
  buffers transitórios na tentativa seguinte.
- O preflight fixa offsets para a capacidade máxima. Chamadas menores reduzem
  tamanhos e preservam esses offsets/lifetimes: um novo first-fit poderia aumentar
  fragmentação mesmo com menos tokens. O plano real continua sendo validado e seu
  extent contabilizado, garantindo a capacidade aceita pelo orçamento inicial.
- O orçamento continua limitado aos buffers CPU explícitos, excluindo Python,
  logits retidos pelo consumidor e PyTorch usado como referência opcional. O KV
  contíguo não é comprimido. Em contexto 2k, os dois bancos F32 reservam
  128 MiB para R0 e 256 MiB para v1, antes dos demais buffers. Não é medição de VRAM.
- M4.01 acrescenta [páginas F32 sob demanda](NEXALM_KV_PAGINADO_CPU.md). Uma página
  agrupa K/V de todas as camadas, com buffers e endereços de heads estáveis. Atenção
  usa tabelas de ponteiros na arena, sem concatenar o prefixo. Append preserva páginas
  existentes; substituição de prompt prepara páginas novas antes de liberar as antigas.
- Reserva de admissão e residência são distintas: reservar `ceil(C/P) + ceil(T/P)`
  páginas garante contexto cheio mais um prefill de até T tokens. Apenas páginas
  usadas recebem alocação. Relatórios incluem o pico da transação, padding por página
  e tabelas; páginas menores não garantem menor custo total. Reset/close liberam páginas,
  e falhas liberam as páginas novas mesmo com traceback retido. Demais codecs, evicção,
  múltiplas sequências e backend GPU continuam em gates futuros.
- O [primeiro codec KV comprimido é Q4](NEXALM_KV_Q4_CPU.md), com grupos por
  token/head reutilizando Q4_GROUPED V1. K é quantizado após RoPE e V após projeção.
  Mesmo páginas parciais já são Q4; tokens novos não alteram escalas ou códigos
  confirmados. A atenção reconstrói valores escalares em double durante a redução,
  sem expandir uma página/head/prefixo em outro buffer.
- Os relatórios Q4 distinguem erro do kernel, erro dos pesos, erro do KV e erro
  combinado. `verified` aprova somente equivalência à referência que usa o mesmo
  codec, não qualidade/perplexidade. Escalas, padding e staging permanecem no
  orçamento. Grupos/cabeças pequenos não garantem ganho físico; TQ01 e tiers
  seguem pendentes. O gate M4 completo exige evidências além desse baseline CPU.
- O [sétimo incremento acrescenta KV Q3](NEXALM_KV_Q3_CPU.md), com identidade
  Q3_GROUPED V1, escala F32 LE por grupo e códigos assinados de três bits LSB-first.
  O código -4 é reservado, caudas/bits sem uso ficam zero. Atenção direta, páginas
  parciais, rollback e reserva preservam os contratos F32/Q4; os layouts JSON
  anteriores não mudam. Q3 aplica-se ao KV, sem alterar os pesos Q4 do bundle.
- Na comparação CPU D64/P16/G32 sob a mesma capacidade, Q3 usa 64 B/token K/V
  contra 80 B Q4 e 512 B F32; uma página física ocupa 1.087/1.343/8.255 B.
  Os três kernels coincidem com seus oracles, mas Q3 introduz maior erro de KV
  nessa fixture sintética. Nenhuma conclusão de qualidade, velocidade ou VRAM.
- O oitavo incremento conclui M1.07 com estado linear MSE, API de contabilidade,
  falhas de alocação/liberação verificadas e preservação dos resultados MSE/Prod.
  Consumidores exclusivamente MSE usam o novo construtor; APIs que expõem Prod
  preservam o construtor legado e oferecem uma opção MSE explícita.
- O nono incremento conclui M1.06: TQ_MSE_SRHT V1 guarda codebook explícito,
  identidade SRHT_XOSHIRO256SS_V1 e registros TQ02. Seed sozinho não fixa centroids
  de Lloyd-Max/libm; importação usa os floats persistidos sem executar esse cálculo.
  Portabilidade dos bytes não promete aritmética idêntica em todo hardware.
  Escrita por vetor, checksums por bloco, publicação atômica e inspeção sem C
  preservam Q4. Bundles e executores rejeitam TQ onde falta kernel.
- Kernels TQ02 usam scratch fornecido pelo chamador e não alocam heap. O wrapper
  serializa chamadas e close, mantém parâmetros públicos imutáveis e verifica o
  orçamento antes do contexto/payload; erro libera buffers mesmo com traceback
  retido. A conversão inclui staging de leitura no relatório, sem confundir
  buffers explícitos com RSS/VRAM. Estado, scratch e reconstrução por vetor
  precisam entrar no plano de M4.02c/M4.03c; atenção TQ, pesos TQ e GPU continuam
  pendentes. A migração de bytes big-endian foi simulada, não executada nesse hardware.

- O décimo incremento integra [KV TQ paginado e atenção CPU](NEXALM_KV_TQ_CPU.md).
  TQ02 é aplicado por token/head, K após RoPE e V após projeção, com um contexto
  MSE/codebook compartilhado na sessão. Dimensão, bits, seed e centroids ficam no
  plano; o codebook gerado é concretizado após admissão e antes da primeira página.
  Formatos F32/Q4/Q3 permanecem iguais. Pesos TQ continuam sem kernel de matriz.
- Atenção TQ reconstrói um head por vez em scratch F32 de 4D bytes e acumula V
  em double com 8D bytes. Esse scratch é compartilhado temporalmente com encode;
  não existe expansão do prefixo. O kernel não aloca heap. Scores F32 do prefixo,
  arena, alinhamento, páginas, contexto e staging de construção entram na admissão.
- A pré-admissão reserva 128 B para a struct, protegidos por assert C, mais sinais
  e codebook/boundaries; antes da construção a API valida o tamanho nativo exato.
  Estado real e reserva são campos separados. Construção lazy ocorre antes de
  páginas/arena; primeira falha desfaz contexto e plano, demais falhas preservam
  o estado confirmado. Reset libera páginas e conserva contexto; close libera ambos.
- Em D64/P16, TQ3 e Q3_GROUPED ocupam 64 B/token K/V e 1.087 B por página,
  mas o pico TQ é maior: 73.618 B contra 72.446 B Q3 pelo contexto/scratch.
  Ambos coincidem com seus oracles; erros de KV são separados do erro do kernel.
  Heads pequenos podem ter maior erro e alinhamento proporcionalmente caro.
  Nenhum ganho geral de qualidade, velocidade ou VRAM é inferido dessa fixture.
- M4.02c/M4.03c passam a identificar a parte TQ implementada; os novos subitens
  M4.02d/M4.03d mantêm tiers e demais codecs pendentes. O gate M4 completo,
  modelo treinado, orquestração Omni e GPU continuam abertos.

- O décimo primeiro incremento implementa [tiers CPU por idade de página](NEXALM_KV_TIERS_CPU.md):
  hot F32, warm Q4 e cold Q3, com política fixa por sessão e identidade de layout
  por página. Todos permanecem residentes na RAM. Atenção mista lê valores dos
  três codecs diretamente; somente páginas completas migram. A última parcial
  continua hot, sem descartar tokens exigidos pela atenção causal.
- A migração ocorre depois de todas as camadas/logits de cada chamada e antes
  do commit. O histórico de chunks afeta os bytes/erros: chamadas grandes podem
  saltar de F32 para Q3; Q4→Q3 recodifica a reconstrução F32 do Q4 atual, sem
  guardar o original. O oracle repete os chunks e as migrações nessa ordem.
- Destinos são alocados separadamente e todas as origens sobrevivem até a
  publicação atômica. Falhas na migração, alocação ou relatório descartam páginas
  novas e buffers mesmo com traceback retido. Reset/close liberam páginas mistas.
  A política não pode ser alterada depois da admissão.
- O orçamento soma capacidade canônica, novas páginas F32, reserva conservadora
  de substitutas, workspace/tabelas/leitura e scratch de migração de 4D+24 B.
  O pico reportado distingue atenção e migração: a arena de atenção é liberada
  antes da recodificação. Estatísticas de migração medem erro local entre origem
  e destino; logits/erro acumulado são avaliados pelo oracle separadamente.
- Na fixture D64/P2/H1/W1/G32, oito tokens terminaram com 1.788 B de páginas,
  contra 4.348 B F32; pico gerenciado de 73.980 B contra 75.323 B F32. O oracle
  coincide com o kernel e mede erro de KV de 0,3118940145 nos logits. Isso não
  demonstra superioridade geral de qualidade/velocidade nem memória GPU.
  Promoção, critérios de qualidade, evicção com backing store e TQ misto seguem
  abertos em M4.05b/M4.03d; este incremento não remove KV causal necessário.

## Backing store de KV CPU

O décimo segundo incremento implementa [evicção/recarga cold Q3](NEXALM_KV_BACKING_CPU.md)
como opção da política por idade. Hot/warm permanecem em RAM; cold usa arquivos
privados de sessão, header/metadata versionados e checksums. Payload inclui K/V e
padding de todas as camadas, sem requantização na recarga. Não é checkpoint
retomável nem cache compartilhável entre modelos/sessões.

- O plano lógico de codecs continua independente da residência. A admissão física
  reserva fontes hot/warm, novas páginas do chunk, destinos de migração, um slot
  Q3, scratch, metadata de I/O e arena. O número de páginas cold não aumenta a
  reserva de payload residente; metadados Python e cache do SO continuam fora.
- Atenção nativa usa duas passagens por camada e estado double por query/head.
  Mantém a aritmética e a ordem de redução do baseline misto, com leituras diretas
  de F32/Q4/Q3. Não armazena scores nem reconstrução do prefixo. Slot e arena
  são liberados antes da migração/persistência.
- A publicação de arquivos antecede o commit da sequência. Falhas anteriores
  preservam fontes/tokens/relatório; fontes antigas só são liberadas após publicar
  o resultado completo. Coleta posterior é tolerante a falhas e retentável, para
  não anunciar erro de decode depois de confirmar um token. Arquivos próprios
  são separados de arquivos alheios; reset/close preservam o diretório pai.
- Na fixture D64/P2/H1/W1/G32 com 512 tokens, residência KV caiu de 49.920 para
  1.406 B e pico gerenciado de 130.431 para 91.899 B. Logits coincidiram exatamente;
  64.262 recargas leram 43.627.130 B. A amostra offloaded foi mais lenta, e contextos
  pequenos podem consumir mais memória por slot/metadata. Não comprova qualidade
  de modelo treinado, desempenho NVMe, RSS ou VRAM.
- M4.05b identifica o backing store CPU concluído; M4.05c preserva promoção e
  critérios de qualidade. Próximo incremento: reuso/promoção de residência com
  slots limitados, antes de confundir esse movimento com aumento de precisão.
  Promoção Q3→F32 não recupera informação descartada.

## Complemento Nexa Omni

O [plano Omni](NexaLang_Plano_Implementacao_Nexa_Omni.pdf) acrescenta orquestração,
Model Packs/ABI e nxpkg, Router/Brain/experts, DAG, Scheduler, memória e Critic.
Os [ajustes específicos](NEXA_OMNI_AJUSTES.md) registram a análise das 14 páginas,
as dependências e a numeração canônica F0–F7 da tabela de §20. O checklist recebeu
os dez itens OMNI, todos pendentes; ler o plano não equivale a implementar seus gates.

V0 usa mensagens tipadas e referências explícitas. Neural Bus latente fica para
pesquisa posterior. O perfil 512 MB serializa residência de experts conforme o
orçamento global. Reuso de KV entre modelos depende de assinatura das entradas,
pesos/adapters e layout, além de qualidade medida; o cache privado de uma sessão
não comprova compatibilidade entre modelos. Não há configuração fixa de Brain
ou especialistas para inferir das definições existentes R0/v1.

## Complemento de dados e treinamento

O [plano de treinamento de 16 páginas](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf)
foi analisado integralmente. Os [ajustes específicos](NEXALM_TREINAMENTO_AJUSTES.md)
preservam R0/v1 e mapeiam seus gates `TRAIN.G0–G8` aos itens existentes, sem
confundir TRAIN.G7 (NAQT) com o G7 anterior (fusões). LLM.02/03/09 foram detalhados
para permitir retomada de dados, tokenizer, shards, trainer e QAT separadamente.
Todos permanecem pendentes; o treino/tokenizer legado não satisfaz o contrato R0.

Coleta começa com schemas, fixtures locais, quotas, proveniência e decisões por
fonte. Trainer pequeno e retomada precedem R0. O documento distingue treinamento
de inferência: não estende o requisito 512 MB ao treino. Q3 KV não fecha o gate de
pesos Q3, e a menção a `.nxm` não introduz outro contêiner além dos formatos já
implementados. Não houve coleta ou treinamento neste incremento.

## Complemento de Conditional Compute

O [plano de 17 páginas](NexaLang_Conditional_Compute_Implementation.pdf) foi
analisado integralmente, incluindo figuras e tabelas. Os
[ajustes específicos](NEXA_CONDITIONAL_COMPUTE_AJUSTES.md) mapeiam C0–C10 para
**CC.01–11**: instrumentação/IR, MoE estático, Top-K/Momentum, cache de experts,
profundidade, caminhos cheap/heavy, early exit, sparsity, budget, NAQT e Omni.
São tarefas pendentes; não há motor condicional no runtime atual.

- Reusar ModelIR/KernelIR, formatos, backends, memória, trainer e scheduler
  existentes ou planejados. `bootstrap/mir.py`, SPIR-V/OpenCL experimental e KV
  comprimido não comprovam as capacidades atribuídas à base pelo novo PDF.
  R0/v1 densos continuam canônicos; novas arquiteturas precisam de versão e treino.
- Na [p.8](NexaLang_Conditional_Compute_Implementation.pdf#page=8), os intervalos
  de memória somam **390–550 MB**. Escolher valores sob um único teto, contando
  routers, combine, slots, fallback e alinhamento. O contrato mínimo de admissão
  de C8 precisa existir desde C0; coordenação adaptativa completa continua em C8.
- Skips e early exit exigem continuidade do KV quando camadas voltam a executar.
  Copiar hidden state não cria o histórico ausente; definir política treinada e
  recomputação admitida antes de habilitar C4/C6. Cheap/heavy preserva tokens e
  posições causais; rotas/histórico relevantes participam da assinatura do cache.
- HOT/PARTIAL/WARM/COLD nesse PDF descrevem residência de **pesos de experts**,
  distinta da política KV F32/Q4/Q3. M4.05b CPU não fecha C3 GPU/prefetch.
  Thresholds, qualidade, custos, ablações e treino progressivo exigem medição;
  `quality_floor=0.98` é exemplo de API, sem significar 98% de acerto.

## Complemento de Plastic Learning

O [plano de 20 páginas](NexaLang_Plastic_Learning_Implementation.pdf) foi
analisado integralmente, com figuras conferidas. Os
[ajustes de integração](NEXA_PLASTIC_LEARNING_AJUSTES.md) registram **PL.01–11**,
preservando P0–P7 e treino T0–T5. P2 foi dividido em aplicação de adapters,
transações e aprendizado/replay validado. Nada foi marcado implementado por leitura.

- Parâmetros totais, ativos e treináveis são distintos. Compute Router escolhe
  execução; Plasticity Router escolhe alvos permitidos. Compartilhar identidade
  de região/expert e budget, sem misturar permissão de forward e permissão de update.
- Usar a sequência textual da [p.5](NexaLang_Plastic_Learning_Implementation.pdf#page=5):
  evidenciar → memória primeiro → candidato isolado → regressão → commit/rejeição.
  A figura dessa página inverte partes da sequência. Na
  [p.17](NexaLang_Plastic_Learning_Implementation.pdf#page=17), adotar a tabela
  P0–P7; a figura termina em P5 e agrupa consolidação.
- O gráfico da [p.12](NexaLang_Plastic_Learning_Implementation.pdf#page=12) soma
  512 MB, mas não possui rótulos de categorias: não atribuir custos por suposição.
  Incluir base/candidato, adapters, gradientes, optimizer, replay, validação,
  staging, versões retidas e possível inferência concorrente nos budgets reais.
- Priorizar metadata/evidências e adapters com base congelada, transações e
  rollback antes de experts treináveis/consolidação. Reusar dados/trainer de
  LLM.02/03/04; o forward nativo atual não possui autograd nem aplica LoRA.
- Publicar versões imutáveis com pai, hashes, evidências e gates. Sessões fixam
  a versão dos pesos/adapters; troca/rollback exige invalidação ou recomputação
  do KV dependente, inclusive arquivos offloaded. Reusar OMNI.08 em vez de
  criar outra assinatura. O cache KV descartável não serve como log persistente.
- Definir ganho, forgetting, drift, localidade e tolerâncias antes dos episódios
  A→B. Replay e avaliação congelada permanecem separados; conteúdos gerados pelo
  próprio modelo não ganham valor de evidência independente por cópia/reingestão.
  T0/T1, pesquisa de experts, NAQT e Omni reutilizam as trilhas LLM/CC/M/OMNI.

Na auditoria dos quatro planos anteriores, foram explicitados critérios já
exigidos: codecs mistos por bloco e formato versionado (M6.02), runtime final
sem dependência de Python (M5.05), importância/budget KV por camada calibrados
(M4.05c/M6.01), negação de capabilities e isolamento/FFI de packs (OMNI.01/04).
A referência do smoke 10–50M foi corrigida para a p.14 do PDF de treinamento.
Isso detalha entregas pendentes; não reabre subitens CPU já comprovados.

## Referências técnicas

Na amostra P01/P23/P28 conferida, P23 precisa distinguir inventor de requerente:
a família EP4730108A1 lista Benjamin Wagner como inventor e Robert Bosch GmbH como
requerente. A folha americana deve ser conferida antes de corrigir definitivamente
a bibliografia US. P28 tem uma relação de continuação indireta.
Fonte: https://patents.google.com/patent/EP4730108A1/en

Os algoritmos serão definidos por contratos, medições e testes próprios. O plano
mantém a revisão de referências/claims antes de um release comercial prevista no PDF.
