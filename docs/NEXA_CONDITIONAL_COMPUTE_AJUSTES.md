# Conditional Compute — integração e critérios de desenvolvimento

Análise integral das 17 páginas do
[Conditional Compute Engine](NexaLang_Conditional_Compute_Implementation.pdf#page=1),
em 2026-09-19. Foram conferidas também as figuras das páginas 1, 2, 4, 7 e 11 e
as tabelas das páginas 8, 14 e 15. O PDF permanece preservado; este documento
registra decisões de planejamento sobre o estado atual do repositório.
Nenhum gate foi concluído pela leitura. O estado executável e as evidências
continuam no [checklist central](BLUEPRINT_512MB_CHECKLIST.md).

## Objetivo e fronteiras

O PDF propõe selecionar experts, camadas, caminhos de tokens e blocos de pesos,
com controle de saída antecipada e de custo. As quatro dimensões aparecem na
figura da [página 2](NexaLang_Conditional_Compute_Implementation.pdf#page=2).
O motor fica entre ModelIR e KernelIR; o runtime escolhe entre rotas válidas,
enquanto planejamento de memória e backend garantem residência e execução.
Essa separação consta do diagrama e da tabela das
[páginas 4–5](NexaLang_Conditional_Compute_Implementation.pdf#page=4).

O objetivo de variar a execução sem trocar o pacote pressupõe um pacote já
preparado e treinado com experts, routers, gates e saídas intermediárias.
Não autoriza aplicar skips ou MoE arbitrariamente a um checkpoint Llama denso.
R0/v1 continuam com suas arquiteturas atuais; uma variante condicional terá
identidade, configuração, tensores e receita próprios. O exemplo
`NexaSparse512`, incluindo `quality_floor=0.98`, é explicitamente ilustrativo,
sem suporte declarado no parser atual e sem significado de “98% de acerto”.
Ver [API proposta, página 14](NexaLang_Conditional_Compute_Implementation.pdf#page=14).

A integração macro/micro da [página 16](NexaLang_Conditional_Compute_Implementation.pdf#page=16)
separa o Router Omni, que escolhe modelos/órgãos, dos routers internos de cada
modelo. Os percentuais e contagens do desenho são exemplos, não configurações
aprovadas nem resultados medidos. Adaptação das rotas durante inferência também
não implica atualizar os pesos do modelo em produção.

## O que pode ser reutilizado

A lista de componentes existentes nas
[páginas 2–3](NexaLang_Conditional_Compute_Implementation.pdf#page=2) precisa das
seguintes qualificações, verificadas nos arquivos e no checkpoint atual:

| Base existente | Uso possível e trabalho ainda necessário |
|---|---|
| `compiler/model_ir.py`, `model_lowering.py` e `model_definition.py` | Grafo validado, shapes, operadores Transformer e DSL declarativa separados do bootstrap. Não têm operações condicionais, experts ou saída antecipada. Estender schemas/lowering com rejeição explícita de variantes incompatíveis. |
| `bootstrap/mir.py` e `nxc`/`nx.py` | Continuam no compilador da linguagem. A MIR da linguagem não substitui ModelIR/KernelIR nem conecta automaticamente os modelos à entrada suportada. Essa integração continua em LLM.01b2/M2.02. |
| NexaPack/bundle e kernels C | Pesos Q4/normas F32, leitura por tiles, validação e baseline CPU. Q3 no KV e TQ02 no armazenamento/KV não fornecem kernels de pesos Q3/TQ ou execução esparsa de experts. Reutilizar M1/M2, mantendo codecs sem kernel indisponíveis. |
| `compiler/planner/memory.py` e planos KV | Admissão, lifetimes, alinhamento e transações CPU. Estender ao conjunto de rotas e buffers do router; nenhuma medição atual demonstra residência de experts em GPU. |
| Tiers e backing store KV | Ownership, checksums, rollback e recarga limitada são padrões reutilizáveis. Seus arquivos guardam páginas KV Q3; não são um cache de pesos de experts nem uma fila de prefetch assíncrona. |
| SPIR-V/OpenCL e demais backends citados | Caminhos experimentais ou gates futuros M2/M7. O novo pipeline ainda não possui a prova GPU/KernelIR exigida pelo PDF. CPU serve como referência de correção. |
| Plano Omni | Compartilhar Pack ABI, scheduler, orçamento e telemetria. Não criar outro gerenciador de pacotes, scheduler global ou corpus de telemetria independente. |

Referências de estado: [ajustes gerais](BLUEPRINT_512MB_AJUSTES.md),
[tiers CPU](NEXALM_KV_TIERS_CPU.md), [backing store KV](NEXALM_KV_BACKING_CPU.md)
e [integração Omni](NEXA_OMNI_AJUSTES.md).

## Contratos que precisam anteceder a implementação

**IR e pacote.** `ConditionalBlock`, `ExpertSpec`, `DepthPlan` e `SparsitySpec`
são esboços nas [páginas 5](NexaLang_Conditional_Compute_Implementation.pdf#page=5),
[8](NexaLang_Conditional_Compute_Implementation.pdf#page=8) e
[10](NexaLang_Conditional_Compute_Implementation.pdf#page=10).
`RouterSpec`, `TokenPolicy`, `BudgetSpec`, `QualityProfile` e políticas de precisão
ou residência não têm schemas completos. Definir versão, IDs estáveis de
blocos/experts/tiles, pesos/aliases, shapes de entrada/saída e de rejoin, kernels
suportados, limites de K/capacidade, estado persistente e rotas de fallback.
Validar JSON e referências sem carregar pesos. Planos rejeitados não podem
virar callbacks dinâmicos fora dos lifetimes e da admissão.

A árvore `compiler/conditional`, `runtime/conditional`, `training/*` e os
arquivos `routing.nxl`/`exits.nxl` das
[páginas 13–14](NexaLang_Conditional_Compute_Implementation.pdf#page=13) é proposta
de organização. Começar com contratos e fixtures no pipeline de modelos;
a gramática pública só deve avançar depois da semântica executável. KernelIR
continua sendo dependência compartilhada M2.01/M2.02, não outro IR paralelo.

**MoE estático e router.** O score das
[páginas 6–7](NexaLang_Conditional_Compute_Implementation.pdf#page=6) combina
relevância, histórico, residência, qualidade, transferência, compute e carga.
Definir escalas/unidades, normalização, coeficientes versionados, desempate
estável e score inválido. Top-K precisa de contrato de dispatch/combine, pesos
das saídas, capacidade de cada expert, tratamento de overflow e fallback que
não descarte tokens silenciosamente. Em decode de uma sequência, esclarecer o
horizonte em que carga e balanceamento são medidos. O mecanismo de afinidade
bidirecional da tabela da página 3 não tem algoritmo ou gate próprio; fica
como experimento posterior à referência Top-2, com ablação independente.

Momentum deve ter decaimento, horizonte, reset por sessão e regra para mudança
de domínio. O custo de trocar expert deve usar bytes/tempos do layout real e
estado de residência. Uma preferência por manter o expert atual não pode
sobrepor restrições de rota, orçamento ou qualidade do modo escolhido.
Pesos, rota, RNG quando houver, momentum e publicação do token precisam de
semântica de falha/retry que evite avançar estado apenas parcialmente.

**Profundidade e estado causal.** O PDF exige continuidade do hidden state e
coerência entre treino e inferência nas
[páginas 8–9](NexaLang_Conditional_Compute_Implementation.pdf#page=8), mas não
define o KV de camadas puladas. Quando um token sai cedo ou uma camada deixa
de executar, faltam entradas de cache que podem ser necessárias se essa
camada voltar em tokens seguintes. Copiar o residual não resolve esse histórico.

Antes de C4/C6, escolher e treinar uma semântica explícita: máscara estável com
transições restritas, manutenção de estado definida pela arquitetura ou
reexecução do prefixo necessária à reentrada. Reexecução, se usada, entra no
budget e preserva posições, RoPE, máscara causal e identidade dos pesos/rotas.
A assinatura do cache deve incluir a política e o histórico relevante de
execução; um KV produzido por outra rota não pode ser reutilizado apenas porque
os IDs/tokenizer coincidem. Estender a `CacheSignature` já prevista em OMNI.08,
sem criar um segundo contrato de compartilhamento. Cobrir prefill, chunks, decode, mudança de máscara,
saída antecipada e retorno ao caminho completo com referência independente.

**Tokens e saída antecipada.** A página 9 apresenta cheap/heavy seguido de
`rejoin()`, não uma licença para remover tokens do contexto causal. Definir
índices, ordem de scatter/gather, caminho vazio, custos do router e preservação
de posições/KV em [C5/C6](NexaLang_Conditional_Compute_Implementation.pdf#page=9).
O exemplo de entropia/confiança exige treino e calibração em split congelado;
calcular concordância com logits finais a cada token em produção eliminaria
parte da economia pretendida. Usar a execução completa como avaliação offline
e definir o estimador disponível no ponto de saída. Profundidade obrigatória
por modo/validador precisa constar da política antes de escolher a rota.

**Esparsidade interna.** `dense`, blocos, N:M e unstructured são opções de
schema, sem implementação automaticamente disponível. M6.06/M6.07 devem
fornecer máscaras, índices, kernels, agrupamento e fallback antes da composição
com experts. Contar metadados e eventual tile denso temporário; zeros lógicos
não demonstram menos bytes físicos nem ganho de velocidade. Essa seleção por
hardware é exigida nas [páginas 9–10](NexaLang_Conditional_Compute_Implementation.pdf#page=9).

## Residência e orçamento conjunto

Os estados da figura e tabela da
[página 7](NexaLang_Conditional_Compute_Implementation.pdf#page=7) são de
**pesos de experts**: HOT em GPU, PARTIAL com tiles em GPU e restante em RAM,
WARM em RAM e COLD em RAM comprimida/NVMe. A figura enfatiza COLD em NVMe,
enquanto a tabela permite também RAM comprimida. Fixar tiers físicos no perfil
selecionado e registrar residência separadamente de codec; não reaproveitar os
nomes hot F32/warm Q4/cold Q3 do KV como se tivessem a mesma semântica.

A tabela de [orçamento, página 8](NexaLang_Conditional_Compute_Implementation.pdf#page=8)
não é uma configuração que possa usar todos os máximos simultaneamente:

| Área | Intervalo sugerido no PDF |
|---|---:|
| Tiles do expert atual | 170–220 MB |
| Cache parcial | 60–90 MB |
| KV/estado | 50–70 MB |
| Ativações/workspace | 50–70 MB |
| Prefetch | 40–60 MB |
| Runtime/reserva | 20–40 MB |
| Soma calculada | **390–550 MB** |

O máximo supera 512 MB em **38 MB**. Cada plano deve escolher valores que
satisfaçam uma restrição global com unidade explícita: `512MB` significa
512.000.000 bytes. Contar escalas, índices, routers/gates, heads de saída,
acumuladores de combine, alignments, buffers de cópia e lifetime das origens
até commit. Tiles compartilhados entre expert ativo e cache parcial não podem
ser contados duas vezes; se forem cópias físicas, ambas entram na soma.

Slots de prefetch, transferências em voo e pressão dos caminhos de fallback
precisam ser admitidos antes da execução. K experts lógicos não exige manter K
experts completos simultaneamente: execução serial por tiles pode atender ao
perfil, desde que preservados o combine e seus buffers. O desenho de prefetch
da página 7 depende de M3 e de eventos/ownership reais; o backing store KV CPU
atual não comprova esse overlap.

`Budget(token)` lista FLOPs, bytes movidos, latência, qualidade e trocas de
expert nas [páginas 10–11](NexaLang_Conditional_Compute_Implementation.pdf#page=10).
Acrescentar a referência ao orçamento global de memória e definir unidades,
escopo por token/prefill e débito de todos os routers. Bytes estimados, leituras
lógicas e transferências físicas medidas são campos distintos. `max_latency_us`
é um alvo de planejamento/medição, não garantia de tempo real produzida por
uma estimativa; registrar violações e limites de cancelamento dos kernels.
Piso de qualidade é avaliado/calibrado por tarefas, não uma garantia online de
correção individual. Na ausência de rota admissível, retornar falha explícita
antes de publicar o token ou usar somente fallback já admitido.

C8 introduz o allocator global tardiamente na tabela, mas C1–C7 já dependem de
admissão conjunta. Antecipar o contrato mínimo e contadores em C0; C8 fecha a
coordenação adaptativa entre K, profundidade, caminho e precisão. Declarar pico
de buffers CPU, RSS, arena de dispositivo e pico do driver separadamente.
Cap artificial em GPU maior e fixture CPU não encerram o gate de hardware 512 MB.
Ver [baselines exigidos, página 15](NexaLang_Conditional_Compute_Implementation.pdf#page=15).

## Treinamento, observabilidade e avaliação

As fases A–E das [páginas 11–12](NexaLang_Conditional_Compute_Implementation.pdf#page=11)
são: referência densa, especialização de experts, Top-K/profundidade,
cheap/heavy/saída e NAQT/esparsidade. Reutilizar os contratos NexaData,
tokenizer, checkpoints e retomada de LLM.02/03 e os contratos NAQT de LLM.09.
A especialização usa inicialização/treino explícitos; acrescentar experts no
manifesto não cria uma especialização demonstrada.

As perdas de tarefa, distilação, balanceamento, diversidade, temporalidade,
budget, calibração de saída e consistência de skip da
[página 12](NexaLang_Conditional_Compute_Implementation.pdf#page=12) devem ter
receita, coeficientes e agenda versionados. Aumentar penalidades de eficiência
após estabilizar capacidade, conforme a página 11. Treinar e habilitar uma
nova dimensão por vez, como exige a
[ordem recomendada, página 15](NexaLang_Conditional_Compute_Implementation.pdf#page=15).
O marco M8/LLM.11 continua pesquisa de arquitetura posterior à base de modelo;
esta análise não substitui o checkpoint treinado e os gates M5.

O ciclo visual da [página 11](NexaLang_Conditional_Compute_Implementation.pdf#page=11)
passa de telemetria para dataset de custo, recalibração e próximo treino.
Isso é uma proposta de ciclo controlado, não atualização automática durante
inferência. Para exemplos que incluam conteúdo do usuário, reutilizar
proveniência, consentimento, remoção de PII/segredos e promoção de dados
aprovados já especificados em LLM.10/OMNI.09.

A matriz das [páginas 15–16](NexaLang_Conditional_Compute_Implementation.pdf#page=15)
exige VRAM, parâmetros ativos, camadas executadas, trocas, bytes host→GPU,
tokens/s, TTFT, perplexidade/tarefas, calibração e balanceamento. Acrescentar
energia/token quando houver medição disponível, como solicita a
[página 9](NexaLang_Conditional_Compute_Implementation.pdf#page=9); sem medidor,
registrar desconhecido. Os termos “significativo”, “baixo” e “qualidade estável”
precisam de limiares por tarefa definidos antes do experimento.

Comparações devem fixar dados/tokenizer, IDs, contexto, política de geração,
precisão, hardware e condições de cache. Distinguir referência densa de mesmo
hidden/depth, Top-1/Top-2 fixos e caminho completo da própria arquitetura
condicional: “mesmas dimensões” não implica mesmo número de parâmetros ou
mesma capacidade. Cada ablação desabilita uma dimensão e mede o custo de
routing/prefetch/calibração, inclusive um cenário em que não há benefício.

Relatórios devem incluir distribuição de K, tokens por expert/domínio,
transições/momentum, histogramas de layers/saída, fração heavy, violações de
capacidade e orçamento, stalls e erro contra referência. Detecção de expert
morto/colapso e testes de mudança de domínio são evidências de comportamento,
não algo concluído por conter campos em um JSON.

## Gates originais e tarefas do checklist

O PDF usa **C0–C10**, sem tickets como `ROUTER-001`. A correspondência abaixo
preserva os nomes das tabelas das
[páginas 14–15](NexaLang_Conditional_Compute_Implementation.pdf#page=14).
Os IDs CC.01–CC.11 foram incorporados ao checklist; as dependências
existentes permanecem abertas até suas próprias provas.

| ID / gate | Entrega específica de Conditional Compute | Aceitação e dependências compartilhadas |
|---|---|---|
| CC.01 / C0 | Instrumentação por camada e contratos de rotas/experts/qualidade/budget mínimo, sem mudar a execução baseline. | Schemas versionados, custos estimados versus medidos separados, limites rejeitados antes de alocar e execução baseline preservada. Reusa M0, M2.02, LLM.04; perfil treinado depende de LLM.04b. |
| CC.02 / C1 | MoE Top-2 estático com dispatch/combine, capacidade e treino/especialização verificáveis. | Oracle numérico, rota completa/fallback, balanceamento e ausência de tokens perdidos. Detalha M8.03/LLM.11; execução nativa pelo KernelIR depende de M2.01/M2.02. |
| CC.03 / C2 | Top-K adaptativo, custos de troca e Expert Momentum versionados. | K válido sob budget, replay/retry determinístico, menor atividade medida e qualidade dentro do limite predefinido. Compara Top-1/Top-2 e sem momentum. |
| CC.04 / C3 | Cache específico de experts HOT/PARTIAL/WARM/COLD e decisão de prefetch por rota. | Identidade/ownership por tile, integridade, transições/falhas, bytes/stalls e prova de residência de experts sob cap real. Reusa M1.10b, M2.06, M3 e M7; cache KV CPU não conclui C3. |
| CC.05 / C4 | Máscara por consulta, gates por camada e continuidade de hidden/KV na reentrada. | Treino coerente, teste causal de mudança de máscara, replay ou recuperação admitidos e redução de camadas com qualidade medida. Reusa M4/M8.03. |
| CC.06 / C5 | Um bloco cheap/heavy com roteamento e rejoin estáveis. | Ordem/posições/KV preservados, caminhos vazios e capacidade testados; economia líquida incluindo router e movimentação, com limite de qualidade. |
| CC.07 / C6 | Heads de saída e estimadores calibrados por token/tarefa/modo. | Profundidade obrigatória respeitada, continuidade KV, métricas de calibração/logits/qualidade em split congelado e profundidade menor medida. Depende de CC.05 e dos dados/treino LLM.02/03. |
| CC.08 / C7 | Compor seleção de experts com sparsity de pesos já suportada. | Mesma semântica com fallback dense, metadados e tiles temporários contabilizados, ablação por hardware com bytes/FLOPs/latência. Reusa M6.06/M6.07; não cria outro kernel esparso paralelo. |
| CC.09 / C8 | Coordenar dinamicamente K, profundidade, heavy path, precisão e swaps no orçamento global. | Contabilidade conservadora, decisões sem rota admissível explícitas, thresholds/qualidade calibrados e trade-offs medidos. Estende o contrato antecipado em CC.01 e custos de M7.09/M8.04. |
| CC.10 / C9 | Receita e pacote condicionais NAQT com custos físicos reais. | QAT versus PTQ com mesma base, rotas e dados, sem expansão integral oculta, qualidade e execução completa sob cap demonstradas. Reusa LLM.09/M8.02, codecs M1/M6 e prova M5. |
| CC.11 / C10 | Integrar routers internos ao orçamento/telemetria do Scheduler Omni. | Seleção macro/micro conjunta, soma real de modelos/KV/workspaces, cancelamento e ablações de duas escalas. Reusa OMNI.03/05/09; não duplica Pack ABI, Router Omni ou gerenciador. |

O Definition of Done da [página 17](NexaLang_Conditional_Compute_Implementation.pdf#page=17)
requer caminho MoE nativo por ModelIR→KernelIR, Top-K, cache, depth, cheap/heavy,
saída calibrada, sparsity com fallback, orçamento, execução 512 MB sem expansão
oculta e baselines/telemetria. Ele não enumera explicitamente a integração Omni,
que continua um gate C10 separado; concluir uma fixture Top-2 não fecha V1.

Ordem proposta: CC.01 com admissão mínima → CC.02 estático → CC.03/CC.04,
com contratos e simulações CPU antes das respectivas provas GPU → CC.05 →
CC.06 → CC.07 → CC.08 → coordenação/calibração CC.09 → CC.10 → CC.11.
Os contratos e instrumentos podem avançar junto à base atual; treino condicional
com qualidade e gates de hardware dependem dos marcos LLM/M5/M8 correspondentes.
O próximo incremento de runtime registrado no checkpoint central permanece
independente desta incorporação documental.

## Referências de pesquisa e limites desta análise

As [páginas 3–4](NexaLang_Conditional_Compute_Implementation.pdf#page=3) apresentam
patentes como mapas de problemas de routing, cache, depth, sparsity e early exit;
a [página 17](NexaLang_Conditional_Compute_Implementation.pdf#page=17) lista os
identificadores e ressalva que isso não demonstra ausência de sobreposição.
Esta análise verificou o conteúdo do PDF local, sem conferir externamente os
números, famílias, titulares, vigência, claims ou resultados atribuídos às fontes.

ADRs, experimentos independentes e a revisão de release já prevista em M9.01–03
continuam sendo a trilha apropriada. Acrescentar momentum ou mudar o nome de
um algoritmo não comprova originalidade jurídica. Nenhum ganho de velocidade,
qualidade ou capacidade de hardware sugerido pelo PDF foi tratado aqui como
resultado reproduzido.
