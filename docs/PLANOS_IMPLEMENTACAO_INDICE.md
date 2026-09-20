# Índice dos planos de implementação

Revisão documental de 2026-09-19: **6 PDFs, 108 páginas**. Os originais foram
preservados. O [checklist central](BLUEPRINT_512MB_CHECKLIST.md) é a fonte única
do estado de desenvolvimento; este índice indica o que consultar por módulo.
Referências `#page=N` usam a página física do PDF, começando em 1. Se o leitor
não abrir no ponto indicado, use o número de página mostrado no link.

## Documentos e cobertura

| Plano | Páginas | Uso na implementação | Itens do checklist | Decisões de integração |
| --- | ---: | --- | --- | --- |
| [Blueprint técnico 512 MB](NexaLang_Blueprint_Tecnico_512MB.pdf) | 19 | IRs, formatos, kernels, memória, streaming, backends e provas de escala | M0–M9 | [Ajustes gerais](BLUEPRINT_512MB_AJUSTES.md) |
| [Primeira LLM](NexaLang_Plano_Implementacao_Primeira_LLM.pdf) | 22 | R0/v1, arquitetura, dados, treino, exportação, gates G0–G8 e release | LLM.01–11; reutiliza M0–M8 | [Integração NexaLM](BLUEPRINT_512MB_AJUSTES.md#complemento-do-plano-da-primeira-llm) e [treinamento](NEXALM_TREINAMENTO_AJUSTES.md) |
| [Nexa Omni](NexaLang_Plano_Implementacao_Nexa_Omni.pdf) | 14 | Packs/ABI, nxpkg, Router/Brain, DAG, memória, Scheduler, Critic e Neural Bus | OMNI.01–10 | [Ajustes Omni](NEXA_OMNI_AJUSTES.md) |
| [Treinamento NexaData/LLMs](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf) | 16 | Proveniência, filtros, tokenizer, shards, trainer, NAQT/QAT, SFT e especialistas | LLM.02/03/04/05/06/07/08/09/10; TRAIN.G0–G8 | [Ajustes de treinamento](NEXALM_TREINAMENTO_AJUSTES.md) |
| [Conditional Compute](NexaLang_Conditional_Compute_Implementation.pdf) | 17 | MoE, Top-K, residência de experts, profundidade, rotas por token, early exit e orçamento | CC.01–13; gates CC.C0–C10 | [Ajustes Conditional Compute](NEXA_CONDITIONAL_COMPUTE_AJUSTES.md) |
| [Plastic Learning](NexaLang_Plastic_Learning_Implementation.pdf) | 20 | Evidências, governança, adapters, transações, experts plásticos e consolidação | PL.01–11; fases PL.P0–P7, treino PL.T0–T5 | [Ajustes Plastic Learning](NEXA_PLASTIC_LEARNING_AJUSTES.md) |

Os quatro primeiros planos já tinham tarefas registradas e foram novamente
conferidos. Conditional Compute e Plastic Learning passam a ter trilhas próprias,
vinculadas à infraestrutura comum. As seções §12 e §13 do PDF de Conditional
Compute atravessam as linhas da tabela C0–C10 e receberam os itens CC.12/CC.13.
Leitura, exemplos de sintaxe e estruturas propostas nos PDFs não significam
implementação concluída.

## Onde consultar cada módulo existente

| Módulo / tarefa | Referências de origem | Como usar |
| --- | --- | --- |
| ModelIR, tensores, HardwareProfile e MemoryPlan — M0, M2 | Blueprint [pp.3–5, §§3–4](NexaLang_Blueprint_Tecnico_512MB.pdf#page=3), [p.14, §6](NexaLang_Blueprint_Tecnico_512MB.pdf#page=14); Primeira LLM [p.6, §4](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=6), [p.11, §9](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=11) | Partir de `compiler/model_ir.py`, `model_lowering.py` e `planner/memory.py`. KernelIR e integração com `nxc` continuam gates próprios. |
| NexaPack, importação, codecs, pacotes executáveis — M1 | Blueprint [p.4, §4.3](NexaLang_Blueprint_Tecnico_512MB.pdf#page=4); Primeira LLM [p.10, §8](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=10); Treinamento [p.10, §16](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf#page=10) | Consultar também os guias [V1](NEXAPACK_V1.md), [TQ](NEXAPACK_TQ_V1.md), [codecs de peso](NEXALM_CODECS_PESOS.md) e [importação](NEXALM_IMPORTACAO.md). Não confundir `.nxm` ilustrativo com formato já suportado. |
| Kernels, streaming e backends — M2/M3/M7 | Blueprint [pp.14–16, §§6–10](NexaLang_Blueprint_Tecnico_512MB.pdf#page=14); Primeira LLM [pp.11–14, §§9–12](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=11) | Contar temporários, cópias e reservas por tier; GPU depende de hardware e telemetria reais. |
| KV, paginação, codecs e residência — M4 | Blueprint [p.15, §7.3](NexaLang_Blueprint_Tecnico_512MB.pdf#page=15); Primeira LLM [p.13, §11](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=13) | Ler [tiers CPU](NEXALM_KV_TIERS_CPU.md) e [backing store CPU](NEXALM_KV_BACKING_CPU.md) antes de modificar o contrato atual. Reuso e residência de páginas cold estão em [reuso CPU](NEXALM_KV_RELOAD_CPU.md); promoção de precisão e critérios de qualidade permanecem em M4.05d. Sequências derivadas com prefixo compartilhado, homogêneas ou por idade, estão em [sequências CPU](NEXALM_KV_SEQUENCIAS_CPU.md); backing store e cancelamento ficaram em M4.06c. |
| Arquiteturas R0/v1 e migração do frontend — LLM.01 | Primeira LLM [pp.3–6, §§1–4](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=3), [pp.21–22, §§19–20](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=21) | Preservar `models/nexalm512/architecture.nxl`; mudanças de shapes/experts criam variantes versionadas. |
| NexaData, tokenizer, `.nxd`, dataloader — LLM.02a–d (tokenizer V1 em [guia](NEXALM_TOKENIZER.md)) | Primeira LLM [p.7, §5](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=7); Treinamento [pp.3–6, §§2–8](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf#page=3), [p.14, §22](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf#page=14) | Usar contratos de proveniência, snapshots, splits, hashes e retomada dos [ajustes de treinamento](NEXALM_TREINAMENTO_AJUSTES.md#contratos-para-implementação). |
| Trainer, checkpoints e avaliação — LLM.03/04/10 | Primeira LLM [p.8, §6](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=8), [pp.16–19, §§14–17](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=16); Treinamento [pp.7–11, §§9–17](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf#page=7), [pp.13–15, §§19–24](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf#page=13) | Começar com fixture pequena e retomada verificável. Os números de tokens dos PDFs são propostas de escala, não comandos para iniciar treino. |
| Precisão, QAT/NAQT, sparsity e custo — M6/M8, LLM.08/09 | Blueprint [p.14, §6](NexaLang_Blueprint_Tecnico_512MB.pdf#page=14), [p.17, §12](NexaLang_Blueprint_Tecnico_512MB.pdf#page=17); Primeira LLM [p.9, §7](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=9), [p.20, §18](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=20); Treinamento [pp.8–9, §§12–13](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf#page=8) | Kernels e codecs compartilhados por CC/PL; não criar implementações concorrentes do mesmo contrato. |
| Prova de modelo completo e distribuição — M5/M9 | Blueprint [pp.17–19, §§11–16](NexaLang_Blueprint_Tecnico_512MB.pdf#page=17); Primeira LLM [p.21, §19](NexaLang_Plano_Implementacao_Primeira_LLM.pdf#page=21); Treinamento [p.15, §24](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf#page=15) | Qualidade de checkpoint treinado, interfaces nativas e execução final independente dos frameworks de referência são gates separados do executor CPU atual. |
| Packs, ABI e nxpkg — OMNI.01/02 | Omni [pp.3–5, §§3–6](NexaLang_Plano_Implementacao_Nexa_Omni.pdf#page=3), [pp.11–12, §§18–20](NexaLang_Plano_Implementacao_Nexa_Omni.pdf#page=11) | Reusar o gerenciador `nxpkg`; manifesto NexaModelBundle não conclui Pack ABI/capabilities. |
| Protocolo, Router/Brain, DAG e Critic — OMNI.03/04/06/07/09 | Omni [pp.6–9, §§7–14](NexaLang_Plano_Implementacao_Nexa_Omni.pdf#page=6), [pp.12–13, §§20–23](NexaLang_Plano_Implementacao_Nexa_Omni.pdf#page=12) | Mensagens tipadas primeiro; F1 não conclui sozinho o DoD V0. |
| Memória compartilhada, Scheduler, cache e Neural Bus — OMNI.05/08/10 | Omni [pp.7–10, §§9–16](NexaLang_Plano_Implementacao_Nexa_Omni.pdf#page=7), [p.12, §20](NexaLang_Plano_Implementacao_Nexa_Omni.pdf#page=12) | Orçamento único, identidades de pesos/adapters e isolamento; Neural Bus continua pesquisa posterior. |

## Como os novos planos se encaixam

Conditional Compute escolhe **o que executar dentro de um modelo**. Omni escolhe
**quais modelos executar**. Plastic Learning escolhe **o que pode ser atualizado**.
Os três compartilham identidades, budgets e telemetria, mas mantêm decisões e
permissões distintas. Os mapas detalhados por fase estão nos ajustes de
[Conditional Compute](NEXA_CONDITIONAL_COMPUTE_AJUSTES.md) e
[Plastic Learning](NEXA_PLASTIC_LEARNING_AJUSTES.md).

| Integração | Dependências reutilizadas | Contrato que precisa ficar explícito |
| --- | --- | --- |
| MoE/rotas condicionais no grafo | M2.01/02, M8.03, LLM.01b2/03/04 | IR deve enumerar rotas e fallback válidos; os exemplos `conditional`, `depth`, `early_exit` ainda são propostas. |
| Cache de experts e pesos | M1/M3/M7, OMNI.05 | Cache de expert contém pesos, não páginas KV. O backing store M4.05b não encerra o gate CC.C3 de residência GPU/prefetch. |
| Skip/early exit e cache causal | M4, OMNI.08 | Definir continuidade do KV e o custo de recomputação quando uma camada volta a executar. Token em caminho leve não pode desaparecer do contexto causal por acidente. |
| Adapters/deltas e atualização de versão | M1.10, LLM.03/04, OMNI.08 | Base imutável, snapshots e publicação atômica; pin de versão da sessão ou reconstrução do KV dependente dos parâmetros alterados. |
| Evidências/replay/telemetria | LLM.02, LLM.10, OMNI.03/05/06 | Proveniência e política de uso antes de promover dados a evidência de treino; separar replay de avaliação congelada. |
| Treino de routers/plasticidade | LLM.03/04/09, M8 | Aproveitar trainer e exportação comuns; manter métricas de qualidade e anti-forgetting antes de escalar. O forward C atual não implementa autograd. |
| Orçamento combinado | M0.04, M2.06, M3, OMNI.05 | KV, pesos, adapters, gradientes, optimizer, staging e rollback entram nos respectivos budgets; inferência e aprendizado concorrentes compartilham o teto. |

## Ordem e retomada

1. Localizar o ID no checklist e abrir este índice, o PDF/páginas indicados e o
   guia de ajustes correspondente. Confirmar pré-requisitos e o estado real no código.
2. Manter a trilha de runtime registrada no checkpoint (M6.01). Em paralelo,
   contratos de CC.C0 e PL.P0/P1 podem começar com fixtures locais e sem treino longo.
3. Antes de rotas condicionais aprendidas, estabelecer teacher/baselines e trainer
   necessários. Adicionar uma dimensão de sparsity por experimento, com fallback.
4. Em Plastic Learning, priorizar evidências, adapters versionados e rollback
   (P0–P2). Pool de experts treináveis e consolidação dependem dessa base; não
   antecipar alteração do Stable Core nem exigir MoE completo para metadata/adapters.
5. Ao parar, registrar ID/subtarefa, arquivos alterados, testes/evidências,
   decisões pendentes e próximo comando no checkpoint. Marcar concluído somente
   quando os critérios de aceite do item forem demonstrados.

Numeração: M/LLM/OMNI continuam estáveis; `TRAIN.G0–G8`, `CC.C0–C10`, `PL.P0–P7`
e `PL.T0–T5` identificam gates/fases dos PDFs. `CC.xx` e `PL.xx` identificam tarefas
no checklist. Os diferentes usos de G0 ou P0 não devem ser fundidos.

## Identificação das fontes revisadas

Os hashes identificam esta revisão dos arquivos. Se um PDF for substituído,
revalidar páginas, propostas e correspondência de itens antes de usar o mapa.

| PDF | SHA-256 |
| --- | --- |
| Blueprint técnico | `8ba9409f35e1530dd229a9816d7486dac6be43a269a58af3972819d33b5d8303` |
| Primeira LLM | `b1e1352b8a48a252e9ebd581cda150852693e2fefb75f908bf500b15111e25ed` |
| Omni | `cb6c0415921821363172722fd26250337f8090ee79fc376892b507a5214cfe08` |
| Treinamento | `e2e342b0b86bcf31cf50e9206418458770f6c515b0d99e7e1369318599face0f` |
| Conditional Compute | `447e24faee2bf5979bdb0255f51ec50811630d3d01e61d859bf8b66b574d89c0` |
| Plastic Learning | `b3f938c151eb84e3c9fd3d4aefc7cf3f79082a56e80c5f48eeb147468ae7f918` |

A revisão cobre os documentos locais; suas bibliografias são referências de
pesquisa, não validação independente de resultados/claims. Conferência bibliográfica
e revisão prevista para release permanecem em M9.02/03, incluindo as listas de
Conditional Compute [pp.3,17](NexaLang_Conditional_Compute_Implementation.pdf#page=17)
e Plastic Learning [p.19](NexaLang_Plastic_Learning_Implementation.pdf#page=19).
