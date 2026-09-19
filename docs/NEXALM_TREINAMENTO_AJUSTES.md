# Integração do plano de treinamento NexaLM

Análise das 16 páginas do
[plano de treinamento](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf), em
2026-09-19. O PDF permanece como fonte original. A implementação e as evidências
ficam no [checklist central](BLUEPRINT_512MB_CHECKLIST.md); ler este plano não
conclui nenhum gate de dados, treino ou GPU.

## Escopo e estado existente

O plano detalha NexaData, treinamento R0, NexaLM-512, NAQT/QAT e especialização
Omni. Dados, trainer e runtime são componentes distintos. O trainer de referência
pode usar PyTorch e hardware maior; o orçamento de 512 MB pertence à inferência,
com unidade, dispositivo e componentes contabilizados explicitamente.

As definições atuais em `models/nexalm512/architecture.nxl` continuam canônicas:
R0 tem 125.854.464 parâmetros e v1 tem 394.331.136, com vocabulário 32768 e quatro
heads KV. O intervalo de 32k–48k tokens e a opção de dois heads KV do novo PDF
não alteram essas configurações automaticamente. Variantes precisam de versão,
novos shapes e contagem de parâmetros.

Há treino e tokenizer legados, mas eles não satisfazem os novos contratos:
`tools/train_mlx.py` usa vocabulário 1024, LayerNorm/GELU, posições aprendidas e
MHA; `tools/train_bpe.py` treina BPE 1024 sobre pretrain/instruct antes do split.
Eles não são o trainer/tokenizer canônico de R0. O pipeline atual importa pesos
Q4/normas F32 e executa KV F32/Q4/Q3; Q3 no KV não equivale a Q3 nos pesos.

## Correspondência dos gates

O PDF de treinamento reutiliza G0–G8 com significados diferentes do plano da
primeira LLM. Referências novas usam **TRAIN.G0–G8**; os IDs anteriores permanecem.

| Novo PDF | Evidência exigida | Checklist existente |
|---|---|---|
| TRAIN.G0 | NexaData reproduzível e tokenizer versionado | LLM.02a–d |
| TRAIN.G1 | R0 treinado com checkpoint e retomada | LLM.03a–b |
| TRAIN.G2 | Logits de checkpoint treinado concordam com o runtime | LLM.04b |
| TRAIN.G3 | Pesos Q4/Q3 sem expansão integral | LLM.05, M1.05 |
| TRAIN.G4 | Working set real dentro de 512 MB | LLM.06, M2.06, M5 |
| TRAIN.G5 | Modelo maior que VRAM com streaming/prefetch | LLM.06, M3 |
| TRAIN.G6 | KV menor com perda de qualidade controlada | LLM.07, M4 |
| TRAIN.G7 | QAT supera PTQ equivalente em qualidade | LLM.09a–b, M8.02 |
| TRAIN.G8 | SFT e interfaces para especialistas | LLM.10, OMNI.03/07 |

LLM.08 continua sendo fusões/tuning e custo medido. O TRAIN.G7 não substitui
esse item: ele corresponde ao G8/LLM.09 do plano anterior. A lista de sprints do
PDF define dependências, sem constituir uma previsão de prazo.

## Contratos para implementação

**LLM.02a — documentos e snapshots.** Versionar schema de documento, origem,
proveniência, licença, idioma/domínio, decisões de aceitação e motivos de rejeição.
Guardar hash de origem e conteúdo normalizado, versão dos filtros, receita de
mistura e seeds. A leitura deve aceitar limites de bytes/documentos/tokens e
retomar deterministicamente. As fontes listadas no PDF são candidatas: a decisão
de ingestão usa os termos e a revisão da fonte efetivamente selecionada, com
allowlist/denylist de licenças e bloqueio de shards não comerciais em produtos.

**LLM.02b — filtros, deduplicação e avaliação.** Definir filtros de qualidade,
PII, segredos e spam, normalização, parâmetros de dedup exato/aproximado e
desempate determinístico. Separar treino/validação/teste
por documento, repositório ou família antes do packing. Congelar avaliação e
registrar decontaminação, estatísticas por fonte e exposição a repetições.

**LLM.02c — tokenizer.** Treinar somente sobre o corpus de treino permitido;
congelar assets, IDs especiais, normalização, versão e hashes. Validar cobertura
PT/EN/código/matemática, round-trip e compatibilidade com o vocabulário configurado.
Medir também bytes/caracteres por token por idioma e domínio.
Tokenizer armazenado como asset no importador ainda não significa tokenização
implementada no runtime final.

**LLM.02d — `.nxd` e dataloader.** Especificar magic/version, endian, largura de
IDs, limites de documentos, BOS/EOS, máscaras, índices e checksums. Publicação
atômica; leitor rejeita truncamentos/corrupção e mantém memória limitada. Testar
retomada da posição de leitura e determinismo do sampler antes de escalar dados.

**LLM.03a — trainer mínimo.** Consumir a configuração arquitetural existente;
guardar a receita de otimização separadamente da DSL. Provar forward/backward,
embedding compartilhado, labels deslocados e máscaras de packing em fixture
pequena, antes do treino R0. Especificar AdamW, scheduler, accumulation, clipping,
precisão de compute, parâmetros mestres, optimizer e acumulação; “BF16 master”
sozinho não define todos esses estados.

**LLM.03b — retomada e exportação.** Checkpoint inclui optimizer/scheduler, RNGs,
sampler, posição nos shards, tokens vistos e hashes de configuração, dados,
tokenizer e receita. Comparar treino contínuo e interrompido dentro da tolerância
declarada. Exportar Safetensors e usar o importador existente; o nome `.nxm` do
PDF é um exemplo futuro, não um novo formato implementado.

**LLM.08 — custos.** Começar com seleção offline baseada em medições do hardware:
memória, transferências e latência. Busca automática e penalidades dentro da loss
são etapas posteriores. Não bloquear o primeiro trainer com um autotuner novo.

**LLM.09a–b — NAQT/QAT.** Definir avanço das fases por tokens ou steps, formatos e
camadas sensíveis. Fake quantization deve representar grupos, arredondamento,
saturação e caudas reais do runtime. Separar receitas para pesos, ativações e KV;
formatos ainda não implementados ficam indisponíveis. Comparar QAT/PTQ com a mesma
base, dados, contexto e precisão, medindo qualidade, memória e desempenho.

**LLM.10 / OMNI.07/09 — SFT e exemplos aprovados.** Separar dados de base, SFT,
preferências e Omni. Definir schema Nexa-Instruct com campo `verified` e método de
verificação dos exemplos sintéticos. Logs/telemetria não viram corpus diretamente:
a promoção para exemplos aprovados registra consentimento, proveniência, avaliação
e remoção de PII/segredos, conforme o contrato do PDF.

## Ambiguidades e decisões de planejamento

- O smoke aparece como 50–100M tokens na [página 6](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf#page=6)
  e 10–50M na [página 14, sprint S4](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf#page=14). Separar
  teste local pequeno de engenharia, experimento do trainer e validação da pipeline,
  cada qual com teto explícito. Não iniciar um treino longo a partir desses intervalos.
  R0 com cerca de 3B e v1 com 20–50B tokens são referências de escala do plano,
  condicionadas a orçamento, dados e gates; não são execuções já configuradas.
- A mistura 35% EN, 30% PT, 15% código, 10% STEM, 5% enciclopédia e 5% Nexa soma
  100%, mas idioma e domínio se sobrepõem. Definir grupos exclusivos de amostragem
  e etiquetas independentes. Em 3B tokens, 5% Nexa seriam 150M tokens: medir dados
  únicos e repetições, com limite de exposição.
- “Texto coerente”, “perda controlada” e “qualidade aceitável” precisam de tarefas,
  splits, métricas, baselines e limites definidos antes de aprovar um experimento.
  Os testes atuais com pesos sintéticos comprovam execução, sem medir linguagem.
- Tokenizer e ancestral comuns entre especialistas não tornam o KV intercambiável.
  OMNI.08 continua exigindo assinatura de pesos/adapters, entradas, posições e layout.
- O requisito GPU precisa de telemetria real; contabilidade de buffers CPU e
  arquivos compactados não concluem TRAIN.G4/G5.

Ordem de execução: contratos/fixtures NexaData → filtros/dedup/decontaminação →
tokenizer/shards/dataloader → trainer pequeno com retomada → R0 conforme orçamento
→ avaliação/exportação → QAT/SFT → especialistas. Essa trilha pode avançar em
paralelo aos kernels e ao runtime. A próxima tarefa de runtime está no checkpoint
central; nenhuma coleta de corpus ou execução de treinamento foi feita neste marco.
