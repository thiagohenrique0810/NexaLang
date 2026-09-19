# Integração do plano Plastic Learning

Análise integral das 20 páginas de
[NexaLang Plastic Learning Engine](NexaLang_Plastic_Learning_Implementation.pdf),
em 2026-09-19. O PDF permanece como fonte; este documento registra contratos e
dependências para implementação. As figuras das páginas 1, 2, 5, 8, 9, 12,
17 e 20 também foram conferidas visualmente. **Esta leitura não implementa aprendizado, adapters,
replay, experts plásticos ou execução GPU.** O estado e as evidências de cada
entrega pertencem ao [checklist central](BLUEPRINT_512MB_CHECKLIST.md).

## Escopo e estado real

O PDF separa parâmetros totais, ativos no forward e treináveis, e propõe duas
decisões distintas: onde executar e onde aprender. A sequência é evidência,
memória, candidato isolado, avaliação e publicação reversível; consolidação na
base é uma etapa posterior. Isso complementa
[Conditional Compute](NEXA_CONDITIONAL_COMPUTE_AJUSTES.md) e
[Omni](NEXA_OMNI_AJUSTES.md),
preservando os contratos de treinamento existentes.
[Fonte: páginas 1–2](NexaLang_Plastic_Learning_Implementation.pdf#page=1),
[pipeline: página 5](NexaLang_Plastic_Learning_Implementation.pdf#page=5).

A inspeção do código encontrou:

- [ModelIR](../compiler/model_ir.py) e [lowering](../compiler/model_lowering.py)
  de Transformer denso, com pesos constantes e operadores de forward. Não há
  neste pipeline `ExpertIR`, mapa de plasticidade, backward, optimizer,
  aplicação LoRA, Learning Gate ou replay. Um `DType`/tier declarado não cria
  o kernel correspondente.
- [NexaModelBundle](../runtime/nexapack/bundle.py) com matrizes Q4, normas F32,
  aliases tied, assets e proveniência. Esse formato não contém um contrato
  executável de adapters ou versões de aprendizado. O atual campo de
  proveniência não implementa validação de evidências.
- [Execução CPU](../runtime/nexapack/transformer.py), KV incremental e
  [tiers por idade](../runtime/nexapack/tiered.py), com recodificação
  transacional. O [backing store](../runtime/nexapack/kv_store.py) de páginas
  cold é privado e descartável; não é banco de evidências, replay, armazenamento
  de experts ou checkpoint persistente de aprendizado.
- Há funções simples de gradiente/SGD em [std/tensor.nxl](../std/tensor.nxl) e
  [treino MLX legado](../tools/train_mlx.py). O modelo MLX usa vocabulário 1024,
  LayerNorm/GELU, posições aprendidas e MHA; não é o trainer canônico R0.
  Esses componentes não comprovam backward de NexaLM, máscara por região,
  replay ou atualização governada. Ver os
  [ajustes de treinamento](NEXALM_TREINAMENTO_AJUSTES.md).

O frontend de modelos permanece independente de `nxc`/`nx.py`/`bootstrap/`.
O checkpoint CPU atual não demonstra Plastic Learning em modelo treinado ou
sob 512 MB de VRAM. Inferência e rollback de KV não estabelecem um protocolo
de alteração dos pesos de um bundle ativo. R0/v1 continuam com suas shapes,
vocabulário 32768 e quatro heads KV canônicos; não têm slots de experts
dormentes implicitamente disponíveis. Acrescentar um pool exige uma variante
arquitetural versionada, coordenada com CC, e seu próprio checkpoint.

## Numeração canônica e correspondência

O PDF não fornece tickets `PL-001` nem equivalentes. Usa fases **P0–P7** para
implementação e **T0–T5** para treinamento. Referências qualificadas como
`PL.P0` e `PL.T0` evitam colisões com outras trilhas. A figura da página 17
termina em P5, agrupando curiosidade e consolidação; a tabela da mesma página
separa P5, P6 e P7. **Adotar a tabela P0–P7**, preservando consolidação e Omni.
[Fonte: página 17](NexaLang_Plastic_Learning_Implementation.pdf#page=17).

Há outra divergência visual na página 5: suas setas passam por regressão,
update de expert, update de adapter e só então memória. Adotar a lista textual
§4.1: memória primeiro, candidato isolado em seguida, regressão depois do treino
e antes da publicação. Memória, adapter e expert são decisões possíveis do
router, não uma sequência obrigatória de três alterações. Os diagramas das
páginas 1, 2 e 20 são visões gerais; não especificam um plano executável nem
impedem que Plastic Experts participem do forward quando permitido.
[Fonte: página 5](NexaLang_Plastic_Learning_Implementation.pdf#page=5).

| Fase do PDF | Entrega exigida | Novos itens PL | Dependências reutilizadas |
|---|---|---|---|
| P0 | Regiões estáveis/plásticas compiladas e validadas | PL.01 | CC.01; LLM.01b2 para integração à linguagem |
| P1 | Learning Gate e evidence store, com aceitação/deferimento/rejeição | PL.02 | LLM.02a–d/10; OMNI.03/06 quando houver integração Omni |
| P2 | Adapters versionados, update reversível completo | PL.03–05 | LLM.03a–b/04b, M0.04, M1.10, OMNI.08 para assinatura/cache |
| P3 | Pool e ciclo de vida de Plastic Experts | PL.06 | CC.02–04; M3; M8.03; OMNI.05/07 |
| P4 | Localização por score com menor interferência | PL.07 | PL.05, CC.01–03; medidas de LLM.04b |
| P5 | Curiosidade e prioridade geram tarefas limitadas | PL.08 | PL.02/07; CC.09 e OMNI.04–06 para agendamento compartilhado |
| P6 | Consolidação, merge/distill e promoção | PL.09 | LLM.03/09, M6.01–03, M8.02–04, CC.10 |
| P7 | Brain/Organs compartilham transações | PL.10 | OMNI.01–07/09, CC.11; cache em OMNI.08 |
| DoD V1 | Ganho, recuperação por routing, rollback e orçamento comprovados | PL.11 | LLM.04b/06, M0.07, M2.06, M5, demais PL relevantes |

As referências `CC.01–CC.11` correspondem, em ordem, aos gates C0–C10 do plano
Conditional Compute. PL estende o contrato de região/expert e a governança;
não cria um segundo router de execução, scheduler, cache de experts ou pacote.
Provas numéricas sintéticas de adapters podem começar antes do treino R0;
o DoD de conhecimento adquirido depende de um checkpoint treinado identificável.

## Módulos e passagem do contrato até a execução

Os nomes seguintes vêm das páginas 13–14. Os caminhos são propostas do PDF,
não diretórios já implementados.

| Componente | Responsabilidade a implementar e limite |
|---|---|
| `ParameterRegion` / `ModelPlasticityConfig` | IDs estáveis, tensores/intervalos físicos, classe, plasticidade/maturidade, proteção, drift, máscara e política de consolidação/rollback. |
| `ExpertIR` | Uma identidade de expert compartilhada com CC; PL acrescenta lifecycle, maturidade e slots de adapters, sem duplicar o inventário. |
| `LearningPolicyIR` | Versões de confiança, tiers de evidência, suites de regressão, quotas de curiosidade e critérios de promoção. |
| `PlasticityMapPass` | Resolve regiões e aliases em armazenamento físico; rejeita sobreposição ou ambiguidade proibida. |
| `LearningTargetValidationPass` | Verifica alvo, proteção, máscara, versão-pai e alcance permitido antes de criar o candidato. |
| `AdapterLayoutPass` | Deriva shapes, precisão, composição, armazenamento e buffers temporários do adapter. |
| `ExpertResidencyPass` | Reutiliza residência/orçamento de CC/M3/Omni; não infere localização física a partir de maturidade. |
| `LearningTransactionInstrumentationPass` | Associa evidência, identidade de execução, candidato, métricas e eventos de commit/rollback. |
| `RegressionProbeInsertionPass` | Deriva probes e custo explícito; instrumentação não pode alterar silenciosamente a saída ou o orçamento. |
| `runtime/learning/gate` e `router` | Decide elegibilidade e `MEMORY_ONLY`, `ADAPTER`, `PLASTIC_EXPERT`, `MATURE_EXPERT`, `DEFER` ou `REJECT`; não publica pesos diretamente. |
| `runtime/learning/localizer` e `replay` | Mede afinidade/sensibilidade/interferência, mantendo amostras limitadas, rastreáveis e separadas da avaliação. |
| `runtime/learning/transactions` | Único publicador do estado de aprendizado; versões imutáveis, log, comparação, conflitos e recuperação de falhas. |
| `runtime/experts/lifecycle` e `learning/consolidator` | Transições de maturidade e consolidação verificadas; slots e crescimento sob quota. |
| `runtime/learning/curiosity` | Converte lacunas em tarefas priorizadas com custo, prazo e cancelamento; não define conhecimento como verdadeiro. |
| `std/ai/learning.nxl`, `nexa learn`, `nexa-eval` | Interfaces futuras sobre os mesmos contratos, sem um segundo caminho que contorne os gates. |

[Fonte: página 13](NexaLang_Plastic_Learning_Implementation.pdf#page=13),
[módulos e tooling: páginas 9 e 14](NexaLang_Plastic_Learning_Implementation.pdf#page=14).

`compiler/model_ir.py` hoje é um arquivo, enquanto o PDF sugere
`compiler/model_ir/plasticity.py`. A primeira extensão pode ficar em módulo
separado, por exemplo `compiler/model_plasticity.py`, com referência validada
no grafo; transformar o módulo existente em pacote exige migração explícita.
Schemas devem rejeitar booleanos em campos inteiros, NaN/Inf, limites inválidos,
referências inexistentes e máscaras incompatíveis. Configuração sem PL continua
com o comportamento denso atual. Suporte parcial deve ser rejeitado no executor,
não aceito e ignorado. A ponte para a linguagem permanece em LLM.01b2.

## Contratos que faltam no PDF

**Proteção e maturidade.** A tabela propõe Stable Core com plasticidade 0–0,05,
mas a base é imutável no aprendizado rápido. Para P0–P2, `protected=true`
prevalece sobre scores, LR, plasticidade e pedidos do router. Gradientes,
weight decay e estado do optimizer não podem atualizar tensores-base por vias
indiretas. Máscaras precisam tratar aliases tied como o mesmo tensor físico.
As faixas de plasticidade e o limiar `domain_coherence >= 0.90` são propostas;
faltam definição, calibração, janela temporal e evidência de utilidade.
[Páginas 3, 8 e 9](NexaLang_Plastic_Learning_Implementation.pdf#page=3).

**Três estados independentes.** `dormant/candidate/plastic/mature/stable`
descreve lifecycle; hot/warm/cold descreve residência, e candidato/confirmado
descreve uma transação. Um expert maduro pode estar cold. O PDF usa Dormant como
slot sem domínio estável e também exige reserva treinada, não pesos vazios:
fixar um template de inicialização treinado e seu hash, distinguindo capacidade
livre de instância criada. Reduzir plasticidade não torna uma região imutável
para sempre, mas reabri-la exige política própria versionada.
O desenho da página 8 ilustra plasticidade 1, aproximadamente 0,8, 0,5, 0,1
e 0 ao longo dos estados, com maturidade crescente; esses exemplos visuais
não definem defaults calibrados ou uma regra obrigatória de atualização.
[Páginas 2, 8 e 10](NexaLang_Plastic_Learning_Implementation.pdf#page=8).

**Memória e confiança.** `MEMORY_ONLY` não confirma a verdade de uma evidência.
Amostras de baixa confiança ficam identificadas como candidatas; recuperação
precisa conservar fonte e incerteza. Hash comprova identidade/integridade de
bytes, não confiabilidade. Evidências derivadas da própria saída não viram
fontes independentes por serem copiadas ou reingeridas. Reusar proveniência,
filtros, licença/escopo e separação de dados de LLM.02; versionar contraditórios,
expiração, deduplicação e motivo de decisão. `DEFER` precisa de política de
retentativa e quota, distinta de rejeição permanente.
LLM.10 já distingue telemetria de exemplos aprovados, com consentimento e
remoção de dados pessoais/segredos; reusar essa fronteira. Labels do Critic
ajudam a priorizar e avaliar candidatos, mas não são fontes de verdade.
[Páginas 5–6](NexaLang_Plastic_Learning_Implementation.pdf#page=5),
[riscos: página 18](NexaLang_Plastic_Learning_Implementation.pdf#page=18).

**Gate completo.** O pseudocódigo `govern` da página 6 omite verificações
explícitas de escopo, proteção, drift e compute presentes na tabela. A
implementação deve cumprir a tabela inteira, antes do treino quando possível
e novamente antes do commit. Rejeição de candidato ainda não publicado é
descarte; rollback de versão publicada é outra operação. A política aprendida
em T4 pode sugerir decisões, mas não desabilitar as restrições obrigatórias.

**Scores e metaplasticidade.** Os coeficientes 0,35/0,30/0,25/−0,10 e a soma
de `activation_score` não definem unidades, normalização nem limiares. Medir
gradientes custa backward, ausente no runtime de forward atual; replay também
não surge de um campo `interference_risk`. Começar por seleção determinística
entre adapters permitidos, registrar componentes ausentes e só acrescentar
scores após validar o instrumento. A recorrência de plasticidade precisa de
cadência, limites, regra para contradições e estabilidade; uso frequente não
é evidência suficiente de qualidade.
[Páginas 4, 7 e 10](NexaLang_Plastic_Learning_Implementation.pdf#page=7).

## Adapters, identidade dos pesos e KV

Para P2, especificar alvo físico, rank, escala, orientação das matrizes,
precisão e ordem de composição dos adapters. Uma opção inicial é calcular
`y = W_base x + (alpha/r) B(Ax)`, com `A[r, entrada]` e `B[saída, r]`.
Essa é uma proposta de contrato, ainda sem kernel no projeto. O caminho Q4
existente calcula a contribuição base; o delta exige novos operadores e
scratch planejado, sem materializar `B A` ou uma cópia densa completa de W.
Comparar adapter nulo e adapter não nulo com oracle independente; rejeitar
rank, shapes, alvo ou base incompatíveis antes de ler payloads.

A identidade de uma versão executável deve vincular hashes dos pesos-base,
grafo/configuração, tokenizer, região/expert, conjunto **ordenado** de adapters
e escalas, política de routing que altera o forward e versões dos layouts.
A transação também referencia evidência, política de aprendizado, dados de
treino/replay, avaliação e estado de retomada do trainer. Não confundir esses
dois níveis de identidade. Reusar Pack ABI/manifesto em M1.10/OMNI.01 e
`CacheSignature` em OMNI.08, em conjunto com CC.01/05/07.

Uma alteração em atenção, FFN, embedding ou adapter pode mudar o KV das camadas
afetadas e das dependentes. **Mesmo codec/layout e mesmas posições não provam
que KV antigo serve para pesos novos.** A estratégia inicial deve fixar uma
versão de pesos/adapters durante toda a sequência. Publicar um update habilita
novas sessões; trocar a versão de uma sessão existente exige reset e novo
prefill, ou um mecanismo de recomputação validado. Reuso parcial fica para uma
prova explícita de dependências. A regra também vale para rollback, troca de
expert e arquivos offloaded: a assinatura atual privada por sessão não resolve
compartilhamento de KV entre versões de aprendizado.

## Publicação transacional e recuperação

`LearningTransaction` contém no PDF ID, versão-pai, conhecimento, alvos, delta,
evidências, métricas antes/depois, drift, forgetting, timestamp e estado.
Acrescentar contratos de integridade, concorrência e recuperação antes de
implementar os comandos de `nexa learn`.
[Fonte: página 9](NexaLang_Plastic_Learning_Implementation.pdf#page=9).

1. Preparar candidato separado da versão ativa, com identidade completa,
   orçamento admitido, artefatos de treino e resultado dos gates congelados.
2. Revalidar `parent_version` no commit. Dois candidatos da mesma base não
   podem sobrescrever silenciosamente um ao outro; serializar inicialmente,
   rejeitando ou reavaliando um candidato obsoleto.
3. Persistir payloads e metadados verificáveis antes de publicar uma referência
   atômica à nova versão. Especificar a garantia de durabilidade por plataforma,
   recuperação após interrupção e tratamento de disco cheio.
4. Manter leitores em versões imutáveis. Só coletar artefatos quando nenhum
   leitor, descendente, janela de rollback ou checkpoint os referenciar.
5. Fazer rollback selecionando uma versão anterior intacta e registrando o
   evento, sem subtrair updates quantizados ou tentar “destreinar”. O estado
   de routing/adapters e a invalidação de KV acompanham a mudança.

O gate diferencia **identidade exata dos artefatos** de **equivalência numérica
da execução**. Bytes/hashes devem ser restaurados exatamente; logits usam
tolerância declarada para o mesmo ambiente e política determinística. Não
prometer reprodução bit a bit entre hardwares, ordens de redução e kernels
distintos. Se o treinamento continuar após rollback, recuperar também o estado
de optimizer/scheduler/RNG/sampler correspondente, conforme LLM.03b.

Os commits atuais de KV fornecem exemplos de ownership e descarte de staging,
mas não constituem esse protocolo persistente. Arquivos privados descartáveis
do cache não podem ser reutilizados como log durável de aprendizado.

## Treinamento, replay e critérios de avaliação

A primeira base não depende de aprendizado online para adquirir linguagem.
Reutilizar a trilha NexaData/tokenizer/trainer; não adicionar um segundo T0.
O trainer de referência pode usar PyTorch. A aplicação do adapter no runtime
CPU deve continuar independente dessa biblioteca.
[Treinamento T0–T5: página 15](NexaLang_Plastic_Learning_Implementation.pdf#page=15).

| Fase | Integração e evidência mínima |
|---|---|
| T0 Base | LLM.02/03/04b: base treinada, versão de dados/tokenizer, retomada e qualidade de referência. |
| T1 Expertization | CC.02–03, M8.03 e OMNI.07: especialização e routing úteis, com comparação densa e balanceamento. |
| T2 Plasticity training | PL.05/06: episódios A→B com máscara/adapters; somente alvos permitidos mudam. |
| T3 Anti-forgetting | PL.05: replay limitado e amostragem reproduzível, distillation/regularização comparadas por ablation. |
| T4 Governance | PL.02/05: candidatos ruins rejeitados; rotulagem de regressão sem contaminar a avaliação final nem contornar gates fixos. |
| T5 Consolidation | PL.09: promoção/merge com avaliação completa, novo baseline versionado e rollback. |

Replay precisa de limites em bytes, tokens e exemplos, política de retenção,
amostragem por domínio, deduplicação e retomada. Amostras usadas para replay ou
localização não podem servir como prova final independente do mesmo update.
Guardar conjuntos congelados para alvo, competências antigas e domínios não
alvo, com horizonte de múltiplos updates. Melhorar um lote memorizado não prova
generalização ou retenção.

Os gates da página 16 são qualitativos. Cada experimento deve fixar antes do
treino: métricas/unidades, tolerância, tamanho da amostra, seeds/repetições,
intervalo de incerteza e limite por domínio. Avaliar ganho alvo, forgetting
contra a base e versões anteriores, efeitos fora do alvo, uso efetivo nas
rotas futuras, latência, memória e drift. Localidade de parâmetros não garante
localidade de comportamento. Drift deve identificar a referência e medir
tanto deslocamento efetivo quanto acúmulo de alterações, sem permitir que
cancelamentos numéricos ocultem uma sequência instável de updates.
[Fonte: página 16](NexaLang_Plastic_Learning_Implementation.pdf#page=16).

## Memória e o requisito de 512 MB

A figura da página 12 contém valores **220, 90, 25, 85, 60 e 32 MB**, somando
512 MB, mas não nomeia as categorias nem fornece legenda. Não atribuir uma
barra a pesos, KV, optimizer ou reserva por suposição. É uma proposta visual,
não um orçamento executável: faltam simultaneidade, unidades precisas, buffers,
alinhamento e overhead. A tabela de residência também não fixa bytes.
[Fonte: página 12](NexaLang_Plastic_Learning_Implementation.pdf#page=12).

O plano anterior trata 512 MB como requisito de inferência e permite treino
em hardware maior. Este PDF permite treinamento de adapters na CPU/RAM, mas
seu DoD exige update sob limite artificial de 512 MB de VRAM. Adotar duas provas:
treino CPU com orçamento host explícito; posteriormente, experimento com GPU
e teto declarado, incluindo inferência concorrente se ela existir. O gate
artificial deve ser identificado como tal e não comprova desempenho de uma
placa física com 512 MB. A prova GPU reutiliza M0.07/M2.06/M5/LLM.06.

Um adapter tem `r * (entrada + saída)` parâmetros; isso não é o custo total de
aprendizado. Por exemplo, parâmetros, gradientes e dois momentos Adam em F32
já requerem 16 bytes por parâmetro, sem ativações, checkpointing, staging,
cópias de publicação ou base. Treinar adapters em várias camadas ainda pode
precisar propagar gradientes pelas camadas congeladas. Contabilizar estado
mestre e precisão efetivamente usados, evitando contar componentes ausentes
ou omitir cópias de compute.

O planner deve somar versão ativa/candidata, replay, artefatos de validação,
gradientes/optimizer, ativações, KV, cópias/slots e alinhamento nos lifetimes
reais. Reservar compute, RAM, VRAM, disco e retenção de versões separadamente;
pausar ou deferir antes de estourar um orçamento. CPU gerenciada não equivale
a RSS; cache do SO e memória do driver continuam identificados separadamente.

## Riscos concretos e aceites

| Risco ou lacuna | Critério verificável antes de avançar |
|---|---|
| Poisoning e autoevidência | Evidência sem origem, duplicada ou gerada pelo próprio modelo não satisfaz diversidade/confiança; decisão e motivos registrados. |
| Alteração da base protegida | Hashes dos tensores-base e aliases permanecem iguais após treino, falha e commit de adapter; casos de máscara/weight decay testados. |
| Candidato validado contra versão errada | Commit de pai obsoleto rejeitado ou reavaliado; testes de duas propostas concorrentes. |
| KV incompatível após update/rollback | Sessões antigas permanecem na versão anterior; troca explícita invalida/recalcula KV, inclusive páginas em disco. |
| Forgetting gradual | Episódios A→B e sequências longas medem tarefas anteriores e domínios fora do alvo; candidato regressivo rejeitado. |
| Publicação interrompida | Falhas em treino, validação, escrita, fsync/publicação e coleta deixam uma versão ativa consistente; recuperação e rollback testados. |
| Growth sem limite | Quotas para adapters, experts, replay, evidências, versões e disco; coleta respeita leitores e janela de rollback. |
| Expert collapse | Utilização/coerência medidas contra baseline; frequência não substitui qualidade nem diversidade. |
| Consolidação irreversível | Novo master/base separado; quantização e ordem de merge explícitas; artefato anterior preservado e comparação antes/depois. |
| Curiosidade sem limite | Tarefas com custo máximo, prioridades, cancelamento e fontes permitidas; nenhuma pesquisa/coleta infinita disparada por incerteza. |

Os riscos de forgetting, poisoning, autorreforço, drift, collapse, crescimento,
rollback e consumo ilimitado são explicitados pelo PDF; os testes acima
transformam essas intenções em critérios de engenharia.
[Fonte: página 18](NexaLang_Plastic_Learning_Implementation.pdf#page=18).

Consolidar adapters por soma, requantização e destilação são operações
diferentes. Merge em Q4 não recupera um master de alta precisão e pode adicionar
erro; ordem e rank também importam. Validar cada política em novo candidato,
preservando a base anterior. Promoção de maturidade e promoção de residência
de M4.05c não devem compartilhar significado ou concluir uma à outra.

## Sequência proposta e retomada

1. **PL.01/P0:** schema e validação de regiões/políticas, com execução densa
   preservada e identidade compartilhada com CC.01. Fixtures locais bastam
   para provar serialização, limites e rejeição de alvos inválidos.
2. **PL.02/P1:** evidence store e gate determinístico sobre fixtures,
   reutilizando LLM.02. Provar decisões e quotas sem atualizar parâmetros.
3. **PL.03/P2:** adapter CPU mínimo, pacote e orçamento; validar forward contra
   oracle com base congelada. Isso prova aplicação de deltas, não aprendizagem.
4. **PL.04/P2:** transações, versões-pai, publicação, rollback e política de
   sessões/KV; testar falhas antes de ligar um trainer.
5. **PL.05/P2/T2–T4:** trainer pequeno, replay e avaliação A→B, com
   LLM.03a–b/04b; evidência de ganho e rejeição de regressões. O P2 completo
   exige o ciclo de aprendizado reversível, não apenas trocar um arquivo.
6. **PL.06–08/P3–P5:** experts plásticos, localização instrumentada e
   curiosidade, nessa ordem de dependência e sobre CC já validado.
7. **PL.09–10/P6–P7:** consolidação e compartilhamento de transações no Omni;
   manter um único contrato de autoria/publicação e isolamento de escopo.
8. **PL.11/DoD V1:** fechar a matriz de ganho/retenção/routing/rollback e
   recursos em checkpoint treinado e ambiente declarado, incluindo a prova
   GPU requerida. Não encerrar M5, LLM.06 ou Omni por simulação CPU.

O próximo marco recomendado pelo PDF é P0–P2 no R0. A adaptação acima começa
pelos contratos e fixtures menores para permitir validação antes de escalar;
não muda automaticamente a próxima tarefa CPU registrada no checklist.
[Síntese e recomendação: página 20](NexaLang_Plastic_Learning_Implementation.pdf#page=20).

## Referências do PDF e alcance desta análise

A página 19 cita US 12,705,547; US 20260187477; US 20240220809; e US 12,530,541
como referências inspiradoras. Nesta tarefa elas foram lidas **como alegações
do documento local**, sem consulta externa, validação dos números/títulos ou
análise de claims. Nenhum benefício técnico, licença ou conclusão jurídica é
inferido dessa bibliografia. A implementação futura precisa de contratos e
medições próprios; o presente registro não encerra a revisão de referências
prevista pelo projeto.
[Fonte: página 19](NexaLang_Plastic_Learning_Implementation.pdf#page=19).

Não houve coleta, treino, alteração de pesos, execução de updates ou mudança
de código durante esta análise documental. Os itens PL permanecem pendentes
até suas respectivas implementações e evidências.
