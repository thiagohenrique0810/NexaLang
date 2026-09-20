# Checklist de desenvolvimento — NexaLang 512 MB

Atualizado em 2026-09-19. Fonte: [Blueprint v0.1](NexaLang_Blueprint_Tecnico_512MB.pdf).
Leia também [os ajustes de engenharia](BLUEPRINT_512MB_AJUSTES.md), o
[plano da primeira LLM](NexaLang_Plano_Implementacao_Primeira_LLM.pdf) e o
[plano Nexa Omni](NexaLang_Plano_Implementacao_Nexa_Omni.pdf), acompanhado dos
[ajustes de integração Omni](NEXA_OMNI_AJUSTES.md), e o
[plano de treinamento](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf), com
[ajustes e correspondência de gates](NEXALM_TREINAMENTO_AJUSTES.md), e os novos
planos de [Conditional Compute](NexaLang_Conditional_Compute_Implementation.pdf)
e [Plastic Learning](NexaLang_Plastic_Learning_Implementation.pdf).
O [índice de implementação](PLANOS_IMPLEMENTACAO_INDICE.md) reúne os seis PDFs,
páginas/seções, módulos, dependências e ajustes a consultar antes de cada tarefa.

## Checkpoint para retomada

**Terceiro incremento CPU implementado e validado:** grafo Transformer
completo, lifetimes derivados, kernels nativos e execução de prefill/decode por IDs
estão integrados aos pacotes Q4. Esse executor permanece como baseline por
recomputação, disponível por padrão na CLI.
M0/M1 ainda têm itens pendentes (GPU, outros codecs e demais importadores);
não são fases encerradas.

**Quarto incremento implementado e validado:** M4.00, KV F32 incremental
com dois bancos, plano explícito de escrita/atenção/commit, RoPE com offset e CLI
`--kv-cache`. Prefill em chunks limita ativações e logits temporários; decode
processa somente o novo ID. Passaram 267 regressões e 110 testes bootstrap, incluindo
equivalência, orçamento, rollback e liberação de buffers. O PDF Nexa Omni foi analisado integralmente;
seus gates foram incorporados abaixo e continuam pendentes de implementação.

**Quinto incremento implementado e validado:** M4.01, páginas KV F32 sob
demanda com plano de segmentos, atenção nativa direta e CLI `--kv-page-tokens`.
Prefill substitui páginas de forma transacional; append preserva endereços e
prefixos. Reset/close liberam todas as páginas. Reserva de capacidade, residência
e pico durante a transação são contabilizados separadamente. Passaram 297 regressões
e 110 testes bootstrap, sem falhas ou skips na suíte completa local.

**Sexto incremento implementado e validado:** KV Q4 paginado por
token/head, páginas parciais já comprimidas e atenção CPU direta sobre códigos e
escalas. A CLI recebe `--kv-codec q4 --kv-group-size G`. O relatório separa erro
de execução, dos pesos, do KV e combinado; quantização não é tratada como exata.
Naquele checkpoint, Q3/TQ01, políticas hot/warm/cold e o gate M4 completo
permaneciam pendentes; Q3 foi implementado no incremento seguinte.
Passaram 324 regressões e 110 testes bootstrap, sem falhas ou skips na suíte completa.

**Sétimo incremento implementado e validado:** KV Q3 paginado, formato Q3_GROUPED V1 por token/head,
quantizador e atenção C direta, CLI `--kv-codec q3`, oracle independente e golden
sem Torch. F32/Q4 preservam seus contratos. Comparação com a mesma capacidade:
64 B/token K/V em Q3 contra 80 B Q4 e 512 B F32, na fixture D64/P16/G32.
Passaram 350 regressões e 110 testes bootstrap, sem falhas ou skips na suíte
completa. Q3 nos pesos, TQ01,
tiers, qualidade de modelo treinado e GPU permanecem pendentes.
O novo plano de treinamento foi analisado e incorporado aos itens LLM existentes,
com ajustes de numeração e subtarefas. Naquele checkpoint havia 24 itens concluídos
e 86 pendentes; a contagem inclui o detalhamento, sem equivaler a percentual ou prazo.

**Oitavo incremento implementado e validado:** M1.07, contexto TurboQuant exclusivo MSE com estado linear,
sem matriz QJL ou alocações Prod. `tq_create` preserva a API e os resultados
legados; `tq_create_mse` é explícito e rejeita Prod sem alocar/escrever. Contabilidade
de memória persistente, ownership e scratch documentados. Consumidores somente
MSE da linguagem e wrapper usam o estado menor. Passaram 364 regressões e 110
testes bootstrap, sem falhas ou skips na suíte completa. Naquele checkpoint havia
**25 itens concluídos e 85 pendentes**. Evidências de validação abaixo.

**Nono incremento implementado e validado:** M1.06, codec TQ_MSE_SRHT V1 no
NexaPack, registros TQ02 little-endian e codebook explícito. Conversão por vetor,
leitura parcial, checksums e migração TQ01 com endian/codebook de origem preservam
Q4 e as APIs legadas. Kernels sem heap e wrapper com orçamento, parâmetros
imutáveis e liberação em falhas. Passaram 396 regressões e 110 testes bootstrap,
sem falhas ou skips. Naquele checkpoint havia **26 concluídos e 84 pendentes**.
Armazenamento TQ ficou disponível; atenção TQ foi integrada no incremento seguinte.

**Décimo incremento implementado e validado:** TQ02 no KV paginado CPU,
quantização por token/head e atenção nativa com reconstrução de um head por vez.
Contexto MSE compartilhado, scratch e acumulador entram no orçamento; páginas
parciais, prefixo imutável, rollback e liberação preservam os contratos anteriores.
CLI `--kv-codec tq --kv-bits 3 --kv-seed 42`; pesos continuam Q4.
Passaram 437 regressões e 110 testes bootstrap, sem falhas ou skips.
M4.02c/M4.03c agora identificam TQ concluído; tiers e demais codecs foram
preservados nos novos M4.02d/M4.03d. Naquele checkpoint: **28 concluídos e 84 pendentes**;
a divisão aumenta a contagem total, sem representar percentual ou prazo.

**Décimo primeiro incremento implementado e validado:** política CPU por idade
de página, hot F32/warm Q4/cold Q3, atenção mista e recodificação transacional.
Páginas de origem permanecem válidas até o commit; falhas descartam os destinos.
Arena, páginas antigas/novas, tabelas e scratch entram no orçamento global.
CLI `--kv-policy age --kv-hot-pages 1 --kv-warm-pages 1`. Todos os tiers ficam
em RAM; promoção, evicção/recarga e TQ misto continuam pendentes.
Passaram **486 regressões + 110 testes bootstrap = 596 testes**, sem falhas ou
skips. M4.02d e a parte de recodificação CPU M4.05a concluídos; o restante de
M4.05 está preservado em M4.05b. Checklist: **30 concluídos e 83 pendentes**.

**Décimo segundo incremento implementado e validado:** backing store privado de páginas cold
Q3 CPU, publicação atômica, checksums e atenção causal completa com um slot de
recarga. Bytes packed e logits foram iguais aos tiers residentes nas fixtures
testadas. Falhas de leitura/escrita/corrupção/cancelamento preservam o estado
confirmado e liberam buffers temporários. CLI `--kv-backing-store DIRETÓRIO`.
M4.05b passa a identificar evicção/recarga CPU; promoção e critérios de qualidade
foram preservados em M4.05c. Checklist: **31 concluídos e 83 pendentes**; a divisão
acrescenta um item, sem equivaler a percentual de conclusão ou prazo.
Passaram **533 regressões + 110 testes bootstrap = 643 testes**, sem falhas ou skips.
Resultados e comandos do incremento estão no registro ao final deste documento.

**Atualização documental de 2026-09-19:** os seis PDFs de `docs/` (108 páginas)
foram conferidos. Conditional Compute (17 páginas) e Plastic Learning (20 páginas)
foram incorporados em tarefas novas, hoje **CC.01–13 e PL.01–11**, com fontes por
página, dependências e critérios de aceite. Os quatro planos anteriores mantêm
suas trilhas, com complementos de contrato e referências corrigidas. Consulte o
[índice por módulo](PLANOS_IMPLEMENTACAO_INDICE.md) e os guias de
[Conditional Compute](NEXA_CONDITIONAL_COMPUTE_AJUSTES.md) e
[Plastic Learning](NEXA_PLASTIC_LEARNING_AJUSTES.md).
Estado atual: **31 concluídos e 105 pendentes, 136 itens**. O aumento vem do
detalhamento do escopo, sem alterar entregas já comprovadas ou indicar percentual
de esforço. Esta revisão não implementa CC/PL e não muda a evidência executável
do décimo segundo incremento. Próxima tarefa CPU permanece M4.05c; CC.01 e
PL.01/02 têm contratos/fixtures que podem avançar em paralelo conforme dependências.

**Décimo terceiro incremento implementado e validado:** cache de recarga com
`--kv-reload-slots N`, admissão por primeiro toque e substituição do slot mais
recente. Páginas cold Q3 passam a ser reutilizadas entre as duas passagens da
atenção, entre camadas e entre chamadas da sessão; os bytes publicados, a ordem
de redução e os logits são idênticos aos de um slot. Falha, cancelamento, reset e
close liberam todos os slots; após o commit só permanecem entradas de páginas
confirmadas. Na fixture D64 de 512 tokens, 256 slots reduziram 64.260 recargas e
43,6 MB lidos para 253 recargas e 171.875 B, elevando o pico gerenciado de
91.899 B para 140.031 B: é troca explícita de I/O por RAM, não ganho automático.
Com um slot e mais de uma página cold não há retenção, e esse caso permanece o
padrão. Passaram **547 regressões + 110 testes bootstrap = 657 testes**, sem
falhas ou skips. M4.05c passa a identificar reuso/promoção de residência; o
restante ficou em M4.05d. Checklist: **32 concluídos e 107 pendentes**.

**Décimo quarto incremento implementado e validado:** sequências derivadas com
prefixo compartilhado no executor paginado homogêneo. `fork()` retém as páginas
completas do prefixo — imutáveis, com contagem de referências — e copia apenas a
página parcial, de modo que cada sequência mantém escrita exclusiva sobre a
própria cauda. Uma escrita que alcance página compartilhada é rejeitada antes de
tocar bytes. Reset, prefill substituto e close de qualquer sequência, em qualquer
ordem, preservam as demais. Com prompt de 256 tokens e dois ramos, o KV residente
somado caiu de 45.662 B para 24.174 B e a amostra local de ~0,63 s para ~0,04 s,
com logits idênticos aos de sequências independentes. Tiers por idade e backing
store recusam `fork` e ficaram em M4.06b. Passaram **558 regressões + 110 testes
bootstrap = 668 testes**, sem falhas ou skips. Checklist: **33 concluídos e 107
pendentes**.

**Décimo quinto incremento implementado e validado:** sequências derivadas sob a
política de idade. A derivada herda os descritores do prefixo e, como a
recodificação cria uma página nova e libera a origem, migrar é privado à
sequência que migra: a página compartilhada continua válida e inalterada para as
demais. A validação de layout passou a comparar a política inteira — hot/warm e
`group_size` não alteram o layout F32, mas decidem como uma página Q4/Q3 herdada
é lida, e a comparação anterior aceitaria uma derivada incompatível. O backing
store continua recusando `fork` e ficou em M4.06c, junto de cancelamento em
andamento e admissão conjunta. Passaram **564 regressões + 110 testes bootstrap =
674 testes**, sem falhas ou skips. Checklist: **34 concluídos e 107 pendentes**.

**Décimo sexto incremento implementado e validado:** NexaTokenizer V1, BPE
byte-level determinístico, e execução a partir de texto. Os 256 primeiros IDs
são bytes, então não há token desconhecido e o round-trip é exato; os especiais
ocupam IDs próprios e **nunca** são produzidos por texto, de modo que um prompt
não confiável não forja um papel de diálogo. A segmentação separa dígitos,
espaços e classes de caractere, e nenhum merge cruza essa fronteira. O treino é
reproduzível — empate resolvido pelos bytes do par, identidade do corpus
independente da ordem dos documentos — e o asset é verificado por SHA-256 no
carregamento. `nexa_run.py --prompt ... --tokenizer DIR` executa prefill/decode
nativos e devolve o texto gerado, exigindo `vocab_size` igual ao do modelo.
Passaram **583 regressões + 110 testes bootstrap = 693 testes**, sem falhas ou
skips. Falta congelar 32768 com corpus real (LLM.02c2). Checklist: **35
concluídos e 107 pendentes**.

**Décimo sétimo incremento implementado e validado:** segundo codec de peso e
despacho por tensor. Até aqui toda matriz era Q4 e o executor só chamava
`nexa_q4_matmul`; agora um tensor pode ser guardado em `RAW_F32_MATRIX`, com
blocos de linhas que são ao mesmo tempo unidade de checksum e de leitura, e um
bundle pode misturar os dois codecs. `nexa_f32_matmul` mantém a ordem de redução
e o acumulador do kernel Q4, então a diferença entre os caminhos é apenas a
quantização: com pesos que Q4 representa exatamente, os logits e o SHA-256 são
idênticos. `--dense-all`/`--dense-tensor` na conversão e verificação bloco a
bloco em `nexa_inspect`. Isso não é formato de distribuição — custa oito vezes a
forma Q4 — mas é a referência exata que faltava e o executor que M6.01/M6.02
precisavam para consumir um precision map. Passaram **595 regressões + 110
testes bootstrap = 705 testes**, sem falhas ou skips. Checklist: **36 concluídos
e 107 pendentes**.

Objetivo completo: modelo importado maior que a VRAM, execução sem PyTorch, pesos
streamados sem expansão integral, KV comprimido e pico de dispositivo comprovado
dentro de 512 MB. Esse objetivo permanece pendente até o gate M5.

Ao retomar:

1. Leia este checkpoint, `BLUEPRINT_512MB_AJUSTES.md` e o
   [índice dos PDFs](PLANOS_IMPLEMENTACAO_INDICE.md); veja `git status --short`.
2. Não descarte mudanças anteriores: há trabalho de estabilização da linguagem no
   mesmo diretório, anterior a este blueprint.
3. Execute os testes do marco tocado e registre os resultados e limitações.
4. Faça a próxima tarefa pendente em ordem de dependência, sem refazer itens concluídos.
   Consulte o PDF/páginas e o guia de ajustes vinculados ao ID escolhido; ao parar,
   registre subtarefa, arquivos, evidências, decisão pendente e próximo comando.
5. Atualize este checkpoint, os comandos reais, os arquivos e o próximo passo.

**Próxima tarefa: M6.01 — calibração de sensibilidade por tensor.** Com o
despacho por codec e a referência densa, dá para medir o efeito real de
quantizar cada tensor: executar o modelo com todos os pesos densos, depois com
um único tensor em Q4, e comparar logits. O custo é O(tensores) execuções;
registrar isso e amostrar quando necessário. Perplexidade de modelo treinado
continua fora do alcance até LLM.04b.

**Também pendente: M5.01 com um modelo real.** Com o tokenizer pronto, falta
importar um modelo de 250–500M por Safetensors (M1.08 já suporta Llama), fixar
revisão/hash em M0.08 e medir prefill/decode e qualidade sem PyTorch. Sem
modelo baixado no repositório: registrar origem, revisão e hashes.

**Também pendente: M4.06c — sequências derivadas sob backing store, cancelamento
e admissão conjunta.** Páginas cold são arquivos de um store privado que os
remove ao fechar; compartilhá-las exige propriedade por referência sobre os
arquivos, ou uma cópia explícita, antes de permitir `fork`. Cancelar uma chamada
em andamento exige executá-la fora da thread que cancela; hoje só há rollback de
falhas e interrupções. Admissão conjunta: cada sessão admite o próprio orçamento,
e o compartilhamento reduz residência real sem reduzir a reserva.

**Também pendente: critérios de qualidade e importância por página, M4.05d.**
Depende de qualidade medida em checkpoint treinado (LLM.04b) e da calibração
M6.01, porque selecionar páginas por importância altera logits.
O número de slots é hoje um parâmetro do operador, sem política que decida
quantos admitir nem quais páginas priorizar. Definir importância por página e
budgets por camada exige a calibração de M6.01 e comparação contra idade
uniforme; medir também um cenário sem ganho. Promoção de **precisão** continua
aberta e não se confunde com residência: converter Q3 para F32 não recupera o
original. Prefetch, leitura assíncrona, TQ misto, checkpoint treinado, pesos TQ,
GPU e o gate M4 completo continuam abertos. NexaData/tokenizer e contratos Omni
F0 podem avançar em paralelo.

Use os guias de [importação](NEXALM_IMPORTACAO.md) e
[execução CPU](NEXALM_EXECUCAO_CPU.md), [KV incremental](NEXALM_KV_CPU.md) e
[KV paginado](NEXALM_KV_PAGINADO_CPU.md), [KV Q4](NEXALM_KV_Q4_CPU.md) e
[KV Q3](NEXALM_KV_Q3_CPU.md), [KV TQ](NEXALM_KV_TQ_CPU.md) e
[tiers CPU](NEXALM_KV_TIERS_CPU.md),
[backing store CPU](NEXALM_KV_BACKING_CPU.md) e
[reuso de páginas cold](NEXALM_KV_RELOAD_CPU.md) e
[sequências derivadas](NEXALM_KV_SEQUENCIAS_CPU.md) para
reproduzir os incrementos atuais.
O [guia TurboQuant MSE](TURBOQUANT_MSE_CPU.md) descreve o estado menor e os contratos
de compatibilidade; o [guia TQ portátil](NEXAPACK_TQ_V1.md) especifica o codec
persistido, memória, conversão e migração que sustentam o próximo incremento.

Dependências futuras: codecs KV paginados → KernelIR e um backend GPU →
streaming/KV comprimido integrado → prova de modelo treinado completo. A escolha do backend
precisa de detecção e benchmark no dispositivo disponível.

## Complemento: primeira LLM (plano de 22 páginas)

O novo PDF define R0 (16 layers, hidden768, query12/KV4, FFN2048, vocab32768,
embedding tied) e v1 (32 layers, hidden1024, query16/KV4, FFN2816, mesmo vocab).
O treinamento pode usar hardware maior; 512 MB continua sendo requisito de
inferência. A ordem abaixo complementa M0–M9; não duplica uma implementação.

- [x] LLM.01a MODEL-001/002 estrutural: configuração R0/v1, contagem exata,
  fonte architecture.nxl e lowering para configuração/shapes/aliases validados.
- [x] LLM.01b1 MODEL-002: grafo de prefill completo em ModelIR, validação de ops,
  lowering R0/v1 e executor CPU consumindo grafo e plano, pesos Q4/normas F32.
- [ ] LLM.01b2 G0 restante: migrar TurboIR legado, integrar frontend da linguagem
  e fechar os contratos completos do plano; pipeline experimental ainda separado de nxc.
- [ ] LLM.02a TRAIN-001: schemas NexaData de documento/snapshot/proveniência/licença,
  allowlist/denylist e bloqueio de shards não comerciais em produto; hashes e
  filtros versionados, quotas e retomada determinística por fonte.
- [ ] LLM.02b TRAIN-001: filtros de qualidade/PII/segredos/spam, idioma/domínio,
  dedup exato/aproximado, split antes
  do packing e decontaminação; mistura PT/EN/código/matemática e repetições medidas.
- [x] LLM.02c1 TRAIN-001: tokenizer BPE byte-level determinístico, asset
  versionado com SHA-256, IDs especiais que texto não produz, segmentação
  declarada, round-trip byte-exato, métricas por domínio e execução a partir
  de texto no runner.
- [ ] LLM.02c2 TRAIN-001: congelar 32768 com o corpus real da mistura, medir
  eficiência por idioma e linguagem de programação, registrar hash do corpus de
  treino do tokenizer e a estratégia de migração antes do modelo base.
- [ ] LLM.02d TRAIN-001: contrato .nxd versionado, shards/checksums/publicação
  atômica e reader/dataloader com memória limitada, máscaras e retomada verificadas.
- [ ] LLM.03a TRAIN-002: trainer PyTorch consome configuração R0, prova pequena de
  forward/backward, labels/máscaras/tied; receita AdamW, precisão e accumulation.
- [ ] LLM.03b TRAIN-002: treino R0 com orçamento, optimizer/scheduler/RNG/sampler/
  posição de leitura no checkpoint, retomada e exportação Safetensors reproduzíveis.
- [x] LLM.04a QA-001 sintético: logits nativos de uma/duas camadas vs PyTorch,
  prefill/decode por recomputação vs cache incremental do oracle, fixture golden sem
  Torch e erro de quantização separado do erro de execução.
- [ ] LLM.04b G1 restante: corpus/tokenizer congelados e checkpoint treinado,
  logits e qualidade de referência; modelos sintéticos não concluem esse gate.
- [ ] LLM.05 G2/G3: Q4 e Q3 no caminho de modelo, com tolerâncias e ganho físico.
- [ ] LLM.06 G4/G5: teto real de GPU e streaming com overlap medido.
- [ ] LLM.07 G6: GQA baseline, KV paginado e Q4/Q3 integrado à atenção.
- [ ] LLM.08 G7: fusões com ganho demonstrado e tuning no hardware; seleção offline
  por memória/transferência/latência medidas antes de busca ou loss de custo.
- [ ] LLM.09a G8: contrato NAQT BF16→Q8→Q4→mixed, avanço por tokens/steps,
  sensitivity map e fake quantization fiel aos formatos implantados.
- [ ] LLM.09b G8: executar curriculum e comparar QAT/PTQ com base/dados/contexto
  iguais; qualidade, memória e desempenho medidos para pesos/ativações/KV.
- [ ] LLM.10 Release v1: treino principal somente após gates, quality PT/EN e
  contexto 2k→4k/8k, checkpoint master auditável e inferência sem PyTorch;
  Nexa-Instruct/SFT com schema verified, sintéticos verificados e promoção de
  telemetria para exemplos aprovados com consentimento e remoção de PII/segredos.
- [ ] LLM.11 Pesquisa v2/v3: state/attention híbrido e sparse MoE depois da v1;
  a trilha CC detalha o motor condicional sem alterar silenciosamente R0/v1.

Os tickets PACK-001/002, KERNEL-001, MEM-001 e BENCH-001 reutilizam o trabalho
M0/M1 existente. Bench de matriz não mede tokens/s/TTFT e cap de buffers CPU
não satisfaz G4. Na sintaxe dos exemplos novos, usar `512MB`/`512MiB`; `512M` é
ambíguo e continua rejeitado. Uma declaração de arquitetura não inicia treino.

O plano de treinamento de 16 páginas detalha essa mesma trilha, sem criar uma
segunda implementação. Seus gates usam o namespace `TRAIN.G0–G8` para evitar
colisão com o plano anterior. LLM.02/03/09 foram divididos em subitens acima;
nenhum está concluído por análise documental. As ambiguidades de smoke, mistura,
vocabulário/heads, precisão e exportação estão nos
[ajustes de treinamento](NEXALM_TREINAMENTO_AJUSTES.md).

## Convenção e gates

`[x]` = implementado e verificado; `[ ]` = pendente, inclusive trabalho parcial.
Nenhum checkbox de GPU é concluído por uma simulação CPU. Um item genérico só fica
concluído quando todos os comportamentos descritos funcionam. Registrar skips.

## Complemento: Nexa Omni (plano de 14 páginas)

A numeração F0–F7 abaixo segue a tabela de §20 do PDF, que inclui Critic; o diagrama
da página 12 omite essa etapa. F1 não conclui sozinho o DoD de V0 da página 13.
Os dez sprints não têm duração definida; estes itens não são uma estimativa de prazo.

- [ ] OMNI.01 F0: Pack ABI versionada, manifesto Model Pack, capabilities,
  variantes, shapes, proveniência, integridade/licença e inspeção sem carregar pesos;
  testar negação de capabilities não concedidas e FFI controlada (PDF Omni p.10, §16).
- [ ] OMNI.02 F0: nxpkg como gerenciador único, instalação local de packs/models,
  resolução/lock, comandos model/doctor/resolve/optimize e erros reproduzíveis.
- [ ] OMNI.03 F1: CognitivePacket tipado, MemoryRef/ArtifactRef, schemas,
  protocolo/ACL, Registry e contratos de Router/Brain/experts/Integrator.
- [ ] OMNI.04 F1: Router seletivo, Brain e executor DAG com três modelos externos,
  rastros, isolamento entre packs e tratamento de falhas; fixtures locais antes da integração real.
- [ ] OMNI.05 F2: memória compartilhada, Scheduler Hot/Warm/Cold, reserva,
  admissão/evicção e budget conjunto; perfil 512 MB com um expert físico por vez.
- [ ] OMNI.06 F3: Critic, verificação, confiança/escalada e telemetria de
  qualidade/latência/memória; fechar DoD V0 incluindo integração Silicon.
- [ ] OMNI.07 F4: Brain e um expert Nexa nativos, pacotes e inferência sem PyTorch,
  com treino/exportação e qualidade medidos.
- [ ] OMNI.08 F5: CacheSignature e reuso seguro de contexto/prefix/KV entre modelos,
  invalidação por pesos/adapters/posições/layout, política/histórico relevante de
  rotas condicionais e versões de aprendizado; ganho sem regressão de qualidade.
- [ ] OMNI.09 F6: roteamento aprendido por qualidade/custo e ablations do Omni V2.
- [ ] OMNI.10 F7: Neural Bus latente entre dois modelos compatíveis, protocolo,
  alinhamento/treino e prova de qualidade; pesquisa posterior ao caminho tipado.

Sandbox/capabilities de FFI, integridade e rastros devem acompanhar os componentes
que introduzem execução ou dados compartilhados. O KV privado de M4.00 não conclui
OMNI.08. R0/v1 não definem automaticamente a arquitetura de Brain/experts.

## Complemento: Conditional Compute (17 páginas)

Fonte: [PDF Conditional Compute](NexaLang_Conditional_Compute_Implementation.pdf).
Consultar os [ajustes e aceites detalhados](NEXA_CONDITIONAL_COMPUTE_AJUSTES.md)
antes de cada módulo. A tabela das pp.14–15 define **CC.C0–C10**; os IDs CC.xx
abaixo detalham essas entregas, reutilizando M/LLM/OMNI. CC.12 e CC.13 cobrem as
seções §12 e §13, que atravessam várias linhas dessa tabela e não têm gate próprio:
perdas/observabilidade de router e a ordem de execução completa. Nenhum item está
implementado pela análise. R0/v1 densos permanecem canônicos; variantes condicionais exigem
configuração, pesos/treino, referência e versão próprios.

- [ ] CC.01 C0: schemas de ConditionalBlock/ExpertSpec/rotas, qualidade e budget
  mínimo; instrumentação por camada, custos estimados/medidos separados e baseline
  preservado. Reusa M0/M2.02/LLM.04; admissão antecede os routers, mesmo com C8 posterior.
  Fonte: [§§3–4, pp.4–6](NexaLang_Conditional_Compute_Implementation.pdf#page=4), [§15, p.14](NexaLang_Conditional_Compute_Implementation.pdf#page=14).
- [ ] CC.02 C1: MoE Top-2 estático com dispatch/combine, capacidade/overflow,
  fallback sem perda de tokens, oracle e especialização/balanceamento medidos.
  Detalha M8.03/LLM.11; caminho nativo por KernelIR depende de M2.01/02.
  Fonte: [§5, pp.6–7](NexaLang_Conditional_Compute_Implementation.pdf#page=6), [§11, pp.11–12](NexaLang_Conditional_Compute_Implementation.pdf#page=11), [C1, p.14](NexaLang_Conditional_Compute_Implementation.pdf#page=14).
- [ ] CC.03 C2: Top-K adaptativo, Expert Momentum, scores normalizados/versionados,
  desempate e retry determinísticos; comparar Top-1/Top-2 e sem momentum sob o mesmo
  budget, com atividade/custo/qualidade medidos. Depende de CC.01/02.
  Fonte: [§5, pp.6–7](NexaLang_Conditional_Compute_Implementation.pdf#page=6).
- [ ] CC.04 C3: cache de pesos de experts HOT/PARTIAL/WARM/COLD, ownership e
  identidade por tile, prefetch e falhas; medir bytes/stalls/residência sob cap GPU.
  Reusa M1.10b/M2.06/M3/M7; backing store KV CPU não conclui cache de experts.
  Fonte: [§6, pp.7–8](NexaLang_Conditional_Compute_Implementation.pdf#page=7).
- [ ] CC.05 C4: máscara por consulta/gates por camada com semântica treinada de
  continuidade hidden/KV; testar mudança de máscara, reentrada causal e custos de
  recomputação; menor profundidade com qualidade medida. Reusa M4/LLM.03/04.
  Fonte: [§7, pp.8–9](NexaLang_Conditional_Compute_Implementation.pdf#page=8).
- [ ] CC.06 C5: bloco cheap/heavy com scatter/gather e rejoin, ordem/posições/KV
  preservados, caminhos vazios/capacidade testados e economia líquida incluindo router.
  Fonte: [§8.1, p.9](NexaLang_Conditional_Compute_Implementation.pdf#page=9).
- [ ] CC.07 C6: heads/estimadores de early exit calibrados em split congelado,
  profundidade obrigatória por modo e continuidade KV; medir logits, calibração,
  profundidade, qualidade e custo. Depende de CC.05 e LLM.02/03/04.
  Fonte: [§8.2–8.3, p.9](NexaLang_Conditional_Compute_Implementation.pdf#page=9).
- [ ] CC.08 C7: compor seleção de experts com sparsity de pesos de M6.06/07,
  fallback dense, metadata e temporários no plano; ablações por hardware medem
  bytes/FLOPs/latência, incluindo cenário sem ganho.
  Fonte: [§9, pp.9–10](NexaLang_Conditional_Compute_Implementation.pdf#page=9).
- [ ] CC.09 C8: coordenar K, profundidade, heavy path, precisão e swaps num budget
  global; ausência de rota admissível explícita, qualidade calibrada e violações
  medidas. Estende CC.01 e custos de M7.09/M8.04, sem prometer tempo real por estimativa.
  Fonte: [§10, pp.10–11](NexaLang_Conditional_Compute_Implementation.pdf#page=10), [§16, p.15](NexaLang_Conditional_Compute_Implementation.pdf#page=15).
- [ ] CC.10 C9: receita/pacote condicionais NAQT, QAT versus PTQ com mesma base,
  rotas e dados, execução sem expansão integral e qualidade/cap de memória
  demonstrados. Reusa LLM.09/M8.02, codecs M1/M6 e prova M5.
  Fonte: [§11.5, p.12](NexaLang_Conditional_Compute_Implementation.pdf#page=12), [C9, p.15](NexaLang_Conditional_Compute_Implementation.pdf#page=15).
- [ ] CC.11 C10: routers internos compartilham orçamento e telemetria com Scheduler
  Omni; seleção macro/micro conjunta, cancelamento e ablações, sem duplicar ABI,
  gerenciador ou scheduler. Depende de OMNI.03/05/09 e motores condicionais validados.
  Fonte: [§18, p.16](NexaLang_Conditional_Compute_Implementation.pdf#page=16).

- [ ] CC.12 §12: perdas auxiliares de router com receita, coeficientes e agenda
  versionados — balanceamento de carga, diversidade de experts, regularizador
  temporal, penalidade de budget, calibração de saída e consistência de skip —
  com ablação por termo sob a mesma base. Entregar o relatório de roteamento de
  cada execução de treino: share de tokens por expert/domínio/dataset, experts
  ativos por token, matriz de transição e frequência de troca, histograma de
  camadas executadas, fração de tokens no caminho pesado, distribuição da
  profundidade de saída e delta de qualidade contra a execução completa. Sem essa
  observabilidade, colapso de router se confunde com ganho de velocidade.
  Depende de CC.02/03/05/07 e reusa LLM.03/10.
  Fonte: [§12–12.1, pp.12–13](NexaLang_Conditional_Compute_Implementation.pdf#page=12).
- [ ] CC.13 §13: pipeline condicional fim a fim como ordem única — prefill, perfil
  de complexidade da consulta, máscara inicial de profundidade, score de
  residência com prefetch, laço por token (orçamento, routers de camada/token,
  Top-K adaptativo, streaming dos tiles ausentes, kernels, confiança de early
  exit, telemetria de momentum/residência) e replanejamento quando o domínio do
  contexto muda. Cada etapa precisa de fallback válido e cancelamento; integra
  CC.01–09 sem substituir os aceites individuais.
  Fonte: [§13, p.13](NexaLang_Conditional_Compute_Implementation.pdf#page=13).

**Gate Conditional V1:** DoD da [p.17](NexaLang_Conditional_Compute_Implementation.pdf#page=17),
com rota nativa ModelIR→KernelIR, Top-K/cache/depth/token/exit/sparsity/budget,
baselines e ablações. C10 Omni é integração adicional explicitada na tabela do
roadmap. Telemetria promovida a treino reusa os controles de LLM.10/OMNI.09.

## Complemento: Plastic Learning (20 páginas)

Fonte: [PDF Plastic Learning](NexaLang_Plastic_Learning_Implementation.pdf).
Consultar os [ajustes de integração](NEXA_PLASTIC_LEARNING_AJUSTES.md) antes de cada
módulo. A tabela da p.17 é canônica: **PL.P0–P7**, com treino **PL.T0–T5** na p.15.
A figura da p.17 agrupa curiosidade/consolidação e omite fases da tabela; não
eliminar P6/P7. P2 foi dividido em aplicação de adapters, transações e aprendizado
validado para permitir retomadas. Novos itens permanecem pendentes.

- [ ] PL.01 P0: ParameterRegion/ModelPlasticityConfig/ExpertIR/LearningPolicyIR e
  passes de mapa/validação/layout/probes; IDs físicos, aliases tied, máscaras,
  proteção e limites validados sem mudar baseline. Compartilhar ExpertIR com CC.01;
  sintaxe pública depende de LLM.01b2.
  Fonte: [§2, p.3](NexaLang_Plastic_Learning_Implementation.pdf#page=3), [§§12–13, pp.13–14](NexaLang_Plastic_Learning_Implementation.pdf#page=13).
- [ ] PL.02 P1: evidence store limitado e Learning Gate com proveniência, confiança,
  escopo, proteção, plasticidade, drift, regressão e orçamento; MEMORY_ONLY/ADAPTER/
  PLASTIC_EXPERT/MATURE_EXPERT/DEFER/REJECT, motivos e quotas. Reusar LLM.02/10;
  bloquear autoevidência sem fonte independente; gate não publica pesos diretamente.
  Fonte: [§§3–5, pp.4–6](NexaLang_Plastic_Learning_Implementation.pdf#page=4), [§17, p.18](NexaLang_Plastic_Learning_Implementation.pdf#page=18).
- [ ] PL.03 P2 (aplicação): contrato CPU de adapters/deltas com rank, shapes,
  precisão, composição ordenada, base Q4 imutável e buffers planejados; oracle
  para adapter nulo/não nulo, sem matriz de pesos integral expandida. Reusa M0.04/M1.10.
  Fonte: [§8, p.9](NexaLang_Plastic_Learning_Implementation.pdf#page=9), [§12, p.13](NexaLang_Plastic_Learning_Implementation.pdf#page=13).
- [ ] PL.04 P2 (transações): LearningTransaction, versões-pai, hashes, publicação/recuperação,
  conflito entre candidatos, log/diff/validate/promote/rollback e coleta respeitando
  leitores; sessão fixa versão de pesos/adapters ou refaz prefill/KV ao trocar.
  Testar falhas/cancelamento/disco cheio e restauração dos artefatos; reusa OMNI.08.
  Fonte: [§8, p.9](NexaLang_Plastic_Learning_Implementation.pdf#page=9), [§17, p.18](NexaLang_Plastic_Learning_Implementation.pdf#page=18).
- [ ] PL.05 P2/T2–T4 (aprendizado): trainer seletivo pequeno, replay limitado/reproduzível e
  avaliações A→B/alvo/domínios anteriores separadas; bases/aliases permanecem iguais,
  candidatos regressivos rejeitados e ganho/forgetting/drift medidos. Reusa
  LLM.03a–b/04b; P2 só fecha com aprendizado reversível completo.
  Fonte: [§§14–15, pp.15–16](NexaLang_Plastic_Learning_Implementation.pdf#page=15).
- [ ] PL.06 P3: Plastic Expert Pool, template treinado e lifecycle
  dormant→candidate→plastic→mature→stable, metaplasticidade por bloco, quotas e
  ativação posterior do conhecimento aprendido; maturidade não determina residência.
  Reusa CC.02–04/M3/M8.03/OMNI.05/07 e depende de PL.03–05.
  Fonte: [§1, p.2](NexaLang_Plastic_Learning_Implementation.pdf#page=2), [§7, p.8](NexaLang_Plastic_Learning_Implementation.pdf#page=8), [§9, p.10](NexaLang_Plastic_Learning_Implementation.pdf#page=10).
- [ ] PL.07 P4: Knowledge Localization por afinidade, sensibilidade de gradiente,
  domínio e interferência/replay; normalizar/calibrar scores e demonstrar menor
  interferência frente ao alvo fixo. Depende de PL.05 e instrumentação CC.01–03.
  Fonte: [§6, p.7](NexaLang_Plastic_Learning_Implementation.pdf#page=7).
- [ ] PL.08 P5: curiosidade/prioridade geram tarefas de pesquisa/aprendizado
  rastreáveis, com fontes, quotas, custos, deferimento e cancelamento; lacuna ou saída
  própria não comprova verdade. Reusa PL.02/07, CC.09 e OMNI.04–06 na integração.
  Fonte: [§4, p.5](NexaLang_Plastic_Learning_Implementation.pdf#page=5), [§13, p.14](NexaLang_Plastic_Learning_Implementation.pdf#page=14), [§§16–17, pp.17–18](NexaLang_Plastic_Learning_Implementation.pdf#page=17).
- [ ] PL.09 P6/T5: consolidação por merge/distill/promoção com novo candidato,
  quantização/ordem explícitas, nova base versionada, quotas e rollback preservado;
  Stable Core só muda após avaliação própria. Reusa LLM.03/09, M6/M8 e CC.10 quando aplicável.
  Fonte: [§7, p.8](NexaLang_Plastic_Learning_Implementation.pdf#page=8), [§14, p.15](NexaLang_Plastic_Learning_Implementation.pdf#page=15), [P6, p.17](NexaLang_Plastic_Learning_Implementation.pdf#page=17).
- [ ] PL.10 P7: Brain/Organs compartilham evidências e transações com escopo,
  autoria, isolamento, orçamento e versão de execução, mantendo routers de compute
  e plasticidade distintos. Reusa OMNI.01–09/CC.11; não duplica protocolo/ABI.
  Fonte: [§10, p.11](NexaLang_Plastic_Learning_Implementation.pdf#page=11), [P7, p.17](NexaLang_Plastic_Learning_Implementation.pdf#page=17).
- [ ] PL.11 DoD V1: checkpoint treinado melhora no alvo e recupera aprendizado via
  routing sem alterar a base; evidências/métricas/rollback, retenção e budgets
  comprovados. Separar prova CPU e cap artificial GPU 512 MB, reutilizando
  LLM.04b/06, M0.07/M2.06/M5; aplicação de adapter sintético não fecha o gate.
  Fonte: [§11, p.12](NexaLang_Plastic_Learning_Implementation.pdf#page=12), [§§15–17, pp.16–18](NexaLang_Plastic_Learning_Implementation.pdf#page=16).

**Ordem inicial PL:** metadata/evidências → adapter CPU → transações/rollback →
trainer/replay/gates A→B; somente depois pool/localização/curiosidade e consolidação.
T0 e T1 reutilizam LLM.02/03/04 e CC/M8; não criar outra trilha de pré-treinamento.
O marco P0–P2 recomendado na [p.20](NexaLang_Plastic_Learning_Implementation.pdf#page=20)
pode começar com contratos/fixtures sem alterar a próxima tarefa de runtime.

## M0 — baseline e contratos (PDF fases 0/1, páginas 3–5, 14–17)

- [x] M0.01 Registrar ajustes ao PDF, escopo das medições e instruções de retomada.
- [x] M0.02 IR mínimo de tensores e MatMul rank 2 (transpose_b), verificação de shapes e JSON.
- [x] M0.03 HardwareProfile CPU com capacidades conhecidas e desconhecidos explícitos.
- [x] M0.04 Parser MB/MiB, lifetimes explícitos [start,end), alinhamento, arenas,
  reserva e rejeição de planos. Derivação serial CPU do grafo concluída em M2.05a;
  alias/in-place e trabalho assíncrono permanecem em M2.05b.
- [x] M0.05 Preservar limite no TurboIR legado e rejeitar execução que o ignore.
- [x] M0.06 Benchmark reproduzível CPU: dados determinísticos, referência numérica,
  tempo, bytes packed, buffers previstos/efetivos e relatório JSON/CSV.
- [ ] M0.07 Telemetria GPU real: pico do driver, cópias, bandwidth, launch overhead.
- [ ] M0.08 Fixar modelos 250M/500M/1B, revisões/hash, tokenizer, datasets e qualidade
  de referência, contexto, batch e tolerâncias.

**Gate CPU de M0 aprovado:** um caso válido executa e um caso acima do limite
falha antes da alocação/carga de payload; relatório distingue contabilidade e
medições. Telemetria e baseline de modelos continuam pendentes.

## M1 — NexaPack e execução CPU packed (PDF fases 2/3)

- [x] M1.01 Especificar/implementar contêiner versionado little-endian, limites,
  offsets, checksums, leitura parcial e escrita atômica.
- [x] M1.02 Codec Q4 por grupo com escalas, padding, erros numéricos e interoperabilidade C/Python.
- [x] M1.03 GEMV/GEMM CPU fundido, sem matriz de pesos desquantizada nem alocação oculta.
- [x] M1.04 Conversor inicial de matriz float32 LE para NexaPack e benchmark em tiles.
- [x] M1.05a Despacho de peso por codec no executor e matrizes `RAW_F32_MATRIX`
  com blocos verificados; bundles mistos, conversão por tensor, verificação
  bloco a bloco e equivalência exata com Q4 quando a quantização é exata.
- [ ] M1.05b Kernels Q2/Q3/Q8 e RAW-F16 de pesos, caudas e comparação numérica,
  usando o despacho por codec já existente.
- [x] M1.06 Integrar armazenamento TurboQuant MSE no NexaPack: TQ02 portátil,
  dimensão/bits/seed/SRHT/codebook explícitos, norma F32LE e migração TQ01 com
  endian de origem. Atenção TQ e kernels de matriz permanecem em gates próprios.
- [x] M1.07 Eliminar estado QJL quadrático em uso MSE sem alterar determinismo do Prod;
  construtor explícito, contabilidade de estado/scratch e falhas de alocação verificados.
- [x] M1.08 Importar Safetensors F32/F16/BF16, único/shards, com arquitetura Llama
  suportada, config, assets de tokenizer, hashes e validação de shapes. Tokenização
  executável e outras variantes arquiteturais continuam fora desse importador.
- [ ] M1.09 Importadores GGUF e ONNX com rejeição explícita de operações incompatíveis.
- [x] M1.10a Múltiplos tensores, manifesto versionado, aliases tied, normas F32,
  checksums, leitura parcial e publicação atômica sem alterar o formato V1 de matriz.
- [ ] M1.10b Pacote executável `.nxb` com plano, kernels, variantes e fallback.
- [ ] M1.11 Tipos `qint<N>`/PackedVector na linguagem e contrato de FFI documentado.

**Gate CPU/Q4 de M1 aprovado:** matriz packed maior que o orçamento de buffers
processada por blocos, resultado comparado à referência e nenhuma expansão
completa. Importação local do subconjunto Llama/Safetensors disponível; outros
codecs, arquiteturas e importadores continuam pendentes.

## M2 — KernelIR e primeiro backend GPU (PDF fases 1/3/4)

- [ ] M2.01 KernelIR executável: load_packed, unpack, dequant, register/shared tiles,
  redução, barreiras, store; validação de dependências e capacidades.
- [ ] M2.02 Definir fronteira ModelIR → MemoryPlan/ExecutionPlan → KernelIR → backend.
- [ ] M2.03 GEMM Q4 em OpenCL/SPIR-V com resultado e memória medidos no hardware.
- [x] M2.04a CPU: RMSNorm, RoPE, atenção causal GQA, SwiGLU, residuais e forward
  completo com uma/duas camadas, comparações numéricas e sanitizers.
- [ ] M2.04b Executar os operadores e a camada Transformer em backend GPU real.
- [x] M2.05a Derivar lifetimes do grafo serial CPU, preservar residuais/saídas e
  impedir alias de operandos/resultados; executor usa os offsets planejados.
- [ ] M2.05b Alias analysis/in-place seguro, trabalho assíncrono e cache/alinhamento
  específicos do dispositivo. Nenhuma reutilização CPU é apresentada como prova GPU.
- [ ] M2.06 Reserva de driver e budget de VRAM aplicado a todas as alocações.
- [ ] M2.07 Reabilitar quantização GPU antiga somente após corrigir normas/writeback
  ou substituí-la por um contrato packed novo, com testes.

**Gate M2:** camada real executada em GPU, referência CPU aprovada e nenhum ganho
de memória alegado apenas a partir do tamanho do arquivo.

## M3 — streaming e tiers (PDF fase 5, P06–08/P21–27)

- [ ] M3.01 Loader mmap/leitura segmentada, residência RAM/GPU/NVMe e evicção segura.
- [ ] M3.02 Double/triple buffering com orçamento para todos os slots.
- [ ] M3.03 Filas de cópia e compute, eventos, dependências e cancelamento/liberação.
- [ ] M3.04 Prefetch por next-use/custo, fallback sem overlap e medição de stalls.
- [ ] M3.05 Executor consome o plano validado; não reconstrói uma execução divergente.
- [ ] M3.06 Testar transferência interrompida, OOM, contexto inválido e reexecução.

**Gate M3:** peso maior que VRAM executa sem OOM, sem expansão integral, com bytes
transferidos e limite máximo documentados.

## M4 — NexaKV real (PDF fase 6, P16–20)

- [x] M4.00 Baseline KV F32 incremental CPU, estado transacional e orçamento
  combinado com ativações/staging; prefill em chunks, decode de um novo ID e
  equivalência ao forward por recomputação. Dois bancos completos F32, sem paginação.
- [x] M4.01 Páginas KV F32 estáveis por layer/head, GQA/MQA, tamanho/capacidade reais,
  atenção direta e residência sob demanda; transações, reset/close e orçamento.
- [x] M4.02a Codec KV Q4_GROUPED por token/head em páginas CPU, escalas/caudas,
  tamanho físico, erro numérico e atomicidade de páginas parciais verificados.
- [x] M4.02b Codec KV Q3_GROUPED V1 por token/head, packing de três bits,
  escalas/caudas, páginas parciais e comparação física F32/Q4/Q3 verificados.
- [x] M4.02c Codec KV TQ_MSE_SRHT V1/TQ02 por token/head, contexto/codebook
  compartilhados, páginas parciais e memória de estado/staging/scratch contabilizada.
- [x] M4.02d Política CPU hot F32/warm Q4/cold Q3 por idade, orçamento global,
  identidade por página e atenção mista; todos os tiers residentes em RAM.
- [x] M4.03a Atenção CPU consome KV Q4 packed diretamente, sem cópia float de
  head/página/prefixo; equivalência à referência com a mesma quantização.
- [x] M4.03b Atenção CPU Q3 direta sobre códigos/escalas, sem expansão de
  head/página/prefixo, oracle independente e compatibilidade F32/Q4 validados.
- [x] M4.03c Atenção CPU TQ02 com reconstrução de um head por vez, sem heap,
  scratch/acumulador planejados, oracle independente e compatibilidade F32/Q4/Q3.
- [ ] M4.03d Atenção packed para demais codecs e despacho conforme seus layouts,
  sem expansão integral e com temporários contabilizados.
- [ ] M4.04 Integrar no prefill/decode real; remover duplicação de cache para métricas.
- [x] M4.05a Recodificação CPU por página, transacional, scratch limitado,
  custo/bytes e erro local medidos, com oracle que reproduz o histórico de chunks.
- [x] M4.05b Evicção/recarga CPU de páginas cold Q3, backing store privado com
  checksum/publicação atômica, atenção causal com um slot, bytes preservados,
  orçamento, I/O, rollback/cancelamento e isolamento/liberação verificados.
- [x] M4.05c Promoção de residência CPU: slots de recarga admitidos, política
  determinística de admissão/substituição sob varredura cíclica, ownership e
  liberação; bytes, ordem de redução e logits preservados, bytes/recargas
  evitados, pico e cenário sem ganho medidos.
- [ ] M4.05d Promoção de precisão e critérios de qualidade para transições;
  custo/orçamento sem descartar contexto causal. Importância por página e budgets
  por camada exigem calibração em M6.01 e comparação com idade uniforme
  (Blueprint pp.9–10, P16–19; p.15, §7.3). Prefetch e leitura assíncrona incluídos.
- [x] M4.06a Múltiplas sequências no executor paginado homogêneo: prefixo
  compartilhado por contagem de referências, cópia apenas da página parcial,
  escrita em página compartilhada rejeitada, capacidade por sequência e
  liberação em qualquer ordem; logits idênticos e residência somada medida.
- [x] M4.06b Sequências derivadas sob a política de idade: descritores herdados,
  migração privada sobre páginas compartilhadas, identidade de layout incluindo
  política e group_size, e relatórios de residência compartilhada/própria.
- [ ] M4.06c Sequências derivadas sob backing store, cancelamento de uma chamada
  em andamento, limites e admissão conjunta por processo, e reuso de prefixo
  entre sessões que não derivam uma da outra.

**Gate M4:** melhoria de bytes/token comprovada no cache usado pela atenção,
com qualidade, latência e temporários contabilizados.

## M5 — prova de modelo completo e ABI (PDF páginas 17/19)

- [ ] M5.01 Modelo fixo 250–500M, batch 1, tokenizer, prefill e decode sem PyTorch.
  Tokenizer e caminho texto→IDs→execução→texto estão em LLM.02c1; falta o modelo
  real importado, sua fixação em M0.08 e a medição de qualidade.
- [ ] M5.02 Modelo ~1B sob 512 MB, pesos streamados e KV comprimido.
- [ ] M5.03 Matriz de provas A–E: 250–500M, 1B, 1–3B offload, modelo próprio,
  e escalabilidade em 8/24 GB; registrar hardware e configurações.
- [ ] M5.04 Qualidade/perplexidade, TTFT, tokens/s, pico, bytes/token, contexto,
  tempo de compilação/tuning/cache e energia quando disponível.
- [ ] M5.05 C ABI de create/load/generate/tensor/unload, códigos de erro e ownership;
  orquestração nativa sem dependência de Python no runtime final (Primeira LLM p.10, §8.3).
- [ ] M5.06 Bindings Python/Rust/C++/Node/Swift/Kotlin e export ONNX/GGUF quando suportado.

**Gate M5:** reprodução documentada em GPU real, sem PyTorch na inferência final,
qualidade aprovada e orçamento respeitado durante prefill e decode.

## M6 — precisão, compressão e fusão (PDF fases 7/8, P01–15/P28/P30)

- [ ] M6.01 Calibração por tensor/grupo: sensibilidade, outliers e orçamento de qualidade.
- [ ] M6.02 PrecisionMap Q2/Q3/Q4/Q8/F16 por tensor/bloco e seleção por custo físico
  medido; misturar codecs dentro do tensor exige formato versionado, identidade/
  offsets por bloco e despacho compatível (Blueprint p.4, §4.3; Primeira LLM p.10, §8).
- [ ] M6.03 CompressionPlanner escolhe codec/sparsity/low-rank sem presumir speedup.
- [ ] M6.04 Fusões dequant+GEMM, RMSNorm+QKV, QKV+RoPE e FFN/SwiGLU por custo.
- [ ] M6.05 StreamingRegions entre operações com dependências e register pressure.
- [ ] M6.06 SparseBlock, block-zero/N:M, activation tile skipping e fallback dense.
- [ ] M6.07 Pruning/low-rank/permutações offline e equivalência/qualidade validadas.
- [ ] M6.08 Otimizações algébricas de grafo com provas locais/testes de tolerância.

## M7 — backends e autotuner (PDF fase 9, P09–10/P27/P29)

- [ ] M7.01 Interface comum de backend e capabilities; SPIR-V/OpenCL preservado.
- [ ] M7.02 NVIDIA NVPTX/PTX + CUDA driver e validação em hardware.
- [ ] M7.03 AMDGPU + ROCm/HIP e validação em hardware; fallback para AMD legacy.
- [ ] M7.04 Metal/MSL e memória unificada com medições específicas.
- [ ] M7.05 CPU SIMD AVX/NEON, alinhamento, threading/NUMA e fallback scalar.
- [ ] M7.06 Microbenchmarks bandwidth/latência/GEMM/decode/shared/launch por hardware.
- [ ] M7.07 Variantes tile M/N/K, stages, workgroup, vector width e split-K.
- [ ] M7.08 Cache por modelo/codec/GPU/driver/versão do compilador, invalidação e tuning budget.
- [ ] M7.09 Solver multiobjetivo com greedy/backtracking inicial e ablations medidos.

## M8 — NexaLM-512 e treino (PDF fase 10, P04/P20/P31)

- [ ] M8.01 Propor arquitetura após gate M5: estados menores, atenção local/SSM,
  embeddings compactos e compartilhamento validado.
- [ ] M8.02 QAT Q2/Q3/Q4, camadas sensíveis, distillation e treino reprodutível.
- [ ] M8.03 Sparse/MoE com router sensível a transferência e parâmetros ativos/token.
- [ ] M8.04 Objetivo de treino inclui working set/bytes/qualidade; precisão de treino adaptativa.
- [ ] M8.05 Demonstrar qualidade por MB superior ao baseline com ablations publicados.

## M9 — documentação, manutenção e referências

- [ ] M9.01 ADRs por contrato/layout/algoritmo, justificativas e medições próprias.
- [ ] M9.02 Conferir bibliografia P01–P31 e listas Conditional Compute/Plastic Learning;
  separar inventor, requerente e família, sem tomar alegações dos PDFs como verificação externa.
- [ ] M9.03 Revisão das features ativadas antes de release comercial, conforme PDF §14.
- [x] M9.04a Regressões CPU, formatos malformados, sanitizers e fixtures determinísticas
  integrados à CI; validação local no macOS.
- [ ] M9.04b Confirmar execução remota da matriz Linux/macOS/Windows Python 3.11/3.12.
- [ ] M9.05 CI/testes em dispositivos GPU e matriz explícita do que foi verificado.
- [ ] M9.06 Guias de migração de formato, exemplos executáveis, compatibilidade e release notes.

## Registro de validação

- Base anterior ao blueprint: 110 testes bootstrap e 81 regressões passaram no macOS;
  102 exemplos compilaram e 9 casos negativos produziram erros esperados.
- Primeiro incremento no macOS ARM64: 21 testes ModelIR/MemoryPlan, 17 NexaPack,
  2 testes nativos Q4 (incluindo ASan/UBSan, proibição de alocação e interoperabilidade),
  e 10 testes de integração/CLI/orçamento legado. São 50 novos testes.
- A suíte original de 110 testes e a descoberta de todas as 131 regressões passaram:
  **241 testes, zero falhas**, no estado final do primeiro incremento.
- `make -C runtime all test`, `compileall` e `git diff --check` passaram.
- Demo `4096 x 128`, Q4 com grupos de 32, batch 1, tile de 64 linhas e orçamento
  de 96 KiB: payload de 327.680 B, equivalente FP32 de 2.097.152 B; arena de
  5.888 B + padding de 63 B + scratch de 65.536 B = limite de buffers de 71.487 B.
  Foram 64 tiles, 327.680 B lidos e erro absoluto máximo zero contra referência Q4.
- Tempos e resultado completos estão em `artifacts/reports/blueprint-first-milestone.json`
  e `.csv` (ignorados pelo Git e reproduzíveis pelos comandos abaixo). Não há prova
  de qualidade de LLM, pico do processo ou desempenho GPU nesse incremento.

Segundo incremento:

- R0/v1 compilados da fonte canônica: 125.854.464 e 394.331.136 parâmetros físicos.
- Fixture sintética: 728 parâmetros, 11 tensores físicos, 12 nomes com o alias de
  saída; Safetensors em dois shards com F32/F16/BF16, sem download ou treino.
- Importação e inspeção completas; payload Q4 de 1.056 B. Benchmark do alias
  `lm_head.weight`, batch 2, matriz 16x8, tile 3: seis tiles e erro absoluto máximo
  zero contra referência Q4. Relatórios em `artifacts/reports/nexalm-architecture.json`
  e `artifacts/reports/nexalm-bundle-q4.json`.
- Interoperabilidade opcional local: os 11 tensores do fixture coincidiram com
  a leitura pela biblioteca oficial Safetensors/PyTorch. Elas não são dependências
  do importador nem do runtime.
- Validação local final no macOS ARM64/Python 3.14.5: **190 regressões + 110 testes
  bootstrap = 300 testes, zero falhas**. As regressões incluem sanitizers C,
  formatos malformados, corrupção, publicação atômica e proteção de memória.
- Os 59 testes adicionados neste incremento abrangem ModelConfig/DSL (16),
  bundle (16), Safetensors/importador (18), integração/CLI (8) e validação Q4
  sem expansão da linha (1). `--verify` rejeita também codecs inválidos com SHA
  correto, mantendo a inspeção padrão sem leitura de payload de pesos.
- Corrigido o fixture HTTP existente: reserva da porta durante compilação,
  prazo de startup de 10 segundos, diagnóstico e encerramento/coleta dos pipes.
  A falha original foi intermitente; startup artificial de 2,5 segundos reproduziu
  a fragilidade anterior e passou após o ajuste. Nenhuma alteração adicional em std/http.
- `compileall`, `git diff --check` e a inspeção completa do bundle gerado passaram.
  A sequência offline foi incluída na CI; execução remota da matriz ainda pendente.

Terceiro incremento:

- Grafo de `15 * layers + 3` operações e lifetimes seriais derivados. Runtime usa
  seus offsets e preserva residuais; constantes são streamadas em buffers separados.
  R0/v1 geraram 243/483 operações em `artifacts/reports/transformer-graphs.json`.
- Fixture de 728 parâmetros, prefill `[1,3]` + decode IDs `5,7`: 18 operações por
  forward, 66.879 B de arena/padding/scratch sob 96 KiB. O último forward leu
  1.200 B de payload Q4 (incluindo checksums de blocos) e 96 B de normas F32.
- PyTorch 2.11.0 local: erro máximo C vs referência Q4 = 0; Q4 vs pesos originais
  = 0,4435420930 nos logits dessa fixture não treinada. Tolerância de execução:
  `1e-5 + 1e-4 * abs(reference)`. Não é prova de perplexidade/qualidade.
- Golden numérico independente preserva a prova em CI sem Torch. Testes cobrem
  também duas camadas, MHA/GQA/MQA, tied/untied, causalidade, RoPE em posições
  não zero, rollback após corrupção e decode vs cache KV real do oracle.
- Kernels novos passaram referência escalar, ASan/UBSan e link com alocação heap
  proibida. `make -C runtime all test` passou. Relatório reproduzível em
  `artifacts/reports/nexalm-forward.json`, ignorado pelo Git.
- Builder preserva DLLs carregadas no Windows publicando nova biblioteca com
  nome único quando necessário. Cinco testes simulam o bloqueio, arquivos
  persistentes e propagação de erros; validação Windows real segue pendente.
- Validação final macOS ARM64/Python 3.14.5: **225 regressões + 110 testes bootstrap
  = 335 testes, zero falhas e zero skips**. Incremento de 35 testes: IR/lifetimes
  (14), kernels (2), forward/oracle (11), CLI/normas (3), publicação de runtime (5).
- Verificação sem site-packages (`python3 -S`): sete testes de forward nativo,
  incluindo golden, passaram; quatro testes PyTorch foram corretamente ignorados.
  PyTorch não foi adicionado às dependências do runtime/compilador.
- `compileall`, `git diff --check`, reprodução CLI com referência e fonte original
  passaram. A CI recebeu grafo/prefill/decode offline; execução remota ainda pendente.

Quarto incremento:

- M4.00: plano KV F32 com dois bancos, agenda explícita de CacheWrite/atenção/commit,
  RoPE com offset e lifetimes derivados. Decode executa embedding/projeções somente
  do ID novo e consulta o prefixo persistente; não existe cópia integral de rollback.
- Prefill em chunks por `max_chunk_length`/`--prefill-chunk-size`; o preflight
  combina maior chunk, scratch do contexto máximo e os dois bancos completos.
  Offsets do plano máximo são preservados em chamadas menores: corrigida a
  fragmentação first-fit que podia rejeitar chunks válidos sob orçamento exato.
  O mesmo ajuste protege o baseline por recomputação.
- Corrigidos reset parcialmente aplicado em caso de falha de planejamento/relatório
  e retenção de arena/scratch por exceções guardadas pelo consumidor. Testes com
  weakrefs comprovam liberação antes de retry dentro do próprio `except`.
- Fixture `[1,3]` + decode `5,7`: quatro posições processadas contra nove do baseline;
  último decode com embedding de uma linha, batch 1, 32 B novos de KV e zero cópias
  do prefixo. Os dois bancos ocupam 256 B de payload, 319 B com alinhamento.
- Pico contabilizado de 67.070 B sob 96 KiB; com chunks limitados a dois tokens,
  66.814 B. Os relatórios `artifacts/reports/nexalm-kv-f32.json` e
  `artifacts/reports/nexalm-kv-chunks.json` têm o mesmo hash dos logits finais.
  Referência PyTorch Q4: erro máximo zero; erro de quantização contra fonte original
  0,4435420930. CPU/sintético, sem prova de qualidade de linguagem ou VRAM.
- Validação macOS ARM64/Python 3.14.5: **267 regressões + 110 testes bootstrap =
  377 testes, zero falhas e zero skips**. Os 42 novos testes cobrem plano (14),
  kernels com cache (2), execução incremental/oracle (12), chunks/orçamento (10)
  e liberação de buffers/reset (4). Kernels passaram ASan/UBSan e proibição de heap.
- Sem site-packages: 11 testes incrementais passaram e um oracle PyTorch foi
  ignorado; runtime não depende de PyTorch. `make -C runtime all test`, `compileall`
  e `git diff --check` passaram. CI inclui prefill em chunks e decode incremental;
  execução remota da matriz permanece pendente.
- PDF Nexa Omni lido integralmente (14 páginas), decisões registradas e dez itens
  OMNI adicionados. Nenhum gate Omni foi marcado como implementado. O checklist
  totaliza **19 itens concluídos e 82 pendentes**, incluindo os dez novos itens;
  esses itens têm tamanhos diferentes e não equivalem a percentual ou prazo.

Quinto incremento:

- M4.01: páginas F32 com buffers K/V por camada/head, segmentos de escrita,
  endereços estáveis no append e atenção consumindo tabelas diretamente. Abertura
  da sessão não aloca páginas; reset/close liberam todas. Substituição de prompt
  preserva páginas antigas até commit, contabilizando o pico de ambas as versões.
- Reserva de admissão inclui `ceil(C/P) + ceil(T/P)` páginas, workspace, tabelas,
  scratch e alinhamento. Tamanhos reais entram no relatório de residência/pico.
  Corrigida também a borda de 65.536 segmentos: admissão considera o pior offset
  de append, para não aceitar capacidade que só funcionaria em prefill alinhado.
- Fixture de 728 parâmetros e IDs `[1,3,5,7]`: erro máximo nativo vs referência
  Q4 PyTorch = 0; quantização vs origem = 0,4435420930. Páginas de dois tokens,
  chunks de dois e decode unitário preservaram o hash dos logits contíguos.
  Relatório: `artifacts/reports/nexalm-kv-paged.json`.
- Comparação com contexto máximo oito, quatro tokens usados e páginas de quatro:
  191 B residentes contra 575 B dos dois bancos; pico gerenciado de 66.814 B contra
  67.070 B, incluindo tabelas, padding e reader. Reserva paginada de 573 B garante
  três páginas simultâneas; não foi contada como alocação física. Evidência em
  `artifacts/reports/nexalm-kv-paged-residency.json` e
  `artifacts/reports/nexalm-kv-contiguous-residency.json`, com hashes de logits iguais.
  Resultados sintéticos CPU não demonstram ganho geral de latência, qualidade ou GPU.
- Validação local macOS ARM64/Python 3.14.5: **297 regressões + 110 testes bootstrap
  = 407 testes, zero falhas e zero skips**. São 30 testes novos: planner paginado
  (14), kernels (2), sessão/CLI/oracle (14). Cobrem MHA/MQA/GQA, fronteiras e caudas,
  páginas não contíguas, JSON adulterado, orçamento exato, checksum após escrita
  da primeira camada, alocação interrompida, falha tardia e tracebacks retidos.
- Atenção paginada passou ASan/UBSan e proibição de heap. `make -C runtime all test`,
  `compileall` e `git diff --check` passaram. Sem site-packages, 13 testes de sessão
  passaram e um oracle Torch foi ignorado. CI recebeu execução paginada offline;
  a matriz remota continua sem confirmação.
- Checklist atual: **20 itens concluídos e 81 pendentes**. M4.02/M4.03, o gate
  completo M4, os gates GPU/qualidade e todos os itens Omni continuam pendentes.

Sexto incremento:

- Concluídos M4.02a/M4.03a: Q4_GROUPED V1 aplicado por token/head em páginas KV;
  K pós-RoPE e V pós-projeção. Grupos novos não alteram códigos/escalas do prefixo.
  Páginas parciais já são Q4; não há banco F32 paralelo nem re-encode ao completar
  a página. O JSON F32 permanece byte a byte compatível.
- Kernel de atenção paginada lê escalas/códigos diretamente em double, sem buffer
  F32 de head/página/prefixo. Quantização usa o encoder C existente e entra na
  contabilidade de tempo de compute. Memória inclui escalas, padding, tabelas,
  staging e reserva para substituição; reserva e residência continuam separadas.
- Fixture tiny de 728 parâmetros, quatro IDs, contexto oito, página oito e grupo
  quatro: 12 B/token Q4 contra 32 B/token F32; payload de página 96 B contra 256 B;
  alocação residente 191 B contra 319 B. Picos gerenciados: 66.814 B contra 66.942 B.
  Reservas KV de admissão: 382 B contra 638 B. Relatórios reproduzíveis em
  `artifacts/reports/nexalm-kv-q4.json` e `artifacts/reports/nexalm-kv-f32-comparison.json`.
- Erro máximo C vs oracle com KV Q4 = 0. Erro dos pesos = 0,4435420930; somente
  KV = 0,2454764843; combinado = 0,4757611454. Os dois modos têm hashes diferentes,
  como esperado para quantização com perda. `verified` não aprova qualidade; apenas
  a equivalência de execução à referência que usa o mesmo codec. Não há prova GPU
  ou de qualidade de modelo treinado. Um teste adicional com D64/P16/G32 executa
  com 1.343 B de páginas Q4 contra 8.255 B F32, incluindo padding físico.
- Validação local macOS ARM64/Python 3.14.5: **324 regressões + 110 testes bootstrap
  = 434 testes, zero falhas e zero skips**. São 27 testes novos: plano (11), kernels
  (2), sessão/CLI/oracle (14). Cobrem grupos ímpares, tails/G>D, MHA/MQA/GQA,
  equivalência entre chunks/páginas, prefixo packed imutável, oracle/golden,
  orçamento exato, checksum/falhas tardias, corrupção/alias e liberação de páginas.
- Teste de erro real do encoder: após K gravado, V com escala subnormal retorna
  erro -5. Prefixo/histórico/relatório permanecem intactos, arena nova é liberada
  mesmo com traceback retido e retry coincide com sessão nova.
- ASan/UBSan, proibição de heap, Clang C11 com `-Wall -Wextra -Werror`,
  `make -C runtime all test`, `compileall` e `git diff --check` passaram. Sem
  site-packages: 11 testes Q4 de sessão passaram e três oracles Torch foram
  ignorados; o golden continuou executando. CI recebeu a sequência Q4 KV offline;
  execução remota da matriz ainda pendente.
- M4.02/M4.03 foram divididos em subitens implementados e restantes, preservando
  o escopo futuro. O checklist agora tem **22 itens concluídos e 81 pendentes**;
  a contagem aumentou pela divisão, não representa percentual ou prazo.

Sétimo incremento:

- Concluídos M4.02b/M4.03b: Q3_GROUPED V1 com escala F32 LE, inteiros assinados
  de três bits LSB-first, código -4 reservado e caudas/padding zero. Quantização
  só dos novos tokens, K após RoPE e V após projeção; atenção lê códigos e escalas
  em double, sem buffer de desquantização. Pesos continuam Q4.
- JSON F32/Q4 preservado byte a byte; API/CLI incluem Q3 com grupo padrão 32.
  Pré-admissão, tabelas, staging, páginas parciais e liberação mantêm os contratos
  anteriores. Teste de underflow real após gravar K confirma rollback, liberação
  com traceback retido e retry; o prefixo packed permanece intacto.
- Fixture wide sintética D64/P16/G32, contexto 16, chunks dois, IDs
  `[1,3,5,7,2,4,6,8]`, mesma capacidade nos três modos: bytes/token K/V
  F32/Q4/Q3 = 512/80/64; payload de página = 8.192/1.280/1.024 B;
  alocação física residente = 8.255/1.343/1.087 B. Q3 reduz 20% do payload e
  cerca de 19,1% da alocação da página contra Q4 nessa configuração.
- Pico gerenciado entre chamadas F32/Q4/Q3 = 79.614/72.702/72.446 B; reserva
  KV = 16.510/2.686/2.174 B. São buffers CPU explícitos, não RSS/VRAM. Todos
  processaram oito tokens sem recomputar o prefixo. Os relatórios estão em
  `artifacts/reports/nexalm-kv-wide-{f32,q4,q3}.json`.
- Erro de execução máximo zero nos três modos contra seus oracles. Na wide,
  erro de KV Q4 = 0,2619032860 e Q3 = 0,5301163346; erro dos pesos não medido
  porque essa fixture não preserva checkpoint original. Na tiny com grupo quatro,
  erro dos pesos = 0,4435420930; KV Q3 = 0,6113268733; combinado = 0,9679856896.
  Relatório tiny: `artifacts/reports/nexalm-kv-q3.json`. Q3/Q4 nessa tiny ocupam
  os mesmos 191 B por página; não se presume ganho físico universal.
- Validação local macOS ARM64/Python 3.14.5: **350 regressões + 110 testes
  bootstrap = 460 testes, zero falhas e zero skips**. Os 26 novos testes cobrem
  plano (9), kernels (2), codec/oracle/sessão/CLI (15). Referência Python independente
  dos bytes, golden PyTorch, GQA/MQA/MHA, grupos ímpares, caudas, bits entre bytes,
  limites, aliases, corrupção, orçamento exato, falhas e páginas imutáveis.
- ASan/UBSan, kernels sem heap, Clang C11 com `-Wall -Wextra -Werror`,
  `make -C runtime all test`, `compileall`, links Markdown e `git diff --check`
  passaram. Sem site-packages: 12 testes passaram e três oracles Torch foram
  ignorados. A CI ganhou uma execução Q3 offline, reproduzida localmente;
  matriz remota ainda não executada por este trabalho.
- Logs: `artifacts/reports/q3-kv-regressions.log`, `q3-kv-bootstrap.log`,
  `q3-kv-without-site-packages.log` e `q3-kv-native-build.log` no mesmo diretório.
  Fixture, relatórios, pesos e bibliotecas permanecem ignorados pelo Git.
- Plano de treinamento lido integralmente (16 páginas), com `TRAIN.G0–G8`
  mapeados ao checklist. LLM.02/03/09 detalhados para dados, tokenizer, shards,
  trainer/retomada e QAT; nenhum gate de treinamento foi concluído e nenhum
  corpus foi coletado. Checklist: **24 concluídos e 86 pendentes**, após divisão
  dos itens Q3 e treino. Próximo incremento de runtime: M1.07, pré-requisito TQ01.

Oitavo incremento:

- Concluído M1.07: `tq_create_mse` mantém signs/codebook/boundaries, sem matriz
  quadrática QJL, codebook Prod ou buffer persistente sem uso. `tq_create` mantém
  seu comportamento legado, incluindo consumo de RNG e suporte Prod. Não existe
  upgrade implícito de MSE para Prod; operações Prod retornam -2 sem alocar ou
  escrever saídas, e os helpers Prod retornam zero no contexto MSE.
- `tq_context_memory_bytes` soma bytes solicitados ao allocator para o estado
  persistente, incluindo struct. Soma validada contra overflow antes de alocar;
  MSE não depende de um limite quadrático. Construção parcial libera seus buffers.
  Header e guia distinguem ownership, contexto compartilhável para leitura,
  heap temporário e buffers do chamador. Scratch serial da quantização = 4*D;
  desquantização usa a saída. O diagnóstico `tq_mse` ainda materializa o batch.
- Integração: builtin `compress::create_mse`, `Quantizer::new_mse`/`with_seed_mse`
  e `CompressedBuffer::new_mse`; construtores anteriores continuam legados. KV
  genérico stdlib, quick helpers e wrapper Python TinyLlama usam contexto MSE.
  Corrigido retorno semântico do builtin para o tipo canônico `u8*`, permitindo
  interoperar com FFI. Novo símbolo resolvido pelo JIT. Nenhum modelo foi baixado
  ou executado; imports Torch/Transformers do wrapper ficam na geração.
- macOS ARM64, bits3/seed42, struct88 B: estado MSE D64/D1024 = 404/4.244 B,
  contra legado 17.072/4.202.672 B. Instrumentação independente confirma a API,
  crescimento linear, scratch 256/4.096 B e pico de construção MSE 412/4.252 B.
  Relatório `artifacts/reports/turboquant-mse-memory.json` reproduzível pelo harness.
- Comparação adicional pela API, D4096/bits3: 16.532 B MSE contra 67.141.808 B
  legado. Três vetores tiveram bytes packed e reconstrução idênticos entre modos
  em D64/D1024/D4096. Relatório `artifacts/reports/turboquant-mse-contexts.json`.
  Valores representam estado do contexto, não memória total, RSS ou VRAM.
- **364 regressões + 110 testes bootstrap = 474 testes, zero falhas e zero skips.**
  Quatorze novos testes: oito de runtime/instrumentação e seis de integração;
  cinco cenários NXL rodaram em native e JIT. Cobertura inclui goldens Prod/MSE
  capturados antes da mudança, seeds negativos, bits1–8, zeros/subnormais,
  raw/packed/paralelo, falha em cada alocação de construção (inclusive Lloyd-Max),
  liberação, retry e rejeição Prod sem efeitos no contexto MSE.
- ASan/UBSan, Clang C11 `-Wall -Wextra -Werror`, `make -C runtime all test`,
  `compileall` e `git diff --check` passaram. Sem site-packages, os oito testes
  MSE nativos passaram sem skips. Exemplos: 102 compilaram, nove erros esperados,
  nove executados e zero falhas. Testes novos integram a descoberta usada na CI;
  matriz remota Linux/macOS/Windows continua pendente.
- Logs em `artifacts/reports/mse-context-{regressions,bootstrap,examples,native-build,without-site-packages}.log`.
  Guia: `docs/TURBOQUANT_MSE_CPU.md`. Checklist: **25 concluídos e 85 pendentes**.
  Este incremento não altera o layout TQ01 host-endian nem integra TQ à atenção
  paginada; M1.06 e M4.02c/M4.03c continuam pendentes.

Nono incremento:

- Concluído M1.06: codec TQ_MSE_SRHT V1, identidade SRHT_XOSHIRO256SS_V1,
  shape por vetor, bits1–8, dimensão potência de dois e seed signed-int32.
  Codebook F32LE persistido, finito/crescente e validado; leitor não roda Lloyd-Max.
  Registros TQ02 contêm norma F32LE e índices LSB-first, padding zero e vetor
  nulo canônico. Formato portátil não promete aritmética idêntica em todo hardware.
- Contêiner V1/Q4 preservado byte a byte, leitura parcial com checksums e writer
  atômico com I/O sem buffering e um vetor por vez. Inspeção valida TQ sem C.
  Migração TQ01 exige endian/seed/dim/bits/codebook originais, sem inferência nem
  requantização. Bundles/executor rejeitam TQ antes de payload/kernel incompatível.
- APIs C TQ02 recebem capacidades e scratch do chamador, sem heap; construção
  com codebook importado não executa Lloyd-Max. Contexto MSE linear e legados
  MSE/Prod preservados. Wrapper serializa chamadas/close, impede reentrada,
  mantém parâmetros públicos imutáveis e admite contexto/buffers pelo orçamento.
  Iteradores/views e buffers são liberados inclusive com exceções retidas.
- Demo 4096x64, bits3/seed42, 64 blocos: entrada F32 de 1.048.576 B, payload
  de 131.072 B e arquivo de 143.360 B, ambos maiores que o orçamento de 96 KiB.
  Contexto 404 B, scratch 256 B, pico calculado do codec 980 B e da conversão
  1.236 B com leitura incluída. São buffers gerenciados, não RSS/VRAM; Python,
  metadados, listas do consumidor, bibliotecas/allocator/cache do SO ficam fora.
- Inspeção verificou 131.072 B. Round-trip por vetor: erro absoluto máximo
  0,7924648523, médio 0,1649009345 e RMSE 0,2061312307 contra entrada sintética.
  Não mede qualidade/perplexidade. Origem big-endian simulada migrou para arquivo
  idêntico ao portable original, com limite de buffers de 104 B; não é teste em
  hardware big-endian. Relatórios `artifacts/reports/tq-portable-{conversion,inspection,roundtrip,migration}.json`.
- Validação macOS ARM64/Python 3.14.5: **396 regressões + 110 testes bootstrap
  = 506 testes, zero falhas e zero skips**. Os 32 testes novos (20 formato/wrapper,
  oito CLI e quatro kernels) também passam com `python3 -S`, sem site-packages.
  Cobrem oracle independente, golden Q4 anterior, bits1–8, seeds negativos,
  C/Python, corrupções, endian, orçamento, atomicidade, concorrência, reentrada,
  limpeza em falhas, capacidades/alias/overflow e falhas de alocação.
- ASan/UBSan, kernels sem heap, Clang C11 `-Wall -Wextra -Werror` e
  `make -C runtime all test` passaram, assim como `compileall`, links Markdown e
  `git diff --check`. A receita TQ adicionada à CI e os exemplos do guia foram
  verificados localmente; execução remota da matriz ainda pendente. Logs no diretório ignorado
  `artifacts/reports/tq-portable-{regressions,bootstrap,native-build}.log`.
- Guia e comandos reproduzíveis: `docs/NEXAPACK_TQ_V1.md`. Checklist:
  **26 concluídos e 84 pendentes**. M4.02c/M4.03c continuam abertos; próxima
  implementação é TQ no KV paginado/atenção CPU, com temporários contabilizados.

Décimo incremento:

- Concluídos M4.02c/M4.03c: KV TQ02 por token/head, K pós-RoPE e V pós-projeção,
  contexto/codebook compartilhado, CLI `--kv-codec tq --kv-bits B --kv-seed S`.
  Defaults 3/42; dimensão potência de dois, seed signed-int32 e codebook explícito.
  JSON F32/Q4/Q3 preservado byte a byte. Páginas parciais já são comprimidas;
  append não requantiza/copia prefixo confirmado. Pesos continuam Q4.
- Pré-admissão independente do C reserva estado linear (struct até 128 B,
  verificada por assert C/API), staging de construção e pior workspace/páginas.
  Contexto é criado depois da admissão e antes da primeira página/payload;
  centroids gerados tornam o plano concreto. Codebook importado não usa Lloyd-Max.
  Relatório separa estado real, reserva e staging. Reset libera páginas e conserva
  contexto; close libera ambos. Primeira falha desfaz também contexto e plano.
- Atenção causal MHA/GQA/MQA usa tabelas paginadas, scores F32 por prefixo,
  scratch de reconstrução F32 de 4D e acumulador double de 8D na arena. Scratch
  de quantização compartilha temporalmente o vetor de reconstrução. Kernel sem
  heap, sem buffer F32 de página/prefixo; K é reconstruído duas vezes, V uma vez.
  Validação cobre contexto, spans/alias, capacidades, páginas visíveis e faixa numérica.
- Comparação wide D64/P16, contexto 16/chunks 2/IDs `[1,3,5,7,2,4,6,8]`, grupo 32,
  TQ 3 bits/seed 42: F32/Q4/Q3/TQ usam 512/80/64/64 B/token K/V, páginas residentes de
  8.255/1.343/1.087/1.087 B e picos gerenciados de 79.614/72.702/72.446/73.618 B.
  TQ tem contexto de 404 B, scratch de 256 B e acumulador de 512 B. Seu payload iguala Q3,
  mas contexto/temporários aumentam o pico total. Orçamento comum de 96 KiB; não RSS/VRAM.
- Erro máximo de execução zero em todos contra seus oracles. Na wide, erros
  máximos de KV Q4/Q3/TQ = 0,2619032860/0,5301163346/0,5143706203. Na tiny D4,
  TQ usa 20 B/token K/V e pico de 67.233 B; erro dos pesos 0,4435420930, KV 1,1905089021,
  combinado 1,1530171633. Quantização é com perda; nenhuma prova de qualidade geral,
  velocidade ou GPU. Relatórios `artifacts/reports/tq-kv-wide-{f32,q4,q3,tq}.json`
  e `artifacts/reports/nexalm-kv-tq-verified.json`.
- Validação macOS ARM64/Python 3.14.5: **437 regressões + 110 testes bootstrap
  = 547 testes, zero falhas e zero skips**. São 41 novos testes: planner (11),
  contexto/lifecycle (9), kernels (7) e oracle/sessão/CLI (14). Golden gerado por oracle
  independente com codebook fixo, sem participação do runtime nativo. Cobertura
  inclui bits 1–8, seeds negativos/extremos, GQA/MQA/MHA, chunks, páginas, prefixo,
  importação/exportação, código/norma corrompidos, NaN e overflow, orçamento exato,
  falhas após K escrito, rollback e liberação com tracebacks retidos.
- ASan/UBSan, heap proibido nos kernels, `make -C runtime all test`, `compileall`,
  links Markdown e `git diff --check` passaram. Sem site-packages, 70 testes TQ
  passaram e três oracles Torch opcionais foram ignorados; o golden executou.
  A CI ganhou execução TQ paginada offline. Matriz remota continua pendente.
- Logs: `artifacts/reports/tq-kv-{regressions,bootstrap,native-build,without-site-packages}.log`.
  Guia: `docs/NEXALM_KV_TQ_CPU.md`. Checklist: **28 concluídos e 84 pendentes**,
  após preservar tiers/demais codecs em subitens próprios. Próximo: M4.02d/M4.05.

Décimo primeiro incremento:

- Concluídos M4.02d/M4.05a: política CPU_PAGE_AGE_F32_Q4_Q3_V1 com contagens hot
  e warm explícitas, cold para as páginas restantes. Idade é distância da página
  mais recente; página parcial continua hot. Descritores imutáveis fixam posição,
  tokens válidos, codec/versão/grupo e hash do layout, que não é checksum do payload.
  Planos/transições têm validação canônica, limites e round-trip JSON rigoroso.
- Atenção mista F32/Q4/Q3 sem heap nem expansão de heads/páginas, redução double,
  scratch F32 de scores e tabelas de ponteiros/codec/capacidade na arena. Codecs
  homogêneos F32/Q4/Q3/TQ preservados. TQ em páginas mistas não foi implementado.
- Recodificação após logits/chunk completo e antes do commit. F32→Q4, F32→Q3
  direto e Q4→Q3 via ponte F32 do valor atual. Origem permanece até a publicação;
  não há cópia oculta do F32 original. Fronteiras de chunk afetam quantização e
  logits; CLI registra token_chunks e o oracle repete as mesmas transições.
- Reserva de páginas = residência canônica máxima + páginas F32 do chunk +
  reserva conservadora de substitutas. Scratch de migração = 4D+24 B; fase de
  atenção libera arena antes de migrar. Relatório separa residência, reserva,
  pico de cada fase, erro local/bytes lógicos/custo de migração e erro nos logits.
  Nenhuma métrica é tratada como RSS/VRAM ou tráfego físico de memória.
- D64/P2/H1/W1/G32, contexto 16/chunks `[2,2,2,1,1]`, oito IDs e 96 KiB:
  residência F32/Q4/Q3/idade = 4.348/1.276/764/1.788 B; payload
  4.096/640/512/1.440 B; picos gerenciados 75.323/72.316/71.932/73.980 B.
  A política terminou Q3/Q3/Q4/F32; cinco recodificações ao longo da execução,
  3.392 B lógicos de origem e 736 B de destino. Migração somou ~0,138 ms nesse
  ensaio isolado, sem constituir benchmark de velocidade. Relatórios em
  `artifacts/reports/kv-tiers-wide-{f32,q4,q3,age}.json`.
- Todos coincidiram com seus oracles: erro máximo de execução zero. Erro de KV
  wide Q4/Q3/idade = 0,2619032860/0,5301163346/0,3118940145. Tiny D4/P1/G3,
  chunks `[[1,3],[5],[7]]`: erro dos pesos 0,4435420930, KV 0,1466720104 e combinado
  0,4435420930; 764 B de páginas, sem ganho físico por causa do alinhamento.
  Dados sintéticos não comprovam qualidade/perplexidade ou benefício geral.
- Validação macOS ARM64/Python 3.14.5: **486 regressões + 110 testes bootstrap
  = 596 testes, zero falhas e zero skips**. São 49 novos testes: plano (18), kernels (10),
  sessão/oracle/CLI (14), lifecycle (7). Incluem ASan/UBSan, heap proibido, aliases,
  capacidades/corrupção/overflow, todas as ordens de páginas, recodificação,
  orçamento exato, política imutável, falhas em buffers/destinos/relatório,
  rollback, liberação com tracebacks retidos e retry.
- Sem site-packages, 46 passaram e três oracles Torch opcionais foram ignorados;
  o golden independente e a CLI sem referência rodaram. `make -C runtime all test`,
  `compileall` e `git diff --check` passaram. CI recebeu cenário age offline;
  matriz remota e dispositivos GPU continuam pendentes.
- Logs: `artifacts/reports/kv-tiers-{regressions,bootstrap,native-build,without-site-packages}.log`.
  Guia: `docs/NEXALM_KV_TIERS_CPU.md`. Checklist: **30 concluídos e 83 pendentes**,
  preservando promoção/evicção/qualidade no subitem restante. Próximo: backing
  store de KV CPU e recarga com memória limitada, dentro de M4.05b.

## Registro do décimo segundo incremento — backing store CPU

- Concluído M4.05b: formato NEXAKV01 V1, identidade por manifesto/descritor,
  checksums de metadata e payload, diretório privado por sessão, publicação
  atômica e I/O limitado. Payload mantém os bytes packed Q3 e padding físico.
  A página lógica independe de residência; cold usa disco, hot/warm usam RAM.
- `compiler/offloaded_kv_plan.py` admite fontes antigas, páginas novas,
  destinos de migração, slot de recarga, I/O e scratch antes de carregar payload.
  `runtime/nexapack/offloaded.py` integra escrita/atenção/migração/evicção/commit,
  e `kv_store.py` mantém propriedade e cleanup retentável dos arquivos próprios.
- `nexa_causal_gqa_attention_page/finish` leem F32/Q4/Q3 diretamente, em duas
  passagens por camada. Estado `8*T*Hq*(D+2)` sem scores do prefixo, sem heap e
  sem expansão integral. Slot e arena são liberados antes da migração/persistência.
- Falhas de leitura, corrupção, escrita, falta de espaço, fsync/rename e
  cancelamento anteriores ao commit preservam fontes/tokens/relatório. Limpeza
  posterior ao commit é tolerante a falhas; retry reconcilia remoções já concluídas.
  Falha de limpeza durante rollback também mantém ownership para close retentável.
- CLI `--kv-backing-store DIRETÓRIO` requer `--kv-policy age`. Relatórios separam
  residência, payload em disco, reserva/uso do slot, pico de buffers, bytes e tempos
  de I/O; não são medições de RSS/VRAM. Oracle segue os mesmos chunks de tiers.
- Prova D64/P2/H1/W1/G32, capacidade 512, chunks de 2: logits exatamente iguais
  aos tiers residentes. KV final 1.406 B contra 49.920 B; pico 91.899 B contra
  130.431 B; reserva admitida 93.136 B contra 212.566 B. Houve 254 evicções e
  64.262 recargas, 43.627.130 B lidos e 172.555 B gravados. Uma amostra local
  observou ~2,76 s offloaded contra ~0,66 s residente: ganho de RAM tem custo de I/O.
  Não é benchmark repetido nem evidência de qualidade/NVMe/GPU.
- Fixture reproduzível: `tests/kv_backing_fixture.py --out DIRETÓRIO`; comparação
  em `artifacts/reports/kv-offloaded-512-{resident,offloaded,comparison}.json`.
  Tiny com oracle: `artifacts/reports/kv-offloaded-tiny.json`.
- Validação macOS ARM64/Python 3.14.5: **533 regressões + 110 bootstrap = 643 testes,
  zero falhas e zero skips**. Os 47 novos testes cobrem plano (3), store (16),
  kernels (10), sessão/oracle/CLI (8) e lifecycle (10). Incluem ASan/UBSan,
  orçamento exato, GQA/MQA/MHA, páginas parciais, bytes preservados, short I/O,
  falhas/cancelamento/cleanup e liberação de owners com tracebacks retidos.
- Sem site-packages, **46 passaram e um oracle Torch opcional foi ignorado**.
  Golden, CLI e kernels continuam independentes de Torch. `make -C runtime all test`,
  `compileall`, links locais e `git diff --check` passaram. CI recebeu cenário
  offline com backing store; fallback sem dir_fd foi exercitado por simulação
  local, sem declarar execução remota Windows/Linux nesta etapa.
- Logs: `artifacts/reports/kv-offloaded-{regressions,bootstrap,native-build,without-site-packages}.log`.
  Guia: [backing store CPU](NEXALM_KV_BACKING_CPU.md). Checklist: **31 concluídos,
  83 pendentes**. M4.05c preserva promoção/qualidade; próximo incremento técnico:
  reuso e promoção de residência de páginas cold com slots admitidos.
- Na conclusão deste incremento, Plastic Learning ainda não estava analisado.
  A revisão documental subsequente incorporou suas tarefas PL e o plano Conditional
  Compute; ver o [índice dos PDFs](PLANOS_IMPLEMENTACAO_INDICE.md). O incremento CPU
  continua sem implementar essas capacidades futuras.

## Registro do décimo terceiro incremento — reuso de páginas cold

- Concluído M4.05c: `ReloadPageCache` com `N` slots admitidos,
  `CPU_RELOAD_FIRST_TOUCH_MRU_V1` e escopo de sessão. Um slot é alocado no
  primeiro miss que precisa dele; com o cache cheio, substitui-se o slot
  carregado por último. Sob a varredura crescente da atenção, LRU erraria em
  todas as páginas; esta política mantém `N-1` fixas e usa a última como passagem.
- `compiler/offloaded_kv_plan.py` recebe `reload_slots`, valida o intervalo contra
  as páginas do contexto, publica `reload_cache_capacity_bytes` e a política, e
  limita `reload_slots_used` às páginas cold da própria transação.
  `runtime/nexapack/reload_cache.py` é dono apenas dos slots; arquivos,
  referências e páginas confirmadas continuam do store e da sessão.
- Bytes, layout, codecs, quantização e a ordem de redução das duas passagens não
  mudaram. Nenhuma leitura foi evitada às custas da ordem de soma da passagem 1.
- Uma leitura que falha descarta sua entrada antes de carregar e só publica após
  verificar o payload. Falha ou cancelamento antes do commit libera todos os
  slots, inclusive os já carregados. Após o commit permanecem apenas entradas de
  páginas finais; prefill substituto, reset e close esvaziam o cache.
- Orçamento: `reserva = page_allocation_limit + N*A_C + migration_scratch + io`.
  Os slots sobrevivem à migração da mesma chamada, então o pico das duas fases
  inclui a residência do cache. Relatórios separam slots, entradas, capacidade,
  acertos, bytes evitados, admissões e substituições; `kv_page_reloads` continua
  contando somente leituras efetivas.
- Prova D64/P2/H1/W1/G32, capacidade 512, chunks de 2, uma camada, 253 páginas
  cold: logits idênticos em 1, 8, 64 e 256 slots e iguais aos tiers residentes.
  Recargas 64.260 → 60.767 → 36.351 → 253; bytes lidos 43.625.778 → 171.875;
  pico gerenciado 91.899 B → 140.031 B; cache residente 191 B → 48.323 B. As 254
  evicções para disco não mudaram. Amostras locais únicas: ~4,2 s contra ~2,3 s;
  não é benchmark repetido nem evidência de NVMe, disco de rede ou GPU.
- Cenário sem ganho registrado em regressão: com uma única página cold, mais
  slots não evitam leitura e apenas aumentam a capacidade admitida. No extremo
  oposto, 256 slots superam o pico de 130.431 B dos tiers residentes do
  incremento anterior — reter tudo reproduz o custo de RAM do modo residente.
- Validação macOS ARM64/Python 3.14.5: **547 regressões + 110 bootstrap = 657
  testes, zero falhas e zero skips**. Os 14 novos testes cobrem o cache (7),
  o plano (2), sessão/CLI (4) e lifecycle (1), incluindo política sob varredura
  cíclica, falha de load, discard/retain, clear, orçamento, equivalência de
  logits/bytes entre contagens de slots e rejeições da CLI.
- Sem site-packages, 28 testes offloaded passaram com um oracle Torch opcional
  ignorado, e os 7 do cache passaram. `make -C runtime all test`, `compileall` e
  `git diff --check` passaram; os kernels C não foram alterados neste incremento.
  A CI recebeu cenário offline com quatro slots.
- Relatórios: `artifacts/reports/kv-reload-slots-{1,8,64,256}.json`.
  Guia: [reuso de páginas cold](NEXALM_KV_RELOAD_CPU.md). Checklist:
  **32 concluídos e 107 pendentes**; a divisão de M4.05 e as duas tarefas CC
  acrescentadas na mesma revisão aumentam a contagem, sem equivaler a percentual
  de conclusão ou prazo. Próximo incremento técnico: M4.05d.

## Registro do décimo quarto incremento — sequências derivadas

- Concluído M4.06a: `PagedTransformerSession.fork()` e `_adopt_prefix` no
  executor paginado homogêneo (F32/Q4/Q3/TQ). A derivada admite o próprio
  orçamento antes de reter ou copiar qualquer página, exige o mesmo manifesto e
  um layout idêntico — inclusive bits/seed/codebook TQ — e recusa um prefixo
  maior que a própria capacidade.
- `_KVPage` passa a contar referências. Páginas completas são retidas, não
  copiadas; a página parcial é copiada, no máximo uma por adoção. `release`
  devolve uma referência e só libera a alocação com o último dono, então reset,
  prefill substituto e close de qualquer sequência preservam as demais.
- Escrita em página compartilhada é rejeitada antes de tocar bytes, com prefixo
  e relatório preservados; é defesa em profundidade, já que a adoção copia a
  página parcial exatamente para que esse caso não ocorra.
- Relatórios separam `kv_shared_page_count`, `kv_shared_allocation_bytes` e
  `kv_owned_allocation_bytes`, e a adoção publica `kv_prefix_adoption` com
  tokens herdados, páginas compartilhadas/copiadas e bytes copiados. Bytes
  compartilhados existem uma vez no processo; somar sequências os conta duas vezes.
- Prova D64, prompt de 256 tokens em chunks de 32, páginas de 16, KV Q4 G32,
  dois ramos de 4 tokens: logits idênticos aos de sequências independentes; KV
  residente somado 45.662 B contra 24.174 B; amostra local ~0,63 s contra
  ~0,04 s, porque o prompt é executado uma vez. Não é benchmark repetido.
- Tiers por idade e backing store recusam `fork` explicitamente e ficaram em
  M4.06b, junto de cancelamento em andamento e admissão conjunta por processo.
- Validação macOS ARM64/Python 3.14.5: **558 regressões + 110 bootstrap = 668
  testes, zero falhas e zero skips**. Os 11 novos cobrem equivalência por codec,
  divergência sem contaminar bytes, sobrevivência a reset/prefill/close em
  qualquer ordem, escrita rejeitada, layout/capacidade, falha de adoção,
  relatórios, fork de fork, recusa dos tiers e a CLI.
- Guia: [sequências derivadas](NEXALM_KV_SEQUENCIAS_CPU.md). Checklist:
  **33 concluídos e 107 pendentes**; M4.06 foi dividido preservando o restante.
  Próximo incremento técnico: M4.06b.

## Registro do décimo quinto incremento — sequências sob tiers

- Concluído M4.06b: `TieredTransformerSession` aceita `fork`. A derivada herda
  `_page_descriptors`, e a recodificação por idade permanece privada porque cria
  a página de destino e apenas libera a origem compartilhada.
- Correção de contrato: a identidade de layout usava só o plano F32, que ignora
  hot/warm e `group_size`. Uma derivada com outra política seria aceita e leria
  páginas Q4/Q3 herdadas com parâmetros errados. A comparação passou a usar o
  plano de tiers completo; o executor homogêneo mantém seu próprio plano.
- Relatórios de tiers ganharam `kv_shared_page_count`,
  `kv_shared_allocation_bytes` e `kv_owned_allocation_bytes`, zerados no reset.
  `--fork-tokens` passou a aceitar `--kv-policy age` e continua recusando
  `--kv-backing-store`.
- Custo registrado: cada sequência pode pagar a mesma recodificação, e duas que
  migrem a mesma página lógica passam a ocupar duas páginas físicas. O ganho de
  compartilhamento diminui conforme os ramos envelhecem de formas diferentes.
- Validação macOS ARM64/Python 3.14.5: **564 regressões + 110 bootstrap = 674
  testes, zero falhas e zero skips**. Os 6 novos cobrem herança de codecs com
  logits idênticos, migração privada, cópia da página parcial hot, relatórios e
  sobrevivência ao reset do pai, rejeição de política divergente e falha de adoção.
- Guia: [sequências derivadas](NEXALM_KV_SEQUENCIAS_CPU.md). Checklist:
  **34 concluídos e 107 pendentes**. Próximo incremento técnico: M4.06c.

## Registro do décimo sexto incremento — tokenizer

- Concluído LLM.02c1: `runtime/nexapack/tokenizer.py` (formato, encode/decode,
  métricas), `compiler/tokenizer_trainer.py` (treino determinístico) e
  `tools/nexa_tokenizer.py` (train/inspect/encode/decode).
- Formato `NexaTokenizer` V1: `manifest.json` com formato, versão, segmentação,
  especiais, identidade do corpus e SHA-256 de cada arquivo; `vocab.bin`
  (NEXATOKV) e `merges.bin` (NEXATOKM) little-endian. O carregamento rejeita
  versão/modelo/segmentação diferentes, checksum ou tamanho divergente, magic
  inválido, truncamento, bytes sobrando, merge fora do vocabulário e ID especial
  que não corresponde à entrada.
- Propriedade de segurança verificada: nenhum texto produz um token especial.
  `<|system|>` num prompt vira bytes literais; papéis só entram por
  prefix/suffix explícitos ou `--bos`.
- `nexa_run.py --prompt TEXTO --tokenizer DIR [--bos]` substitui `--tokens`,
  valida `vocab_size` contra o modelo e acrescenta `prompt`, `tokenizer` (com
  hashes), `generated_text` e `decoded_text` ao relatório. Os modos de KV
  existentes continuam disponíveis nesse caminho.
- Validação macOS ARM64/Python 3.14.5: **583 regressões + 110 bootstrap = 693
  testes, zero falhas e zero skips**. Os 19 novos cobrem determinismo e
  independência de ordem, corpus insuficiente, round-trip byte-exato com fuzz
  determinístico, especiais, segmentação, merges que não cruzam fronteira,
  métricas, dez mutações de asset, CLI e a integração texto→modelo→texto.
- Guia: [tokenizer](NEXALM_TOKENIZER.md). Checklist: **35 concluídos e 107
  pendentes**; LLM.02c foi dividido preservando o congelamento em LLM.02c2.

## Registro do décimo sétimo incremento — codecs de peso

- Concluído M1.05a: `RAW_F32_MATRIX` no bundle, `nexa_f32_matmul` no runtime
  nativo e despacho por codec no executor, inclusive para embedding.
- Blocos de linhas são a unidade de checksum e de leitura: uma leitura parcial
  não poderia verificar os bytes consumidos, então `--block-rows` da escrita
  define o tile do executor. A abertura exige que os blocos cubram cada linha
  exatamente uma vez, em ordem.
- Equivalência provada sem Torch: com pesos múltiplos de 0,5 em grupos cujo
  máximo é 3,5 — escala exata de 0,5 — denso e Q4 produzem os mesmos logits e o
  mesmo SHA-256. Com pesos arbitrários, a diferença é o erro de quantização.
- `tools/nexa_convert.py --dense-all/--dense-tensor`, `nexa_inspect --verify`
  com `dense_payload_bytes_read` e relatórios que separam bytes Q4 de bytes raw.
- Custo declarado: oito vezes a forma Q4 em disco, tile de
  `block_rows × cols × 4` no orçamento, e um bloco inteiro lido por token de
  embedding alcançado. É formato de referência e calibração, não de distribuição.
- Validação macOS ARM64/Python 3.14.5: **595 regressões + 110 bootstrap = 705
  testes, zero falhas e zero skips**. Os 12 novos cobrem equivalência exata,
  determinismo, bundle misto, leitura seletiva do embedding, corrupção e
  truncamento de bloco, manifesto com blocos inválidos, bloco acima do limite do
  leitor, mapas de codec inválidos e as três CLIs.
- Guia: [codecs de peso](NEXALM_CODECS_PESOS.md). Checklist: **36 concluídos e
  107 pendentes**; M1.05 foi dividido preservando Q2/Q3/Q8/F16 em M1.05b.

## Comandos para validar e retomar

```sh
python3 -m unittest discover -s tests -p 'test_*regressions.py' -v
python3 tests/run_tests.py
make -C runtime all test

# Formato portátil TQ e migração, sem dependências Python externas.
python3 -S -m unittest discover -s tests -p 'test_tq_portable*regressions.py' -v
# Gerar fixture e converter: veja a receita em docs/NEXAPACK_TQ_V1.md.

# Na primeira geração, use uma saída nova. Se já existir, omita --generate-demo.
python3 tools/nexa_bench.py --generate-demo --pack artifacts/models/blueprint-first-milestone.nxp --rows 4096 --cols 128 --tile-rows 64 --memory-budget 96KiB --verify --report artifacts/reports/blueprint-first-milestone.json --csv artifacts/reports/blueprint-first-milestone.csv

# Deve falhar com status diferente de zero antes da carga do payload.
python3 tools/nexa_bench.py --pack artifacts/models/blueprint-first-milestone.nxp --memory-budget 1KiB

# Definições estruturais R0/v1.
python3 tools/nexa_model.py models/nexalm512/architecture.nxl --out artifacts/reports/nexalm-architecture.json

# Nas duas primeiras linhas, as pastas de saída devem ser novas.
python3 tests/model_checkpoint_fixture.py --out artifacts/checkpoints/nexalm-tiny
python3 tools/nexa_convert.py --checkpoint artifacts/checkpoints/nexalm-tiny --out artifacts/models/nexalm-tiny --group-size 4 --block-rows 3
python3 tools/nexa_inspect.py artifacts/models/nexalm-tiny --verify
python3 tools/nexa_bench.py --bundle artifacts/models/nexalm-tiny --tensor lm_head.weight --batch 2 --tile-rows 3 --memory-budget 96KiB --verify --report artifacts/reports/nexalm-bundle-q4.json

# KV TQ paginado e atenção CPU, sem PyTorch na execução.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 2 --kv-codec tq --kv-bits 3 --kv-seed 42 --prefill-chunk-size 2 --memory-budget 96KiB --tile-rows 3 --report artifacts/reports/nexalm-kv-tq.json

# Política CPU de idade; recodificação e atenção mista sem PyTorch.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --kv-cache --kv-page-tokens 1 --kv-policy age --kv-hot-pages 1 --kv-warm-pages 1 --kv-group-size 3 --prefill-chunk-size 2 --memory-budget 96KiB --tile-rows 3 --report artifacts/reports/kv-tiers-tiny.json

# Evicção de cold Q3 e atenção causal com um slot de recarga.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --kv-cache --kv-page-tokens 1 --kv-policy age --kv-backing-store artifacts/kv-store/demo --kv-group-size 3 --prefill-chunk-size 2 --memory-budget 96KiB --tile-rows 3 --report artifacts/reports/kv-offloaded-tiny.json

# Reuso de páginas cold entre passagens, camadas e chamadas.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --kv-cache --kv-page-tokens 1 --kv-policy age --kv-backing-store artifacts/kv-store/reuso --kv-reload-slots 4 --kv-group-size 3 --prefill-chunk-size 2 --memory-budget 96KiB --tile-rows 3 --report artifacts/reports/kv-reload-tiny.json
python3 -m unittest discover -s tests -p 'test_reload_cache_regressions.py' -v

# Sequência derivada que continua o prefixo sem recomputá-lo.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5,7 --kv-cache --kv-page-tokens 2 --kv-codec q4 --kv-group-size 4 --max-sequence-length 8 --tile-rows 3 --memory-budget 1MiB --fork-tokens 2,4 --report artifacts/reports/kv-sequencias-tiny.json
python3 -m unittest discover -s tests -p 'test_paged_sequences_regressions.py' -v

# Codec de peso por tensor: referência densa e bundle misto.
python3 tools/nexa_convert.py --checkpoint artifacts/checkpoints/nexalm-tiny --out artifacts/models/nexalm-dense --dense-all --block-rows 3
python3 tools/nexa_inspect.py artifacts/models/nexalm-dense --verify
python3 tools/nexa_run.py artifacts/models/nexalm-dense --tokens 1,3 --decode-tokens 5 --tile-rows 3 --memory-budget 4MiB
python3 -m unittest discover -s tests -p 'test_dense_weights_regressions.py' -v

# Tokenizer: treino determinístico, verificação e execução a partir de texto.
python3 tools/nexa_tokenizer.py train --corpus CORPUS --out artifacts/tokenizers/demo --vocab-size 400
python3 tools/nexa_tokenizer.py inspect artifacts/tokenizers/demo --samples SAMPLES.json
python3 tools/nexa_run.py MODELO --prompt "O NexaLang compila" --tokenizer artifacts/tokenizers/demo --bos --generate 4 --kv-cache --kv-page-tokens 2 --max-sequence-length 48 --tile-rows 4 --memory-budget 8MiB
python3 -m unittest discover -s tests -p 'test_tokenizer_regressions.py' -v

# Sequência derivada sob a política de idade, com migração privada.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --kv-cache --kv-page-tokens 1 --kv-policy age --kv-hot-pages 1 --kv-warm-pages 1 --kv-group-size 3 --max-sequence-length 8 --tile-rows 3 --memory-budget 1MiB --fork-tokens 7,2
python3 -m unittest discover -s tests -p 'test_tiered_sequences_regressions.py' -v
python3 -S -m unittest discover -s tests -p 'test_offloaded*regressions.py' -v
python3 -S -m unittest discover -s tests -p 'test_kv_store_regressions.py' -v
python3 -S -m unittest discover -s tests -p 'test_streaming_kv_kernels_regressions.py' -v

# Grafo completo e forward nativo, sem PyTorch.
python3 tools/nexa_model.py models/nexalm512/architecture.nxl --sequence-length 4 --out artifacts/reports/transformer-graphs.json
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-forward.json

# Opcional, com PyTorch local, separa erro de execução e de quantização.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --tile-rows 3 --memory-budget 96KiB --verify --reference-checkpoint artifacts/checkpoints/nexalm-tiny --report artifacts/reports/nexalm-forward.json

# KV F32 incremental nativo; prefill/decode sem recomputar posições anteriores.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3 --decode-tokens 5,7 --kv-cache --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-f32.json

# Prefill em chunks; mesmo contexto e hash de logits com ativações menores.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --prefill-chunk-size 2 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-chunks.json

# Evidência sem dependências opcionais de site-packages (oracle Torch é ignorado).
python3 -S -m unittest discover -s tests -p 'test_incremental_transformer_regressions.py' -v

# Páginas F32 sob demanda, atenção direta e prefill em chunks.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 2 --prefill-chunk-size 2 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-paged.json

# Regressões específicas de plano, kernels e sessão paginada.
python3 -m unittest discover -s tests -p 'test_*paged*regressions.py' -v
python3 -S -m unittest discover -s tests -p 'test_paged_transformer_regressions.py' -v

# Q4 KV por token/head; páginas parciais já comprimidas, sem re-encode do prefixo.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 8 --kv-codec q4 --kv-group-size 4 --prefill-chunk-size 2 --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-q4.json

# Opcional: separa erro de execução, pesos, KV e combinado usando PyTorch local.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 8 --kv-codec q4 --kv-group-size 4 --prefill-chunk-size 2 --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB --verify --reference-checkpoint artifacts/checkpoints/nexalm-tiny --report artifacts/reports/nexalm-kv-q4.json

# Inclui golden Q4 KV sem dependência de Torch.
python3 -S -m unittest discover -s tests -p 'test_q4_paged_transformer_regressions.py' -v

# Q3 KV; acrescente --verify e --reference-checkpoint para diagnóstico opcional.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 8 --kv-codec q3 --kv-group-size 4 --prefill-chunk-size 2 --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-q3.json
python3 -m unittest discover -s tests -p 'test_q3*regressions.py' -v
python3 -S -m unittest discover -s tests -p 'test_q3_paged_transformer_regressions.py' -v

# Estado TurboQuant MSE linear, goldens legados, falhas e integração native/JIT.
python3 -m unittest discover -s tests -p 'test_turboquant_mse_regressions.py' -v
python3 -m unittest discover -s tests -p 'test_mse_integration_regressions.py' -v
python3 -S -m unittest discover -s tests -p 'test_turboquant_mse_regressions.py' -v
make -C runtime all test
artifacts/build/runtime/test_tq_mse_regressions memory > artifacts/reports/turboquant-mse-memory.json
```

No Windows, substitua `python3` por `python`; `python runtime/build_runtime.py` é
a alternativa portátil ao build do Makefile. A regressão Python compila/executa
os testes C e integra a matriz da CI.

## Arquivos do primeiro incremento

- `compiler/model_ir.py`, `compiler/hardware_profile.py`, `compiler/planner/memory.py`.
- `runtime/nexapack/format.py`, `runtime/nexapack/q4.c`, `runtime/nexapack/q4.h`.
- `tools/nexa_bench.py`, `tools/nexa_convert.py` e testes de regressão correspondentes.

- `runtime/nexapack/executor.py`: execução por tiles usando os offsets do plano real.
- `tests/test_model_ir_regressions.py`, `tests/test_nexapack_regressions.py`,
  `tests/test_q4_runtime_regressions.py`, `tests/test_blueprint_regressions.py`.
- `docs/NEXAPACK_V1.md`: especificação de bytes, APIs e escopo de memória.

## Arquivos do segundo incremento

- `compiler/model_config.py`, `compiler/model_definition.py`, `models/nexalm512/`.
- `compiler/importers/safetensors.py`, `compiler/importers/llama.py`.
- `runtime/nexapack/bundle.py` e integração no executor existente.
- `tools/nexa_model.py`, `tools/nexa_inspect.py`, conversor e benchmark ampliados.
- `tests/model_checkpoint_fixture.py`, regressões model_config/model_bundle/
  safetensors/model_pipeline e sequência offline na CI.
- `docs/NEXALM_IMPORTACAO.md`: contrato, limites e comandos reproduzíveis.

## Arquivos do terceiro incremento

- `compiler/model_lowering.py`, operadores novos em `compiler/model_ir.py` e
  `tools/nexa_model.py --sequence-length`.
- `runtime/nexapack/transformer.c/.h`, `transformer.py`, helper Q4 de embedding e
  `ModelBundleReader.read_f32_into`, integração nos builds Python/Makefile.
- `tools/nexa_run.py`, `tests/transformer_reference.py`, regressões
  transformer_ir/transformer_kernels/transformer_forward/transformer_cli e golden JSON.
- `tests/test_runtime_build_regressions.py`: publicação de bibliotecas em uso.
- `docs/NEXALM_EXECUCAO_CPU.md`, sequência de forward offline na CI.

## Arquivos do quarto incremento

- `compiler/kv_plan.py`: layout persistente, agenda explícita e lifetimes por chunk.
- `runtime/nexapack/incremental.py`, hooks no executor `transformer.py` e kernels
  com offset RoPE/atenção sobre prefixo persistente em `transformer.c/.h`.
- `runtime/nexapack/format.py`: liberação do scratch após falhas de leitura/checksum.
- `tools/nexa_run.py --kv-cache --prefill-chunk-size`, sequência incremental na CI.
- Regressões `test_kv_plan`, `test_kv_kernels`, `test_incremental_transformer`,
  `test_chunked_prefill` e `test_buffer_lifetime`, todas com sufixo `_regressions.py`.
- `docs/NEXALM_KV_CPU.md`, `docs/NEXA_OMNI_AJUSTES.md` e os gates OMNI deste checklist.

## Arquivos do quinto incremento

- `compiler/paged_kv_plan.py`: layout por página, segmentos de escrita, lifetimes,
  planos JSON estritos e limites de admissão inclusive append desalinhado.
- `runtime/nexapack/paged.py`: alocação sob demanda, tabelas planejadas, commit,
  rollback e liberação de páginas; FFI ampliada em `transformer.py`.
- `runtime/nexapack/transformer.c/.h`: atenção paginada sem heap ou concatenação.
- `tools/nexa_run.py --kv-page-tokens`, sequência paginada na CI e
  `tests/test_paged_kv_plan_regressions.py`, `tests/test_paged_kv_kernels_regressions.py`,
  `tests/test_paged_transformer_regressions.py`.
- `docs/NEXALM_KV_PAGINADO_CPU.md`: API, layout, memória e reproduções comparáveis.

## Arquivos do sexto incremento

- `compiler/paged_kv_plan.py`: codec Q4 e metadados/layout/bindings por head,
  preservando o JSON F32 anterior.
- `runtime/nexapack/paged.py`: quantização dos segmentos nas próprias páginas;
  FFI/contabilidade de tempo em `transformer.py` e hook ajustado em `incremental.py`.
- `runtime/nexapack/transformer.c/.h`: atenção Q4 paginada, validação de prefixo e
  redução direta de escala/código; reutiliza quantizador existente em `q4.c`.
- `tools/nexa_run.py --kv-codec --kv-group-size`, sequência Q4 KV na CI.
- `tests/transformer_reference.py`: oracle de KV quantizado independente do C,
  diagnóstico separado de erros; golden e regressões em `test_q4_kv_plan_regressions.py`,
  `test_q4_kv_kernels_regressions.py`, `test_q4_paged_transformer_regressions.py`.
- `docs/NEXALM_KV_Q4_CPU.md`: bytes, contratos, memória e evidência reproduzível.

Sétimo incremento acrescenta:

- `compiler/paged_kv_plan.py`: Q3_GROUPED V1 e tamanho físico de três bits,
  preservando JSON F32/Q4.
- `runtime/nexapack/transformer.c/.h`: tamanho de row, quantizador Q3 e atenção
  paginada; `transformer.py` e `paged.py` integram FFI, escrita e despacho.
- `tools/nexa_run.py --kv-codec q3`, sequência offline Q3 na CI.
- `tests/q3_reference.py`, `tests/transformer_reference.py` e os três arquivos
  `test_q3_*regressions.py`: codec independente, golden, planner, kernels,
  sessão, CLI, orçamento e rollback.
- `docs/NEXALM_KV_Q3_CPU.md`: formato, API, medições e reprodução F32/Q4/Q3.

Oitavo incremento acrescenta:

- `runtime/turboquant.c/.h`: criação MSE explícita, cálculo de estado persistente,
  validação de tamanhos e contrato de ownership/scratch.
- `runtime/turboquant_mse_regressions.c`, `tests/test_turboquant_mse_regressions.py`
  e target em `runtime/Makefile`: allocations instrumentadas, goldens, falhas e
  relatório reproduzível de memória.
- `bootstrap/codegen.py`/`semantic.py`, `std/compress.nxl`, `std/kv_cache_quant.nxl`
  e `tools/chat_tinyllama_turboquant.py`: novos construtores e consumidores MSE.
- `tests/test_mse_integration_regressions.py`: execução native/JIT e wrapper sem modelo.
- `docs/TURBOQUANT_MSE_CPU.md`: APIs, compatibilidade, medições e próximos limites.

Nono incremento acrescenta:

- `runtime/nexapack/tq.py`, APIs TQ02 em `runtime/turboquant.c/.h` e codec TQ em
  `format.py`: centroids persistidos, conversão/validação e migração explícita.
- `tools/nexa_convert.py`/`nexa_inspect.py`, três arquivos `test_tq_portable_*`
  e `docs/NEXAPACK_TQ_V1.md`: CLI, memória, interoperabilidade e contratos.

Décimo incremento acrescenta:

- `compiler/paged_kv_plan.py`: layout TQ02 por token/head e identidade do codebook.
- `runtime/nexapack/tq_attention.c/.h`, target `nexa_tq_attention` e getters/reserva
  do contexto TurboQuant: atenção paginada com scratch do chamador, sem heap.
- `runtime/nexapack/tq_kv.py`/`paged.py`: admissão, contexto compartilhado, arena,
  escrita TQ02, transações, relatório e liberação; flags TQ em `tools/nexa_run.py`.
- `tests/tq_reference.py`/`transformer_reference.py` e quatro arquivos de regressão
  `test_tq_kv_{plan,context,kernels}_regressions.py`,
  `test_tq_paged_transformer_regressions.py`: oracle independente e provas CPU.
- `docs/NEXALM_KV_TQ_CPU.md` e sequência TQ paginada offline na CI.

Décimo primeiro incremento acrescenta:

- `compiler/tiered_kv_plan.py`: política, descritores, identidades, transições e
  reserva de páginas antiga/nova/substituta.
- `runtime/nexapack/transformer.c/.h/.py`: atenção mista e recodificação por head,
  capacidades/alias/erros, estatísticas locais e FFI; sem heap nos kernels.
- `runtime/nexapack/tiered.py`: sessão, publicação transacional, memória por fase,
  métricas, reset/close; `tools/nexa_run.py --kv-policy age` e cenário de CI.
- `tests/tiered_reference.py` e quatro arquivos `test_tiered_*regressions.py`:
  oracle por chunks, golden sem Torch, plano, kernels, lifecycle e integração.
- `docs/NEXALM_KV_TIERS_CPU.md`: contrato, sequência, memória, erros e reprodução.

Décimo terceiro incremento acrescenta:

- `compiler/offloaded_kv_plan.py`: `reload_slots`, capacidade do cache, política
  publicada e slots usados por transição.
- `runtime/nexapack/reload_cache.py`: slots, admissão/substituição, contadores e
  liberação; `runtime/nexapack/offloaded.py` integra cache, relatório e transação.
- `tools/nexa_run.py --kv-reload-slots`, totais de reuso e cenário de CI.
- `tests/test_reload_cache_regressions.py` e acréscimos em
  `test_offloaded_kv_plan_regressions.py`, `test_offloaded_transformer_regressions.py`
  e `test_offloaded_kv_lifecycle_regressions.py`.
- `docs/NEXALM_KV_RELOAD_CPU.md`: política, ownership, memória, métricas e limites.

Décimo quarto incremento acrescenta:

- `runtime/nexapack/paged.py`: contagem de referências em `_KVPage`, `fork`,
  `_adopt_prefix`, rejeição de escrita em página compartilhada e campos de
  residência compartilhada/própria no relatório.
- `runtime/nexapack/tiered.py` e `offloaded.py`: adoção sob tiers com migração
  privada, identidade de layout pela política e recusa explícita no backing store.
- `tools/nexa_run.py --fork-tokens` e bloco `derived_sequence` no relatório.
- `tests/test_paged_sequences_regressions.py` e
  `tests/test_tiered_sequences_regressions.py`: equivalência por codec,
  isolamento de bytes, migração privada, ownership, limites, falhas,
  relatórios e CLI.
- `docs/NEXALM_KV_SEQUENCIAS_CPU.md`: contrato, custo, evidências e limites.

Décimo sétimo incremento acrescenta:

- `runtime/nexapack/bundle.py`: escrita/validação/leitura de `RAW_F32_MATRIX`
  por blocos e `tensor_codecs` no writer.
- `runtime/nexapack/transformer.c/.h`: `nexa_f32_matmul`; `transformer.py`
  despacha por codec e contabiliza os dois caminhos de leitura.
- `compiler/importers/llama.py`, `tools/nexa_convert.py --dense-all/--dense-tensor`
  e `tools/nexa_inspect.py` com verificação bloco a bloco.
- `tests/test_dense_weights_regressions.py` e `docs/NEXALM_CODECS_PESOS.md`.

Décimo sexto incremento acrescenta:

- `runtime/nexapack/tokenizer.py`, `compiler/tokenizer_trainer.py` e
  `tools/nexa_tokenizer.py`: formato, treino, codec, métricas e CLI.
- `tools/nexa_run.py --prompt/--tokenizer/--bos` e campos de texto no relatório.
- `tests/test_tokenizer_regressions.py` e `vocab_size`/`max_position_embeddings`
  parametrizáveis na fixture de bundle.
- `docs/NEXALM_TOKENIZER.md`: formato, segmentação, reprodutibilidade e limites.

Limites atuais: DSL e pipeline de modelos separados de nxc; executor C scalar
orquestrado em Python, uma sequência, baseline por recomputação e KV incremental
opcional F32 (dois bancos/páginas), Q4/Q3/TQ paginado ou política mista F32/Q4/Q3
por idade em RAM, com backing store opcional para evicção/recarga de cold Q3 CPU.
Páginas cold podem ser reutilizadas em slots admitidos, e sequências derivadas
compartilham o prefixo paginado, homogêneo ou por idade. Pesos Q3/TQ, TQ misto,
promoção de precisão, prefetch, múltiplas sequências e compartilhamento de
prefixos permanecem pendentes. O cache privado CPU não implementa residência
GPU de experts, roteamento condicional ou Plastic Learning.
Há tokenizer byte-level com execução a partir de texto, ainda sem vocabulário
congelado em corpus real.
Forward/logits validados em pesos sintéticos pequenos; treinamento,
qualidade de modelo real, KernelIR/backend GPU, qint na sintaxe e Omni permanecem pendentes.
