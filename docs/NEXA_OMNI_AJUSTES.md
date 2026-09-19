# Nexa Omni — integração ao plano de desenvolvimento

Referência: [plano Nexa Omni](NexaLang_Plano_Implementacao_Nexa_Omni.pdf), 14 páginas.
Este complemento registra decisões de integração; o PDF original foi preservado.
Os itens OMNI no [checklist](BLUEPRINT_512MB_CHECKLIST.md) acompanham a implementação.

## Escopo e relação com a base existente

O Omni acrescenta orquestração multi-modelo à infraestrutura de execução: Pack ABI,
Model Packs, Router, Brain, Registry, Scheduler, DAG, memória, Critic e Integrator.
`nxpkg` permanece como gerenciador único. O caminho V0 usa mensagens tipadas,
`CognitivePacket`, referências a memória/artefatos e schemas explícitos. Neural Bus
com estados latentes é pesquisa posterior, não um pré-requisito da primeira versão.

O plano sugere um Fast Router de 20–50M parâmetros ou um classificador menor.
Não fixa layers, hidden size ou vocabulário de Brain/experts, portanto não se devem
inventar configurações para eles. As definições R0/v1 existentes continuam válidas
para seus próprios marcos; não representam, por si, Brain e especialistas treinados.

O NexaModelBundle oferece pesos, shapes, aliases e proveniência para um futuro
Model Pack. Ainda faltam manifesto de capacidades/qualidade/residência, variantes,
calibração, assinatura/licença, ABI e integração ao `nxpkg`. A inspeção sem carregar
pesos deve ser preservada.

Os exemplos de manifestos e comandos das páginas 4–5 e 11–12 são propostas de API.
Hoje `nxpkg` não instala automaticamente seções `packs/models`; os novos comandos
`model`, `doctor`, `resolve` e `optimize` exigem implementação e testes próprios.

## Numeração canônica dos gates

Na página 12, o diagrama F0–F6 omite Critic, enquanto a tabela de §20 contém F0–F7.
O checklist adota **a tabela de §20**, que preserva essa etapa explicitamente:

| Gate | Entrega necessária |
|---|---|
| F0 | Pack ABI, manifesto e instalação/lock local pelo nxpkg. |
| F1 | Router/Brain/Registry/Packet com três modelos externos executando DAG simples. |
| F2 | Memória, Scheduler e tiers Hot/Warm/Cold sob orçamento declarado. |
| F3 | Critic, verificações e telemetria; escalada conforme confiança. |
| F4 | Brain e um expert Nexa nativos, sem PyTorch no runtime. |
| F5 | Reuso de contexto/prefix/KV com ganho medido e sem regressão de qualidade. |
| F6 | Omni V2 com roteamento aprendido por qualidade/custo. |
| F7 | Prova de Neural Bus latente entre dois modelos compatíveis. |

F1 é um gate intermediário. O Definition of Done de V0 da página 13 também inclui
memória, scheduler, Critic, telemetria e integração Silicon; um DAG isolado não
conclui esse DoD. Os dez sprints listados não têm duração definida e não fornecem
uma estimativa de calendário.

## Orçamento global e cache

O perfil 512 MB mantém um expert físico de cada vez; o de 8 GB prevê Brain com
um a três experts. Paralelismo lógico pode ser serializado fisicamente conforme
o orçamento. O Scheduler deve somar modelos residentes, KV, ativações, staging,
temporários, alinhamento e reservas por tier.

Uma prova sintética de buffers CPU pode validar um contrato de F2, mas não mede
RSS/VRAM nem encerra M5 do blueprint. Por exemplo, os dois bancos KV F32 atuais
reservam 128 MiB para R0 e 256 MiB para v1 em contexto 2.048, antes dos outros
buffers. Compartilhar orçamento exige contabilizar esse custo inteiro.

F5 precisa de `CacheSignature` que identifique pesos e adapters relevantes,
dependências anteriores, tokenizer/IDs, posições, RoPE/máscara, layout e codec.
Tokenizer e família iguais, isoladamente, não provam que dois modelos produziram
o mesmo KV. O cache incremental privado da sessão atual é uma base; compartilhá-lo
entre modelos é outro contrato e permanece pendente.

## Ordem de implementação

Concluir e manter verificável a base de execução CPU/KV, depois iniciar contratos
F0 sem duplicar o gerenciador de pacotes. Protocolo/ACL, routing e DAG devem ser
testados com fixtures locais antes de integrar três modelos externos. Brain deve
ser acionado seletivamente; memória e orçamento precisam ser compartilhados com
o runtime. Sandbox, capabilities/FFI, integridade, rastros e métricas acompanham
os componentes que introduzem execução e dados compartilhados.

Somente após o DoD correspondente devem avançar os Model Packs nativos, cache
entre modelos, roteamento aprendido e Neural Bus. Nenhum desses gates foi marcado
como concluído pela leitura do documento ou pelos testes do cache de uma sessão.
