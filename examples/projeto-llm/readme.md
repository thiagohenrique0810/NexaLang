PROMPT MESTRE — PROJETO COMPLETO
Papel

Você é um arquiteto principal de sistemas, engenheiro de compiladores, engenheiro de runtime de IA, engenheiro de GPU e líder técnico de produto.
Sua tarefa é criar e executar integralmente um projeto chamado TurboLang LLM Runtime, que consiste em uma stack própria para inferência de LLM de alta performance.

Você deve agir como uma equipe completa de:

arquiteto de software
engenheiro backend
engenheiro de compiladores
engenheiro de performance
engenheiro de GPU
engenheiro DevOps
QA
documentador técnico

Seu objetivo é entregar um sistema funcional, executável localmente, modular e preparado para evolução.

Objetivo do projeto

Construir uma plataforma chamada TurboLang LLM Runtime, composta por:

Uma linguagem própria/DSL
para descrever políticas de inferência e otimização de LLM.
Um compilador da DSL
que transforma o código da linguagem em uma representação intermediária e gera plano de execução.
Um runtime de inferência
para rodar LLMs com foco em:
baixa latência
alto throughput
uso eficiente de VRAM
quantização agressiva
compressão de KV cache
execução otimizada em GPU
Um servidor de inferência
com API HTTP compatível com uso simples estilo chat/completions.
Ferramentas de benchmark
para medir:
TTFT
tokens por segundo
latência p50/p95
consumo de VRAM
eficiência por batch
Visão do produto

O sistema não deve ser uma LLM treinada do zero inicialmente.
Ele deve ser uma infraestrutura própria de inferência eficiente para modelos open-weight, com possibilidade futura de:

fine-tuning
distillation
kernels customizados
backend próprio
compressão mais agressiva
suporte MoE

O primeiro foco é inferência eficiente, porque esse é o ponto com maior retorno prático.

Diretrizes obrigatórias

Você deve:

criar um projeto completo de engenharia
gerar todos os arquivos necessários
escrever o código
organizar a estrutura de pastas
documentar
criar scripts de build
criar ambiente Docker quando apropriado
criar testes
criar exemplos
executar localmente
validar funcionamento
corrigir erros encontrados
repetir até deixar operacional

Não peça confirmação a cada etapa.
Tome decisões técnicas razoáveis e siga em frente.
Explique o que está fazendo em cada fase.

Nome do projeto

TurboLang LLM Runtime

Stack obrigatória
Linguagens
Rust para runtime, parser, orquestração de alto desempenho e partes críticas
Python para integração com modelos, benchmark, tooling e prototipação
Triton para kernels de GPU quando necessário
MLIR/LLVM como base conceitual da camada de compilação e IR
FastAPI ou framework similar para API HTTP
Opcional, se necessário
C++ apenas se houver necessidade real de integração de baixo nível
PyTorch para carregar modelos base e validar a execução
Docker e Docker Compose para ambiente reprodutível
Arquitetura obrigatória
Módulo 1 — DSL própria

Criar uma linguagem chamada TurboLang.

Ela deve permitir descrever:

modelo usado
estratégia de quantização
política de KV cache
modo de attention
speculative decoding
scheduler
limites de batch
alvo de hardware
Exemplo de sintaxe esperada
model "llama3-8b" {
  weights quantized int4
  kv_cache turboquant bits=3
  attention flash streaming=true
  decode speculative drafter="tiny-draft"
  scheduler continuous_batching max_batch=32 max_ctx=8192
  target gpu "cuda"
}

Você deve:

definir gramática
criar lexer
criar parser
criar AST
validar sintaxe
gerar mensagens de erro claras
Módulo 2 — IR / plano de execução

Criar uma representação intermediária chamada TurboIR.

TurboIR deve representar:

configurações do modelo
tipo de quantização
política de memória
plano de prefill
plano de decode
compressão de cache
agendamento de lotes
fallback CPU/GPU

O compilador deve converter:
TurboLang → AST → TurboIR → Execution Plan

Módulo 3 — Runtime de inferência

Criar o runtime chamado TurboRuntime.

Responsabilidades:

carregar modelos base
preparar pesos
aplicar quantização
gerenciar KV cache
aplicar compressão estilo TurboQuant no KV cache
executar prefill e decode
implementar batching contínuo
permitir speculative decoding básico
medir métricas de execução

A compressão tipo TurboQuant deve ser tratada como estratégia de compressão/extrema compactação de cache para reduzir uso de memória e acelerar inferência quando possível, coerente com o uso recente dessa abordagem em KV cache.

Módulo 4 — Kernels e otimização

Criar uma camada de kernels chamada TurboKernels.

Objetivos:

implementar operações críticas otimizadas
preparar integração com Triton
encapsular:
attention path
matmul path
dequant path
KV cache pack/unpack
fused ops sempre que possível

Atenção rápida e kernels especializados seguem sendo uma base importante para performance moderna em inferência de LLM, e Triton é um caminho prático para construir isso.

Primeiro entregue uma versão funcional mesmo que nem tudo esteja completamente otimizado.
Depois evolua para otimizações.

Módulo 5 — API server

Criar um servidor chamado TurboServe.

Endpoints mínimos:

GET /health
GET /metrics
POST /load-model
POST /compile
POST /generate
POST /chat

Requisitos:

JSON simples
streaming opcional
timeout configurável
logs estruturados
métricas de performance
Módulo 6 — Benchmark suite

Criar um pacote TurboBench para medir:

TTFT
latência média
latência p95
tokens por segundo
throughput por batch
uso de VRAM
uso de RAM
impacto da quantização
impacto da compressão do KV cache
comparação entre modos

Salvar resultados em:

JSON
CSV
tabela legível no terminal
Escopo funcional da V1

A V1 deve fazer o seguinte:

Receber um arquivo .tl da linguagem TurboLang
Fazer parse e validação
Gerar TurboIR
Traduzir TurboIR em plano de execução
Carregar um modelo open-weight pequeno ou médio
Aplicar quantização inicial
Gerenciar KV cache comprimido
Executar geração de texto
Expor API HTTP
Rodar benchmark básico
Gerar documentação de uso
Restrições importantes
Não tentar treinar LLM do zero agora
Não depender de cluster distribuído na V1
Não exigir múltiplas GPUs na V1
Priorizar execução local com 1 GPU quando houver
Ter fallback parcial para CPU onde possível
Código deve ser modular e preparado para expansão
Objetivos de performance da V1

Metas:

inicializar sem travar
servir respostas com API funcional
rodar modelo de teste localmente
mostrar ganho mensurável entre:
modo sem compressão
modo com compressão de KV cache
modo com quantização mais agressiva

Se alguma meta não puder ser atingida por limitação do ambiente, documente claramente.

Estrutura de pastas desejada
turbolang-llm-runtime/
  README.md
  ROADMAP.md
  ARCHITECTURE.md
  Makefile
  docker-compose.yml
  .env.example

  turbolang/
    grammar/
    parser/
    ast/
    validator/

  turboir/
    ir/
    planner/

  turboruntime/
    core/
    scheduler/
    kv_cache/
    quant/
    model_loader/
    decode/
    prefill/

  turbokernels/
    triton/
    fallback/
    fused/

  turboserve/
    api/
    schemas/
    middleware/

  turbobench/
    scripts/
    reports/

  examples/
    basic.tl
    quantized.tl
    turboquant_kv.tl

  tests/
    parser/
    ir/
    runtime/
    api/
    benchmarks/
Entregáveis obrigatórios

Você deve entregar:

1. Documento de arquitetura

Explicando:

visão geral
fluxo de compilação
fluxo de inferência
decisões técnicas
riscos
próximos passos
2. Código-fonte completo
3. Exemplos de arquivos .tl
4. Script de execução local

Exemplos:

make dev
make run
make bench
5. Dockerização
6. Testes automatizados

Cobrir:

parser
validação
geração de IR
API
geração simples
7. Benchmarks reproduzíveis
8. README excelente

Com:

instalação
uso
troubleshooting
exemplos de chamadas
estrutura do projeto
Fases de execução

Você deve executar na seguinte ordem:

Fase 1 — Planejamento interno
definir arquitetura final
listar componentes
identificar dependências
descrever abordagem
Fase 2 — Scaffold do projeto
criar estrutura de pastas
arquivos iniciais
configuração de build
ambiente base
Fase 3 — DSL
definir gramática
parser
AST
validação
arquivos de exemplo
Fase 4 — IR e planner
TurboIR
transformação AST → IR
plano de execução
Fase 5 — Runtime básico
model loader
geração simples
ciclo prefill/decode básico
Fase 6 — Quantização e KV cache
quantização inicial
compressão de cache
estratégias configuráveis
Fase 7 — API server
endpoints
schemas
logs
healthcheck
Fase 8 — Benchmarks
scripts
medições
relatórios
Fase 9 — Docker e testes
ambiente reproduzível
testes automatizados
Fase 10 — Execução e validação
subir projeto
compilar exemplo TurboLang
carregar modelo
executar geração
rodar benchmark
corrigir falhas encontradas
Comportamento esperado durante a execução

Enquanto desenvolve, você deve:

mostrar quais arquivos está criando
mostrar conteúdo importante
explicar decisões
rodar testes
mostrar erros reais
corrigir erros
continuar até concluir

Nunca responda só com teoria.
Sempre avance implementando.

Critérios de qualidade

O projeto deve ser:

limpo
modular
bem documentado
fácil de expandir
executável
com logs claros
com tratamento de erro razoável
Requisitos extras desejáveis

Se possível, incluir:

suporte a configuração por YAML além da DSL
endpoint de profiling
métricas Prometheus
tracing básico
modo dry-run do compilador
visualização textual do TurboIR
Regras de implementação
use nomes consistentes
não gerar código descartável sem motivo
prefira clareza e modularidade
documente funções importantes
não esconder falhas
sempre que uma biblioteca falhar, registre isso e proponha alternativa
manter o sistema rodável
Modelos para teste

Escolha automaticamente um modelo open-weight pequeno ou médio que seja simples de testar localmente, preferindo algo com boa compatibilidade em PyTorch/Transformers.

O sistema deve abstrair o modelo, para permitir trocar depois.

Saídas esperadas ao final

Ao terminar, você deve me entregar:

árvore final do projeto
principais arquivos criados
comandos para rodar
resultado de um teste funcional
resultado de benchmark básico
próximos passos para V2
PROMPT EXTRA — MODO MAIS AGRESSIVO

Se você quiser mandar a IA agir de forma ainda mais executiva, cole isso junto no final:

Não pare em planejamento. Execute o projeto de ponta a ponta.
Crie os arquivos completos.
Implemente o código.
Rode os comandos necessários.
Mostre os erros encontrados.
Corrija automaticamente.
Continue iterando até obter uma versão funcional.
Sempre priorize um MVP executável antes de refinamentos avançados.
Sempre que houver dúvida técnica, escolha a opção mais simples que preserve a arquitetura.
VERSÃO MAIS CURTA PARA COLAR DIRETO

Se quiser uma versão resumida:

Crie e execute um projeto completo chamado TurboLang LLM Runtime.

Objetivo:
Construir uma stack própria para inferência eficiente de LLM com:
- DSL própria chamada TurboLang
- compilador TurboLang -> AST -> TurboIR -> Execution Plan
- runtime de inferência
- quantização
- compressão de KV cache estilo TurboQuant
- continuous batching
- speculative decoding básico
- API HTTP
- benchmark suite
- Docker
- testes
- documentação

Stack:
- Rust
- Python
- Triton
- MLIR/LLVM como base conceitual de compilação
- FastAPI
- PyTorch/Transformers quando necessário

Entregue:
- arquitetura
- estrutura de pastas
- código completo
- parser e AST
- IR e planner
- runtime
- kernels
- API
- benchmarks
- exemplos .tl
- Docker
- README
- testes
- execução funcional

Exemplo de sintaxe TurboLang:
model "llama3-8b" {
  weights quantized int4
  kv_cache turboquant bits=3
  attention flash streaming=true
  decode speculative drafter="tiny-draft"
  scheduler continuous_batching max_batch=32 max_ctx=8192
  target gpu "cuda"
}

Não pare em planejamento.
Implemente.
Execute.
Teste.
Corrija falhas.
Entregue uma V1 funcional.