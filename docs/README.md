# Documentação NexaLang

Esta pasta reúne os guias do projeto e a documentação histórica da linguagem
NexaLang em formato HTML.

O compilador está em estágio experimental. Consulte o `readme.md` da raiz para instalação, garantias atuais e limitações. Os exemplos HTML ainda não constituem uma especificação de conformidade.

## Desenvolvimento de IA com orçamento de 512 MB

- [Índice de implementação: todos os PDFs, módulos, páginas e dependências](PLANOS_IMPLEMENTACAO_INDICE.md)
- [Blueprint original](NexaLang_Blueprint_Tecnico_512MB.pdf)
- [Plano da primeira LLM](NexaLang_Plano_Implementacao_Primeira_LLM.pdf)
- [Plano de treinamento NexaData/LLMs](NexaLang_Plano_Implementacao_Treinamento_LLMs.pdf)
- [Treinamento: ajustes, contratos e correspondência de gates](NEXALM_TREINAMENTO_AJUSTES.md)
- [Ajustes técnicos e contratos](BLUEPRINT_512MB_AJUSTES.md)
- [Checklist de desenvolvimento e ponto de retomada](BLUEPRINT_512MB_CHECKLIST.md)
- [NexaLM: arquitetura e importação local](NEXALM_IMPORTACAO.md)
- [NexaLM: forward e decode CPU](NEXALM_EXECUCAO_CPU.md)
- [NexaLM: KV F32 incremental e prefill em chunks](NEXALM_KV_CPU.md)
- [NexaLM: KV paginado F32 e residência sob demanda](NEXALM_KV_PAGINADO_CPU.md)
- [NexaLM: KV Q4 paginado, atenção packed e erro numérico](NEXALM_KV_Q4_CPU.md)
- [NexaLM: KV Q3 paginado e comparação F32/Q4/Q3](NEXALM_KV_Q3_CPU.md)
- [NexaLM: KV TQ paginado, atenção CPU e memória do codec](NEXALM_KV_TQ_CPU.md)
- [NexaLM: política hot/warm/cold CPU e recodificação transacional](NEXALM_KV_TIERS_CPU.md)
- [NexaLM: evicção/recarga CPU de páginas KV com backing store](NEXALM_KV_BACKING_CPU.md)
- [TurboQuant: contexto MSE linear, memória e compatibilidade](TURBOQUANT_MSE_CPU.md)
- [NexaPack V1: matriz e execução Q4](NEXAPACK_V1.md)
- [NexaPack TQ: formato portátil, conversão e migração TQ01](NEXAPACK_TQ_V1.md)
- [Plano Nexa Omni — PDF original](NexaLang_Plano_Implementacao_Nexa_Omni.pdf)
- [Nexa Omni: ajustes, dependências e gates](NEXA_OMNI_AJUSTES.md)
- [Plano Conditional Compute — PDF original](NexaLang_Conditional_Compute_Implementation.pdf)
- [Conditional Compute: módulos, orçamento, continuidade KV e gates C0–C10](NEXA_CONDITIONAL_COMPUTE_AJUSTES.md)
- [Plano Plastic Learning — PDF original](NexaLang_Plastic_Learning_Implementation.pdf)
- [Plastic Learning: evidências, adapters, transações e gates P0–P7](NEXA_PLASTIC_LEARNING_AJUSTES.md)

Os seis PDFs foram analisados e vinculados ao checklist. As trilhas CC e PL
acrescentam 22 tarefas pendentes; cada módulo referencia as páginas de origem,
dependências e critérios de aceite. Antes de iniciar uma tarefa, consulte o
índice e o guia de ajustes correspondente. Os exemplos de sintaxe e os módulos
novos propostos pelos PDFs ainda exigem implementação.

A execução CPU oferece `--kv-cache` para decode incremental com dois bancos KV
F32 e `--prefill-chunk-size` para limitar os chunks do prompt. `--kv-page-tokens`
seleciona páginas F32 alocadas sob demanda, com atenção direta sobre suas tabelas.
`--kv-codec q4` ou `--kv-codec q3` comprime o KV por token/head e ativa atenção
direta sobre os grupos compactados.
`--kv-codec tq` usa registros TQ02 e reconstrói um head por vez em scratch
planejado; seu contexto MSE e acumulador também entram no orçamento.
`--kv-policy age` mantém páginas recentes em F32, intermediárias em Q4 e antigas
em Q3, com atenção mista e migração transacional após cada chunk. Por padrão,
todos ficam na RAM; `--kv-backing-store DIRETÓRIO` descarrega páginas cold Q3
e usa um único slot para recarga, preservando bytes e o contexto causal completo.
O orçamento inclui esses buffers; o modo padrão por recomputação permanece disponível. Os guias
distinguem essa contabilidade de RSS/VRAM e mantêm GPU, demais codecs/tiers KV, treinamento e
orquestração Omni como gates pendentes.

## Acesso à Documentação

Abra o arquivo `index.html` em seu navegador para visualizar a documentação completa.

### Visualização Local

```bash
# No Windows
start docs/index.html

# No Linux/Mac
open docs/index.html
# ou
xdg-open docs/index.html
```

### Servidor Local (Recomendado)

Para melhor experiência, você pode servir a documentação através de um servidor HTTP local:

```bash
# Python 3
python -m http.server 8000

# Node.js (com http-server)
npx http-server docs -p 8000

# PHP
php -S localhost:8000 -t docs
```

Depois acesse: `http://localhost:8000/index.html`

## Conteúdo

A documentação inclui:

- ✅ Introdução à linguagem
- ✅ Guia de instalação
- ✅ Sintaxe básica
- ✅ Tipos de dados
- ✅ Funções
- ✅ Structs e métodos
- ✅ Enums e pattern matching
- ✅ Generics
- ✅ Ownership & Borrowing
- ✅ Regions (Arenas)
- ✅ GPU Kernels
- ✅ Controle de fluxo
- ✅ Arrays & Slices
- ✅ Ponteiros
- ✅ Standard Library
- ✅ Ferramentas
- ✅ Exemplos práticos

## Atualizações

A documentação é atualizada conforme a linguagem evolui. Para contribuir, edite o arquivo `index.html` e faça um pull request.
