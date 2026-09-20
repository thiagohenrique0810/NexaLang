# Calibração por tensor e PrecisionMap

O décimo oitavo incremento mede, tensor a tensor, o que a quantização custa.
São duas medições separadas de propósito, porque respondem coisas diferentes:

1. **Estática** — distribuição, outliers e erro de round-trip do codec. Lê o
   checkpoint, não executa nada, custa uma passagem por tensor.
2. **Sensibilidade** — executa o modelo com **todos** os pesos densos e depois
   com **um único** tensor empacotado, em cada codec medido, e mede quanto os
   logits andaram.

Nenhuma das duas é perplexidade. O erro estático não diz como o erro se propaga
pelo grafo, e um delta de logits sobre pesos não treinados não fala de qualidade
de resposta. São proxies com limites declarados; viram afirmação de qualidade
apenas com um checkpoint treinado (LLM.04b).

## Por que a medição fim a fim existe

Na fixture D8/H4/KV2 com grupo 4, o erro estático dos tensores é praticamente
igual — todos por volta de 3,5% de RMSE relativo — enquanto o impacto real nos
logits varia mais de três vezes:

| Tensor | RMSE relativo do codec | RMSE nos logits | KiB economizados |
| --- | ---: | ---: | ---: |
| `model.embed_tokens.weight` | 0,0370 | 0,1896 | 0,31 |
| `model.layers.0.self_attn.o_proj.weight` | 0,0346 | 0,1592 | 0,16 |
| `model.layers.0.mlp.down_proj.weight` | 0,0359 | 0,1012 | 0,31 |
| `model.layers.0.mlp.up_proj.weight` | 0,0369 | 0,0794 | 0,31 |

Ordenar tensores pelo erro do codec daria uma ordem diferente da real. É
exatamente por isso que o precision map precisa da medição fim a fim, e não só
da estatística barata. Estes números vêm de pesos sintéticos: valem como
demonstração do método, não como recomendação de precisão para um modelo real.

## Como a variante é montada

Reescrever o modelo inteiro por tensor seria O(tensores) conversões. Em vez
disso, a calibração escreve **uma vez** o bundle denso e **uma vez** o
empacotado, e monta cada variante ligando (hard link, com cópia como fallback)
os payloads já escritos, trocando apenas a entrada daquele tensor no manifesto.

Isso depende do despacho por codec de [codecs de peso](NEXALM_CODECS_PESOS.md):
um bundle pode misturar `RAW_F32_MATRIX` e `Q4_GROUPED`, e o executor lê cada
tensor pelo seu próprio caminho.

A montagem é atômica: a variante é construída num diretório `.partial` e
renomeada ao final, então uma falha de link, cópia ou espaço não deixa um bundle
incompleto para trás.

## Conjunto de calibração

`--tokens` é repetível: cada ocorrência é um prompt do conjunto, e
`--prompt-label` dá nome a cada um, em ordem. O relatório registra o conjunto
inteiro — rótulo, IDs e comprimento — porque **o que foi medido depende do que
foi executado**.

A agregação é o RMSE combinado de todos os prompts, e o relatório traz também o
delta de cada prompt e qual foi o **pior**. Um codec aceitável na média pode
quebrar um domínio, e um prompt só não distingue os dois casos. Na fixture com
três prompts, a sensibilidade de um mesmo tensor em Q4 variou de 0,34 a 0,81
conforme o prompt — a média sozinha esconderia isso.

## Métricas do relatório

Por tensor: `statistics` (min, max, média, RMS, desvio, `max_abs`,
`median_row_max_abs` e `outlier_ratio`), `dense_bytes` e um bloco `codecs` com
uma entrada por codec medido. Cada entrada traz `quantization` (erro máximo,
RMSE, RMSE relativo e SNR em dB), `packed_bytes`, `saved_bytes`, `sensitivity`
(delta máximo, RMSE e RMSE relativo dos logits) e `cost_per_saved_kib` — o
custo em logits por KiB economizado, que é a ordenação que o plano usa.

`--codec` restringe os codecs medidos; sem ele, a calibração mede Q3, Q4 e Q8.
Numa fixture com grupo 8, o modelo inteiro num codec só moveu os logits 0,686
(Q3), 0,447 (Q4) e 0,017 (Q8) de RMSE — a escala de degraus que o plano usa.

`outlier_ratio` compara o pico da pior linha com o da linha mediana: é o que
torna um tensor difícil de quantizar com uma escala só.

Do modelo: `reference` (SHA-256 dos logits densos), `all_packed` (o mesmo com
tudo em Q4, e o delta correspondente) e `most_sensitive`. `quality_measured`
permanece `false`, com a nota explicando o que falta.

## Execução

```bash
# Passe estático: sem execução, ordena pelo erro relativo do codec.
python3 tools/nexa_calibrate.py --checkpoint CHECKPOINT --tokens 1,3,5 --static-only

# Sensibilidade completa: uma execução por tensor medido.
python3 tools/nexa_calibrate.py --checkpoint CHECKPOINT --tokens 1,3,5 \
  --group-size 4 --block-rows 3 --tile-rows 3 --memory-budget 8MiB \
  --report artifacts/reports/calibracao.json

# Restringir aos tensores em questão mantém o custo proporcional.
python3 tools/nexa_calibrate.py --checkpoint CHECKPOINT --tokens 1,3 \
  --tensor model.embed_tokens.weight --tensor model.layers.0.mlp.down_proj.weight
```

`--work-dir` preserva os bundles intermediários para inspeção; sem ele, tudo
vive num diretório temporário que é removido ao final.

## PrecisionMap: escolher codecs sob um teto de bytes

`select_precision` consome esse relatório. Cada tensor começa no **codec mais
barato medido** e, enquanto houver orçamento, aplica-se o degrau com melhor
erro evitado por byte extra, em qualquer ponto do modelo. Um tensor pode subir
mais de um degrau (Q4 → Q8 → denso, ou direto para denso), e um degrau que não
melhora nada nunca é comprado: opções dominadas — que custam mais e erram igual
ou mais — são descartadas antes da escolha.

```bash
python3 tools/nexa_precision.py plan --calibration artifacts/reports/calibracao.json \
  --budget 2KiB --out artifacts/precision/map.json
python3 tools/nexa_precision.py show artifacts/precision/map.json
python3 tools/nexa_convert.py --checkpoint CHECKPOINT --out artifacts/models/planejado \
  --precision-map artifacts/precision/map.json --group-size 4 --block-rows 3
```

O mapa é versionado (`schema_version`, `policy_id`) e carrega a proveniência da
decisão: checkpoint, tokens de calibração, `group_size`, codecs medidos,
baseline no codec mais barato, bytes planejados, sobra, contagem por codec e a
lista de degraus com o RMSE que cada um evita.
Um mapa com versão, política ou campos diferentes é recusado em vez de
reinterpretado.

**O que a estimativa não é.** `estimated_avoided_rmse_sum` soma sensibilidades
medidas **individualmente**, e erros de quantização não se somam: quantizar dois
tensores não é a soma de quantizar cada um. O número ordena planos; não prevê a
qualidade do conjunto, e `quality_measured` permanece `false`. A seleção também
é gulosa sobre uma razão — com escolha binária por tensor, é heurística, não
ótimo.

O espaço de escolha tem hoje quatro pontos por tensor: Q3, Q4, Q8 e denso. Na
fixture com grupo 8, o teto mínimo planeja tudo em Q3; mais 100 bytes já
misturam Q3, Q4 e Q8; e tetos maiores sobem para denso onde o ganho por byte é
maior. Q2 e RAW-F16 entram em M1.05d e ampliam a escala sem mudar o contrato.

Um codec grosseiro nem sempre é o mais barato: com grupos pequenos, a escala de
quatro bytes domina e Q3 ocupa o mesmo que Q4 errando mais. A fronteira remove
essa opção antes da escolha, sem precisar de regra especial.

## Teto de qualidade em vez de teto de bytes

`--max-rmse` inverte a pergunta: em vez de "o melhor plano que cabe em N bytes",
"o plano mais barato que fica abaixo deste erro". Os degraus são aplicados na
mesma ordem, e a busca para assim que a estimativa atinge o teto.

```bash
python3 tools/nexa_precision.py plan --calibration REPORT --max-rmse 0.05 --out map.json
```

A estimativa combina as sensibilidades medidas **uma de cada vez** como se
fossem independentes (raiz da soma dos quadrados). Isso é uma suposição, não uma
medição: serve para comparar planos, não como número de qualidade do modelo.
`--budget` e `--max-rmse` são mutuamente exclusivos, e a proveniência registra
qual dos dois limitou o plano, a estimativa resultante e se o teto foi atingido.

## Limites

O custo é **uma execução do modelo por tensor e por codec medido**, mais uma
referência densa e uma por codec. Para um modelo grande isso é caro em tempo e em disco (o bundle
denso ocupa oito vezes o empacotado), então use `--tensor` ou rode por camada.

O conjunto de calibração define o que está sendo medido. Ele agora aceita vários
prompts e reporta o pior, mas continua sendo escolhido por quem chama: uma
mistura representativa por idioma e domínio depende do corpus de LLM.02, e
nenhuma agregação corrige um conjunto que não representa o uso real.

Calibração por grupo dentro do tensor — em vez de um codec por tensor inteiro —
continua em M6.02c, junto da seleção por bloco.

Validar um plano de verdade exige comparar qualidade entre mapas num modelo
treinado, o que depende de LLM.04b. O
[checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
próxima tarefa.
