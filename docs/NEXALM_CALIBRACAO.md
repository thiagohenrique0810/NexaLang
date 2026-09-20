# Calibração por tensor: erro do codec e sensibilidade nos logits

O décimo oitavo incremento mede, tensor a tensor, o que a quantização custa.
São duas medições separadas de propósito, porque respondem coisas diferentes:

1. **Estática** — distribuição, outliers e erro de round-trip do codec. Lê o
   checkpoint, não executa nada, custa uma passagem por tensor.
2. **Sensibilidade** — executa o modelo com **todos** os pesos densos e depois
   com **um único** tensor empacotado, e mede quanto os logits andaram.

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

## Métricas do relatório

Por tensor: `statistics` (min, max, média, RMS, desvio, `max_abs`,
`median_row_max_abs` e `outlier_ratio`), `quantization` (erro máximo, RMSE,
RMSE relativo e SNR em dB), `dense_bytes`, `packed_bytes`, `saved_bytes`,
`sensitivity` (delta máximo, RMSE e RMSE relativo dos logits) e
`cost_per_saved_kib` — o custo em logits por KiB economizado, que é a ordenação
que um precision map precisa.

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

## Limites

O custo é **uma execução do modelo por tensor medido**, mais duas de
referência. Para um modelo grande isso é caro em tempo e em disco (o bundle
denso ocupa oito vezes o empacotado), então use `--tensor` ou rode por camada.

O prompt de calibração define o que está sendo medido: tokens diferentes
exercitam caminhos diferentes. Um único prompt curto não representa uma mistura
de domínios — escolher o conjunto de calibração faz parte de M6.01b, junto com
o orçamento de qualidade e a calibração por grupo dentro do tensor.

A seleção automática de codec por tensor sob orçamento — o PrecisionMap — é
M6.02, e usa exatamente o `cost_per_saved_kib` deste relatório. O
[checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
próxima tarefa.
