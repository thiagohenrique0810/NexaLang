# weights/ — pesos, checkpoints e bundles

Nada de binário aqui entra no Git. Só este README e os `.gitkeep` são versionados;
todo o resto é ignorado, pelo mesmo motivo dos PDFs de referência: são artefatos
grandes, reproduzíveis a partir de uma origem declarada, e alguns têm licença
própria que não é a deste repositório.

## Estrutura

```text
weights/
  synthetic/    checkpoints sintéticos gerados aqui (pesos pseudoaleatórios)
  checkpoints/  checkpoints reais baixados (Safetensors + config + tokenizer)
  bundles/      bundles NexaPack e contêineres .nxb convertidos de qualquer um
```

## O que um checkpoint sintético prova, e o que não prova

Um checkpoint sintético tem a **forma** de um modelo real — mesma contagem de
parâmetros, mesmos shapes, mesma aritmética — e pesos pseudoaleatórios de semente
declarada. Serve para medir **física**: bytes no disco, residência em RAM, pico da
arena, tempo de prefill e decode, e se o conjunto cabe em 512 MB.

Não diz **nada** sobre qualidade. Perplexidade, coerência e sensibilidade por
tensor medidas sobre pesos aleatórios não transferem para um modelo treinado —
é a mesma distinção que os relatórios já publicam como `quality_measured: false`.
Medir sensibilidade de codec aqui produziria um número que parece uma calibração
e não é.

Para qualidade é preciso um checkpoint **treinado**: ou baixado em
`checkpoints/`, com origem, revisão e hashes registrados, ou produzido pelo gate
de treinamento (LLM.03/LLM.04), que ainda não existe.

## Registro de origem

Todo checkpoint real baixado precisa de um `origin.json` ao lado, com a origem,
a revisão exata, a licença e o SHA-256 de cada arquivo — para que o que foi
medido possa ser reproduzido. Um checkpoint sem isso não deve ser usado em
nenhuma medição publicada.
