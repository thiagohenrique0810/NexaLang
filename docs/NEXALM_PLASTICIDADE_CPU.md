# Regiões de parâmetro e adapters low-rank (CPU)

Contratos implementados em PL.01a e PL.03a. Consultar antes
[os ajustes do plano](NEXA_PLASTIC_LEARNING_AJUSTES.md), que permanecem a fonte
de correspondência com o PDF. **Nada aqui treina, publica pesos, versiona
aprendizado ou altera o manifesto do bundle.** Não há backward, optimizer,
Learning Gate, replay ou transação: PL.01a declara *onde* um modelo poderia
aprender e PL.03a define *como* um delta já existente entra no forward.

## PL.01a — `compiler/model_plasticity.py`

### O que existe

`ParameterRegion` (dataclass congelado) com `id`, `domain`, `region_class`,
`plasticity`, `maturity`, `protected`, `tensors` e `provenance`.
`ModelPlasticityConfig` é o contêiner versionado (`SCHEMA_VERSION = 1`), com
`to_dict`/`to_json`/`from_dict`/`from_json`. A serialização usa `sort_keys=True`
e `allow_nan=False`; a desserialização reusa `model_ir._load_json`, então chave
JSON duplicada e literais `NaN`/`Infinity` são recusados — `json.loads` puro
aceitaria os dois em silêncio.

`resolve_plasticity_map(config, plasticity)` resolve as regiões contra um
`ModelConfig` e devolve um `PlasticityMap`.

### Campos deliberadamente ausentes

`importance`, `drift_budget`, `update_count`, `last_update` e `residency`
aparecem no PDF e **não estão aqui**. Nenhum dos cinco tem produtor nem
consumidor no repositório hoje: seriam campos que nada escreve e nada lê, isto
é, campos infalsificáveis, e um schema feito só deles concordaria apenas
consigo mesmo. `importance` e `drift_budget` pertencem ao Learning Gate
(PL.02), `update_count`/`last_update` à transação (PL.04) e `residency` ao eixo
de residência de CC/M3 — que é independente do lifecycle, como o próprio plano
registra (um expert maduro pode estar cold).

### Regras que olham o modelo, não a si mesmas

- **Aliases tied.** Um span sobre `lm_head.weight` resolve para
  `model.embed_tokens.weight` quando `tie_word_embeddings=True`. Duas regiões,
  uma por cada nome, são recusadas como sobrepostas nesse modelo e **aceitas**
  no modelo untied. O mesmo documento JSON dá respostas opostas conforme o
  `ModelConfig`; é isso que separa uma validação real de um round-trip.
- **Partição exata.** Escalares cobertos (somados dos spans) mais não
  atribuídos (somados do *complemento* dos spans, percorrendo as lacunas por
  tensor) igualam `ModelConfig.parameter_count()`, que conta embeddings
  compartilhados uma única vez. Na fixture tiny: **728 escalares em 11 tensores
  físicos** com tied, **856 em 12** sem tied. O oráculo é anterior a este módulo
  e não o chama. Calcular o não atribuído como `total - coberto` seria uma
  tautologia e está coberto por teste.
- **Proteção.** `protected=True` prevalece, como o plano exige para P0–P2, e por
  isso uma região protegida precisa declarar `plasticity=0.0`: a configuração
  não pode afirmar as duas coisas. `validate_learning_target` recusa um alvo que
  caia em linhas protegidas, inclusive quando o alvo foi escrito pelo alias.
- Booleanos são recusados em campos numéricos, faixas fora de `[0, 1]` também, e
  uma faixa de linhas além do fim do tensor é erro, não truncamento.

### Baseline preservado

O módulo não é importado pelo lowering, pelo planner, pelo writer de bundle nem
pelo executor. Um `PlasticityMap` resolvido não altera o grafo lowered, as
requisições de ativação nem o `MemoryPlan.to_dict()` de uma sessão — há teste
para os três. Os 831 testes anteriores continuam com as mesmas expectativas.

## PL.03a — `runtime/learning/adapter.py`

### Semântica fixada

```
y = W_base x + (alpha / rank) * B (A x)
```

com `A[rank, in_features]` e `B[out_features, rank]`, ambas row-major. O payload
é um arquivo float32 little-endian com A e depois B, de exatamente
`rank * (in_features + out_features) * 4` bytes — nada mais. `W_base` continua
empacotada e imutável; `B A` nunca é materializada e nenhuma cópia densa da
matriz base é criada.

**Ordem canônica de acumulação** (adição em ponto flutuante não é associativa,
então a ordem é parte do contrato):

1. adapters na ordem declarada, cada um completo antes do próximo;
2. `t = A x` reduzido em coordenada de entrada crescente, em double, com um
   arredondamento por elemento — o mesmo kernel denso do caminho base;
3. a escala multiplica `t`, o intermediário de tamanho `rank`, e não a saída:
   custa `length * rank` arredondamentos em vez de `length * out_features`;
4. `B t` reduzido em coordenada de rank crescente, e a soma com a saída base em
   índice plano crescente, `posição * out_features + feature`.

### Adapter nulo é idêntico byte a byte

Com `B` inteiramente zero, ou com `alpha = 0`, cada elemento do delta é `+0.0`
(um acumulador double que começa em `0.0` nunca produz `-0.0`, e a saída de um
matmul do runtime também não), e `valor + 0.0` devolve o mesmo valor para todo
float finito que o caminho base produz. O `logits_sha256` é idêntico ao da
mesma sessão sem adapter. Com **zero** adapters o `MemoryPlan.to_dict()`, o
bloco `io` e o relatório são os de hoje, sem chave nova.

### Validação antes de ler payload

Recusado antes de qualquer byte de payload: rank fora de `[1, 64]` ou maior que
a menor dimensão do alvo; shapes declaradas que não casam com o tensor-alvo lido
do `ModelBundleReader`; alvo inexistente; alvo que não seja peso de MatMul de
projeção; `precision` diferente de `f32`; `alpha` não finito; alvos, ids ou
ordens repetidos no conjunto; tamanho de arquivo diferente do exigido.

Os alvos permitidos são **derivados do grafo lowered**, não de uma lista:
`matmul_weight_targets` pega os constantes consumidos por ops MatMul e remove os
consumidos por Embedding e a cabeça de saída. Isso importa porque, com tied, a
MatMul de `logits` consome o próprio tensor de embedding — adaptar ali mudaria
duas operações de uma vez. V1 recusa embedding e `lm_head` nos dois casos.

### Contabilidade

`io["adapter_payload_bytes_read"]` só aparece quando há adapter e vale
exatamente `rank * (in_features + out_features) * 4` por passagem de grafo, por
adapter. O relatório ganha um bloco `adapters` com semântica, ordem de
acumulação, layout do payload e os alvos vinculados.

O payload é lido direto para um slot da arena (`__adapter_payload`), sem cópia
intermediária; `__adapter_low` e `__adapter_delta` completam os três buffers
pedidos ao planner, e só quando existe adapter vinculado.

### Oráculo independente

`tests/adapter_reference.py` recalcula o forward em float64 puro, sem PyTorch,
**com decodificador Q4_GROUPED próprio**. Não importa `decode_q4_row` nem
qualquer kernel do runtime: um oráculo que usasse o mesmo decodificador e a
mesma ordem provaria apenas que uma chamada é igual a si mesma. Há teste que lê
o fonte do oráculo e falha se esses imports aparecerem.

### Números medidos

Fixture de teste (alvo `mlp.up_proj` 12×8, rank 2, 4 tokens):

| Medida | Valor |
|---|---|
| Payload declarado, em disco e lido por prefill | 160 B (`2*(8+12)*4`) |
| Digest do adapter nulo vs. baseline | idêntico |
| Maior mudança absoluta de logit com adapter não nulo | 0,10984230 |
| Runtime vs. oráculo float64 (baseline e adaptado) | erro máximo **0,0** |
| Crescimento da arena | 2944 B → 3840 B (+896 B para 384 B pedidos) |

Fixture maior (2 camadas, hidden 128, intermediate 256, alvo 256×128, rank 8,
16 tokens): payload 12288 B, **9,4%** dos 131072 B da mesma matriz em F32 denso;
prefill 7,80 ms → 8,74 ms, **+12,0%**.

O erro máximo **zero** contra o oráculo é agradável mas merece leitura honesta:
o oráculo compartilha zero código com o runtime, e por isso o acordo bit a bit
mostra que o contrato aritmético está inteiramente especificado — não que o
oráculo seja uma medida independente de *qualidade*. É uma fixture de 1 e de 2
camadas; um modelo mais profundo pode divergir por `cos`/`exp` de libm
diferentes, e a comparação continua tolerante a 1e-5/1e-4 por isso.

Os 9,4% da fixture maior também não são uma economia: são a fração que o
payload ocupa frente a uma matriz pequena. Com shapes reais a fração cai, mas
isso é aritmética da fórmula, não uma medição feita aqui.

## Limites conhecidos

- **A soma final do delta é um laço Python**, porque os kernels nativos recusam
  uma saída que aliasa uma entrada e não há kernel de acumulação escalada. É a
  maior parte dos +12% medidos. Um `nexa_adapter_accumulate` em C removeria o
  laço sem mudar a ordem declarada; fica para PL.03b.
- **Um adapter por tensor-alvo.** Compor dois deltas sobre a mesma projeção
  exige uma regra de composição declarada; inventá-la aqui criaria uma regra que
  nenhum teste poderia conferir contra algo fora deste arquivo.
- **Payload inteiro residente** durante a aplicação de cada adapter. Para rank
  baixo isso é pequeno, mas não é streaming por blocos como os pesos base.
- **Sem persistência versionada.** Não há identidade de versão executável
  ligando hashes de base, tokenizer, conjunto ordenado de adapters e política de
  routing; isso é PL.04, e sem ele **trocar adapters numa sessão viva não é
  suportado** — a sessão fixa seu conjunto na construção, como o plano exige
  para KV.
- **Nenhuma qualidade de modelo medida.** Os deltas destes testes são ruído
  determinístico; provam aritmética e contabilidade, não aprendizado.
- `ExpertIR`, `LearningPolicyIR`, probes de regressão e layout de adapter
  derivado por passe continuam pendentes em PL.01b/PL.02.
