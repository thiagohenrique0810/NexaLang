# NexaKV CPU: despacho por codec e páginas Q8

O vigésimo quarto incremento acrescenta `--kv-codec q8` e, mais importante, o
**despacho por layout** que M4.03d pedia: um kernel de atenção paginada que lê
qualquer codec suportado através dos mesmos acessores por linha, em vez de um
kernel duplicado por codec.

## Por que um kernel compartilhado

Até aqui cada codec de KV tinha seu próprio kernel de atenção
(`..._paged_q4`, `..._paged_q3`, `..._paged_tq`), com a mesma estrutura de
duas passagens e diferindo apenas em como uma linha é lida. Acrescentar Q8
assim significaria uma quarta cópia de oitenta linhas.

O runtime já tinha acessores genéricos por codec — `kv_row_layout`, `kv_lane`
e `valid_kv_rows` — usados pela atenção mista dos tiers. `nexa_causal_gqa_attention_paged_codec`
usa esses mesmos acessores num kernel homogêneo, recebendo o id do codec
(0 F32, 3 Q3, 4 Q4, 8 Q8). Os kernels dedicados continuam existindo e em uso;
uma regressão executa os dois caminhos sobre os mesmos bytes e exige o mesmo
resultado, porque um despacho novo não pode mudar um número que já tinha kernel.

## Q8 no cache

Escala float32 por grupo e um byte com sinal por coordenada — o mesmo layout do
codec de pesos, agora por token e head. `nexa_q8_quantize` escreve as páginas;
a atenção lê os códigos direto, sem expandir página, head ou prefixo.

Medido na fixture tiny com páginas de 2 tokens e grupo 4, prefill de três tokens
mais um decode:

| Codec KV | Bytes por token | Erro máximo nos logits |
| --- | ---: | ---: |
| Q3 | 12 B | 0,506 |
| Q4 | 12 B | 0,245 |
| Q8 | 16 B | 0,0096 |
| F32 | 32 B | 0 |

Com grupo 4, Q3 e Q4 custam o mesmo — a escala de quatro bytes domina, como
acontece nos pesos. Q8 custa metade do F32 e erra 25 vezes menos que Q4.

## Uso

```bash
python3 tools/nexa_run.py MODELO --tokens 1,3,5 --decode-tokens 7 \
  --kv-cache --kv-page-tokens 2 --kv-codec q8 --kv-group-size 4 \
  --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB
```

Prefill em chunks continua equivalente ao prefill único: as páginas são
quantizadas por token, então um prefixo já escrito nunca é requantizado.

## Limites

Este incremento cobre o **modo homogêneo paginado**. A política de idade
(`--kv-policy age`) mantém seus três tiers fixos F32/Q4/Q3, e o backing store
continua gravando cold Q3: colocar Q8 numa política de tiers é decidir qual
degrau ele substitui, o que pertence a M4.05d junto dos critérios de qualidade.

Meia precisão e Q2 **no cache** continuam fora: o KV é escrito a cada token, e
um codec sem escala por grupo muda o contrato de escrita por página. TQ mantém
seu kernel próprio, por causa do contexto e do codebook compartilhados. O
[checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
próxima tarefa.
