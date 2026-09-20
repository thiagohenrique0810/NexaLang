# StreamingRegions: onde dá para andar por tiles de linhas

Este incremento entrega M6.05a — **detecção, análise apenas**. Nada é executado
em streaming; o executor continua avaliando tensores inteiros.

A manchete honesta é o oposto da empolgante. Na arquitetura real, os logits são
**97,7% da arena de prefill**, e eles são saída do grafo: não encolhem. Por isso
as StreamingRegions compram **2,29%**, não 94%.

```bash
python3 tools/nexa_graph.py regions \
  --definition models/nexalm512/architecture.nxl --model NexaLM512_R0 \
  --sequence-length 512 --tile-rows 1
```

## A regra de corte, verificada no código

Lendo `compiler/model_ir.py` ao longo do eixo de tokens: Embedding pega uma
linha de peso por token, RMSNorm reduz sobre as features da própria linha,
MatMul reduz sobre K dentro da própria linha, RoPE rotaciona lanes dentro da
própria linha, Add e SwiGLU são elementwise. Todos computam a linha *t* só a
partir da linha *t*.

`CausalAttention` não: a linha *t* lê as linhas 0..*t* de key e value. Logo
**atenção encerra uma região e nunca fica dentro de uma.**

Duas condições que é fácil perder:

1. **Operando que não acompanha o eixo tem de ser inteiro.** O peso de um
   MatMul, o vetor de um RMSNorm. Só constantes e entradas do grafo valem. No
   grafo Llama isso nunca falha — todo segundo operando é constante — então a
   regressão constrói um grafo onde o segundo operando é ativação, para a
   condição poder falhar.
2. **Independente por linha não é cego à posição.** RoPE não lê outra linha e
   ainda assim precisa saber *qual* linha está rotacionando. O driver carrega o
   offset absoluto do tile; `evaluate_op(..., row_offset=)` existe por isso.
   Esquecer o offset gira cada tile como se a sequência recomeçasse.

## Prova: identidade byte a byte

O grafo tiny é avaliado duas vezes — inteiro, e com cada região detectada
dirigida tile a tile — e **todos os tensores** são serializados com
`struct.pack("<f", ...)` e comparados byte a byte, para T em {1, 2, 3, 4, 5, 6}
sobre S=6. Sem tolerância.

Dois controles negativos impedem que isso seja vácuo: o driver sem o offset de
linha **tem de divergir**, e uma região desenhada por cima da atenção **tem de
divergir**. Nos dois casos o primeiro tile continua certo — linhas 0..1 só
atendem a 0..1 — e o resto muda, que é exatamente o formato do erro esperado.

## O que uma região publica

`StreamingRegion` é congelado: nome, intervalo semiaberto `[start, end)` de
índices de operação, eixo, extensão em linhas, `tile_rows`, os tensores que
cruzam a fronteira contra os interiores, as constantes lidas inteiras, e
`live_tile_buffers` — quantos buffers de tile interiores estão vivos ao mesmo
tempo.

Esse último número é o argumento inteiro a favor do tiling. Na região do MLP do
grafo tiny: **10 tensores interiores, 4 vivos simultaneamente.**

`detect_streaming_regions(graph)` não escolhe altura de tile; ela reporta onde
streaming é possível. `derive_streamed_activation_requests(graph, regions, T)`
devolve as mesmas requisições que `derive_activation_requests`, com os
interiores dimensionados a T linhas e os tempos de vida intactos.

## O número medido

Arena planejada pelo mesmo `MemoryPlanner` do runtime, T=1:

| Modelo | S | Regiões | Pico | Com streaming | Economia | Logits/pico |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixture tiny | 4 | 3 | 832 | 512 | **38,46%** | 25,00% |
| fixture tiny | 16 | 3 | 3.136 | 2.432 | 22,45% | 26,53% |
| NexaLM512_R0 | 4 | 17 | 536.576 | 527.360 | 1,72% | 97,71% |
| NexaLM512_R0 | 128 | 17 | 17.170.432 | 16.780.288 | 2,27% | 97,71% |
| NexaLM512_R0 | 512 | 17 | 68.681.728 | 67.111.936 | **2,29%** | 97,71% |
| NexaLM512_v1 | 512 | 33 | 69.206.016 | 67.112.960 | 3,02% | 96,97% |

Os 38% da fixture tiny são o aviso, não o resultado: ali o vocabulário tem 13
entradas e os logits são 25% da arena. Escolher aquela fixture para anunciar o
ganho seria escolher o número. Numa arquitetura real o vocabulário é 32.768 e a
matriz de logits sozinha é quase toda a arena.

**A economia fica presa em ~2% enquanto não existir um consumidor de logits em
streaming.** Um decodificador que consumisse os logits linha a linha — ou que só
pedisse a última linha, que é tudo que o decode usa — mudaria o denominador.
Isso é outro trabalho, em outro módulo.

O relatório também traz `activation_request_bytes` ao lado do pico, e os dois
discordam de propósito: o planejador já reusa ativações mortas, então encolher
um buffer só ajuda onde ele era o mais alto.

## Limites

Nada executa em streaming. Não há tiling nos kernels C, não há laço de tiles no
executor, e `derive_streamed_activation_requests` não é consumida por
`runtime/nexapack`. O driver tile a tile vive na suíte de regressão, como prova
de que a regra de corte está certa — não como caminho de execução. Isso fica
preservado em M6.05c.

Só o eixo de tokens é analisado. Tiling por features, split-K e pressão de
registradores por operador não entram aqui.

O [checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte e a próxima tarefa.
