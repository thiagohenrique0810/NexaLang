# NexaTokenizer V1: BPE byte-level determinístico

O décimo sexto incremento acrescenta o tokenizer que faltava entre texto e o
executor CPU. Com ele, `tools/nexa_run.py --prompt "texto" --tokenizer DIR`
codifica, executa prefill/decode nativos e devolve o texto gerado, sem PyTorch
em nenhuma etapa. É o pré-requisito de M5.01; o modelo treinado continua
pendente.

Este guia descreve o formato, a segmentação, o treinamento reproduzível e os
limites. Vocabulário congelado de 32768 com corpus real permanece em LLM.02c2.

## Modelo do tokenizer

BPE sobre **bytes**. Os 256 primeiros IDs são os bytes, então:

- Não existe token desconhecido. Qualquer texto — inclusive emoji, CJK, bytes
  de controle ou UTF-8 que o corpus nunca viu — codifica e volta idêntico.
- `decode_bytes` de uma sequência produzida por `encode` devolve exatamente os
  bytes originais. `decode` aplica UTF-8 com política de erro explícita, porque
  uma sequência cortada no meio de um caractere não é UTF-8 válido.

### Tokens especiais

Os IDs de 256 em diante começam com os especiais, na ordem declarada:
`<|pad|>`, `<|bos|>`, `<|eos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`,
`<|tool|>`, `<|tool_result|>`, `<|memory|>`, `<|route|>`, `<|json|>`, `<|end|>`.

Eles cobrem os papéis e formatos que o plano Omni exige e **nunca são produzidos
por texto**. Escrever `<|system|>` num prompt codifica como aqueles bytes
literais; o marcador de papel só entra por `prefix`/`suffix` explícitos, ou por
`--bos` na CLI. Isso não é detalhe de implementação: é o que impede uma entrada
não confiável de forjar um papel no diálogo, e está coberto por regressão.

### Segmentação

`class_runs_with_leading_space_and_single_digits_v1`. Um segmento é um espaço
opcional à esquerda seguido de uma sequência de uma classe de caractere
(letra, dígito, espaço, outro). Um merge **nunca** cruza fronteira de segmento.

- Dígitos saem um a um: `2024` são quatro tokens, e números não viram pedaços
  arbitrários do corpus.
- Runs do mesmo caractere de espaço ficam juntos, então indentação e linhas em
  branco são explícitas; espaços de tipos diferentes separam.
- `segment(" 2024-01-02")` → `[" 2", "0", "2", "4", "-", "0", "1", "-", "0", "2"]`.

Como a segmentação faz parte do manifesto, um asset treinado com outra regra é
rejeitado no carregamento em vez de produzir IDs silenciosamente diferentes.

## Treinamento reproduzível

`compiler/tokenizer_trainer.py` conta segmentos do corpus, normaliza cada
documento para NFC e aplica merges gulosos. O empate de frequência é resolvido
pelos bytes do par, nunca pela ordem de iteração de um dicionário, então o mesmo
corpus produz o mesmo vocabulário, a mesma ordem de merges e os mesmos IDs em
qualquer máquina. A identidade do corpus é um SHA-256 dos hashes dos documentos
**ordenados**, de modo que a ordem dos shards não muda o hash.

Um corpus pequeno não alcança o alvo: o manifesto registra
`requested_vocab_size`, `vocab_size` e `reached_target` em vez de fingir que
chegou. Nada quebra — os tokens de byte garantem cobertura total.

## Formato do asset

```text
tokenizer/
  manifest.json   # formato, versão, segmentação, especiais, tamanhos e SHA-256
  vocab.bin       # NEXATOKV + versão/contagem + (u16 tamanho, bytes) por token
  merges.bin      # NEXATOKM + versão/contagem + pares (u32, u32) em ordem de rank
```

Tudo little-endian. O carregamento confere tamanho e SHA-256 de cada arquivo
contra o manifesto, rejeita versão/modelo/segmentação diferentes, magic
inválido, truncamento, bytes sobrando, merge apontando para fora do vocabulário,
token duplicado e ID especial que não corresponde à sua entrada. O manifesto
guarda a identidade do corpus de treino, como pede o plano de treinamento.

## Uso

```bash
# Treinar a partir de um diretório de .txt ou de um .jsonl com campo text.
python3 tools/nexa_tokenizer.py train --corpus data/corpus --out artifacts/tokenizers/v1 --vocab-size 32768

# Verificar o asset e medir eficiência por domínio (bytes/token, chars/token).
python3 tools/nexa_tokenizer.py inspect artifacts/tokenizers/v1 --samples samples.json

# Codificar e decodificar; encode falha se o round-trip não for exato.
python3 tools/nexa_tokenizer.py encode artifacts/tokenizers/v1 --text "memória 512" --prefix "<|bos|>"
python3 tools/nexa_tokenizer.py decode artifacts/tokenizers/v1 --ids 257,79,398 --skip-special
```

Execução a partir de texto, com qualquer modo de KV já suportado:

```bash
python3 tools/nexa_run.py MODELO --prompt "O NexaLang compila" \
  --tokenizer artifacts/tokenizers/v1 --bos --generate 4 \
  --kv-cache --kv-page-tokens 2 --max-sequence-length 48 \
  --tile-rows 4 --memory-budget 8MiB
```

O relatório ganha `prompt`, `tokenizer` (com os SHA-256 dos arquivos),
`generated_text` e `decoded_text`. O runner exige que
`tokenizer.vocab_size == config.vocab_size`: um asset de outro vocabulário
indexaria os embeddings errados em silêncio, então é erro, não aviso.

## Limites

O vocabulário é uma **proposta a congelar**, não uma escolha validada: 32768 com
corpus real, medição de eficiência por idioma e por linguagem de programação, e
a decisão de congelar ficaram em LLM.02c2. Trocar o tokenizer depois do modelo
base exige estratégia de migração explícita — os IDs especiais são estáveis
justamente para não renumerar uma família inteira.

O encoder é Python e usa cache por segmento; é adequado a prompts, não a
tokenizar um corpus de bilhões de tokens, o que pertence ao pipeline NexaData
(LLM.02d). Não há normalização além de NFC no treino, nem truncamento
automático: quem chama decide o limite de contexto. O
[checklist](BLUEPRINT_512MB_CHECKLIST.md) registra a suíte, os comandos e a
próxima tarefa.
