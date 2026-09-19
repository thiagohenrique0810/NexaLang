# KV Q4 paginado em CPU

O modo Q4 comprime as chaves e valores usados pela atenção, preservando as
[páginas e transações existentes](NEXALM_KV_PAGINADO_CPU.md). O kernel consulta
diretamente escalas e códigos compactados; não aloca uma cópia F32 do prefixo.
Pesos Q4 e KV Q4 são escolhas distintas. A representação dos pesos permanece igual.

## API e reprodução

```sh
# Após gerar/importar a fixture tiny conforme o guia de importação.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 8 --kv-codec q4 --kv-group-size 4 --prefill-chunk-size 2 --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-q4.json

# Comparação F32 com os mesmos limites e tamanho de página.
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 8 --prefill-chunk-size 2 --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB --report artifacts/reports/nexalm-kv-f32-comparison.json
```

`--kv-codec q4` exige `--kv-cache` e `--kv-page-tokens`. `--kv-group-size` só é
aceito para Q4 e [Q3](NEXALM_KV_Q3_CPU.md); seu padrão é 32 e o limite é 1.048.576. Grupos maiores que a dimensão
do head são válidos, mas armazenam padding. Não existe escolha automática de grupo
por qualidade ou redução de memória nesta etapa. O modo padrão continua F32.

Na API, use `PagedTransformerSession(..., kv_codec="q4", kv_group_size=32)`.
`prefill`, `append`, `decode`, `reset` e `close` preservam seus contratos anteriores.
Configurações ou orçamentos inválidos são rejeitados antes de alocar páginas ou
carregar pesos. O plano JSON F32 continua compatível; Q4 declara codec/layout,
dimensão de head, grupos, bytes por head/token e versão explicitamente.

## Contrato dos bytes

O layout é `[token, head KV, grupo dentro do head]`, separadamente para K e V de
cada camada. Cada head usa o codec `Q4_GROUPED` versão 1 do NexaPack:

- Grupo de `G` coordenadas: escala F32 little-endian seguida de `ceil(G/2)` bytes.
- Escala `float32(max(abs(grupo))/7)`; arredondamento para o inteiro mais próximo,
  com empates para longe de zero e saturação em `[-7,7]`.
- Primeiro valor no nibble baixo; negativos em complemento de dois. Código -8
  é inválido. Coordenadas ausentes e nibble alto sem uso ficam zerados.
- Grupos zerados têm escala e códigos zero. Entradas não finitas e underflow da
  escala de grupos não zero produzem erro. O kernel rejeita escalas/códigos/padding
  inválidos no prefixo visível.

K é quantizado **após RoPE** e V após sua projeção. Cada token/head recebe suas
próprias escalas: a chegada de novos tokens não modifica valores já confirmados.
Páginas parcialmente preenchidas já armazenam Q4; não há uma página F32 paralela,
nem re-encode do prefixo quando a página fica cheia. Tokens futuros ou sujos após
uma falha não são lidos até serem sobrescritos e incluídos em uma nova chamada.

O quantizador C existente escreve os segmentos diretamente em suas páginas. A
atenção reconstrói `escala * código` em registradores escalares double durante o
produto e a soma de V, sem arredondamento intermediário para F32. O scratch contém
exponenciais F32 e as reduções usam double, como no kernel anterior. O resultado
final é F32. Não existe buffer de desquantização por head ou por prefixo.

## Memória e transações

Com dimensão de head `D`, `Hkv` heads, `L` camadas e `P` tokens por página:

```text
bytes por grupo = 4 + ceil(G/2)
bytes por head = ceil(D/G) * bytes por grupo
bytes por token de um buffer K ou V = Hkv * bytes por head
payload por página = 2 * L * P * Hkv * bytes por head
alocação por página = 2 * L * align64(P * Hkv * bytes por head) + 63
```

Os buffers de camada são alinhados, mas escalas dentro de rows podem ficar
desalinhadas: o codec lê bytes little-endian, sem converter endereços em `float*`.
Escalas, padding, tabelas de ponteiros e páginas de staging entram no orçamento.
Reservar a capacidade máxima continua diferente de alocar suas páginas.

Append só escreve o sufixo novo; prefill substitui o prompt com páginas novas e
libera as antigas após commit. Falhas preservam o prefixo e liberam páginas novas,
inclusive com exceções retidas. Essas garantias são compartilhadas com o modo F32.

O relatório acrescenta `kv_codec`/`kv_group_size`. Em `memory`,
`kv_encoded_bytes_per_token` inclui K/V, todas as camadas, escalas e padding de
grupos; `kv_valid_prefix_bytes` usa esse tamanho. `kv_valid_prefix_f32_bytes` informa
o equivalente F32 e `kv_full_dequantized_buffer_bytes` é zero. A alocação física
residente inclui também padding de páginas e posições ainda não usadas.
`io.kv_bytes_written` mede bytes codificados escritos; `kv_source_f32_bytes_quantized`
mede as ativações novas fornecidas ao quantizador. Não somar ambos como KV residente.

Para `D=64`, `G=32`, cada head ocupa 40 B por token, contra 256 B em F32, antes de
alinhamento de página. Heads muito pequenos, grupos inadequados ou páginas pequenas
podem não reduzir a alocação física. Não se assume um ganho universal de oito vezes.
O teto permanece limitado a buffers CPU gerenciados; não mede RSS ou VRAM.

## Erro numérico e evidência

Para reproduzir as medições com PyTorch local e o checkpoint original da fixture:

```sh
python3 tools/nexa_run.py artifacts/models/nexalm-tiny --tokens 1,3,5 --decode-tokens 7 --kv-cache --kv-page-tokens 8 --kv-codec q4 --kv-group-size 4 --prefill-chunk-size 2 --max-sequence-length 8 --tile-rows 3 --memory-budget 96KiB --verify --reference-checkpoint artifacts/checkpoints/nexalm-tiny --report artifacts/reports/nexalm-kv-q4.json
```

A referência implementa as equações em PyTorch e o codec KV em Python, sem chamar
os kernels C. Seus pesos/cache/temporários ficam fora do orçamento de execução.
Um golden produzido pela referência permite verificar logits Q4 KV sem instalar Torch.

Os campos de validação distinguem:

| Campo | Comparação |
|---|---|
| `execution_error` | Kernel nativo vs referência com os mesmos pesos e KV Q4. |
| `quantization_error` | Pesos Q4 vs pesos originais, mantendo KV F32 nos dois lados. |
| `kv_quantization_error` | KV Q4 vs KV F32, mantendo os mesmos pesos Q4. |
| `combined_quantization_error` | Pesos e KV Q4 vs pesos originais e KV F32. |

`verified=true` indica que o erro de execução ficou dentro da tolerância
`1e-5 + 1e-4 * abs(referência)`. Não aprova a perda de qualidade causada pela
quantização. Erros de representações diferentes são medidos separadamente, sem
um limiar de qualidade arbitrário. Qualidade/perplexidade de modelo treinado segue
pendente.

Na fixture sintética de 728 parâmetros, quatro IDs, contexto máximo oito, `P=8`,
`G=4` e chunks de dois tokens, foram obtidos:

| Medida | F32 KV | Q4 KV |
|---|---:|---:|
| Bytes codificados por token, todas as camadas K/V | 32 | 12 |
| Payload da página residente | 256 B | 96 B |
| Alocação residente com padding | 319 B | 191 B |
| Pico gerenciado entre chamadas | 66.942 B | 66.814 B |
| Reserva KV de admissão | 638 B | 382 B |

O erro máximo de execução Q4 foi zero. O erro dos pesos foi 0,4435420930; o erro
adicional do KV, 0,2454764843; o erro combinado, 0,4757611454. Esses erros não são
aditivos e o hash dos logits Q4 KV difere do F32 KV. São medidas de uma fixture não
treinada, sem conclusão sobre velocidade ou qualidade de linguagem.

O [Q3 CPU também está implementado](NEXALM_KV_Q3_CPU.md), com formato próprio e
comparação F32/Q4/Q3. O [checklist](BLUEPRINT_512MB_CHECKLIST.md) mantém TQ01, tiers hot/warm/cold,
evicção/re-encode, múltiplas sequências e GPU como etapas futuras. A prova Q4 CPU
conclui somente os subitens específicos de codec e atenção direta para Q4.
