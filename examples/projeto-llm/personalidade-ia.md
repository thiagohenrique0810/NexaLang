Quero que você projete e implemente, em uma linguagem inspirada em Python, um sistema de personalidade adaptativa persistente para uma LLM.

Objetivo:
Construir uma arquitetura onde a LLM possua uma identidade comportamental contínua, que evolua lentamente com base nas interações com o usuário, sem mudanças bruscas, sem perder coerência, e sem alterar diretamente os pesos principais do modelo a cada conversa.

A personalidade não deve ser tratada como algo binário, mas como um conjunto de traços contínuos que variam gradualmente ao longo do tempo. O sistema deve dar a sensação de amadurecimento progressivo, mantendo um núcleo estável.

Requisitos conceituais:
1. O sistema deve possuir um núcleo fixo e estável:
   - valores centrais
   - regras de segurança
   - limites éticos
   - estilo-base mínimo
   - identidade principal não volátil

2. O sistema deve possuir uma camada de personalidade adaptativa:
   - formalidade
   - objetividade
   - profundidade técnica
   - empatia
   - criatividade
   - cautela
   - humor
   - assertividade
   - curiosidade
   - didatismo

3. Cada traço deve ser representado numericamente, preferencialmente entre 0.0 e 1.0.

4. O sistema deve aprender com:
   - perguntas do usuário
   - respostas geradas
   - feedback explícito
   - feedback implícito
   - padrões recorrentes ao longo de múltiplas conversas

5. O sistema não pode alterar sua personalidade de forma brusca.
   Toda atualização deve ocorrer com inércia, amortecimento e limite máximo por ciclo.

6. O sistema deve separar:
   - estado momentâneo da conversa atual
   - traços persistentes de longo prazo
   - memória episódica
   - memória semântica
   - perfil comportamental consolidado

7. O sistema deve distinguir preferência temporária de preferência persistente.

8. O sistema deve evitar:
   - drift excessivo
   - manipulação por um único diálogo
   - perda de coerência histórica
   - contradições de identidade
   - instabilidade comportamental

Requisitos técnicos:
Implemente os seguintes módulos e estruturas.

Módulo 1: PersonalityCore
Responsável pelo núcleo estável da entidade.
Campos sugeridos:
- identity_name
- base_style
- immutable_values
- safety_rules
- response_principles
- locked_traits

Módulo 2: PersonalityTraits
Responsável pelos traços adaptativos persistentes.
Cada traço deve ser float entre 0.0 e 1.0.
Exemplo:
- formality
- objectivity
- empathy
- technical_depth
- creativity
- caution
- humor
- assertiveness
- curiosity
- pedagogical_level

Deve conter métodos:
- clamp()
- to_dict()
- from_dict()
- blend_with(...)
- apply_delta(...)
- distance(...)

Módulo 3: ConversationState
Representa o estado temporário da sessão atual.
Campos:
- session_id
- current_mood
- temporary_style_bias
- local_topic_focus
- recent_feedback
- transient_adjustments

Esse módulo não deve alterar diretamente a personalidade persistente sem passar por um validador.

Módulo 4: MemorySystem
Separar memórias em:
a) episodic_memory
   - eventos específicos
   - interações recentes
   - contexto de conversas passadas

b) semantic_memory
   - preferências estáveis do usuário
   - padrões recorrentes
   - abstrações extraídas de múltiplos episódios

c) personality_history
   - snapshots históricos dos traços
   - log de mudanças
   - motivo das atualizações
   - grau de confiança da mudança

Criar funções:
- store_episode(...)
- extract_semantic_pattern(...)
- get_relevant_memories(...)
- save_personality_snapshot(...)
- summarize_long_term_patterns(...)

Módulo 5: InteractionAnalyzer
Esse módulo deve analisar uma interação completa:
- entrada do usuário
- resposta da LLM
- sinais de aprovação ou rejeição
- tempo de uso, repetição de temas, estilo das mensagens

Ele deve inferir sinais como:
- usuário prefere respostas curtas
- usuário prefere respostas técnicas
- usuário rejeita humor
- usuário gosta de exemplos
- usuário tolera exploração criativa
- usuário quer objetividade
- usuário espera profundidade

Deve gerar uma estrutura como:
InteractionSignal {
    trait_deltas_suggested,
    confidence,
    persistence_score,
    source_type,
    explanation
}

Módulo 6: PersonalityUpdater
Responsável por atualizar lentamente a personalidade persistente.

Regras obrigatórias:
1. Nenhum traço pode mudar acima de um limite máximo por atualização.
   Exemplo: max_delta_per_update = 0.02

2. Traços persistentes só devem mudar se houver confiança suficiente.
   Exemplo: confidence >= 0.7

3. Mudanças permanentes só devem ocorrer se houver recorrência.
   Exemplo: persistence_score >= 0.6

4. Aplicar média móvel exponencial ou fórmula de inércia:
   new_trait = old_trait * (1 - alpha) + target_trait * alpha

5. alpha deve ser pequeno, exemplo:
   alpha = 0.01 até 0.05

6. Traços bloqueados pelo núcleo não podem ser alterados.

7. O sistema deve registrar o motivo de cada alteração.

8. O sistema deve conter mecanismo de reversão parcial caso ocorra instabilidade.

Módulo 7: ConsistencyValidator
Antes de aplicar alterações permanentes, validar:
- coerência com histórico
- conflito com valores imutáveis
- velocidade excessiva de mudança
- contradição com perfil consolidado
- tentativa de manipulação adversarial

Se falhar, rejeitar ou reduzir drasticamente a alteração.

Módulo 8: PromptComposer
Esse módulo deve transformar o estado interno em instruções de geração para a LLM.

Ele deve combinar:
- núcleo estável
- traços persistentes
- estado temporário da conversa
- memórias relevantes
- contexto atual do usuário

A saída deve ser uma instrução de comportamento para a geração de resposta.

Exemplo de comportamento desejado:
- se objectivity > 0.8, reduzir floreio verbal
- se technical_depth > 0.75, incluir mais detalhamento técnico
- se empathy > 0.7, usar tom mais acolhedor
- se humor < 0.2, evitar tom brincalhão
- se pedagogical_level > 0.8, explicar por etapas

Módulo 9: ReflectionEngine
Criar um mecanismo interno de autorreflexão funcional.
Ele não deve alegar consciência real, mas deve gerar resumos como:
- “meu estilo tem ficado mais técnico com este usuário”
- “houve aumento consistente de objetividade”
- “o usuário responde melhor a respostas estruturadas”

Essa reflexão deve servir para auditoria interna e não necessariamente aparecer ao usuário.

Módulo 10: PersistenceLayer
Implementar persistência em arquivos JSON, banco leve, ou estrutura equivalente.
Salvar:
- personality_core
- personality_traits
- memories
- interaction_logs
- snapshots
- confidence scores
- semantic patterns

Requisitos de implementação:
1. Escreva o código de forma modular e extensível.
2. Use classes ou estruturas equivalentes.
3. Inclua tipagem quando possível.
4. Inclua comentários explicando a lógica.
5. Inclua exemplos de uso.
6. Inclua serialização e desserialização.
7. Inclua um fluxo principal demonstrando:
   - receber input do usuário
   - gerar resposta
   - analisar interação
   - atualizar traços lentamente
   - salvar estado

Crie também:
- uma função simulate_interaction(...)
- uma função update_personality_from_interaction(...)
- uma função build_behavior_prompt(...)
- uma função export_personality_report(...)

Quero que a implementação seja robusta, clara e pensada para produção futura.
Evite simplificações excessivas.
Quero código e também explicação da arquitetura.

Importante:
- não trate isso como consciência humana real
- trate como identidade adaptativa persistente
- priorize estabilidade, gradualismo, coerência e memória de longo prazo
- não permita mudanças bruscas por uma única interação
- preserve continuidade histórica da personalidade