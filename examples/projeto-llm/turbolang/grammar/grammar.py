"""TurboLang grammar rules.

TurboLang Grammar (EBNF):

    program     ::= model_block*
    model_block ::= 'model' STRING '{' directive* '}'
    directive   ::= keyword param*
    param       ::= IDENTIFIER '=' value
                  | value
    value       ::= STRING | INTEGER | FLOAT | BOOLEAN | IDENTIFIER
    keyword     ::= 'weights' | 'kv_cache' | 'attention' | 'decode'
                  | 'scheduler' | 'target'
"""

GRAMMAR_EBNF = """
program     ::= model_block*
model_block ::= 'model' STRING '{' directive* '}'
directive   ::= keyword param*
param       ::= IDENTIFIER '=' value | value
value       ::= STRING | INTEGER | FLOAT | BOOLEAN | IDENTIFIER
keyword     ::= 'weights' | 'kv_cache' | 'attention' | 'decode'
              | 'scheduler' | 'target'
"""

# Valid directive keywords and their expected parameter structure
DIRECTIVE_SCHEMAS = {
    'weights': {
        'positional': ['strategy', 'dtype'],
        'named': {},
        'valid_strategies': ['quantized', 'full', 'mixed'],
        'valid_dtypes': ['int4', 'int8', 'fp16', 'bf16', 'fp32'],
    },
    'kv_cache': {
        'positional': ['strategy'],
        'named': {'bits': int, 'max_size': int},
        'valid_strategies': ['turboquant', 'standard', 'paged', 'none'],
    },
    'attention': {
        'positional': ['variant'],
        'named': {'streaming': bool, 'window': int},
        'valid_variants': ['flash', 'standard', 'linear', 'sliding_window'],
    },
    'decode': {
        'positional': ['mode'],
        'named': {'drafter': str, 'k': int, 'temperature': float},
        'valid_modes': ['autoregressive', 'speculative', 'parallel'],
    },
    'scheduler': {
        'positional': ['policy'],
        'named': {'max_batch': int, 'max_ctx': int, 'timeout_ms': int},
        'valid_policies': ['continuous_batching', 'static', 'dynamic'],
    },
    'target': {
        'positional': ['device', 'backend'],
        'named': {'memory_limit': str},
        'valid_devices': ['gpu', 'cpu', 'auto'],
    },
}
