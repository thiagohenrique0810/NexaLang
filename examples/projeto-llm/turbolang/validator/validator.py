"""TurboLang Semantic Validator — validates AST against grammar rules."""

from turbolang.ast.nodes import Program, ModelBlock, Directive
from turbolang.grammar.grammar import DIRECTIVE_SCHEMAS


class ValidationError(Exception):
    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        super().__init__(f"Validation error at L{line}:{col}: {message}")


class ValidationWarning:
    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.message = message
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Warning L{self.line}:{self.col}: {self.message}"


class Validator:
    def __init__(self):
        self.warnings: list[ValidationWarning] = []

    def validate(self, program: Program) -> list[ValidationWarning]:
        self.warnings = []
        if not program.models:
            raise ValidationError("Program must contain at least one model block", 1, 1)

        seen_names = set()
        for model in program.models:
            if model.name in seen_names:
                raise ValidationError(
                    f"Duplicate model name: {model.name!r}", model.line, model.col
                )
            seen_names.add(model.name)
            self._validate_model(model)

        return self.warnings

    def _validate_model(self, model: ModelBlock):
        if not model.name:
            raise ValidationError("Model block must have a name", model.line, model.col)

        if not model.directives:
            self.warnings.append(ValidationWarning(
                f"Model {model.name!r} has no directives", model.line, model.col
            ))
            return

        seen_directives = set()
        for directive in model.directives:
            if directive.keyword in seen_directives:
                self.warnings.append(ValidationWarning(
                    f"Duplicate directive {directive.keyword!r} in model {model.name!r}",
                    directive.line, directive.col
                ))
            seen_directives.add(directive.keyword)
            self._validate_directive(directive, model.name)

    def _validate_directive(self, directive: Directive, model_name: str):
        schema = DIRECTIVE_SCHEMAS.get(directive.keyword)
        if schema is None:
            raise ValidationError(
                f"Unknown directive: {directive.keyword!r}", directive.line, directive.col
            )

        # Validate positional params
        expected_positional = schema.get('positional', [])
        if len(directive.positional) > len(expected_positional):
            self.warnings.append(ValidationWarning(
                f"Directive {directive.keyword!r} has {len(directive.positional)} positional params, "
                f"expected at most {len(expected_positional)}",
                directive.line, directive.col
            ))

        # Validate named params
        valid_named = schema.get('named', {})
        for name, value in directive.named.items():
            if name not in valid_named:
                self.warnings.append(ValidationWarning(
                    f"Unknown parameter {name!r} in directive {directive.keyword!r}",
                    directive.line, directive.col
                ))
            else:
                expected_type = valid_named[name]
                if expected_type == int and not isinstance(value, int):
                    raise ValidationError(
                        f"Parameter {name!r} in {directive.keyword!r} expects integer, "
                        f"got {type(value).__name__}",
                        directive.line, directive.col
                    )
                elif expected_type == bool and not isinstance(value, bool):
                    raise ValidationError(
                        f"Parameter {name!r} in {directive.keyword!r} expects boolean, "
                        f"got {type(value).__name__}",
                        directive.line, directive.col
                    )

        # Validate specific enum values
        if directive.positional:
            first_pos = directive.positional[0]
            # Find the valid_* key in the schema
            valid_key = None
            for k in schema:
                if k.startswith('valid_'):
                    valid_key = k
                    break
            if valid_key and valid_key in schema:
                valid_values = schema[valid_key]
                if isinstance(first_pos, str) and first_pos not in valid_values:
                    raise ValidationError(
                        f"Invalid value {first_pos!r} for {directive.keyword!r}. "
                        f"Valid options: {', '.join(valid_values)}",
                        directive.line, directive.col
                    )
