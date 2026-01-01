class CompilerError(Exception):
    def __init__(self, message, line=None, column=None, file=None, hint=None, error_code=None):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.file = file
        self.hint = hint
        self.error_code = error_code

    def __str__(self):
        loc = ""
        if self.file:
            loc += f"{self.file}:"
        if self.line:
            loc += f"{self.line}:"
        if self.column:
            loc += f"{self.column}:"
        
        if loc:
            return f"{loc} {self.message}"
        return self.message
