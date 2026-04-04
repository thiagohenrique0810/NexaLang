"""NexaLang Language Server Protocol (LSP) server.

A lightweight LSP server that provides:
 - Diagnostics (syntax + semantic errors)
 - Hover information (type info at cursor)
 - Go-to-definition
 - Document symbols
 - Completion

Communicates via stdin/stdout using the JSON-RPC 2.0 LSP protocol.
"""

import sys
import os
import json
import re
import traceback

# Add bootstrap to path
BOOTSTRAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bootstrap')
sys.path.insert(0, BOOTSTRAP_DIR)

from lexer import Lexer
from n_parser import Parser, FunctionDef, StructDef, EnumDef, TraitDef, ImplDef, VarDecl, ExternBlock
from semantic import SemanticAnalyzer
from errors import CompilerError

# ── JSON-RPC Transport ───────────────────────────────────────────────────

def read_message():
    """Read a JSON-RPC message from stdin (Content-Length header protocol)."""
    headers = {}
    while True:
        line = sys.stdin.buffer.readline().decode('utf-8')
        if line == '\r\n' or line == '\n':
            break
        if ':' in line:
            key, value = line.split(':', 1)
            headers[key.strip()] = value.strip()

    length = int(headers.get('Content-Length', 0))
    if length == 0:
        return None

    body = sys.stdin.buffer.read(length).decode('utf-8')
    return json.loads(body)


def send_message(msg):
    """Send a JSON-RPC message to stdout."""
    body = json.dumps(msg)
    header = f"Content-Length: {len(body.encode('utf-8'))}\r\n\r\n"
    sys.stdout.buffer.write(header.encode('utf-8'))
    sys.stdout.buffer.write(body.encode('utf-8'))
    sys.stdout.buffer.flush()


def send_response(req_id, result):
    send_message({"jsonrpc": "2.0", "id": req_id, "result": result})


def send_error(req_id, code, message):
    send_message({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def send_notification(method, params):
    send_message({"jsonrpc": "2.0", "method": method, "params": params})


# ── LSP Server State ────────────────────────────────────────────────────

class NexaLSP:
    def __init__(self):
        self.documents = {}   # uri -> source text
        self.ast_cache = {}   # uri -> ast
        self.symbols = {}     # uri -> list of symbols
        self.diagnostics = {} # uri -> list of diagnostics
        self.running = True

    # ── Analysis ─────────────────────────────────────────────────────

    def analyze_document(self, uri, source):
        """Full lexer + parser + semantic analysis, collecting diagnostics."""
        self.documents[uri] = source
        diagnostics = []
        ast = []
        symbols = []

        # Lex
        try:
            tokens = Lexer(source).tokenize()
        except Exception as e:
            line = getattr(e, 'line', 1) or 1
            diagnostics.append(self._make_diagnostic(line - 1, 0, str(e), 1))
            self._publish_diagnostics(uri, diagnostics)
            return

        # Parse
        try:
            parser = Parser(tokens)
            ast = parser.parse()
            self.ast_cache[uri] = ast
        except Exception as e:
            msg = str(e)
            line = 0
            # Extract line from parser error messages
            m = re.search(r'(\d+):(\d+)', msg)
            if m:
                line = int(m.group(1)) - 1
            diagnostics.append(self._make_diagnostic(line, 0, msg, 1))
            self._publish_diagnostics(uri, diagnostics)
            return

        # Collect symbols from AST
        symbols = self._collect_symbols(ast)
        self.symbols[uri] = symbols

        # Semantic analysis
        try:
            sa = SemanticAnalyzer()
            sa.current_file_path = self._uri_to_path(uri)
            sa.current_dir = os.path.dirname(sa.current_file_path) if sa.current_file_path else '.'
            sa.analyze(ast)

            # Collect warnings
            for (msg, line, col) in sa.warnings:
                diagnostics.append(self._make_diagnostic(
                    (line or 1) - 1, (col or 1) - 1, msg, 2  # Warning
                ))

        except CompilerError as e:
            line = (e.line or 1) - 1
            col = (e.column or 1) - 1
            msg = e.message
            if e.hint:
                msg += f" (hint: {e.hint})"
            diagnostics.append(self._make_diagnostic(line, col, msg, 1))
        except Exception as e:
            # Non-fatal: log but still provide partial diagnostics
            pass

        self._publish_diagnostics(uri, diagnostics)

    def _make_diagnostic(self, line, col, message, severity):
        """Create an LSP Diagnostic object. severity: 1=Error, 2=Warning, 3=Info, 4=Hint"""
        return {
            "range": {
                "start": {"line": max(0, line), "character": max(0, col)},
                "end": {"line": max(0, line), "character": col + 20}
            },
            "severity": severity,
            "source": "nexalang",
            "message": message
        }

    def _publish_diagnostics(self, uri, diagnostics):
        self.diagnostics[uri] = diagnostics
        send_notification("textDocument/publishDiagnostics", {
            "uri": uri,
            "diagnostics": diagnostics
        })

    # ── Symbols ──────────────────────────────────────────────────────

    def _collect_symbols(self, ast):
        """Extract document symbols from AST."""
        symbols = []
        for node in ast:
            if isinstance(node, FunctionDef):
                params_str = ', '.join(f"{p[0]}: {p[1]}" for p in node.params)
                ret = f" -> {node.return_type}" if node.return_type and node.return_type != 'void' else ''
                detail = f"fn({params_str}){ret}"
                symbols.append({
                    "name": node.name,
                    "kind": 12,  # Function
                    "detail": detail,
                    "range": self._node_range(node),
                    "selectionRange": self._node_range(node),
                })
            elif isinstance(node, StructDef):
                fields_str = ', '.join(f"{n}: {t}" for n, t in node.fields)
                symbols.append({
                    "name": node.name,
                    "kind": 23,  # Struct
                    "detail": f"struct {{ {fields_str} }}",
                    "range": self._node_range(node),
                    "selectionRange": self._node_range(node),
                })
            elif isinstance(node, EnumDef):
                variants = ', '.join(v[0] for v in node.variants)
                symbols.append({
                    "name": node.name,
                    "kind": 10,  # Enum
                    "detail": f"enum {{ {variants} }}",
                    "range": self._node_range(node),
                    "selectionRange": self._node_range(node),
                })
            elif isinstance(node, TraitDef):
                symbols.append({
                    "name": node.name,
                    "kind": 11,  # Interface
                    "detail": f"trait ({len(node.methods)} methods)",
                    "range": self._node_range(node),
                    "selectionRange": self._node_range(node),
                })
            elif isinstance(node, ImplDef):
                trait_str = f" {node.trait_name} for " if node.trait_name else " "
                symbols.append({
                    "name": f"impl{trait_str}{node.struct_name}",
                    "kind": 5,  # Class
                    "detail": f"{len(node.methods)} methods",
                    "range": self._node_range(node),
                    "selectionRange": self._node_range(node),
                })
        return symbols

    def _node_range(self, node):
        line = max(0, getattr(node, 'line', 1) - 1)
        col = max(0, getattr(node, 'column', 1) - 1)
        return {
            "start": {"line": line, "character": col},
            "end": {"line": line, "character": col + len(getattr(node, 'name', ''))}
        }

    # ── Hover ────────────────────────────────────────────────────────

    def handle_hover(self, uri, position):
        """Get type info at cursor position."""
        source = self.documents.get(uri, '')
        if not source:
            return None

        line = position['line']
        char = position['character']
        lines = source.splitlines()
        if line >= len(lines):
            return None

        # Extract word at cursor
        line_text = lines[line]
        word = self._word_at(line_text, char)
        if not word:
            return None

        # Search AST for type info
        ast = self.ast_cache.get(uri, [])
        info = self._find_symbol_info(ast, word)

        if info:
            return {
                "contents": {
                    "kind": "markdown",
                    "value": f"```nexalang\n{info}\n```"
                }
            }

        # Check keywords
        keyword_docs = {
            'fn': 'Function declaration',
            'let': 'Variable binding (immutable by default)',
            'mut': 'Mutable binding modifier',
            'struct': 'Struct type definition',
            'enum': 'Enum type definition',
            'trait': 'Trait (interface) definition',
            'impl': 'Implementation block',
            'match': 'Pattern matching expression',
            'if': 'Conditional expression',
            'while': 'While loop',
            'for': 'For loop (iterator)',
            'return': 'Return statement',
            'kernel': 'GPU kernel function',
            'async': 'Asynchronous function',
            'await': 'Await async result',
            'pub': 'Public visibility modifier',
            'use': 'Import declaration',
            'mod': 'Module declaration',
            'extern': 'External FFI block',
            'type': 'Type alias',
            'break': 'Break from loop',
            'continue': 'Continue to next iteration',
        }

        if word in keyword_docs:
            return {
                "contents": {
                    "kind": "markdown",
                    "value": f"**{word}** — {keyword_docs[word]}"
                }
            }

        return None

    def _word_at(self, line, col):
        """Extract identifier at given column."""
        if col >= len(line):
            return None
        start = col
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
            start -= 1
        end = col
        while end < len(line) and (line[end].isalnum() or line[end] == '_'):
            end += 1
        word = line[start:end]
        return word if word else None

    def _find_symbol_info(self, ast, name):
        """Find type/signature info for a symbol in the AST."""
        for node in ast:
            if isinstance(node, FunctionDef) and node.name == name:
                params = ', '.join(f"{p[0]}: {p[1]}" for p in node.params)
                ret = f" -> {node.return_type}" if node.return_type and node.return_type != 'void' else ''
                prefix = 'async ' if node.is_async else ''
                prefix += 'kernel ' if node.is_kernel else ''
                return f"{prefix}fn {name}({params}){ret}"
            elif isinstance(node, StructDef) and node.name == name:
                fields = '\n    '.join(f"{n}: {t}," for n, t in node.fields)
                generics = f"<{', '.join(g[0] for g in node.generics)}>" if node.generics else ''
                return f"struct {name}{generics} {{\n    {fields}\n}}"
            elif isinstance(node, EnumDef) and node.name == name:
                variants_str = '\n    '.join(
                    f"{v[0]}({', '.join(v[1])})," if v[1] else f"{v[0]},"
                    for v in node.variants
                )
                return f"enum {name} {{\n    {variants_str}\n}}"
            elif isinstance(node, TraitDef) and node.name == name:
                methods = '\n    '.join(f"fn {m.name}(...);" for m in node.methods)
                return f"trait {name} {{\n    {methods}\n}}"
            elif isinstance(node, ImplDef):
                for method in node.methods:
                    if method.name == name:
                        params = ', '.join(f"{p[0]}: {p[1]}" for p in method.params)
                        ret = f" -> {method.return_type}" if method.return_type and method.return_type != 'void' else ''
                        return f"fn {node.struct_name}::{name}({params}){ret}"
        return None

    # ── Go-to-definition ─────────────────────────────────────────────

    def handle_goto_definition(self, uri, position):
        """Jump to definition of symbol at cursor."""
        source = self.documents.get(uri, '')
        lines = source.splitlines()
        line = position['line']
        char = position['character']
        if line >= len(lines):
            return None

        word = self._word_at(lines[line], char)
        if not word:
            return None

        ast = self.ast_cache.get(uri, [])
        for node in ast:
            if isinstance(node, (FunctionDef, StructDef, EnumDef, TraitDef)):
                if node.name == word:
                    return {
                        "uri": uri,
                        "range": self._node_range(node)
                    }
            elif isinstance(node, ImplDef):
                if node.struct_name == word:
                    return {"uri": uri, "range": self._node_range(node)}
                for method in node.methods:
                    if method.name == word:
                        return {"uri": uri, "range": self._node_range(method)}

        return None

    # ── Completion ───────────────────────────────────────────────────

    def handle_completion(self, uri, position):
        """Provide code completions."""
        source = self.documents.get(uri, '')
        lines = source.splitlines()
        line = position['line']
        char = position['character']

        items = []

        # Keywords
        keywords = [
            'fn', 'let', 'mut', 'if', 'else', 'while', 'for', 'return',
            'struct', 'enum', 'trait', 'impl', 'match', 'pub', 'use', 'mod',
            'extern', 'async', 'await', 'kernel', 'break', 'continue', 'type',
            'true', 'false', 'self',
        ]
        for kw in keywords:
            items.append({"label": kw, "kind": 14, "detail": "keyword"})

        # Built-in types
        types = ['i32', 'i64', 'u8', 'f32', 'f64', 'bool', 'string', 'void']
        for t in types:
            items.append({"label": t, "kind": 25, "detail": "type"})

        # Symbols from AST
        ast = self.ast_cache.get(uri, [])
        for node in ast:
            if isinstance(node, FunctionDef):
                params = ', '.join(f"{p[0]}: {p[1]}" for p in node.params)
                items.append({
                    "label": node.name,
                    "kind": 3,  # Function
                    "detail": f"fn({params})",
                    "insertText": f"{node.name}($0)",
                    "insertTextFormat": 2  # Snippet
                })
            elif isinstance(node, StructDef):
                items.append({
                    "label": node.name,
                    "kind": 22,  # Struct
                    "detail": f"struct ({len(node.fields)} fields)"
                })
            elif isinstance(node, EnumDef):
                items.append({
                    "label": node.name,
                    "kind": 13,  # Enum
                    "detail": f"enum ({len(node.variants)} variants)"
                })
                for vname, _ in node.variants:
                    items.append({
                        "label": f"{node.name}::{vname}",
                        "kind": 20,  # EnumMember
                    })
            elif isinstance(node, TraitDef):
                items.append({
                    "label": node.name,
                    "kind": 8,  # Interface
                })

        # Built-in functions
        builtins = [
            ('print', 'fn(value) — Print to stdout'),
            ('assert!', 'macro(condition, message) — Assert with message'),
            ('cast', 'fn<T>(value) — Type cast'),
            ('sizeof', 'fn<T>() — Size of type in bytes'),
        ]
        for name, detail in builtins:
            items.append({"label": name, "kind": 3, "detail": detail})

        return {"isIncomplete": False, "items": items}

    # ── Helpers ──────────────────────────────────────────────────────

    def _uri_to_path(self, uri):
        if uri.startswith('file://'):
            return uri[7:]
        return uri

    # ── Request Handler ──────────────────────────────────────────────

    def handle(self, msg):
        method = msg.get('method')
        params = msg.get('params', {})
        req_id = msg.get('id')

        if method == 'initialize':
            send_response(req_id, {
                "capabilities": {
                    "textDocumentSync": {
                        "openClose": True,
                        "change": 1,  # Full sync
                        "save": {"includeText": True}
                    },
                    "hoverProvider": True,
                    "completionProvider": {
                        "triggerCharacters": ['.', ':', '<'],
                        "resolveProvider": False
                    },
                    "definitionProvider": True,
                    "documentSymbolProvider": True,
                    "diagnosticProvider": {
                        "interFileDependencies": False,
                        "workspaceDiagnostics": False
                    }
                },
                "serverInfo": {
                    "name": "nexalang-lsp",
                    "version": "0.1.0"
                }
            })

        elif method == 'initialized':
            pass  # Client acknowledged

        elif method == 'shutdown':
            send_response(req_id, None)

        elif method == 'exit':
            self.running = False

        elif method == 'textDocument/didOpen':
            td = params['textDocument']
            self.analyze_document(td['uri'], td['text'])

        elif method == 'textDocument/didChange':
            td = params['textDocument']
            changes = params.get('contentChanges', [])
            if changes:
                self.analyze_document(td['uri'], changes[-1]['text'])

        elif method == 'textDocument/didSave':
            td = params['textDocument']
            text = params.get('text')
            if text:
                self.analyze_document(td['uri'], text)

        elif method == 'textDocument/didClose':
            td = params['textDocument']
            uri = td['uri']
            self.documents.pop(uri, None)
            self.ast_cache.pop(uri, None)
            self.symbols.pop(uri, None)
            self._publish_diagnostics(uri, [])

        elif method == 'textDocument/hover':
            td = params['textDocument']
            result = self.handle_hover(td['uri'], params['position'])
            send_response(req_id, result)

        elif method == 'textDocument/definition':
            td = params['textDocument']
            result = self.handle_goto_definition(td['uri'], params['position'])
            send_response(req_id, result)

        elif method == 'textDocument/completion':
            td = params['textDocument']
            result = self.handle_completion(td['uri'], params['position'])
            send_response(req_id, result)

        elif method == 'textDocument/documentSymbol':
            td = params['textDocument']
            uri = td['uri']
            result = self.symbols.get(uri, [])
            send_response(req_id, result)

        elif req_id is not None:
            # Unknown request
            send_error(req_id, -32601, f"Method not found: {method}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    server = NexaLSP()
    while server.running:
        try:
            msg = read_message()
            if msg is None:
                break
            server.handle(msg)
        except Exception as e:
            # Log to stderr (VS Code reads it)
            sys.stderr.write(f"LSP Error: {e}\n")
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()

if __name__ == '__main__':
    main()
