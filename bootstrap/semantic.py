from lexer import Lexer
from n_parser import Parser, FunctionDef, StructDef, EnumDef, ImplDef, TraitDef, MatchExpr, CaseArm, ArrayLiteral, IndexAccess, UnaryExpr, VariableExpr, IfStmt, WhileStmt, ForStmt, VarDecl, Assignment, CallExpr, MemberAccess, MethodCall, ReturnStmt, BinaryExpr, RegionStmt, FloatLiteral, CharLiteral, IntegerLiteral, BreakStmt, ContinueStmt, UseStmt, TypeAlias
from errors import CompilerError
import sys
import copy

class SemanticAnalyzer:
    def __init__(self):
        # Symbol table entries: {name: {'type': type, 'moved': bool, 'readers': int, 'writer': bool, 'is_ref': bool}}
        self.scopes = [{}] 
        self.warnings = [] # list of (msg, line, col)
        self.borrow_cleanup_stack = [] # Stack of lists of (borrowed_var_name, is_mut)
        self.current_function = None
        self.struct_methods = {}  # {struct_name: {method_name: FunctionDef}}
        self.impls = set() # {(struct_name, trait_name)}
        self.aliases = {} # {alias_name: full_qualified_name}
        self.loop_stack = [] # list of labels (None if no label)
        self.functions = set(['print', 'gpu::global_id', 'gpu::dispatch', 'panic', 'assert', 'slice_from_array', 'fs::read_file', 'fs::write_file', 'fs::append_file', '__nexa_panic', '__nexa_assert'])
        self.function_defs = {} # name -> list of FunctionDef
        self.structs = {} # name -> {field: type}
        self.struct_defs = {} # name -> StructDef (for privacy check)
        self.struct_used = {} # name -> bool
        self.enums = {} # name -> {variant: [payload_types]}
        self.enum_defs = {} # name -> EnumDef (for privacy check)
        self.enum_used = {} # name -> bool
        self.generic_functions = {} # name -> node
        self.generic_structs = {} # name -> node
        self.generic_enums = {} # name -> node
        self.traits = {} # name -> {method_name: method_signature_node}
        self.trait_defs = {} # name -> TraitDef
        self.current_module = ""
        self.lambda_count = 0
        self.tests = [] # list of test function names
        self.lambda_base_scopes = [] # Stack of len(self.scopes) when lambda started
        self.lambda_capture_stack = [] # Stack of dicts: {name: type}
        self.lambda_count = 0


    def get_suggestion(self, name, possibilities):
        import difflib
        matches = difflib.get_close_matches(name, possibilities, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def error(self, message, node=None, hint=None, error_code=None):
        line = node.line if node else None
        column = node.column if node else None
        raise CompilerError(message, line, column, hint=hint, error_code=error_code)

    def resolve_type_name(self, name):
        if not name: return name
        # Normalize: remove spaces
        name = name.replace(' ', '')
        
        # Mark as used if it's a known struct/enum
        base_name = name.split('<')[0] if '<' in name else name
        if base_name in self.struct_used: self.struct_used[base_name] = True
        if base_name in self.enum_used: self.enum_used[base_name] = True
        
        # 1. Alias Check (Direct or Recursive)
        if name in self.aliases:
            return self.resolve_type_name(self.aliases[name])
            
        # 2. Generics Check
        if '<' in name and name.endswith('>'):
            try:
                base = name.split('<', 1)[0]
                resolved_base = self.resolve_type_name(base)
                
                inside = name[name.find('<')+1:-1]
                # Split by comma but respect nested brackets
                args = []
                current = ""
                depth = 0
                for char in inside:
                    if char == '<': depth += 1
                    elif char == '>': depth -= 1
                    elif char == ',' and depth == 0:
                        args.append(current.strip())
                        current = ""
                        continue
                    current += char
                args.append(current.strip())
                
                resolved_args = [self.resolve_type_name(a) for a in args]
                return f"{resolved_base}<{','.join(resolved_args)}>"
            except:
                return name # Fallback if parsing fails
        
        # 3. Canonicalize :: -> _ (with Alias resolution)
        if '::' in name:
             parts = name.split('::')
             # Start resolving from the root
             if parts[0] in self.aliases:
                 resolved_root = self.resolve_type_name(parts[0])
                 # If the resolved root already contains underscores (mangled), proceed
                 parts[0] = resolved_root
             
             return "_".join(parts)
            
        # 4. Local Module Lookup
        if self.current_module:
             # Try prefixing with current module (mangled)
             local_name = f"{self.current_module.replace('::', '_')}_{name}"
             
             # Check if this local name exists in known types (Enums, Structs)
             if local_name in self.structs or local_name in self.enums or local_name in self.generic_structs or local_name in self.generic_enums:
                  return local_name
        
        return name

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def suggest_name(self, name, candidates):
        best_match = None
        min_dist = float('inf')
        for candidate in candidates:
            dist = self.levenshtein_distance(name, candidate)
            if dist < min_dist:
                min_dist = dist
                best_match = candidate
        
        limit = max(3, len(name) // 3)
        if min_dist <= limit and min_dist < len(name): 
             return best_match
        return None

    def is_copy_type(self, type_name: str) -> bool:
        if not type_name: return False
        if type_name in ('i32', 'i64', 'u64', 'u8', 'bool', 'f32', 'string', 'char'):
            return True
        if type_name.endswith('*'):
            return True
        return False

    def move_var(self, name: str, node=None):
        var_info = self.lookup(name)
        if not var_info:
            self.error(f"Semantic Error: Move of undefined variable '{name}'", node)

        if var_info.get('moved'):
            self.error(f"Ownership Error: Use of moved variable '{name}'", node)

        if var_info.get('readers', 0) > 0:
            self.error(f"Borrow Error: Cannot move '{name}' because it is borrowed", node)
        if var_info.get('writer'):
            self.error(f"Borrow Error: Cannot move '{name}' because it is borrowed mutable", node)

        var_info['moved'] = True

    def check_privacy(self, target_node, target_name):
        target_mod = getattr(target_node, 'module', "")
        if not target_mod: return

        current_mod = ""
        if self.current_function:
            current_mod = getattr(self.current_function, 'module', "")
        
        if current_mod == target_mod: return
        
        if not getattr(target_node, 'is_pub', False):
            raise Exception(f"Privacy Error: '{target_name}' is private to module '{target_mod}'")

    def get_mangled_name(self, name, params):
        if name in ('print', 'panic', 'assert', 'malloc', 'free', 'realloc', 'memcpy') or name.startswith('gpu::') or name.startswith('fs::'):
            return name
        
        # Don't mangle if already mangled
        if "__args__" in name:
            return name

        param_types = []
        for i, (pn, pt) in enumerate(params):
            # Normalize self types for consistent mangling
            if i == 0 and pn == 'self':
                # Methods always mangle as having a pointer to the struct
                # except if we ever support by-value self, but for now...
                clean_t = "SELF_PTR"
            else:
                clean_t = pt.replace('*', '_ptr').replace('<', '_L_').replace('>', '_R_').replace('&', '_ref_').replace(' ', '_').replace(',', '_')
            param_types.append(clean_t)
        
        res = name
        if param_types:
            res = f"{name}__args__{'__'.join(param_types)}"
            
        return res

    def resolve_overload(self, base_name, arg_types, node):
        candidates = []
        if base_name in self.function_defs:
            candidates = self.function_defs[base_name]
        
        if not candidates:
            # Try suggestion
            possibilities = list(self.functions) + list(self.structs.keys())
            suggestion = self.get_suggestion(base_name, possibilities)
            hint = f"did you mean '{suggestion}'?" if suggestion else None
            self.error(f"Unknown function or struct: '{base_name}'", node, hint=hint, error_code="E0004")

        # 1. Exact match (strongest)
        for cand in candidates:
            # Handle vararg
            if getattr(cand, 'is_vararg', False):
                if len(arg_types) < len(cand.params): continue
            elif len(cand.params) != len(arg_types): 
                continue
                
            match = True
            for i, atype in enumerate(arg_types):
                if i < len(cand.params):
                    ptype = cand.params[i][1]
                if ptype != atype:
                    match = False; break
                else:
                    # Vararg part
                    break
            if match:
                if getattr(cand, 'is_extern', False):
                    mangled = base_name # Don't mangle extern functions
                else:
                    mangled = self.get_mangled_name(base_name, cand.params)
                cand.mangled_name = mangled
                return cand, mangled

        # 2. Match with compatibility/coercion
        for cand in candidates:
            if getattr(cand, 'is_vararg', False):
                if len(arg_types) < len(cand.params): continue
            elif len(cand.params) != len(arg_types): 
                continue
                
            match = True
            for i, atype in enumerate(arg_types):
                if i < len(cand.params):
                    ptype = cand.params[i][1]
                if not self.check_type_compatibility(ptype, atype, node):
                    match = False; break
                else:
                    # Vararg part - assume compatible for now in bootstrap
                    break
            if match:
                if getattr(cand, 'is_extern', False):
                    mangled = base_name # Don't mangle extern functions
                else:
                    mangled = self.get_mangled_name(base_name, cand.params)
                cand.mangled_name = mangled
                return cand, mangled

        # Error if no match
        self.error(f"No overload of '{base_name}' matches arguments: ({', '.join(arg_types)})", node)

    def generate_derive(self, node, trait):
        from n_parser import ImplDef, FunctionDef, CallExpr, StringLiteral, MemberAccess, VariableExpr, ReturnStmt, IntegerLiteral, MethodCall
        if trait == 'Debug':
            # Generate Debug for Struct
            if isinstance(node, StructDef):
                # fn debug_print(&self)
                body = []
                body.append(CallExpr("print", [StringLiteral(f"{node.name} {{ ")]))
                for i, (fname, ftype) in enumerate(node.fields):
                    body.append(CallExpr("print", [StringLiteral(f"{fname}: ")]))
                    # print(self.field) - only works if field is primitive!
                    # For now, let's assume primitives
                    body.append(CallExpr("print", [MemberAccess(VariableExpr("self"), fname)]))
                    if i < len(node.fields) - 1:
                        body.append(CallExpr("print", [StringLiteral(", ")]))
                body.append(CallExpr("print", [StringLiteral(" }\n")]))
                # No explicit return needed for void functions if we handle it in codegen
                # or use ReturnStmt(None)
                
                method = FunctionDef("debug_print", [("self", f"&{node.name}")], "void", body)
                return [ImplDef(node.name, [method])]
        
        elif trait == 'Clone':
            if isinstance(node, StructDef):
                # fn clone(&self) -> Self
                args = []
                for fname, ftype in node.fields:
                    # Recursive clone for each field: self.field.clone()
                    field_access = MemberAccess(VariableExpr("self"), fname)
                    # For primitives we could just use them directly, 
                    # but calling .clone() is safer if we have overloaded it.
                    # Simplified: if it's primitive i32/f32/u8/bool just copy
                    if ftype in ('i32', 'f32', 'u8', 'bool', 'i64'):
                        args.append(field_access)
                    else:
                        args.append(MethodCall(field_access, "clone", []))
                
                body = [ReturnStmt(CallExpr(node.name, args))]
                method = FunctionDef("clone", [("self", f"&{node.name}")], node.name, body)
                return [ImplDef(node.name, [method])]
                
        return []

    def visit_AwaitExpr(self, node):
        if not self.current_function or not self.current_function.is_async:
            self.error("Await is only allowed inside async functions", node)
        
        inner_t = self.visit(node.value)
        # In a real implementation, Await expects a Future<T> and returns T.
        # For this bootstrap, if it's Task<T>, return T.
        if inner_t.startswith('Task<') and inner_t.endswith('>'):
            res_t = inner_t[5:-1]
            node.type_name = res_t
            return res_t
            
        node.type_name = inner_t
        return inner_t

    def visit_MacroCallExpr(self, node):
        import os
        from n_parser import StringLiteral, IntegerLiteral
        
        if node.name == 'include_str':
            if len(node.args) != 1: self.error("include_str! expects 1 argument", node)
            path_node = node.args[0]
            if not isinstance(path_node, StringLiteral): self.error("include_str! expects string literal", node)
            
            # Find relative to source
            base_dir = self.current_dir if hasattr(self, 'current_dir') else "."
            full_path = os.path.join(base_dir, path_node.value)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                node.expanded = StringLiteral(content)
                return 'string'
            except Exception as e:
                self.error(f"Could not read file '{full_path}': {e}", node)
                
        elif node.name == 'env':
            if len(node.args) != 1: self.error("env! expects 1 argument", node)
            var_node = node.args[0]
            if not isinstance(var_node, StringLiteral): self.error("env! expects string literal", node)
            
            val = os.environ.get(var_node.value, "")
            node.expanded = StringLiteral(val)
            return 'string'

        elif node.name == 'file':
            node.expanded = StringLiteral(getattr(self, 'current_file_path', 'unknown'))
            return 'string'

        elif node.name == 'line':
            node.expanded = IntegerLiteral(node.line or 0)
            return 'i32'

        elif node.name == 'column':
            node.expanded = IntegerLiteral(node.column or 0)
            return 'i32'

        elif node.name == 'panic':
            if len(node.args) != 1: self.error("panic! expects 1 argument", node)
            from n_parser import CallExpr, StringLiteral, IntegerLiteral, BlockStmt
            msg = node.args[0]
            
            # Expand to: __nexa_panic(msg, file, line)
            expansion = CallExpr("__nexa_panic", [msg, StringLiteral(getattr(self, 'current_file_path', 'unknown')), IntegerLiteral(node.line or 0)])
            node.expanded = expansion
            return self.visit(expansion)

        elif node.name == 'assert':
            if len(node.args) != 2: self.error("assert! expects 2 arguments: condition, message", node)
            from n_parser import IfStmt, CallExpr, StringLiteral, IntegerLiteral, BlockStmt
            cond = node.args[0]
            msg = node.args[1]
            
            # Expand to: __nexa_assert(cond, msg, file, line)
            expansion = CallExpr("__nexa_assert", [cond, msg, StringLiteral(getattr(self, 'current_file_path', 'unknown')), IntegerLiteral(node.line or 0)])
            node.expanded = expansion
            return self.visit(expansion)
            
        else:
            self.error(f"Unknown macro: {node.name}!", node)

    def analyze(self, ast):
        self.ast_root = ast
        
        # Register built-ins in function_defs for overload resolution
        for bf in self.functions:
            if bf not in self.function_defs:
                # Create a list with a dummy FunctionDef for built-ins
                self.function_defs[bf] = [FunctionDef(bf, [], 'void', None)]

        # 0. Process Derives
        new_nodes = []
        for node in ast:
            if isinstance(node, (StructDef, EnumDef)):
                for attr_name, attr_args in getattr(node, 'attrs', []):
                    if attr_name == 'derive':
                        for trait in attr_args:
                            generated = self.generate_derive(node, trait)
                            if generated:
                                new_nodes.extend(generated)
        ast.extend(new_nodes)
        
        # Pass 1: Collect Types (Structs, Enums, Traits)
        for node in ast:
            name = type(node).__name__
            if name == 'StructDef':
                 if node.generics:
                     self.generic_structs[node.name] = node
                 else:
                     fields = {name: type_ for name, type_ in node.fields}
                     self.structs[node.name] = fields
                     self.struct_defs[node.name] = node
                     self.struct_used[node.name] = getattr(node, 'is_pub', False) or node.name.startswith('std_')
            elif name == 'EnumDef':
                 if node.generics:
                     self.generic_enums[node.name] = node
                 else:
                     variants = {vname: payloads for vname, payloads in node.variants}
                     self.enums[node.name] = variants
                     self.enum_defs[node.name] = node
                     self.enum_used[node.name] = getattr(node, 'is_pub', False) or node.name.startswith('std_')
            elif name == 'TraitDef':
                 self.trait_defs[node.name] = node
                 methods = {m.name: m for m in node.methods}
                 self.traits[node.name] = {'methods': methods, 'types': node.associated_types}
            elif name == 'TypeAlias':
                 self.aliases[node.alias] = node.original_type

        # Pass 2: Collect Functions and Impls
        for node in ast:
            name = type(node).__name__
            if name == 'FunctionDef':
                if node.generics:
                    self.generic_functions[node.name] = node
                else:
                    self.functions.add(node.name)
                    if node.name not in self.function_defs:
                        self.function_defs[node.name] = []
                    self.function_defs[node.name].append(node)
            elif name == 'ExternBlock':
                 for func in node.functions:
                      func.module = getattr(node, 'module', '')
                      func.is_extern = True # Mark as extern
                      self.canonicalize_type_refs(func)
                      if func.name not in self.function_defs:
                          self.function_defs[func.name] = []
                      self.function_defs[func.name].append(func)
                      self.functions.add(func.name)
            elif name == 'ImplDef':
                 self.register_impl_methods(node)
        
        # Inject Built-ins
        if 'Slice' not in self.generic_structs and 'Slice' not in self.structs:
            self.generic_structs['Slice'] = StructDef('Slice', [('ptr', 'T*'), ('len', 'i32')], generics=[('T', None, False)])
        if 'Task' not in self.generic_structs and 'Task' not in self.structs:
            self.generic_structs['Task'] = StructDef('Task', [('coro_handle', 'u8*'), ('done', 'bool')], generics=[('T', None, False)])
        if 'Buffer' not in self.generic_structs and 'Buffer' not in self.structs:
            self.generic_structs['Buffer'] = StructDef('Buffer', [('ptr', 'T*'), ('len', 'i32')], generics=[('T', None, False)])
        self.structs['Arena'] = {'chunk': 'u8*', 'offset': 'i32', 'capacity': 'i32'}

        # 1.5 Process Imports (Use Statements)
        for node in ast:
             if type(node).__name__ == "UseStmt":
                  self.visit_UseStmt(node)

        # 1.6 Canonicalize Types in Definitions (Module Scoping)
        for node in ast:
             self.canonicalize_type_refs(node)

        # 2. Analyze bodies
        for node in ast:
            if type(node).__name__ == "UseStmt": continue
            if isinstance(node, FunctionDef) and getattr(node, 'is_lambda', False): continue
            self.visit(node)
        
        # 3. Dead Code Analysis
        for name, funcs in self.function_defs.items():
             if name == 'main' or name.endswith('::main'): continue
             
             # Safety: ensure funcs is a list
             if not isinstance(funcs, list):
                 funcs = [funcs]
                 
             for func in funcs:
                  if not func.used and not func.is_pub and not name.startswith('std_'):
                       # Check if it's a method implementation or trait method (simplification: skip methods for now or refine)
                       if '::' in name: continue # Skip methods for now to avoid noise
                       if self.current_module == 'std': continue # Skip std lib internal
                       
                       # Find original line/col? FuncDef has it.
                       self.warnings.append((f"Dead code: Function '{name}' is never used", func.line, func.column))

        for name, used in self.struct_used.items():
             if not used and not name.startswith('std_'):
                  node = self.struct_defs[name]
                  self.warnings.append((f"Dead code: Struct '{name}' is never used", node.line, node.column))

        for name, used in self.enum_used.items():
             if not used and not name.startswith('std_'):
                  node = self.enum_defs[name]
                  self.warnings.append((f"Dead code: Enum '{name}' is never used", node.line, node.column))

    def canonicalize_type_refs(self, node):
        prefix = getattr(node, 'module', '')
        self.current_module = prefix # Set context for resolution
        name = type(node).__name__
        if name == 'FunctionDef':
             node.return_type = self.resolve_type_name(self.mangle_type_if_local(node.return_type, prefix))
             for i, (pname, ptype) in enumerate(node.params):
                  node.params[i] = (pname, self.resolve_type_name(self.mangle_type_if_local(ptype, prefix)))
        elif name == 'StructDef':
             node.fields = [(n, self.resolve_type_name(self.mangle_type_if_local(t, prefix))) for n, t in node.fields]
             self.structs[node.name] = {n: self.resolve_type_name(self.mangle_type_if_local(t, prefix)) for n, t in node.fields}
        elif name == 'TypeAlias':
             node.original_type = self.resolve_type_name(self.mangle_type_if_local(node.original_type, prefix))
             if prefix:
                 self.aliases[f"{prefix}_{node.alias}"] = node.original_type
             else:
                 self.aliases[node.alias] = node.original_type
        elif name == 'ExternBlock':
             for func in node.functions:
                  func.module = prefix
                  self.canonicalize_type_refs(func)
                  if func.name not in self.function_defs:
                      self.function_defs[func.name] = []
                  self.function_defs[func.name].append(func)
                  self.functions.add(func.name)
        elif name == 'ImplDef':
             for method in node.methods:
                  method.return_type = self.resolve_type_name(self.mangle_type_if_local(method.return_type, prefix))
                  for i, (pname, ptype) in enumerate(method.params):
                       method.params[i] = (pname, self.resolve_type_name(self.mangle_type_if_local(ptype, prefix)))

    def visit_TypeAlias(self, node):
        pass

    def mangle_type_if_local(self, type_name, prefix):
        if not type_name: return type_name
        if '<' in type_name and type_name.endswith('>'):
             base = type_name.split('<', 1)[0]
             base = self.mangle_type_if_local(base, prefix)
             inside = type_name[type_name.find('<')+1:-1]
             args = [self.mangle_type_if_local(x.strip(), prefix) for x in inside.split(',')]
             return f"{base}<{','.join(args)}>"
             
        mangled = f"{prefix.replace('::', '_')}_{type_name}"
        if mangled in self.structs or mangled in self.enums:
             return mangled
        return type_name

    def is_deeply_concrete(self, t):
        if not t: return True
        if '<' in t:
             base = t.split('<')[0]
             inside = t[t.find('<')+1:-1]
             args = self.split_generic_args(inside)
             return self.is_deeply_concrete(base) and all(self.is_deeply_concrete(a) for a in args)
        if len(t) == 1 and t.isupper(): return False
        if t in ('T', 'U', 'V', 'E', 'K', 'Self'): return False
        return True

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def visit_ExternBlock(self, node): pass
    def visit_TraitDef(self, node): pass
    def visit_StructDef(self, node): pass
    def visit_EnumDef(self, node): pass
    def visit_ImplDef(self, node):
        prev_mod = self.current_module
        if getattr(node, 'module', None):
             self.current_module = node.module
             
        struct_name = node.struct_name
        if struct_name == 'std_hash_i32': struct_name = 'i32'
        if struct_name == 'std_hash_string': struct_name = 'string'
             
        # Resolve associated types and Self inside this impl
        mapping = {"Self": struct_name}
        if hasattr(node, 'associated_types'):
            for assoc_name, assoc_type in node.associated_types.items():
                 mapping[f"Self_{assoc_name}"] = assoc_type
             
        for method in node.methods:
             # Handle self parameters BEFORE submap/mangling
             if method.params:
                  pname, ptype = method.params[0]
                  if pname == 'self':
                       if ptype == 'Self': method.params[0] = ('self', struct_name)
                       elif ptype == '&Self': method.params[0] = ('self', f"{struct_name}*")
                       elif ptype == '&mut Self': method.params[0] = ('self', f"{struct_name}*")

             # Update signature (other params)
             method.return_type = self.apply_submap(method.return_type, mapping)
             new_params = []
             for i, (pn, pt) in enumerate(method.params):
                  # Skip self for submap as we handled it
                  if i == 0 and pn == 'self':
                       new_params.append((pn, pt))
                  else:
                       new_params.append((pn, self.apply_submap(pt, mapping)))
             method.params = new_params
             
             # Update body
             self.substitute_generics(method.body, mapping)
             
             self.visit(method)
        
        self.current_module = prev_mod

    def visit_UseStmt(self, node):
        full_name = "::".join(node.path)
        if node.is_glob:
            prefix = full_name
            for item in self.ast_root:
                # We use name-based matching for glob imports
                # instead of relying on a potentially missing 'module' attribute.
                # All mangled names are 'prefix_itemname'
                mangled_prefix = prefix.replace('::', '_') + "_"
                if getattr(item, 'name', '').startswith(mangled_prefix):
                    if getattr(item, 'is_pub', False):
                        full_mangled_name = item.name
                        short_name = full_mangled_name[len(mangled_prefix):]
                        self.aliases[short_name] = full_mangled_name
        else:
            alias = node.path[-1]
            self.aliases[alias] = full_name

    def register_impl_methods(self, node):
        struct_name = node.struct_name
        if struct_name == 'std_hash_i32': struct_name = 'i32'
        if struct_name == 'std_hash_string': struct_name = 'string'
        
        if struct_name not in self.struct_methods:
            self.struct_methods[struct_name] = {}
            
        if node.trait_name:
             if node.trait_name not in self.traits:
                  self.error(f"Semantic Error: Unknown trait '{node.trait_name}'", node)
             trait_info = self.traits[node.trait_name]
             trait_methods = trait_info['methods']
             expected_types = trait_info['types']
             
             impl_method_names = set(m.name for m in node.methods)
             
             missing = []
             for m_name, m_node in trait_methods.items():
                  if m_name not in impl_method_names:
                       if m_node.body is not None:
                            new_method = copy.deepcopy(m_node)
                            node.methods.append(new_method)
                       else:
                            missing.append(m_name)

             if missing:
                  self.error(f"Semantic Error: Implementation of trait '{node.trait_name}' for '{struct_name}' is missing methods: {', '.join(missing)}", node)
                  
             # Check associated types
             for assoc_type in expected_types:
                  if assoc_type not in node.associated_types:
                       self.error(f"Semantic Error: Implementation of trait '{node.trait_name}' for '{struct_name}' is missing associated type '{assoc_type}'", node)
             self.impls.add((struct_name, node.trait_name))
             
        # Resolve associated types and Self in signatures
        mapping = {"Self": struct_name}
        if hasattr(node, 'associated_types'):
            for assoc_name, assoc_type in node.associated_types.items():
                 mapping[f"Self_{assoc_name}"] = assoc_type

        for method in node.methods:
             # Update signature
             method.return_type = self.apply_submap(method.return_type, mapping)
             new_params = []
             for pn, pt in method.params:
                  new_params.append((pn, self.apply_submap(pt, mapping)))
             method.params = new_params
             
             if method.params:
                  pname, ptype = method.params[0]
                  if pname == 'self':
                       if ptype == 'Self': method.params[0] = ('self', struct_name)
                       elif ptype in ('&Self', '&mut Self'): method.params[0] = ('self', f"{struct_name}*")
             
             if method.name not in self.struct_methods[struct_name]:
                 self.struct_methods[struct_name][method.name] = []
             self.struct_methods[struct_name][method.name].append(method)
             
             # Also register in global function defs for direct calls (Type_method)
             mangled_base = f"{struct_name}_{method.name}"
             method.name = mangled_base # Update method name to mangled base
             if mangled_base not in self.function_defs:
                 self.function_defs[mangled_base] = []
             self.function_defs[mangled_base].append(method)
             self.functions.add(mangled_base)

    def check_trait_impl(self, type_name, trait_name):
        if (type_name, trait_name) in self.impls: return True
        if '<' in type_name:
             base = type_name.split('<')[0]
             if (base, trait_name) in self.impls: return True
        return False

    def visit_MemberAccess(self, node):
        obj_type = self.visit(node.object)
        base_type = obj_type.lstrip('&').replace('mut', '').rstrip('*')
        
        # Try full type first (monomorphized)
        if base_type in self.structs:
             fields_dict = self.structs[base_type]
             lookup_type = base_type
        else:
             base = base_type.split('<')[0] if '<' in base_type else base_type
             lookup_type = self.resolve_type_name(base)
             
             if lookup_type in self.structs: fields_dict = self.structs[lookup_type]
             elif lookup_type in self.generic_structs: fields_dict = {f[0]: f[1] for f in self.generic_structs[lookup_type].fields}
             else: raise Exception(f"Type Error: access member '{node.member}' on non-struct type '{obj_type}' (lookup={lookup_type})")
        
        if node.member not in fields_dict:
             suggestion = self.get_suggestion(node.member, list(fields_dict.keys()))
             hint = f"did you mean '.{suggestion}'?" if suggestion else None
             self.error(f"Type Error: Struct '{lookup_type}' has no member '{node.member}'", node, hint=hint, error_code="E0006")
        
        node.struct_type = lookup_type
        node.type_name = fields_dict[node.member]
        return node.type_name

    def visit_MethodCall(self, node):
        receiver_type = self.visit(node.receiver)
        base_type = receiver_type.lstrip('&').replace('mut ', '').rstrip('*')
        lookup_type = base_type.split('<')[0] if '<' in base_type else base_type
            
        if lookup_type not in self.struct_methods:
            raise Exception(f"Semantic Error: Type '{base_type}' has no methods")
        
        methods = self.struct_methods[lookup_type]
        if node.method_name not in methods:
            suggestion = self.get_suggestion(node.method_name, list(methods.keys()))
            hint = f"did you mean '.{suggestion}()'?" if suggestion else None
            self.error(f"Method '{node.method_name}' not found on type '{base_type}'", node, hint=hint, error_code="E0005")
        
        arg_types = [self.visit(arg) for arg in node.args]
        
        # Overload resolution for methods
        candidates = methods[node.method_name]
        best_cand = None
        
        # 1. Exact match
        for cand in candidates:
            method_params = cand.params[1:] if cand.params and cand.params[0][0] == 'self' else cand.params
            if len(method_params) != len(arg_types): continue
            match = True
            for (pname, ptype), atype in zip(method_params, arg_types):
                if ptype != atype:
                    match = False; break
            if match:
                best_cand = cand; break
        
        # 2. Match with compatibility/coercion
        if not best_cand:
            for cand in candidates:
                method_params = cand.params[1:] if cand.params and cand.params[0][0] == 'self' else cand.params
                if len(method_params) != len(arg_types): continue
                match = True
                for (pname, ptype), atype in zip(method_params, arg_types):
                    if not self.check_type_compatibility(ptype, atype, node):
                        match = False; break
                if match:
                    best_cand = cand; break
        
        if not best_cand:
            self.error(f"No overload of method '{node.method_name}' on type '{base_type}' matches arguments: ({', '.join(arg_types)})", node)

        node.struct_type = lookup_type
        node.receiver_type = receiver_type
        node.method_def = best_cand # Store for codegen
        best_cand.used = True
        
        # Mangle name
        node.method_name = self.get_mangled_name(f"{lookup_type}_{node.method_name}", best_cand.params)
        
        ret_type = best_cand.return_type
        if '<' in receiver_type:
             struct_def = None
             if lookup_type in self.generic_structs: 
                  struct_def = self.generic_structs[lookup_type]
             elif lookup_type in self.generic_enums:
                  struct_def = self.generic_enums[lookup_type]
             
             if struct_def:
                 args_str = receiver_type[receiver_type.find('<')+1:-1]
                 args = self.split_generic_args(args_str)
                 
                 mapping = dict(zip([g[0] for g in struct_def.generics], args))
                 ret_type = self.apply_submap(ret_type, mapping)
             
        node.return_type = ret_type
        return ret_type

    def visit_ArrayLiteral(self, node):
        if not node.elements: raise Exception("Semantic Error: Empty array literals not supported")
        first_type = self.visit(node.elements[0])
        for el in node.elements[1:]:
            if self.visit(el) != first_type:
                 raise Exception(f"Type Error: Array elements must be same type")
        return f"[{first_type}:{len(node.elements)}]"

    def visit_IndexAccess(self, node):
        obj_type = self.visit(node.object)
        if self.visit(node.index) != 'i32': raise Exception("Type Error: Array index must be i32")
        if obj_type.startswith('[') and obj_type.endswith(']'):
            elem_type = obj_type[1:-1].split(':', 1)[0]
            node.type_name = elem_type
            return elem_type
        if obj_type.endswith('*'):
            ty = obj_type[:-1]
            node.type_name = ty
            return ty
        if obj_type.startswith("Slice<") and obj_type.endswith(">"):
            inner = obj_type[len("Slice<"):-1]
            node.type_name = inner
            return inner
        raise Exception(f"Type Error: Indexing non-array type '{obj_type}'")

    def visit_UnaryExpr(self, node):
        t = self.visit(node.operand)
        if node.op == '*':
            if t.endswith('*'):
                node.type_name = t[:-1]
                return node.type_name
            elif t.startswith('&'):
                if t.startswith('&mut '):
                    node.type_name = t[len('&mut '):]
                else:
                    node.type_name = t[1:]
                return node.type_name
            else:
                raise Exception(f"Type Error: Cannot dereference non-pointer. Type: {t}")
        elif node.op in ('&', '&mut'):
            if isinstance(node.operand, VariableExpr):
                var = self.lookup(node.operand.name)
                if not var: raise Exception(f"Undefined var '{node.operand.name}'")
                if node.op == '&mut':
                    if var['writer'] or var['readers'] > 0: raise Exception("Borrow Error")
                    var['writer'] = True
                else:
                    if var['writer']: raise Exception("Borrow Error")
                    var['readers'] += 1
                if 'active_borrows' not in self.scopes[-1]: self.scopes[-1]['active_borrows'] = []
                self.scopes[-1]['active_borrows'].append((node.operand.name, 'writer' if node.op == '&mut' else 'reader'))
            node.type_name = f"{t}*"
            return node.type_name
        elif node.op == '!':
            if t != 'bool':
                raise Exception(f"Type Error: Logical NOT requires bool, got {t}")
            node.type_name = 'bool'
            return 'bool'
        elif node.op == '-':
            if t not in ('i32', 'i64', 'f32'):
                raise Exception(f"Type Error: Negation requires numeric type, got {t}")
            node.type_name = t
            return t
        return t

    def visit_MatchExpr(self, node):
        expr_type = self.visit(node.value)
        node.enum_name = expr_type # Set for codegen
        if expr_type and '<' in expr_type: self.instantiate_generic_type(expr_type)
        
        base = expr_type.split('<')[0] if '<' in expr_type else expr_type
        
        if expr_type in self.enums:
             variants = self.enums[expr_type]
        elif base in self.generic_enums:
             # Generic Enum Match (e.g. inside impl<T> Result<T>)
             enum_def = self.generic_enums[base]
             variants = {vname: payloads for vname, payloads in enum_def.variants}
        else:
             raise Exception(f"Match expression must be an Enum. Type: {expr_type}")
             
        covered = set()
        for case in node.cases:
            if case.variant_name not in variants: raise Exception(f"Enum has no variant '{case.variant_name}'")
            covered.add(case.variant_name)
            self.enter_scope()
            if case.var_names:
                payloads = variants[case.variant_name]
                if len(case.var_names) != len(payloads): raise Exception("Payload mismatch")
                for i, vname in enumerate(case.var_names): self.declare_variable(vname, payloads[i])
            self.visit(case.body)
            self.exit_scope()
        if len(covered) != len(variants):
             raise Exception("Match not exhaustive")

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method in SemanticAnalyzer")

    def instantiate_generic_type(self, name):
        if not name: return
        if name.endswith('*'): return self.instantiate_generic_type(name[:-1])
        if name.startswith('[') and name.endswith(']'): return self.instantiate_generic_type(name[1:-1].split(':', 1)[0].strip())
        if '<' not in name: return
        base = name.split('<', 1)[0]
        if base in self.generic_structs: self.instantiate_generic_struct(name)
        elif base in self.generic_enums: self.instantiate_generic_enum(name)

    def instantiate_generic_struct(self, name):
        if name in self.structs: return
        base_name, rest = name.split('<', 1)
        args = [a.strip() for a in self.split_generic_args(rest[:-1])]
        def_node = self.generic_structs[base_name]
        for (gn, gb, is_const), at in zip(def_node.generics, args):
             if is_const:
                  if not at.isdigit(): raise Exception(f"Const generic argument '{at}' must be integer literal")
             elif gb and not self.check_trait_impl(at, gb): raise Exception(f"Bounds check failed for {at}")
        mapping = dict(zip([g[0] for g in def_node.generics], args))
        
        prev_mod = self.current_module
        if getattr(def_node, 'module', None): self.current_module = def_node.module
        new_fields = [(n, self.resolve_type_name(self.apply_submap(t, mapping))) for n, t in def_node.fields]
        for _, fty in new_fields: self.instantiate_generic_type(fty)
        self.current_module = prev_mod

        # Only append to AST if args are concrete
        is_concrete = all(self.is_deeply_concrete(arg) for arg in args)
        if is_concrete:
             self.structs[name] = dict(new_fields)
             self.ast_root.append(StructDef(name, new_fields))

    def instantiate_generic_enum(self, name):
        if name in self.enums: return
        base_name, rest = name.split('<', 1)
        args = [a.strip() for a in self.split_generic_args(rest[:-1])]
        def_node = self.generic_enums[base_name]
        for (gn, gb, is_const), at in zip(def_node.generics, args):
             if is_const:
                  if not at.isdigit(): raise Exception(f"Const generic argument '{at}' must be integer literal")
             elif gb and not self.check_trait_impl(at, gb): raise Exception(f"Bounds check failed for {at}")
        mapping = dict(zip([g[0] for g in def_node.generics], args))
        
        prev_mod = self.current_module
        if getattr(def_node, 'module', None): self.current_module = def_node.module
        new_variants = [(vn, [self.resolve_type_name(self.apply_submap(p, mapping)) for p in ps]) for vn, ps in def_node.variants]
        for _, ps in new_variants:
             for p in ps: self.instantiate_generic_type(p)
        self.current_module = prev_mod

        # Only append to AST if args are concrete
        is_concrete = all(self.is_deeply_concrete(arg) for arg in args)
        if is_concrete:
             self.enums[name] = {v: ps for v, ps in new_variants}
             self.ast_root.append(EnumDef(name, new_variants))

    def instantiate_generic_function(self, name):
        if name in self.functions: return
        base_name, rest = name.split('<', 1)
        args = [a.strip() for a in self.split_generic_args(rest[:-1])]
        if base_name not in self.generic_functions: return self.instantiate_generic_type(name)
        def_node = self.generic_functions[base_name]
        for (gn, gb, is_const), at in zip(def_node.generics, args):
             if is_const:
                  if not at.isdigit(): raise Exception(f"Const generic argument '{at}' must be integer literal")
             elif gb and not self.check_trait_impl(at, gb): raise Exception(f"Bounds check failed for {at}")
        mapping = dict(zip([g[0] for g in def_node.generics], args))
        
        prev_mod = self.current_module
        if getattr(def_node, 'module', None): self.current_module = def_node.module
        
        new_params = [(pn, self.resolve_type_name(self.apply_submap(pt, mapping))) for pn, pt in def_node.params]
        new_ret = self.resolve_type_name(self.apply_submap(def_node.return_type, mapping))
        
        for _, pty in new_params: self.instantiate_generic_type(pty)
        self.instantiate_generic_type(new_ret)
        
        import copy
        new_body = copy.deepcopy(def_node.body)
        self.substitute_generics(new_body, mapping)
        
        self.current_module = prev_mod
        
        # Only append to AST if args are concrete
        is_concrete = all(self.is_deeply_concrete(arg) for arg in args)
        if is_concrete:
             new_func = FunctionDef(name, new_params, new_ret, new_body, def_node.is_kernel)
             self.ast_root.append(new_func)
             self.functions.add(name)
             if name not in self.function_defs:
                 self.function_defs[name] = []
             self.function_defs[name].append(new_func)

    def visit_IfStmt(self, node):
        if self.visit(node.condition) != 'bool': raise Exception("If condition must be bool")
        self.enter_scope()
        for s in node.then_branch: self.visit(s)
        self.exit_scope()
        
        if node.else_branch:
             self.enter_scope()
             for s in node.else_branch: self.visit(s)
             self.exit_scope()

    def visit_ForStmt(self, node):
        self.enter_scope()
        self.loop_stack.append(node.label)
        
        if node.is_iterator:
            coll_type = self.visit(node.start_expr)
            node.iterator_type = coll_type
            
            if '<' in coll_type:
                base = coll_type.split('<')[0]
                args_str = coll_type[coll_type.find('<')+1:-1]
                args = self.split_generic_args(args_str)
            else:
                base = coll_type
                args = []
                
            methods = self.struct_methods.get(base, {})
            method = methods.get('next')
            if not method:
                 raise Exception(f"Type '{coll_type}' does not implement 'next()'")
                 
            ret_type = method.return_type
            
            real_ret = ret_type
            if args:
                struct_def = self.generic_structs.get(base)
                if struct_def:
                     mapping = {}
                     for (gname, bound, is_const), gval in zip(struct_def.generics, args):
                           mapping[gname] = gval
                     real_ret = self.apply_submap(ret_type, mapping)
            
            item_type = None
            if '<' in real_ret:
                 base_ret = real_ret.split('<')[0]
                 if base_ret in ('Option', 'std_option_Option'):
                      item_type = real_ret[real_ret.find('<')+1:-1]
            
            if not item_type:
                 raise Exception(f"Iterator next() must return Option<T>, got {real_ret}")
            
            node.item_type = item_type # Store for codegen
            
            self.declare_variable(node.var_name, item_type, node=node)
            for s in node.body: self.visit(s)
        else:
            if self.visit(node.start_expr) != 'i32' or self.visit(node.end_expr) != 'i32': 
                raise Exception("For loop range must be i32")
            self.declare_variable(node.var_name, 'i32', node=node)
            for s in node.body: self.visit(s)
            
        self.loop_stack.pop()
        self.exit_scope()

    def visit_WhileStmt(self, node):
        if self.visit(node.condition) != 'bool': raise Exception("While condition must be bool")
        self.loop_stack.append(node.label)
        self.enter_scope()
        for s in node.body: self.visit(s)
        self.loop_stack.pop()
        self.exit_scope()

    def visit_BreakStmt(self, node):
        if not self.loop_stack: raise Exception("Break outside loop")
        if node.label:
            if node.label not in self.loop_stack:
                raise Exception(f"Break to unknown label '{node.label}'")

    def visit_ContinueStmt(self, node):
        if not self.loop_stack: raise Exception("Continue outside loop")
        if node.label:
            if node.label not in self.loop_stack:
                raise Exception(f"Continue to unknown label '{node.label}'")

    def visit_FunctionDef(self, node):
        if node.generics:
            self.generic_functions[node.name] = node
            return
        
        prev_mod = self.current_module
        if getattr(node, 'module', None):
             self.current_module = node.module

        # Mark as test if attribute exists
        for attr_name, attr_args in getattr(node, 'attrs', []):
            if attr_name == 'test':
                self.tests.append(node.name)

        self.current_function = node
        self.enter_scope()
        for i, (pn, pt) in enumerate(node.params):
            pt = self.resolve_type_name(pt)
            node.params[i] = (pn, pt)
            if '<' in pt: self.instantiate_generic_type(pt)
            self.declare_variable(pn, pt, node=node)
        
        # Mangle name for overloading (except main and externs)
        if node.name != 'main' and not node.name.endswith('::main') and node.body is not None:
            node.name = self.get_mangled_name(node.name, node.params)
            self.functions.add(node.name) # Add mangled name to known functions
            
        # For async functions, the effective return type from the outside is a Future/Task
        # but internally we check against the declared return type.
        if node.is_async:
            # Inject Task<T> or similar if we want strict typing for caller
            # For now, let's just make sure it's valid.
            pass

        for s in node.body: self.visit(s)
        self.exit_scope()
        self.current_function = None
        self.current_module = prev_mod

    def check_type_compatibility(self, expected, actual, node):
        if expected == actual:
            return True
            
        # Handle Generics (e.g. Vec<T> == Vec<i32> if T is generic in current context? No, that's already handled elsewhere)
        if '<' in expected and '<' in actual:
             if expected.split('<')[0] == actual.split('<')[0]:
                  return True
                  
        # Coercion
        numeric_types = ('i32', 'i64', 'u8', 'f32')
        if expected in numeric_types and actual in numeric_types:
             # Upcasts (allowed, no warning usually, or maybe warning if we want to be strict)
             if (expected == 'i64' and actual == 'i32') or (expected == 'i64' and actual == 'u8') or (expected == 'i32' and actual == 'u8'):
                  return True
             if (expected == 'f32' and actual in ('i32', 'i64', 'u8')):
                  return True
             
             # Downcasts (warning for potential precision loss)
             if (expected == 'i32' and actual == 'i64') or (expected == 'u8' and actual in ('i32', 'i64', 'i32')):
                  self.warnings.append((f"Potential precision loss: coercing {actual} to {expected}", node.line, node.column))
                  return True
             if (expected in ('i32', 'i64', 'u8') and actual == 'f32'):
                  self.warnings.append((f"Potential precision loss: coercing {actual} to {expected} (float to int)", node.line, node.column))
                  return True
                  
        # String to u8* or i8* coercion
        if expected in ('u8*', 'i8*', '*u8', '*i8') and actual == 'string':
             return True
        if expected == 'string' and actual in ('u8*', 'i8*', '*u8', '*i8'):
             return True

        return False

    def visit_VarDecl(self, node):
        if node.type_name: node.type_name = self.resolve_type_name(node.type_name)
        init_t = self.visit(node.initializer)
        if node.type_name is None: node.type_name = init_t
        if '<' in node.type_name: self.instantiate_generic_type(node.type_name)
        if init_t != node.type_name and not self.check_type_compatibility(node.type_name, init_t, node):
             self.error(f"Type Error: {init_t} != {node.type_name}", node, error_code="E0002")
        if isinstance(node.initializer, VariableExpr) and not self.is_copy_type(init_t): self.move_var(node.initializer.name)
        self.declare_variable(node.name, node.type_name, node=node)

    def visit_Assignment(self, node):
        val_t = self.visit(node.value)
        if isinstance(node.target, VariableExpr):
             v = self.lookup(node.target.name)
             if not v: raise Exception("Undefined var")
             if v['readers'] > 0: raise Exception("Variable is borrowed")
             if val_t != v['type'] and not self.check_type_compatibility(v['type'], val_t, node):
                  self.error("Type mismatch in assignment", node, error_code="E0002")
             v['moved'] = False
        else:
             target_t = self.visit(node.target)
             if target_t != val_t and not self.check_type_compatibility(target_t, val_t, node):
                  self.error("Type mismatch in assignment", node, error_code="E0002")
        if isinstance(node.value, VariableExpr) and not self.is_copy_type(val_t): self.move_var(node.value.name)

    def visit_BinaryExpr(self, node):
        l, r = self.visit(node.left), self.visit(node.right)
        if node.op in ('PLUS', 'MINUS'):
            if l.endswith('*') and r in ('i32', 'i64'): return l
            if r.endswith('*') and l in ('i32', 'i64') and node.op == 'PLUS': return r
        if l != r and not self.check_type_compatibility(l, r, node) and not self.check_type_compatibility(r, l, node):
            self.error(f"Type mismatch: {l} {node.op} {r}", node, error_code="E0002")
        if node.op in ('PLUS', 'MINUS', 'STAR', 'SLASH', 'PERCENT'): return l
        if node.op in ('EQEQ', 'LT', 'GT', 'LTE', 'GTE', 'NEQ', 'AND', 'OR'): return 'bool'
        raise Exception("Unsupported binary op")

    def visit_IntegerLiteral(self, node): return 'i32'
    def visit_FloatLiteral(self, node): return 'f32'
    def visit_BooleanLiteral(self, node): return 'bool'
    def visit_StringLiteral(self, node): return 'string'
    def visit_CharLiteral(self, node): return 'char'
    
    def visit_LambdaExpr(self, node):
        lambda_name = f"__lambda_{self.lambda_count}"
        self.lambda_count += 1
        node.lambda_name = lambda_name
        
        resolved_params = []
        for pname, ptype in node.params:
            rt = self.resolve_type_name(ptype) if ptype else 'i32'
            resolved_params.append((pname, rt))
            
        ret_type = self.resolve_type_name(node.return_type) if node.return_type else 'i32'
        
        # Create FunctionDef
        func_node = FunctionDef(lambda_name, resolved_params, ret_type, node.body, is_pub=True)
        func_node.lambda_node = node
        func_node.is_lambda = True
        func_node.line = node.line
        func_node.column = node.column
        
        # Register
        self.function_defs[lambda_name] = func_node
        self.functions.add(lambda_name)
        
        if isinstance(self.ast_root, list):
            self.ast_root.append(func_node)
            
        # Capture Management
        self.lambda_base_scopes.append(len(self.scopes))
        self.lambda_capture_stack.append({})
        
        self.visit_FunctionDef(func_node)
        
        node.captures = self.lambda_capture_stack.pop()
        self.lambda_base_scopes.pop()
        
        param_types = [p[1] for p in resolved_params]
        # Type of a lambda (pointer to it)
        return f"fn({','.join(param_types)})->{ret_type}"

    def visit_VariableExpr(self, node):
        if '::' in node.name:
            parts = node.name.rsplit('::', 1)
            prefix = self.resolve_type_name(parts[0])
            suffix = parts[1]
            node.name = f"{prefix}::{suffix}"
            if '<' in prefix: self.instantiate_generic_type(prefix)
            if prefix in self.enums:
                 variants = self.enums[prefix]
                 if suffix in variants:
                      payloads = variants[suffix]
                      if payloads: 
                           raise Exception(f"Variant '{node.name}' expects arguments, used as value")
                      return prefix
            
            if '<' in prefix and prefix.split('<')[0] in self.generic_enums:
                 return prefix

        v, v_depth = self.lookup_with_depth(node.name)
        if not v:
            possibilities = []
            for scope in self.scopes:
                possibilities.extend([k for k in scope.keys() if k != 'active_borrows'])
            possibilities.extend(list(self.functions))
            possibilities.extend(list(self.structs.keys()))
            
            suggestion = self.get_suggestion(node.name, possibilities)
            hint = f"did you mean '{suggestion}'?" if suggestion else None
            self.error(f"Undefined variable: '{node.name}'", node, hint=hint, error_code="E0001")
        
        if self.lambda_base_scopes:
             captured = False
             # If variable is in a scope prior to the current lambda's base scope, it's a capture.
             # We must mark it as captured for ALL lambdas between the variable's scope and use.
             for i in range(len(self.lambda_base_scopes) - 1, -1, -1):
                  if v_depth < self.lambda_base_scopes[i]:
                        self.lambda_capture_stack[i][node.name] = v['type']
                        captured = True
             if captured:
                  node.is_capture = True

        if v['moved']: self.error(f"Use of moved variable '{node.name}'", node, error_code="E0007")
        v['used'] = True
        return v['type']

    def process_fn_call(self, node, callee_type):
        if not callee_type.startswith('fn('):
             raise Exception(f"Type Error: Cannot call non-function type '{callee_type}'")
        main_part, _, ret_type = callee_type.rpartition(')->')
        for arg in node.args: self.visit(arg)
        return ret_type

    def visit_CallExpr(self, node):
        if not isinstance(node.callee, (str, VariableExpr)):
             callee_type = self.visit(node.callee)
             return self.process_fn_call(node, callee_type)

        if isinstance(node.callee, VariableExpr):
             name = node.callee.name
             v = self.lookup(name)
             if v:
                  return self.process_fn_call(node, v['type'])
             node.callee = name
        
        callee = node.callee
        if callee in self.aliases:
            callee = self.aliases[callee]
            node.callee = callee
        
        # Handle built-in intrinsics and special internal functions
        if callee in ('print', 'panic', 'assert', 'slice_from_array', 'fs::read_file', 'fs::write_file', 'fs::append_file', 'malloc', 'free', 'realloc', 'memcpy', '__nexa_panic', '__nexa_assert', 'gpu::dispatch', 'gpu::global_id'):
            if callee == 'gpu::dispatch':
                # O primeiro argumento de gpu::dispatch é um nome de função (kernel), não uma variável
                if len(node.args) > 0:
                    # Skip visit para o primeiro arg se for o nome do kernel
                    for i in range(1, len(node.args)):
                        self.visit(node.args[i])
                return 'void'
            
            for a in node.args: self.visit(a)
            if callee == 'fs::read_file': 
                self.instantiate_generic_type('Buffer<u8>')
                node.type_name = 'Buffer<u8>'
                return 'Buffer<u8>'
            if callee in ('malloc', 'realloc'): return 'u8*'
            if callee == 'gpu::global_id': return 'i32'
            return 'void'

        # Double Colon Logic (also handles resolved aliases from glob imports)
        generics_mapping = {}
        # Double Colon Logic (also handles resolved aliases from glob imports)
        if isinstance(callee, str) and '::' in callee:
            parts = callee.rsplit('::', 1); prefix = self.resolve_type_name(parts[0]); suffix = parts[1]
            callee = f"{prefix}::{suffix}"
            node.callee = callee
            if '<' in prefix: 
                 self.instantiate_generic_type(prefix)
                 # Capture Generics Mapping for Static Methods
                 base = prefix.split('<')[0]
                 if base in self.generic_structs:
                      struct_def = self.generic_structs[base]
                      args_str = prefix[prefix.find('<')+1:-1]
                      args = self.split_generic_args(args_str)
                      for (gname, bound, is_const), gval in zip(struct_def.generics, args):
                           generics_mapping[gname] = gval

            if prefix in self.enums:
                variants = self.enums[prefix]
                if suffix not in variants: raise Exception("Variant not found")
                payloads = variants[suffix]
                if len(node.args) != len(payloads): raise Exception("Arg mismatch")
                self.enter_scope()
                for i, arg in enumerate(node.args):
                    if self.visit(arg) != payloads[i]: raise Exception("Type mismatch")
                    if isinstance(arg, VariableExpr) and not self.is_copy_type(payloads[i]): self.move_var(arg.name)
                self.exit_scope(); return prefix
            
            # Support variants of non-concrete generic enums
            base_prefix = prefix.split('<')[0]
            if base_prefix in self.generic_enums:
                 # We can't type-check the payload thoroughly without instantiation,
                 # but for now we accept it and return the prefix.
                 for arg in node.args: self.visit(arg)
                 return prefix
            
            # Check for mangled function or struct
            mangled_base = f"{prefix}_{suffix}"
            if mangled_base not in self.function_defs and '<' in prefix:
                 # Check if this is a generic method we haven't monomorphized yet
                 base_prefix = prefix.split('<')[0]
                 if base_prefix in self.generic_structs or base_prefix in self.generic_enums:
                      self.instantiate_generic_function(mangled_base)
            
            if mangled_base not in self.function_defs and '<' in prefix:
                 # Fallback to base name mangling if monomorphized function doesn't exist
                 mangled_base = f"{prefix.split('<')[0]}_{suffix}"
            
            if mangled_base in self.function_defs:
                 arg_types = [self.visit(arg) for arg in node.args]
                 func_def, mangled_full = self.resolve_overload(mangled_base, arg_types, node)
                 node.callee = mangled_full
                 func_def.used = True
                 
                 ret = func_def.return_type
                 if generics_mapping: ret = self.apply_submap(ret, generics_mapping)
                 return ret
            
            if mangled_base in self.structs or mangled_base in self.generic_structs:
                 callee = mangled_base; node.callee = mangled_base

        if callee == 'print':
            for a in node.args: self.visit(a)
            return 'void'
        if callee in ('fs::read_file', 'fs::write_file', 'fs::append_file', 'malloc', 'free', 'realloc', 'panic', 'assert', 'memcpy'):
            # Simplified intrinsics check
            if callee in self.function_defs:
                 self.function_defs[callee].used = True
            for a in node.args: self.visit(a)
            if callee == 'fs::read_file': 
                self.instantiate_generic_type('Buffer<u8>')
                node.type_name = 'Buffer<u8>'
                return 'Buffer<u8>'
            if callee in ('malloc', 'realloc'): return 'u8*'
            return 'void'
        if callee == 'gpu::global_id': return 'i32'
        if callee == 'gpu::dispatch':
             if len(node.args) >= 1:
                  self.visit(node.args[0])
                  if len(node.args) == 2: self.visit(node.args[1])
             return 'void'
        if isinstance(callee, str) and callee.startswith('cast<'): self.visit(node.args[0]); return callee[5:-1]
        if isinstance(callee, str) and callee.startswith('sizeof<'): return 'i32'

        # Try local module lookup if not found
        if callee not in self.functions and callee not in self.structs and '<' not in callee:
             if self.current_module:
                  local = f"{self.current_module.replace('::', '_')}_{callee}"
                  if local in self.functions or local in self.structs or local in self.generic_functions or local in self.generic_structs:
                       callee = local
                       node.callee = local
        
        # If still not found, try to find a suggestion
        if callee not in self.functions and callee not in self.structs and callee not in self.generic_functions and callee not in self.generic_structs and '<' not in callee:
             possibilities = list(self.functions) + list(self.structs.keys()) + list(self.generic_functions.keys()) + list(self.generic_structs.keys())
             suggestion = self.get_suggestion(callee, possibilities)
             hint = f"did you mean '{suggestion}'?" if suggestion else None
             self.error(f"Unknown function or struct: '{callee}'", node, hint=hint, error_code="E0004")

        # Generic Struct Instantiation (e.g. Wrapper<T>)
        if '<' in callee:
             base = callee.split('<')[0]
             resolved_base = self.resolve_type_name(base)
             if resolved_base in self.generic_structs:
                  callee = f"{resolved_base}<{callee[callee.find('<')+1:]}"
                  node.callee = callee
                  self.instantiate_generic_struct(callee)

        if callee in self.structs or (callee in self.generic_structs) or ('<' in callee and callee.split('<')[0] in self.generic_structs):
            # Constructor call
            struct_name = callee
            if '<' in struct_name:
                 base = struct_name.split('<')[0]
                 if base in self.generic_structs:
                      self.instantiate_generic_struct(struct_name)
            
            if struct_name not in self.structs:
                 # Check if base exists but not instantiated
                 base = struct_name.split('<')[0] if '<' in struct_name else struct_name
                 if base in self.generic_structs:
                      # Still not instantiated? maybe args are not concrete
                      # We return the generic type name
                      for a in node.args: self.visit(a)
                      return struct_name
                 raise Exception(f"Semantic Error: Unknown struct '{struct_name}'")

            # Mark struct as used
            base_struct = struct_name.split('<')[0]
            if base_struct in self.struct_used: self.struct_used[base_struct] = True
            
            fields = self.structs[struct_name]
            # ... rest of constructor logic (unchanged)
            if len(node.args) != len(fields): raise Exception(f"Arg mismatch for '{struct_name}' constructor")
            self.enter_scope()
            for i, (fname, ftype) in enumerate(fields.items()):
                arg_t = self.visit(node.args[i])
                if arg_t != ftype: 
                    raise Exception(f"Type mismatch field '{fname}': expected {ftype}, got {arg_t}")
            self.exit_scope()
            return struct_name
            
        if '<' in callee:
            base = callee.split('<')[0]
            if base in self.generic_functions:
                 self.instantiate_generic_function(callee)

        if callee in self.functions:
            arg_types = [self.visit(arg) for arg in node.args]
            func_def, mangled_name = self.resolve_overload(callee, arg_types, node)
            
            node.callee = mangled_name
            func_def.used = True
            
            # Re-visit args in scope if needed? No, already visited.
            # But we should check for moves if not copy.
            for i, arg in enumerate(node.args):
                 if isinstance(arg, VariableExpr) and not self.is_copy_type(arg_types[i]):
                      self.move_var(arg.name)

            ret = func_def.return_type
            if generics_mapping: ret = self.apply_submap(ret, generics_mapping)
            
            if func_def.is_async:
                # Wrap in Task<T>
                task_type = f"Task<{ret}>"
                self.instantiate_generic_type(task_type)
                return task_type
                
            return ret

        raise Exception(f"Semantic Error: Unknown function or struct definition '{callee}'")

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        current_scope = self.scopes[-1]
        if 'active_borrows' in current_scope:
            for var_name, role in current_scope['active_borrows']:
                var_info = self.lookup(var_name)
                if var_info:
                    if role == 'reader': var_info['readers'] -= 1
                    elif role == 'writer': var_info['writer'] = False
        
        # Check for unused variables
        for name, info in current_scope.items():
             if name == 'active_borrows': continue
             if not info['used'] and not name.startswith('_'):
                  self.warnings.append((f"Unused variable '{name}'", info['line'], info['col']))

        self.scopes.pop()

    def declare_variable(self, name, type_name, node=None):
        if name in self.scopes[-1]:
             raise Exception(f"Semantic Error: Variable '{name}' already declared in this scope")
        line = node.line if node else 0
        col = node.column if node else 0
        self.scopes[-1][name] = {'type': type_name, 'moved': False, 'readers': 0, 'writer': False, 'used': False, 'line': line, 'col': col}

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope: return scope[name]
        return None

    def lookup_with_depth(self, name):
        for i, scope in enumerate(reversed(self.scopes)):
            if name in scope:
                return scope[name], (len(self.scopes) - 1 - i)
        return None, None

    def visit_BlockStmt(self, node):
        self.enter_scope()
        for s in node.stmts: self.visit(s)
        self.exit_scope()

    def visit_RegionStmt(self, node):
        self.enter_scope(); self.declare_variable(node.name, 'Arena', node=node)
        for s in node.body: self.visit(s)
        self.exit_scope()

    def visit_ReturnStmt(self, node):
        if node.value:
            t = self.visit(node.value)
            if isinstance(node.value, VariableExpr) and not self.is_copy_type(t):
                 if '::' not in node.value.name:
                      self.move_var(node.value.name)

    def substitute_generics(self, node, mapping):
        if isinstance(node, list):
            for x in node: self.substitute_generics(x, mapping)
            return
        if not hasattr(node, '__dict__'): return

        # 1. Update Type Annotations recursively
        # We handle all attributes that might carry a type name string
        for attr in ('type_name', 'struct_type', 'item_type', 'option_type', 'iterator_type', 'enum_name'):
            if hasattr(node, attr):
                val = getattr(node, attr)
                if isinstance(val, str):
                    setattr(node, attr, self.apply_submap(val, mapping))

        if isinstance(node, CallExpr):
            if isinstance(node.callee, str) and '<' in node.callee:
                 node.callee = self.apply_submap(node.callee, mapping)
            elif hasattr(node.callee, 'name'): # VariableExpr
                 if '<' in node.callee.name:
                      node.callee.name = self.apply_submap(node.callee.name, mapping)
                 elif node.callee.name in mapping:
                      node.callee.name = mapping[node.callee.name]

        if hasattr(node, 'name') and not isinstance(node, (FunctionDef, StructDef, EnumDef)):
             val = getattr(node, 'name')
             if isinstance(val, str):
                  if val in mapping:
                       setattr(node, 'name', mapping[val])
                  elif '<' in val:
                       setattr(node, 'name', self.apply_submap(val, mapping))

        # 2. Recurse on Children
        for k, v in node.__dict__.items():
            if k in ('type_name', 'struct_type', 'item_type', 'option_type', 'iterator_type', 'enum_name', 'callee', 'name', 'module'): continue 
            if isinstance(v, (list, object)) and not isinstance(v, (str, int, bool, float)) and v is not None:
                self.substitute_generics(v, mapping)

    def apply_submap(self, t, mapping):
        if not t: return t
        # Normalize: remove spaces
        t = t.replace(' ', '')
        if t.startswith('&mut'): return '&mut' + self.apply_submap(t[4:], mapping)
        if t.startswith('&'): return '&' + self.apply_submap(t[1:], mapping)
        if t.endswith('*'): return self.apply_submap(t[:-1], mapping) + '*'
        if '<' in t and t.endswith('>'):
            b = t[:t.find('<')]; i = t[t.find('<')+1:-1]
            args = [self.apply_submap(x.strip(), mapping) for x in self.split_generic_args(i)]
            return f"{b}<{','.join(args)}>"
        return mapping.get(t, t)

    def split_generic_args(self, s):
        args = []
        depth = 0
        current = ""
        for char in s:
            if char == '<': depth += 1
            elif char == '>': depth -= 1
            
            if char == ',' and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += char
        if current: args.append(current.strip())
        return args
