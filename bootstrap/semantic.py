from lexer import Lexer
from n_parser import Parser, FunctionDef, StructDef, EnumDef, ImplDef, MatchExpr, CaseArm, ArrayLiteral, IndexAccess, UnaryExpr, VariableExpr, IfStmt, WhileStmt, ForStmt, VarDecl, Assignment, CallExpr, MemberAccess, MethodCall, ReturnStmt, BinaryExpr, RegionStmt, FloatLiteral, CharLiteral, IntegerLiteral, BreakStmt, ContinueStmt
from errors import CompilerError
import sys
import copy
class SemanticAnalyzer:
    def __init__(self):
        # Symbol table entries: {name: {'type': type, 'moved': bool, 'readers': int, 'writer': bool, 'is_ref': bool}}
        self.scopes = [{}] 
        self.borrow_cleanup_stack = [] # Stack of lists of (borrowed_var_name, is_mut)
        self.current_function = None
        self.struct_methods = {}  # {struct_name: {method_name: FunctionDef}}
        self.loop_depth = 0

    def error(self, message, node=None, hint=None):
        line = node.line if node else None
        column = node.column if node else None
        raise CompilerError(message, line, column, hint=hint)

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
        
        # Threshold: allow distance up to 3 or 1/3 length
        limit = max(3, len(name) // 3)
        if min_dist <= limit and min_dist < len(name): 
             return best_match
        return None

    def is_copy_type(self, type_name: str) -> bool:
        """
        Bootstrap 'Copy' rule:
        - Primitive scalars are copy: i32, i64, u8, bool, f32
        - Pointers are copy: T*
        - 'string' (i8*) is treated as copy in the bootstrap
        Everything else is treated as move-only (structs/enums/arrays).
        """
        if not type_name:
            return False
        if type_name in ('i32', 'i64', 'u64', 'u8', 'bool', 'f32', 'string', 'char'):
            return True
        if type_name.endswith('*'):
            return True
        return False

    def move_var(self, name: str, node=None):
        """
        Marks a variable as moved (consumed) if it is move-only.
        Also enforces: cannot move while borrowed (readers/writer active).
        """
        var_info = self.lookup(name)
        if not var_info:
            self.error(f"Semantic Error: Move of undefined variable '{name}'", node)

        if var_info.get('moved'):
            self.error(f"Ownership Error: Use of moved variable '{name}'", node)

        # Disallow moving while borrowed (bootstrap rule).
        if var_info.get('readers', 0) > 0:
            self.error(f"Borrow Error: Cannot move '{name}' because it is borrowed", node)
        if var_info.get('writer'):
            self.error(f"Borrow Error: Cannot move '{name}' because it is borrowed mutable", node)

        var_info['moved'] = True

    def check_privacy(self, target_node, target_name):
        target_mod = getattr(target_node, 'module', "")
        # If target has no module (e.g. built-ins), it's public.
        if not target_mod: return

        current_mod = ""
        if self.current_function:
            current_mod = getattr(self.current_function, 'module', "")
        
        # If accessing member of same module, allow.
        # Check prefix matching? name mangling handles unique definition?
        # module "A" accessing "A" is fine.
        if current_mod == target_mod: return
        
        # If different module, target MUST be pub.
        if not getattr(target_node, 'is_pub', False):
            raise Exception(f"Privacy Error: '{target_name}' is private to module '{target_mod}'")

    def analyze(self, ast):
        self.ast_root = ast
        # 1. Collect function and struct names
        self.functions = set(['print', 'gpu::global_id', 'gpu::dispatch', 'panic', 'assert', 'slice_from_array', 'fs::read_file', 'fs::write_file', 'fs::append_file'])
        self.function_defs = {} # name -> FunctionDef
        self.structs = {} # name -> {field: type}
        self.struct_defs = {} # name -> StructDef (for privacy check)
        self.enums = {} # name -> {variant: [payload_types]}
        self.enum_defs = {} # name -> EnumDef (for privacy check)
        self.generic_functions = {} # name -> node
        self.generic_structs = {} # name -> node
        self.generic_enums = {} # name -> node
        
        for node in ast:
            if isinstance(node, FunctionDef):
                if node.generics:
                    self.generic_functions[node.name] = node
                else:
                    self.functions.add(node.name)
                    self.function_defs[node.name] = node
            elif isinstance(node, StructDef):
                 if node.generics:
                     self.generic_structs[node.name] = node
                 else:
                     fields = {name: type_ for name, type_ in node.fields}
                     self.structs[node.name] = fields
                     self.struct_defs[node.name] = node
            elif isinstance(node, EnumDef):
                 if node.generics:
                     self.generic_enums[node.name] = node
                 else:
                     # Variants: list of (name, payload_types)
                     variants = {vname: payloads for vname, payloads in node.variants}
                     self.enums[node.name] = variants
                     self.enum_defs[node.name] = node
            elif isinstance(node, ImplDef):
                 pass # We handle ImplDef in a second pass or directly in visit loop?
                 # Actually, methods should be available before visiting bodies.
                 # So we should process ImplDef here (extract method signatures).
                 self.register_impl_methods(node)
        
        # Built-in generic Buffer<T> (for GPU/heterogeneous APIs):
        # struct Buffer<T> { ptr: *T, len: i32 }
        # This stays as a generic template and gets monomorphized when used.
        if 'Buffer' not in self.generic_structs and 'Buffer' not in self.structs:
            self.generic_structs['Buffer'] = StructDef(
                'Buffer',
                fields=[('ptr', 'T*'), ('len', 'i32')],
                generics=['T'],
            )

        # Built-in generic Slice<T> (for views into arrays/buffers):
        # struct Slice<T> { ptr: *T, len: i32 }
        if 'Slice' not in self.generic_structs and 'Slice' not in self.structs:
            self.generic_structs['Slice'] = StructDef(
                'Slice',
                fields=[('ptr', 'T*'), ('len', 'i32')],
                generics=['T'],
            )

        # Inject Built-in Arena
        self.structs['Arena'] = {'chunk': 'u8*', 'offset': 'i32', 'capacity': 'i32'}

        # 2. Analyze bodies
        for node in ast:
            self.visit(node)

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def visit_StructDef(self, node):
        pass # Already collected

    def visit_EnumDef(self, node):
        pass # Already collected

    def visit_ImplDef(self, node):
        # Visit methods to check bodies
        for method in node.methods:
             self.visit(method) # visit_FunctionDef

    def register_impl_methods(self, node):
        struct_name = node.struct_name
        
        # Initialize methods dict for this struct if needed
        if struct_name not in self.struct_methods:
            self.struct_methods[struct_name] = {}
        
        # Mangle method names and fix 'self' types
        for method in node.methods:
             # Check for duplicate?
             
             # Rewrite 'self'
             if method.params:
                  pname, ptype = method.params[0]
                  if pname == 'self':
                       # method.params is list of tuples (name, type)
                       # tuples are immutable. Need to replace tuple in list.
                       if ptype == 'Self': 
                           method.params[0] = ('self', struct_name)
                       elif ptype == '&Self': 
                           method.params[0] = ('self', f"{struct_name}*")
                       elif ptype == '&mut Self':
                           method.params[0] = ("self", f"{struct_name}*")
             
             
             # Register method in struct_methods for resolution
             self.struct_methods[struct_name][method.name] = method
             
             # Mangle name
             mangled_name = f"{struct_name}_{method.name}"
             method.name = mangled_name # Update AST for CodeGen/Semantic visitation
             
             self.functions.add(mangled_name)
             self.function_defs[mangled_name] = method

    def visit_MemberAccess(self, node):
        # 1. Evaluate object
        obj_type = self.visit(node.object)
        
        # 2. Check if object is a struct or pointer to struct
        base_type = obj_type
        if base_type.startswith('&'):
             if base_type.startswith('&mut '):
                  base_type = base_type[5:]
             else:
                  base_type = base_type[1:]
        
        if base_type.endswith('*'):
             base_type = base_type[:-1]

        lookup_type = base_type
        if lookup_type not in self.structs and '<' in lookup_type:
            lookup_type = lookup_type.split('<')[0]

        if lookup_type in self.structs:
             fields_dict = self.structs[lookup_type]
        elif lookup_type in self.generic_structs:
             struct_def = self.generic_structs[lookup_type]
             fields_dict = {f[0]: f[1] for f in struct_def.fields}
        else:
             raise Exception(f"Type Error: access member '{node.member}' on non-struct type '{obj_type}'")
        
        # 3. Check if member exists
        if node.member not in fields_dict:
             hint = None
             suggestion = self.suggest_name(node.member, fields_dict.keys())
             if suggestion:
                 hint = f"Did you mean '{suggestion}'?"
             self.error(f"Type Error: Struct '{lookup_type}' has no member '{node.member}'", node, hint=hint)
        
        # Annotate node for CodeGen
        node.struct_type = lookup_type
             
        ty = fields_dict[node.member]
        node.type_name = ty
        return ty

    def visit_MethodCall(self, node):
        # Resolve obj.method(args) and annotate for code generation
        # 1. Get receiver type
        receiver_type = self.visit(node.receiver)
        
        # Strip references to get base type
        base_type = receiver_type.lstrip('&')
        if base_type.startswith('mut '):
            base_type = base_type[4:]
        if base_type.endswith('*'):
            base_type = base_type[:-1]
        
        # 2. Look up method
        lookup_type = base_type
        if lookup_type not in self.struct_methods and '<' in lookup_type:
            lookup_type = lookup_type.split('<')[0]
            
        if lookup_type not in self.struct_methods:
            raise Exception(f"Semantic Error: Type '{base_type}' has no methods")
        
        methods = self.struct_methods[lookup_type]
        if node.method_name not in methods:
            hint = None
            suggestion = self.suggest_name(node.method_name, methods.keys())
            if suggestion:
                hint = f"Did you mean '{suggestion}'?"
            try:
                self.error(f"Semantic Error: Method '{node.method_name}' not found on type '{base_type}'", node, hint=hint)
            except CompilerError as e:
                raise e # Propagate CompilerError
            except Exception as e:
                # Fallback if self.error wasn't used or raised generic Exception
                raise Exception(f"Semantic Error: Method '{node.method_name}' not found on type '{base_type}'{(' (Hint: ' + hint + ')') if hint else ''}")
        
        method_def = methods[node.method_name]
        
        # 3. Type-check arguments
        # Special case for generic methods in bootstrap:
        # If method belongs to a generic struct, we should ideally monomorphize it.
        # For now, we'll allow it if it's 'Vec' or other builtins.
        
        # 4. Annotate for codegen
        node.struct_type = lookup_type # Use the type name where methods are registered
        node.receiver_type = receiver_type
        node.return_type = method_def.return_type
        
        return method_def.return_type


    def visit_ArrayLiteral(self, node):
        if not node.elements:
             raise Exception("Semantic Error: Empty array literals not supported (cannot infer type)")
        
        first_type = self.visit(node.elements[0])
        for el in node.elements[1:]:
            type_ = self.visit(el)
            if type_ != first_type:
                 raise Exception(f"Type Error: Array elements must be same type. Expected '{first_type}', got '{type_}'")
                 
        size = len(node.elements)
        return f"[{first_type}:{size}]"

    def visit_IndexAccess(self, node):
        # 1. Analyze object
        obj_type = self.visit(node.object)
        
        # 2. Analyze index
        index_type = self.visit(node.index)
        if index_type != 'i32':
             raise Exception(f"Type Error: Array index must be i32, got {index_type}")

        # 3. Arrays: [T:N]
        if obj_type.startswith('[') and obj_type.endswith(']'):
            # Parse type string [T:N]
            content = obj_type[1:-1]
            elem_type, _size_str = content.split(':', 1)
            node.type_name = elem_type
            return elem_type

        # 4. Pointers: T*
        # Allow pointer indexing as syntactic sugar for ptr_offset + deref.
        if obj_type.endswith('*'):
            ty = obj_type[:-1]
            node.type_name = ty
            return ty

        # 5. Slices: Slice<T>
        if isinstance(obj_type, str) and obj_type.startswith("Slice<") and obj_type.endswith(">"):
            inner = obj_type[len("Slice<"):-1]
            node.type_name = inner
            return inner

        raise Exception(f"Type Error: Indexing non-array/non-pointer type '{obj_type}'")

    def visit_UnaryExpr(self, node):
        operand_type = self.visit(node.operand)
        
        if node.op == '*':
            # Dereference: operand must be a pointer type "T*"
            if not operand_type.endswith('*'):
                 raise Exception(f"Type Error: Cannot dereference non-pointer type '{operand_type}'")
            ty = operand_type[:-1]
            node.type_name = ty
            return ty
            
        elif node.op == '&':
            # Address Of: return "T*"
            if isinstance(node.operand, VariableExpr):
                var_name = node.operand.name
                var_info = self.lookup(var_name)
                if not var_info:
                     raise Exception(f"Semantic Error: Borrow of undefined variable '{var_name}'")
                
                # Check Borrow Rules
                if var_info['writer']:
                     raise Exception(f"Borrow Error: Cannot borrow '{var_name}' as immutable because it is already borrowed as mutable")
                
                # Register Reader
                
                # Register Reader
                var_info['readers'] += 1
                
                # Add to current scope cleanup
                # We need to associate this borrow with the current scope lifetime.
                # Simplest way: self.scopes[-1] needs a 'borrows' list?
                # Using self.borrow_cleanup_stack might be tricky if scopes handle it.
                # Let's add 'active_borrows' to self.scopes entries.
                if 'active_borrows' not in self.scopes[-1]:
                     self.scopes[-1]['active_borrows'] = []
                self.scopes[-1]['active_borrows'].append((var_name, 'reader'))
                
            ty = f"{operand_type}*"
            node.type_name = ty
            return ty
            
        elif node.op == '&mut':
             # Mutable Reference
             if isinstance(node.operand, VariableExpr):
                var_name = node.operand.name
                var_info = self.lookup(var_name)
                if not var_info:
                     raise Exception(f"Semantic Error: Borrow of undefined variable '{var_name}'")
                
                # Check Borrow Rules
                if var_info['writer']:
                     raise Exception(f"Borrow Error: Cannot borrow '{var_name}' as mutable because it is already borrowed as mutable")
                if var_info['readers'] > 0:
                     raise Exception(f"Borrow Error: Cannot borrow '{var_name}' as mutable because it is already borrowed as immutable")

                # Register Writer
                var_info['writer'] = True
                
                if 'active_borrows' not in self.scopes[-1]:
                     self.scopes[-1]['active_borrows'] = []
                self.scopes[-1]['active_borrows'].append((var_name, 'writer'))
                
             ty = f"{operand_type}*"
             node.type_name = ty
             return ty
            
        node.type_name = operand_type
        return operand_type

    def visit_MatchExpr(self, node):
        """
        Type-check `match` expressions and bind variant payload variables (single payload for now).

        Notes:
        - Supports generic enums by instantiating `Option<i32>`-style types on demand.
        - Enforces basic exhaustiveness (all variants must be covered).
        - Uses enter_scope/exit_scope to properly release borrows via `active_borrows`.
        """
        expr_type = self.visit(node.value)

        # Instantiate generic enum if needed (e.g., Option<i32>)
        if expr_type and '<' in expr_type:
            self.instantiate_generic_type(expr_type)

        if expr_type not in self.enums:
            raise Exception(f"Type Error: Match expression must be an Enum, got '{expr_type}'")

        node.enum_name = expr_type
        variants = self.enums[expr_type]

        covered = set()
        for case in node.cases:
            if case.variant_name not in variants:
                raise Exception(f"Semantic Error: Enum '{expr_type}' has no variant '{case.variant_name}'")

            covered.add(case.variant_name)
            payloads = variants[case.variant_name]

            # Bind payload variable if present (single payload supported)
            self.enter_scope()
            if case.var_names:
                if len(case.var_names) != len(payloads):
                     raise Exception(f"Semantic Error: Variant '{case.variant_name}' has {len(payloads)} fields, but matched with {len(case.var_names)} variables")
                
                for i, vname in enumerate(case.var_names):
                     self.declare_variable(vname, payloads[i])
            else:
                if len(payloads) > 0:
                     # Payload ignored
                     pass

            self.visit(case.body)
            self.exit_scope()

        if len(covered) != len(variants):
            missing = set(variants.keys()) - covered
            raise Exception(f"Semantic Error: Match not exhaustive. Missing: {missing}")

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        current_scope = self.scopes[-1]
        
        # Release borrows tied to this scope
        if 'active_borrows' in current_scope:
            for var_name, role in current_scope['active_borrows']:
                var_info = self.lookup(var_name)
                if var_info:
                    if role == 'reader':
                        var_info['readers'] -= 1
                    elif role == 'writer':
                        var_info['writer'] = False
                        
        self.scopes.pop()

    def declare_variable(self, name, type_name):
        if name in self.scopes[-1]:
             raise Exception(f"Semantic Error: Variable '{name}' already declared in this scope")
        # Initialize borrow state: readers=0 (set needed?), writer=False
        self.scopes[-1][name] = {
            'type': type_name, 
            'moved': False,
            'readers': 0,
            'writer': False
        }

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def generic_visit(self, node):
        print(f"GENERIC VISIT CAUGHT: {type(node).__name__}", flush=True)
        if type(node).__name__ == 'ForStmt':
             print("Dispatching to visit_ForStmt manually", flush=True)
             return self.visit_ForStmt(node)
             
        raise Exception(f"No visit_{type(node).__name__} method in SemanticAnalyzer")

    def instantiate_generic_type(self, name_with_args):
        if not name_with_args:
            return

        # Handle pointer types like "Vec<i32>*": instantiate the inner type ("Vec<i32>")
        # instead of accidentally trying to monomorphize a synthetic struct named "Vec<i32>*".
        if name_with_args.endswith('*'):
            self.instantiate_generic_type(name_with_args[:-1])
            return

        # Handle array types like "[T:3]" or "[Vec<i32>:3]" by instantiating the element type.
        if name_with_args.startswith('[') and name_with_args.endswith(']'):
            content = name_with_args[1:-1]
            elem_type = content.split(':', 1)[0].strip()
            self.instantiate_generic_type(elem_type)
            return

        if '<' not in name_with_args:
            return

        base_name = name_with_args.split('<', 1)[0]
        
        if base_name in self.generic_structs:
             self.instantiate_generic_struct(name_with_args)
        elif base_name in self.generic_enums:
             self.instantiate_generic_enum(name_with_args)
        else:
             # Could be unknown or not generic type (handled elsewhere)
             pass

    def instantiate_generic_struct(self, name_with_args):
        # name_with_args: "Box<i32>"
        if name_with_args in self.structs:
            return # Already instantiated
            
        if '<' not in name_with_args:
             return # Not generic
             
        base_name, rest = name_with_args.split('<', 1)
        rest = rest[:-1] # Remove trailing >
        args = [a.strip() for a in rest.split(',')]
        
        if base_name not in self.generic_structs:
            raise Exception(f"Semantic Error: Unknown generic struct '{base_name}'")
            
        def_node = self.generic_structs[base_name]
        
        if len(args) != len(def_node.generics):
            raise Exception(f"Type Error: Generic '{base_name}' expects {len(def_node.generics)} args, got {len(args)}")
            
        # Monomorphize
        mapping = dict(zip(def_node.generics, args))
        
        new_fields = []
        for fname, ftype in def_node.fields:
            new_type = self.apply_submap(ftype, mapping)
            new_fields.append((fname, new_type))
            
        # Register new concrete struct
        self.structs[name_with_args] = dict(new_fields)
        
        new_def = StructDef(name_with_args, new_fields)
        self.ast_root.append(new_def)

    def instantiate_generic_enum(self, name_with_args):
        # name_with_args: "Option<i32>"
        if name_with_args in self.enums: return
        if '<' not in name_with_args: return
        
        base_name, rest = name_with_args.split('<', 1)
        rest = rest[:-1]
        args = [a.strip() for a in rest.split(',')]
        
        if base_name not in self.generic_enums:
             raise Exception(f"Semantic Error: Unknown generic enum '{base_name}'")
             
        def_node = self.generic_enums[base_name]
        
        if len(args) != len(def_node.generics):
             raise Exception(f"Type Error: Generic '{base_name}' expects {len(def_node.generics)} args, got {len(args)}")
             
        mapping = dict(zip(def_node.generics, args))
        
        new_variants = []
        for vname, payloads in def_node.variants:
             new_payloads = []
             for ptype in payloads:
                 new_payloads.append(self.apply_submap(ptype, mapping))
             new_variants.append((vname, new_payloads))
             
        self.enums[name_with_args] = {v: ps for v, ps in new_variants}
        
        # Append to AST for CodeGen
        new_def = EnumDef(name_with_args, new_variants)
        self.ast_root.append(new_def)

    def instantiate_generic_function(self, name_with_args):
        # name_with_args: "id<i32>"
        if name_with_args in self.functions:
            return
            
        if '<' not in name_with_args:
             return
             
        base_name, rest = name_with_args.split('<', 1)
        rest = rest[:-1]
        args = [a.strip() for a in rest.split(',')]
        
        if base_name not in self.generic_functions:
             if base_name in self.generic_structs:
                  self.instantiate_generic_struct(name_with_args)
                  return
             elif base_name in self.generic_enums:
                  self.instantiate_generic_enum(name_with_args)
                  return
             else:
                  raise Exception(f"Semantic Error: Unknown generic function or struct '{base_name}'")

        def_node = self.generic_functions[base_name]
        
        if len(args) != len(def_node.generics):
             raise Exception(f"Type Error: Generic '{base_name}' expects {len(def_node.generics)} args, got {len(args)}")
             
        mapping = dict(zip(def_node.generics, args))
        
        new_params = []
        for pname, ptype in def_node.params:
            new_params.append((pname, self.apply_submap(ptype, mapping)))
                
        new_return = self.apply_submap(def_node.return_type, mapping)
            
        # Body substitution
        import copy
        new_body = copy.deepcopy(def_node.body)
        
        self.substitute_generics(new_body, mapping)
        
        new_func = FunctionDef(name_with_args, new_params, new_return, new_body, def_node.is_kernel)
        # No debug print
        self.ast_root.append(new_func)
        self.functions.add(name_with_args)
        self.function_defs[name_with_args] = new_func
        # No debug print



    def visit_ForStmt(self, node):
        # 1. Analyze Range
        start_type = self.visit(node.start_expr)
        end_type = self.visit(node.end_expr)
        
        if start_type != 'i32' or end_type != 'i32':
             raise Exception(f"Type Error: For loop range must be i32, got '{start_type}..{end_type}'")
             
        # 2. Scope for Loop Variable
        self.enter_scope()
        self.loop_depth += 1
        
        # Declare loop variable (immutable in body? usually yes, but for now just standard var)
        self.declare_variable(node.var_name, 'i32')
        
        # 3. Analyze Body
        for stmt in node.body:
             self.visit(stmt)
             
        self.loop_depth -= 1
        self.exit_scope()

    def visit_WhileStmt(self, node):
        if self.visit(node.condition) != 'bool':
            raise Exception("Type Error: While condition must be bool")
            
        self.enter_scope()
        self.loop_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.loop_depth -= 1
        self.exit_scope()

    def visit_BreakStmt(self, node):
        if self.loop_depth == 0:
            raise Exception("Semantic Error: 'break' outside of loop")

    def visit_ContinueStmt(self, node):
        if self.loop_depth == 0:
            raise Exception("Semantic Error: 'continue' outside of loop")


    def visit_FunctionDef(self, node):
        if node.generics:
             # Skip body analysis for generic definitions.
             # They are analyzed only when instantiated.
             self.functions.add(node.name)
             self.generic_functions[node.name] = node
             return

        self.current_function = node
        self.enter_scope()
        
        # Register parameters
        for pname, ptype in node.params:
             if '<' in ptype:
                  self.instantiate_generic_type(ptype)
             self.declare_variable(pname, ptype)
             
        for stmt in node.body:
            self.visit(stmt)
        self.exit_scope()
        self.current_function = None

    def visit_VarDecl(self, node):
        # 1. Analyze initializer
        init_type = self.visit(node.initializer)
        
        # 2. Infer or Check type
        if node.type_name is None:
             node.type_name = init_type
        
        # Instantiate generic type if needed
        if node.type_name and '<' in node.type_name:
             self.instantiate_generic_type(node.type_name)
        
        # No debug print

        if init_type != node.type_name:
             # Relax check for generic structs in bootstrap
             is_generic_match = False
             if '<' in init_type and '<' in node.type_name:
                 if init_type.split('<')[0] == node.type_name.split('<')[0]:
                     is_generic_match = True
             
             if not is_generic_match:
                 raise Exception(f"DEBUG: init_type='{init_type}', node.type='{node.type_name}'. Type Error: Cannot assign type '{init_type}' to variable '{node.name}' of type '{node.type_name}'")

        # Ownership: `let y = x` moves `x` if it's move-only.
        if isinstance(node.initializer, VariableExpr) and not self.is_copy_type(init_type):
            self.move_var(node.initializer.name)
        
        # 3. Declare variable
        self.declare_variable(node.name, node.type_name)

    def visit_Assignment(self, node):
        value_type = self.visit(node.value)
        
        if isinstance(node.target, VariableExpr):
             name = node.target.name
             var_info = self.lookup(name)
             if not var_info:
                 raise Exception(f"Semantic Error: Assignment to undefined variable '{name}'")
             
             # Borrow Check: No mutation while borrowed
             if var_info['readers'] > 0:
                  raise Exception(f"Borrow Error: Cannot assign to '{name}' because it is borrowed")
             
             target_type = var_info['type']
             if target_type != value_type:
                  # Relax check for generic structs
                  is_generic_match = False
                  if '<' in value_type and '<' in target_type:
                      if value_type.split('<')[0] == target_type.split('<')[0]:
                          is_generic_match = True
                  
                  if not is_generic_match:
                      raise Exception(f"Type Error: Cannot assign type '{value_type}' to variable '{name}' of type '{target_type}'")
             var_info['moved'] = False

        elif isinstance(node.target, MemberAccess):
             # Visit target
             target_type = self.visit(node.target)
             if target_type != value_type:
                 raise Exception(f"Type Error: Mismatch assignment to member {target_type} != {value_type}")

        elif isinstance(node.target, UnaryExpr):
             if node.target.op == '*':
                  # *ptr = val
                  # visit(*ptr) returns type T (dereferenced type)
                  # visit(node.target) checks valid dereference.
                  target_type = self.visit(node.target)
                  
                  if target_type != value_type:
                       raise Exception(f"Type Error: Cannot assign '{value_type}' to dereference of type '{target_type}'")
             else:
              raise Exception("Invalid assignment target")

        elif isinstance(node.target, IndexAccess):
            # arr[i] = x  OR  ptr[i] = x
            target_elem_type = self.visit(node.target)  # visit_IndexAccess returns element type
            if target_elem_type != value_type:
                raise Exception(f"Type Error: Cannot assign '{value_type}' to indexed element of type '{target_elem_type}'")
              
        elif isinstance(node.target, MemberAccess):
             # Assign to struct field
             # Visit MemberAccess (which checks validity and returns field type)
             # Note: visit_MemberAccess evaluates the OBJECT and checks field existence.
             target_type = self.visit(node.target)
             
             if target_type != value_type:
                  raise Exception(f"Type Error: Cannot assign '{value_type}' to member of type '{target_type}'")
        else:
             raise Exception("Invalid assignment target")

        # Ownership: assigning from a variable moves it if move-only (x = y)
        # (also applies to `*ptr = y` and `obj.field = y`).
        if isinstance(node.value, VariableExpr) and not self.is_copy_type(value_type):
            self.move_var(node.value.name)

    def visit_BinaryExpr(self, node):
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        
        # Pointer Arithmetic: Pointer + Integer or Pointer - Integer
        if node.op in ('PLUS', 'MINUS'):
            if left_type.endswith('*') and right_type in ('i32', 'i64'):
                node.type_name = left_type
                return left_type
            if right_type.endswith('*') and left_type in ('i32', 'i64') and node.op == 'PLUS':
                node.type_name = right_type
                return right_type

        if left_type != right_type:
             raise Exception(f"Type Error: Operands must be same type. Got '{left_type}' and '{right_type}'")

        # Arithmetic
        if node.op in ('PLUS', 'MINUS', 'STAR', 'SLASH', 'PERCENT'):
            if left_type not in ('i32', 'i64', 'u64', 'u8', 'f32') and not left_type.endswith('*'):
                raise Exception(f"Type Error: Arithmetic only supported for primitives or pointers. Got '{left_type}'")
            node.type_name = left_type
            return left_type

        # Comparisons
        if node.op in ('EQEQ', 'LT', 'GT', 'LTE', 'GTE', 'NEQ'):
            if left_type not in ('i32', 'i64', 'u8', 'char', 'f32', 'bool'):
                raise Exception(f"Type Error: Comparison not supported for type '{left_type}'")
            node.type_name = 'bool'
            return 'bool'

        raise Exception(f"Semantic Error: Unsupported binary operator '{node.op}'")

    def visit_IntegerLiteral(self, node):
        node.type_name = 'i32'
        return 'i32' # Default

    def visit_FloatLiteral(self, node):
        node.type_name = 'f32'
        return 'f32'

    def visit_BooleanLiteral(self, node):
        node.type_name = 'bool'
        return 'bool'

    def visit_StringLiteral(self, node):
        node.type_name = 'string'
        return 'string'

    def visit_CharLiteral(self, node):
        node.type_name = 'char'
        return 'char'

    def visit_VariableExpr(self, node):
        var_info = self.lookup(node.name)
        if not var_info:
             # Gather candidates
             candidates = []
             for scope in self.scopes:
                 candidates.extend(scope.keys())
             
             hint = None
             suggestion = self.suggest_name(node.name, candidates)
             if suggestion:
                 hint = f"Did you mean '{suggestion}'?"
             
             self.error(f"Semantic Error: Undefined variable '{node.name}'", node, hint=hint)
             
        if var_info['moved']:
            self.error(f"Ownership Error: Use of moved variable '{node.name}'", node)
            
        ty = var_info['type']
        node.type_name = ty
        return ty

    def visit_CallExpr(self, node):
        callee_name = None
        
        # 1. Resolve Callee Name or Handle Method Call
        if isinstance(node.callee, str):
            callee_name = node.callee
        elif isinstance(node.callee, VariableExpr):
            callee_name = node.callee.name
        
        if callee_name and '::' in callee_name:
            # Check for Struct::new
            parts = callee_name.split('::')
            struct_name = parts[0]
            # Strip generics from struct name for lookup
            if '<' in struct_name: struct_base = struct_name.split('<')[0]
            else: struct_base = struct_name
            method_name = parts[1]
            mangled_name = f"{struct_base}_{method_name}"
            # Check if this mangled name exists
            if mangled_name in self.functions:
                callee_name = mangled_name
        
        if callee_name:
            node.callee = callee_name

        if isinstance(node.callee, MemberAccess):
            # ... and so on ... 
        
             obj_node = node.callee.object
             method_name = node.callee.member
             obj_type = self.visit(obj_node)
             base_type = obj_type
             if base_type.endswith('*'): base_type = base_type[:-1]
             mangled_name = f"{base_type}_{method_name}"
             
             if mangled_name in self.functions:
                  target_func = self.function_defs[mangled_name]
                  if target_func.params and target_func.params[0][0] == 'self':
                      self_param_type = target_func.params[0][1]
                      arg_expr = obj_node
                      if obj_type == base_type and self_param_type.endswith('*'):
                           arg_expr = UnaryExpr('&', obj_node)
                      elif obj_type.endswith('*') and not self_param_type.endswith('*'):
                           arg_expr = UnaryExpr('*', obj_node)
                      node.args.insert(0, arg_expr)
                  callee_name = mangled_name
                  node.callee = callee_name 
             else:
                  raise Exception(f"Method '{method_name}' not found for type '{base_type}'")

        if not callee_name:
             raise Exception(f"Semantic Error: Call to complex expression or unknown method: {node.callee}")

        # Intrinsics
        if callee_name == 'print':
            self.enter_scope()
            for arg in node.args:
                self.visit(arg)
            self.exit_scope()
            node.type_name = 'void'
            return 'void'
        elif callee_name == 'fs::read_file':
            # fs::read_file(path: string) -> Buffer<u8>
            if len(node.args) != 1:
                raise Exception("fs::read_file expects 1 arg: path (string)")
            path_ty = self.visit(node.args[0])
            if path_ty != 'string':
                raise Exception(f"Type Error: fs::read_file expects string path, got '{path_ty}'")
            ret_ty = "Buffer<u8>"
            self.instantiate_generic_type(ret_ty)
            node.type_name = ret_ty
            return ret_ty
        elif callee_name == 'fs::write_file' or callee_name == 'fs::append_file':
            if len(node.args) != 3: raise Exception(f"{callee_name} expects 3 args: (path, data, len)")
            path_ty = self.visit(node.args[0])
            data_ty = self.visit(node.args[1])
            len_ty = self.visit(node.args[2])
            if path_ty != 'string': raise Exception(f"Type Error: {callee_name} expects string path")
            if data_ty != 'string' and data_ty != 'u8*': raise Exception(f"Type Error: {callee_name} expects data as string or u8*")
            if len_ty != 'i32': raise Exception(f"Type Error: {callee_name} expects len as i32")
            node.type_name = 'void'
            return 'void'
        elif callee_name == 'slice_from_array':
            if len(node.args) != 1: raise Exception("slice_from_array expects 1 arg")
            arg_ty = self.visit(node.args[0])
            if not arg_ty.endswith('*'): raise Exception("slice_from_array expects pointer")
            elem_type = arg_ty[:-1].split('[')[1].split(':')[0]
            size = int(arg_ty[:-1].split(':')[1][:-1])
            slice_ty = f"Slice<{elem_type}>"
            self.instantiate_generic_type(slice_ty)
            node.type_name = slice_ty
            node.slice_elem_type = elem_type
            node.slice_len = size
            return slice_ty
        elif callee_name == 'memcpy':
            if len(node.args) != 3: raise Exception("memcpy requires 3 args")
            self.enter_scope()
            self.visit(node.args[0])
            self.visit(node.args[1])
            size_ty = self.visit(node.args[2])
            self.exit_scope()
            if size_ty != 'i32': raise Exception("memcpy size must be i32")
            node.type_name = 'void'
            return 'void'
        elif callee_name == 'malloc':
             self.enter_scope()
             size_type = self.visit(node.args[0])
             self.exit_scope()
             node.type_name = 'u8*'
             return 'u8*'
        elif callee_name == 'free':
             self.enter_scope()
             self.visit(node.args[0])
             self.exit_scope()
             node.type_name = 'void'
             return 'void'
        elif callee_name == 'realloc':
             self.enter_scope()
             self.visit(node.args[0])
             self.visit(node.args[1])
             self.exit_scope()
             node.type_name = 'u8*'
             return 'u8*'
        elif callee_name == 'panic':
            self.enter_scope()
            self.visit(node.args[0])
            self.exit_scope()
            node.type_name = 'void'
            return 'void'
        elif callee_name == 'assert':
            self.enter_scope()
            self.visit(node.args[0])
            self.visit(node.args[1])
            self.exit_scope()
            node.type_name = 'void'
            return 'void'
        elif callee_name == 'gpu::global_id':
             node.type_name = 'i32'
             return 'i32'
        elif callee_name == 'gpu::dispatch':
            node.type_name = 'void'
            return 'void'
        elif callee_name.startswith('cast<'):
             self.visit(node.args[0])
             return callee_name[5:-1]
        elif callee_name.startswith('sizeof<'):
             return 'i32'



        # Try resolving relative to current module
        if callee_name not in self.function_defs and \
           callee_name not in self.structs and \
           callee_name not in self.generic_structs and \
           self.current_function and getattr(self.current_function, 'module', ""):
             mod_prefix = self.current_function.module
             mangled = f"{mod_prefix}_{callee_name}"
             print(f"DEBUG: Trying resolve '{callee_name}' in '{mod_prefix}' -> '{mangled}'", flush=True)
             
             # Check functions
             if mangled in self.function_defs:
                  callee_name = mangled
                  node.callee = mangled
             # Check structs
             elif mangled in self.structs or mangled in self.generic_structs:
                  callee_name = mangled
                  node.callee = mangled

        # Generic Instantiation
        if '<' in callee_name:
             self.instantiate_generic_function(callee_name)

        # Regular Function Call
        if callee_name in self.function_defs:
            fn = self.function_defs[callee_name]
            self.check_privacy(fn, callee_name)
            if len(node.args) != len(fn.params):
                 raise Exception(f"Function identity '{callee_name}' expects {len(fn.params)} args, got {len(node.args)}")
            
            self.enter_scope()
            for i, arg in enumerate(node.args):
                 arg_ty = self.visit(arg)
                 _pname, ptype = fn.params[i]
                 if isinstance(arg, VariableExpr) and not self.is_copy_type(ptype):
                      self.move_var(arg.name)
            self.exit_scope()
            return fn.return_type

        # Struct Constructor
        if callee_name in self.structs or callee_name in self.generic_structs:
            if callee_name in self.struct_defs:
                self.check_privacy(self.struct_defs[callee_name], callee_name)
            elif callee_name in self.generic_structs:
                self.check_privacy(self.generic_structs[callee_name], callee_name)

            self.enter_scope()
            for arg in node.args:
                arg_ty = self.visit(arg)
                if isinstance(arg, VariableExpr) and not self.is_copy_type(arg_ty):
                    self.move_var(arg.name, arg)
            self.exit_scope()
            return callee_name  # In an impl<T>, this refers to the current generic type

        # Enum Variant
        if '::' in callee_name:
            parts = callee_name.rsplit('::', 1)
            enum_name = parts[0]
            variant_name = parts[1]
            if '<' in enum_name: self.instantiate_generic_type(enum_name)
            if enum_name in self.enums:
                 variants = self.enums[enum_name]
                 if variant_name not in variants: raise Exception(f"Enum has no variant '{variant_name}'")
                 payloads = variants[variant_name]
                 if len(node.args) != len(payloads): raise Exception("Enum arg mismatch")
                 self.enter_scope()
                 for i, arg in enumerate(node.args):
                      self.visit(arg)
                      if isinstance(arg, VariableExpr) and not self.is_copy_type(payloads[i]):
                           self.move_var(arg.name, arg)
                 self.exit_scope()
                 return enum_name

        # raise Exception(...) - Moved to end
        if node.callee == 'slice_from_array':
            # slice_from_array(&arr) -> Slice<T>
            if len(node.args) != 1:
                raise Exception("slice_from_array expects 1 arg: &arr where arr is [T:N]")
            arg_ty = self.visit(node.args[0])
            if not (isinstance(arg_ty, str) and arg_ty.endswith('*')):
                raise Exception("slice_from_array expects a pointer to array: use slice_from_array(&arr)")
            inner = arg_ty[:-1]
            if not (inner.startswith('[') and inner.endswith(']')):
                raise Exception("slice_from_array expects &arr where arr is [T:N]")
            content = inner[1:-1]
            elem_type, size_str = content.split(':', 1)
            elem_type = elem_type.strip()
            size = int(size_str.strip())

            slice_ty = f"Slice<{elem_type}>"
            self.instantiate_generic_type(slice_ty)
            node.type_name = slice_ty
            node.slice_elem_type = elem_type
            node.slice_len = size
            return slice_ty
        elif node.callee == 'memcpy':
            if len(node.args) != 3: raise Exception("memcpy requires 3 args")
            # Validation logic? dest and src should be pointers.
            self.enter_scope()
            dest_ty = self.visit(node.args[0])
            src_ty = self.visit(node.args[1])
            size_ty = self.visit(node.args[2])
            self.exit_scope()
            if size_ty != 'i32': raise Exception("memcpy size must be i32")
            node.type_name = 'void'
            return 'void'
            
        elif node.callee == 'malloc':
             if len(node.args) != 1: raise Exception("malloc takes 1 arg")
             self.enter_scope()
             size_type = self.visit(node.args[0])
             self.exit_scope()
             if size_type != 'i32': raise Exception("malloc expects i32")
             node.type_name = 'u8*'
             return 'u8*'
             
        elif node.callee == 'free':
             if len(node.args) != 1: raise Exception("free takes 1 arg")
             self.enter_scope()
             self.visit(node.args[0])
             self.exit_scope()
             node.type_name = 'void'
             return 'void'
             
        elif node.callee == 'realloc':
             if len(node.args) != 2: raise Exception("realloc takes 2 args")
             self.enter_scope()
             self.visit(node.args[0])
             self.visit(node.args[1])
             self.exit_scope()
             node.type_name = 'u8*'
             return 'u8*'
             
        elif node.callee == 'panic':
            if len(node.args) != 1: raise Exception("panic expects 1 arg")
            self.enter_scope()
            self.visit(node.args[0])
            self.exit_scope()
            node.type_name = 'void'
            return 'void'
            
        elif node.callee == 'assert':
            if len(node.args) != 2: raise Exception("assert expects 2 args")
            self.enter_scope()
            self.visit(node.args[0])
            self.visit(node.args[1])
            self.exit_scope()
            node.type_name = 'void'
            return 'void'
            
        elif node.callee == 'gpu::global_id':
             node.type_name = 'i32'
             return 'i32'

        elif node.callee == 'gpu::dispatch':
            # gpu::dispatch(kernel_fn, threads?)
            # This is a bootstrap/mock dispatch; the first arg is treated as a function reference.
            if len(node.args) not in (1, 2):
                raise Exception("gpu::dispatch expects 1 or 2 args: (kernel_fn, threads?)")

            fn_arg = node.args[0]
            if not isinstance(fn_arg, VariableExpr):
                raise Exception("gpu::dispatch first argument must be a function name (identifier)")

            # The function name is a reference, not a variable lookup.
            fn_name = fn_arg.name
            if fn_name not in self.functions and fn_name not in self.function_defs and fn_name not in self.generic_functions:
                raise Exception(f"Semantic Error: gpu::dispatch unknown function '{fn_name}'")

            # Optional threads argument
            if len(node.args) == 2:
                threads_ty = self.visit(node.args[1])
                if threads_ty != 'i32':
                    raise Exception(f"Type Error: gpu::dispatch threads must be i32, got '{threads_ty}'")

            return 'void'
             
        elif node.callee.startswith('cast<'):
             target_type = node.callee[5:-1]
             if len(node.args) != 1: raise Exception("cast takes 1 arg")
             self.enter_scope()
             self.visit(node.args[0])
             self.exit_scope()
             return target_type
             
        elif node.callee.startswith('sizeof<'):
             return 'i32'
             


        # Generic Instantiation if needed
        if '<' in node.callee:
             self.instantiate_generic_function(node.callee)
        
        # 1) Regular function call
        # Note: `self.functions` may include intrinsics; those returned above.
        if node.callee in self.function_defs:
            fn = self.function_defs[node.callee]
            # Visit and apply moves based on parameter types (by-value move-only).
            self.enter_scope()
            for i, arg in enumerate(node.args):
                arg_ty = self.visit(arg)
                if i < len(fn.params):
                    _pname, ptype = fn.params[i]
                    if isinstance(arg, VariableExpr) and not self.is_copy_type(ptype):
                        self.move_var(arg.name, arg)
            self.exit_scope()
            return self.function_defs[node.callee].return_type

        # 2) Struct constructor call (TypeName(...))
        if node.callee in self.structs:
            # Keep permissive in bootstrap: only validate arg expressions,
            # since field ordering/overloads are still evolving.
            self.enter_scope()
            for arg in node.args:
                arg_ty = self.visit(arg)
                if isinstance(arg, VariableExpr) and not self.is_copy_type(arg_ty):
                    self.move_var(arg.name, arg)
            self.exit_scope()
            return node.callee
            
        # Double Colon Logic (Enums or Static Methods)
        if '::' in callee_name:
            parts = callee_name.rsplit('::', 1)
            prefix = parts[0]
            suffix = parts[1]
            
            # 1. Try Enum Variant
            if '<' in prefix: self.instantiate_generic_type(prefix)
            
            if prefix in self.enums:
                 variants = self.enums[prefix]
                 if suffix not in variants:
                      raise Exception(f"Semantic Error: Enum '{prefix}' has no variant '{suffix}'")
                 
                 payloads = variants[suffix]
                 if len(node.args) != len(payloads):
                      raise Exception(f"Semantic Error: Variant '{callee_name}' expects {len(payloads)} arguments, got {len(node.args)}")
                 
                 self.enter_scope()
                 for i, arg in enumerate(node.args):
                      arg_type = self.visit(arg)
                      if arg_type != payloads[i]:
                           raise Exception(f"Type Error: Variant '{suffix}' arg {i} expected '{payloads[i]}', got '{arg_type}'")

                      # Move payload variables if the payload type is move-only
                      if isinstance(arg, VariableExpr) and not self.is_copy_type(payloads[i]):
                          self.move_var(arg.name, arg)
                 self.exit_scope()
                 return prefix
            
            # 2. Try Static Method (Struct_method)
            mangled = f"{prefix}_{suffix}"
            if mangled in self.function_defs:
                 fn = self.function_defs[mangled]
                 callee_name = mangled
                 node.callee = callee_name # Update for CodeGen
                 
                 if len(node.args) != len(fn.params):
                      raise Exception(f"Static Method '{callee_name}' expects {len(fn.params)} args, got {len(node.args)}")
                 
                 self.enter_scope()
                 for i, arg in enumerate(node.args):
                      arg_ty = self.visit(arg)
                      _pname, ptype = fn.params[i]
                      if isinstance(arg, VariableExpr) and not self.is_copy_type(ptype):
                           self.move_var(arg.name, arg)
                 self.exit_scope()
                 return fn.return_type
            
            # 3. Try Namespaced Struct Constructor
            if mangled in self.structs or mangled in self.generic_structs:
                if mangled in self.struct_defs:
                     self.check_privacy(self.struct_defs[mangled], mangled)
                elif mangled in self.generic_structs:
                     self.check_privacy(self.generic_structs[mangled], mangled)
                
                node.callee = mangled # Update AST
                
                self.enter_scope()
                for arg in node.args:
                    arg_ty = self.visit(arg)
                    if isinstance(arg, VariableExpr) and not self.is_copy_type(arg_ty):
                         self.move_var(arg.name, arg)
                self.exit_scope()
                return mangled
            
            raise Exception(f"Semantic Error: Unknown function or type '{callee_name}'")

        # Gather all functions
        candidates = list(self.functions) + list(self.function_defs.keys()) + list(self.generic_functions.keys())
        hint = None
        suggestion = self.suggest_name(node.callee, candidates)
        if suggestion:
             hint = f"Did you mean '{suggestion}'?"
        self.error(f"Semantic Error: Unknown function '{node.callee}'", node, hint=hint)

    def visit_IfStmt(self, node):
        cond_type = self.visit(node.condition)
        # In C-like, i32 is valid bool, but let's be strict if we had bool type.
        # For now, we allow i32 as condition.
        
        self.enter_scope()
        for stmt in node.then_branch:
            self.visit(stmt)
        self.exit_scope()
        
        if node.else_branch:
            self.enter_scope()
            for stmt in node.else_branch:
                self.visit(stmt)
            self.exit_scope()



    def visit_RegionStmt(self, node):
        # No debug print
        self.enter_scope()
        # Declare the region variable (Arena)
        self.declare_variable(node.name, 'Arena')
        
        for stmt in node.body:
            self.visit(stmt)
        self.exit_scope()

    def visit_ReturnStmt(self, node):
        if node.value:
            ret_ty = self.visit(node.value)
            if isinstance(node.value, VariableExpr) and not self.is_copy_type(ret_ty):
                self.move_var(node.value.name, node.value)
        # Should check against function return type

    def substitute_generics(self, node, mapping):
        # Recursive substitution of types in AST nodes
        if isinstance(node, list):
             for x in node: self.substitute_generics(x, mapping)
             return
             
        if isinstance(node, VarDecl):
             if node.type_name:
                 node.type_name = self.apply_submap(node.type_name, mapping)
             if node.initializer:
                 self.substitute_generics(node.initializer, mapping)
        
        elif isinstance(node, CallExpr):
             # Handle TurboFish or generic calls
             if '<' in node.callee:
                  # ID<T> -> ID<i32>
                  node.callee = self.apply_submap(node.callee, mapping)
             
             # Handle intrinsics
             if 'cast<' in node.callee or 'sizeof<' in node.callee or 'ptr_offset<' in node.callee:
                 node.callee = self.apply_submap(node.callee, mapping)

             for arg in node.args:
                  self.substitute_generics(arg, mapping)
                  
        elif isinstance(node, IfStmt):
             self.substitute_generics(node.condition, mapping)
             self.substitute_generics(node.then_branch, mapping)
             if node.else_branch: self.substitute_generics(node.else_branch, mapping)
             
        elif isinstance(node, WhileStmt):
             self.substitute_generics(node.condition, mapping)
             self.substitute_generics(node.body, mapping)
             
        elif isinstance(node, ReturnStmt):
             if node.value: self.substitute_generics(node.value, mapping)
             
        elif isinstance(node, Assignment):
             if isinstance(node.target, (UnaryExpr, IndexAccess, MemberAccess)):
                  self.substitute_generics(node.target, mapping) 
             self.substitute_generics(node.value, mapping)
             
        elif isinstance(node, BinaryExpr):
             self.substitute_generics(node.left, mapping)
             self.substitute_generics(node.right, mapping)
             
        elif isinstance(node, UnaryExpr):
             self.substitute_generics(node.operand, mapping)
             
        elif isinstance(node, MatchExpr):
             self.substitute_generics(node.value, mapping)
             for case in node.cases:
                 self.substitute_generics(case.body, mapping)
        
        elif isinstance(node, MemberAccess):
             self.substitute_generics(node.object, mapping)
             
        elif isinstance(node, IndexAccess):
             self.substitute_generics(node.object, mapping)
             self.substitute_generics(node.index, mapping)
             
    def apply_submap(self, type_str, mapping):
        if not type_str: return type_str
        
        # Pointer
        if type_str.endswith('*'):
             inner = type_str[:-1]
             return self.apply_submap(inner, mapping) + '*'
             
        # Generic: Name<Args>
        if '<' in type_str and type_str.endswith('>'):
             base = type_str[:type_str.find('<')]
             inside = type_str[type_str.find('<')+1:-1]
             args = [x.strip() for x in inside.split(',')]
             new_args = [self.apply_submap(a, mapping) for a in args]
             return f"{base}<{','.join(new_args)}>"
             
        # Base type
        if type_str in mapping:
             return mapping[type_str]
             
        return type_str

    def visit_BlockStmt(self, node):
        self.enter_scope()
        for stmt in node.stmts:
             self.visit(stmt)
        self.exit_scope()
