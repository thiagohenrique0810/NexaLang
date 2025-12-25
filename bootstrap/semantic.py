from lexer import Lexer
from parser import Parser, FunctionDef, StructDef, EnumDef, MatchExpr, CaseArm, VarDecl, IfStmt, WhileStmt
import sys
import copy
class SemanticAnalyzer:
    def __init__(self):
        self.scopes = [{}] # List of dictionaries {name: {'type': type, 'moved': bool}}
        self.current_function = None

    def analyze(self, ast):
        self.ast_root = ast
        # 1. Collect function and struct names
        self.functions = set(['print', 'gpu::global_id', 'panic', 'assert'])
        self.function_defs = {} # name -> FunctionDef
        self.structs = {} # name -> {field: type}
        self.structs = {} # name -> {field: type}
        self.structs = {} # name -> {field: type}
        self.enums = {} # name -> {variant: [payload_types]}
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
            elif isinstance(node, EnumDef):
                 if node.generics:
                     self.generic_enums[node.name] = node
                 else:
                     # Variants: list of (name, payload_types)
                     variants = {vname: payloads for vname, payloads in node.variants}
                     self.enums[node.name] = variants

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

    def visit_MemberAccess(self, node):
        # 1. Evaluate object
        obj_type = self.visit(node.object)
        
        # 2. Check if object is a struct
        if obj_type not in self.structs:
             raise Exception(f"Type Error: access member '{node.member}' on non-struct type '{obj_type}'")
        
        # 3. Check if member exists
        fields = self.structs[obj_type]
        if node.member not in fields:
             raise Exception(f"Type Error: Struct '{obj_type}' has no member '{node.member}'")
        
        # Annotate node for CodeGen
        node.struct_type = obj_type
             
        return fields[node.member]

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
        
        # 2. Verify it is an array
        if not (obj_type.startswith('[') and obj_type.endswith(']')):
             raise Exception(f"Type Error: Indexing non-array type '{obj_type}'")
        
        # Parse type string [T:N]
        content = obj_type[1:-1]
        elem_type, size_str = content.split(':')
        
        # 3. Analyze index
        index_type = self.visit(node.index)
        if index_type != 'i32':
             raise Exception(f"Type Error: Array index must be i32, got '{index_type}'")
             
        return elem_type

    def visit_MatchExpr(self, node):
        # 1. Analyze value
        val_type = self.visit(node.value)
        if val_type not in self.enums:
             raise Exception(f"Semantic Error: Match on non-enum type '{val_type}'")
             
        # Annotate for CodeGen
        node.enum_name = val_type
        
        enum_variants = self.enums[val_type]
        
        # 2. Check coverage (basic)
        covered = set()
        
        for case in node.cases:
            if case.variant_name not in enum_variants:
                 raise Exception(f"Semantic Error: Enum '{val_type}' has no variant '{case.variant_name}'")
            
            covered.add(case.variant_name)
            
            # Check payload count
            expected_payloads = enum_variants[case.variant_name]
            # Case has 0 or 1 variable binding for now (we parsed var_name as single string)
            # If enum has payload, we must bind it? Or can ignore?
            # Creating scope for case body
            self.scopes.append({})
            
            if case.var_name:
                if len(expected_payloads) == 0:
                     raise Exception(f"Semantic Error: Variant '{case.variant_name}' has no payload, but matched with variable '{case.var_name}'")
                elif len(expected_payloads) > 1:
                     # Nested tuple unpacking not yet supported
                     pass 
                
                # Register bound variable
                # Assuming single payload for now
                payload_type = expected_payloads[0]
                self.scopes[-1][case.var_name] = {'type': payload_type, 'moved': False}
            
            # Visit body
            self.visit(case.body)
            
            self.scopes.pop()
            
        # Check exhaustive (simple check)
        if len(covered) != len(enum_variants):
             missing = set(enum_variants.keys()) - covered
             raise Exception(f"Semantic Error: Match not exhaustive. Missing: {missing}")

    def enter_scope(self):
        self.scopes.append({})

    def exit_scope(self):
        self.scopes.pop()

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method in SemanticAnalyzer")

    def instantiate_generic_type(self, name_with_args):
        if '<' not in name_with_args: return

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
            # Substitute T
            # Simple substitution for now. Nested logic needed for Box<Box<T>>
            # For Phase 2.9, simplistic replacement
            if ftype in mapping:
                new_type = mapping[ftype]
            else:
                new_type = ftype
            new_fields.append((fname, new_type))
            
        # Register new concrete struct
        self.structs[name_with_args] = dict(new_fields)
        
        # We also need to add a StructDef to AST for CodeGen?
        # Yes, CodeGen iterates AST.
        # But analyze() modifies `self.structs` which Semantic uses.
        # CodeGen uses `codegen.struct_types` populated by `visit_StructDef`.
        # CodeGen NEEDS `visit_StructDef` called for "Box<i32>".
        # So we must create a new StructDef node and append to AST or visit it manually?
        # CodeGen logic runs AFTER Semantic.
        # So if we append to AST, CodeGen will see it.
        # But `self.analyze(ast)` doesn't easily return new nodes to append.
        # We can append to `self.ast` if we stored it?
        # Or `node.parent`? No parent links.
        # Hack: SemanticAnalyzer can have `generated_nodes`. CodeGen main loop can be updated?
        # Or better: `semantic.analyze` returns `augmented_ast`?
        # `main.py`: `semantic.analyze(ast)` -> `codegen.generate(ast)`.
        # If `semantic` modifies `ast` (it's a list), it works.
        # So: self.ast_root.append(...)
        
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
                 if ptype in mapping:
                      new_payloads.append(mapping[ptype])
                 else:
                      new_payloads.append(ptype)
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
             # Might be struct instantiation? Code below handles that.
             # Or error?
             # Let's check generic functions.
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
        
        # Monomorphize Function
        # Need to deep copy body?
        # For simplicity, we just clone AST nodes if possible or re-create them.
        # Python shallow copy might not be enough if we modify types in place.
        # We need to substitute types in params, return type, and BODY VAR DECLS.
        # This is recursive substitution. Complex.
        # Alternative: Re-parse? No source.
        # Alternative: Helper `substitute_types(node, mapping)`.
        
        new_params = []
        for pname, ptype in def_node.params:
            if ptype in mapping:
                new_params.append((pname, mapping[ptype]))
            else:
                new_params.append((pname, ptype))
                
        new_return = def_node.return_type
        if new_return in mapping:
            new_return = mapping[new_return]
            
        # Body substitution
        # We need a deep copy of body.
        # Let's use `copy` module?
        import copy
        new_body = copy.deepcopy(def_node.body)
        
        # Now traverse new_body and replace types.
        # We need a visitor for substitution?
        # Or just rely on Type Inference!
        # If we replace declaration types, inference handles the rest?
        # `let x: T = ...` -> `let x: i32 = ...`.
        # `VarDecl.type_name` needs change.
        # `CallExpr`? If inside we call `id<T>(...)`, it becomes `id<i32>(...)`.
        # Complex.
        # Phase 2.9 goal is simple Generics.
        # Let's assume naive substitution on `VarDecl` in body is enough for now?
        # And `match` arms?
        
        self.substitute_generics(new_body, mapping)
        
        new_func = FunctionDef(name_with_args, new_params, new_return, new_body, def_node.is_kernel)
        self.ast_root.append(new_func)
        self.functions.add(name_with_args)
        self.function_defs[name_with_args] = new_func

    def substitute_generics(self, nodes, mapping):
        # Helper to recursively substitute types in body
        for node in nodes:
            if isinstance(node, VarDecl):
                if node.type_name in mapping:
                    node.type_name = mapping[node.type_name]
                # Also handle generic structs: Box<T> -> Box<i32>
                if node.type_name and '<' in node.type_name and any(k in node.type_name for k in mapping):
                     # Naive string replacement for simple cases
                     for k, v in mapping.items():
                         node.type_name = node.type_name.replace(f"<{k}>", f"<{v}>")
                         # Also handle Box<T> -> Box<i32> check split
                         # This is crude. Ideally we parse type string.
                         
            elif isinstance(node, IfStmt):
                self.substitute_generics(node.then_block, mapping)
                if node.else_block:
                    self.substitute_generics(node.else_block, mapping)
            elif isinstance(node, WhileStmt):
                self.substitute_generics(node.body, mapping)

    def declare(self, name, type_name):
        if name in self.scopes[-1]:
            raise Exception(f"Semantic Error: Variable '{name}' already declared in this scope.")
        self.scopes[-1][name] = {'type': type_name, 'moved': False}

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.enter_scope()
        
        # Register parameters
        for pname, ptype in node.params:
             if '<' in ptype:
                  self.instantiate_generic_type(ptype)
             self.declare(pname, ptype)
             
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

        if init_type != node.type_name:
             raise Exception(f"Type Error: Cannot assign type '{init_type}' to variable '{node.name}' of type '{node.type_name}'")
        
        # 3. Declare variable
        self.declare(node.name, node.type_name)

    def visit_Assignment(self, node):
        # 1. Check if variable exists
        var_info = self.lookup(node.name)
        if not var_info:
            raise Exception(f"Semantic Error: Assignment to undefined variable '{node.name}'")
        
        # NOTE: Assigning to a moved variable is valid (re-initialization).
        # We just need to ensure the variable is considered 'alive' (not moved) after this.
            
        # 2. Analyze value
        val_type = self.visit(node.value)
        
        # 3. Check type mismatch
        if var_info['type'] != val_type:
            raise Exception(f"Type Error: Cannot assign type '{val_type}' to variable '{node.name}' of type '{var_info['type']}'")
            
        # Revive variable
        var_info['moved'] = False

    def visit_BinaryExpr(self, node):
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        
        if left_type != right_type:
             raise Exception(f"Type Error: Operands must be same type. Got '{left_type}' and '{right_type}'")
             
        # For now, everything is i32
        if left_type != 'i32':
             raise Exception(f"Type Error: Arithmetic only supported for i32. Got '{left_type}'")
             
        return 'i32' # Result type

    def visit_IntegerLiteral(self, node):
        return 'i32'

    def visit_StringLiteral(self, node):
        return 'string'

    def visit_VariableExpr(self, node):
        var_info = self.lookup(node.name)
        if not var_info:
             raise Exception(f"Semantic Error: Undefined variable '{node.name}'")
             
        if var_info['moved']:
            raise Exception(f"Ownership Error: Use of moved variable '{node.name}'")
            
        return var_info['type']

    def visit_CallExpr(self, node):
        if '<' in node.callee:
             self.instantiate_generic_function(node.callee)
             
        if node.callee in self.functions:
            if node.callee == 'print':
                for arg in node.args:
                    self.visit(arg)
                return 'void'
            elif node.callee == 'panic':
                if len(node.args) != 1:
                    raise Exception("Semantic Error: panic() expects 1 argument (message)")
                self.visit(node.args[0])
                return 'void'
            elif node.callee == 'assert':
                if len(node.args) != 2:
                    raise Exception("Semantic Error: assert() expects 2 arguments (condition, message)")
                cond_type = self.visit(node.args[0])
                # We don't have explicit bool type in semantic yet? visit_BinaryExpr returns 'i32'.
                # Let's assume i32 for now (0=false, !0=true)
                self.visit(node.args[1])
                return 'void'
            elif node.callee == 'gpu::global_id':
                return 'i32'
            else:
                 # User function call
                 for arg in node.args:
                     self.visit(arg)
                 
                 # Look up return type
                 if node.callee in self.function_defs:
                     return self.function_defs[node.callee].return_type
                 else:
                     return 'void' # Should not happen if in self.functions
        elif node.callee in self.structs:
            # Struct Instantiation
            fields = self.structs[node.callee]
            if len(node.args) != len(fields):
                raise Exception(f"Semantic Error: Struct '{node.callee}' expects {len(fields)} arguments, got {len(node.args)}")
            
            for arg in node.args:
                self.visit(arg)
            return node.callee
            
        # Check for Enum Variant: Enum::Variant
        elif '::' in node.callee:
            parts = node.callee.rsplit('::', 1)
            enum_name = parts[0]
            variant_name = parts[1]
            
            # Instantiate generic enum if needed
            if '<' in enum_name:
                self.instantiate_generic_type(enum_name)
            
            if enum_name in self.enums:
                 variants = self.enums[enum_name]
                 if variant_name not in variants:
                      raise Exception(f"Semantic Error: Enum '{enum_name}' has no variant '{variant_name}'")
                 
                 payloads = variants[variant_name]
                 if len(node.args) != len(payloads):
                      raise Exception(f"Semantic Error: Variant '{node.callee}' expects {len(payloads)} arguments, got {len(node.args)}")
                 
                 for i, arg in enumerate(node.args):
                      arg_type = self.visit(arg)
                      if arg_type != payloads[i]:
                           raise Exception(f"Type Error: Variant '{variant_name}' arg {i} expected '{payloads[i]}', got '{arg_type}'")
                 
                 return enum_name
            else:
                 raise Exception(f"Semantic Error: Unknown function or type '{node.callee}'")

        raise Exception(f"Semantic Error: Unknown function '{node.callee}'")

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

    def visit_WhileStmt(self, node):
        self.visit(node.condition)
        self.enter_scope()
        for stmt in node.body:
            self.visit(stmt)
        self.exit_scope()

    def visit_MatchExpr(self, node):
        expr_type = self.visit(node.value)
        
        if '<' in expr_type:
             self.instantiate_generic_type(expr_type)
             
        if expr_type not in self.enums:
            raise Exception(f"Type Error: Match expression must be an Enum, got '{expr_type}'")
            
        node.enum_name = expr_type
        variants = self.enums[expr_type]
        
        for case in node.cases:
            if case.variant_name not in variants:
                raise Exception(f"Semantic Error: Enum '{expr_type}' has no variant '{case.variant_name}'")
                
            payloads = variants[case.variant_name]
            
            # Check pattern args? 
            # Current Match Case syntax: Variant(var)
            # We assume single variable binding for payload?
            # Or if payload is empty, no variable.
            
            if payloads:
                 # Expecting variable binding
                 if not case.var_name:
                      # If payload exists but no var bound, maybe allow? (ignore payload)
                      # For now, require binding if payload exists?
                      # Or maybe syntax is just Variant without parens?
                      # Lexer/Parser: Variant or Variant(var)
                      pass
                      
                 if case.var_name:
                      # Register variable in case scope
                      # Enter scope for case
                      # But semantic analyzer visit structure for case body?
                      # We need to wrap body visit in scope
                      pass
                      
            else:
                 if case.var_name:
                      raise Exception(f"Type Error: Variant '{case.variant_name}' has no payload, but variable '{case.var_name}' bound")

            # Check Duplicate Scope?
            # We need to visit body with Bound Variable
            self.enter_scope()
            if case.var_name:
                 # Assume single payload for now
                 if len(payloads) != 1:
                      # TODO: logic for multiple payloads or struct payloads
                      pass
                 self.declare(case.var_name, payloads[0])
            
            self.visit(case.body)
            self.exit_scope()

    def visit_ReturnStmt(self, node):
        self.visit(node.value)
        # Should check against function return type
