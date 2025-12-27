from lexer import Lexer
from parser import Parser, FunctionDef, StructDef, EnumDef, MatchExpr, CaseArm, ArrayLiteral, IndexAccess, UnaryExpr, VariableExpr, IfStmt, WhileStmt, VarDecl, Assignment, CallExpr, MemberAccess, ReturnStmt, BinaryExpr
import sys
import copy
class SemanticAnalyzer:
    def __init__(self):
        # Symbol table entries: {name: {'type': type, 'moved': bool, 'readers': int, 'writer': bool, 'is_ref': bool}}
        self.scopes = [{}] 
        self.borrow_cleanup_stack = [] # Stack of lists of (borrowed_var_name, is_mut)
        self.current_function = None

    def analyze(self, ast):
        self.ast_root = ast
        # 1. Collect function and struct names
        self.functions = set(['print', 'gpu::global_id', 'panic', 'assert'])
        self.function_defs = {} # name -> FunctionDef
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
             raise Exception(f"Type Error: Array index must be i32, got {index_type}")
             
        base_elem_type = obj_type[1:].split(':')[0] # Use obj_type instead of undefined array_type
        return base_elem_type

    def visit_UnaryExpr(self, node):
        operand_type = self.visit(node.operand)
        
        if node.op == '*':
            # Dereference: operand must be a pointer type "T*"
            if not operand_type.endswith('*'):
                 raise Exception(f"Type Error: Cannot dereference non-pointer type '{operand_type}'")
            return operand_type[:-1]
            
        elif node.op == '&':
            # Address Of: return "T*"
            if isinstance(node.operand, VariableExpr):
                var_name = node.operand.name
                var_info = self.lookup(var_name)
                if not var_info:
                     raise Exception(f"Semantic Error: Borrow of undefined variable '{var_name}'")
                
                # Check Borrow Rules
                try:
                    if var_info['writer']:
                         raise Exception(f"Borrow Error: Cannot borrow '{var_name}' as immutable because it is already borrowed as mutable")
                except KeyError:
                    print(f"DEBUG: KeyError accessing 'writer' for '{var_name}'. Info: {var_info}")
                    raise
                
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
                
            return f"{operand_type}*"
            
        return operand_type

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
                self.declare(case.var_name, payload_type)
            
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

    def declare(self, name, type_name):
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
        print(f"DEBUG: Instantiated '{name_with_args}' with return type '{new_return}'")
        self.ast_root.append(new_func)
        self.functions.add(name_with_args)
        self.function_defs[name_with_args] = new_func
        print(f"DEBUG: Key '{name_with_args}' added to function_defs with type '{new_return}'")


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
        if node.generics:
             # Skip body analysis for generic definitions.
             # They are analyzed only when instantiated.
             self.functions.add(node.name)
             self.generic_functions[node.name] = node
             return

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
        
        print(f"DEBUG: VarDecl '{node.name}' type='{node.type_name}' init_type='{init_type}'")

        if init_type != node.type_name:
             raise Exception(f"DEBUG: init_type='{init_type}', node.type='{node.type_name}'. Type Error: Cannot assign type '{init_type}' to variable '{node.name}' of type '{node.type_name}'")
        
        # 3. Declare variable
        self.declare(node.name, node.type_name)

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
             if var_info['writer']:
                  raise Exception(f"Borrow Error: Cannot assign to '{name}' because it is borrowed mutable")

             target_type = var_info['type']
             
             # Check type mismatch
             if target_type != value_type:
                  # Allow explicit pointer assignment?
                  pass
             if target_type != value_type:
                  raise Exception(f"Type Error: Cannot assign type '{value_type}' to variable '{name}' of type '{target_type}'")
                  
             # Revive variable
             var_info['moved'] = False
             
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
              
        elif isinstance(node.target, MemberAccess):
             # Assign to struct field
             # Visit MemberAccess (which checks validity and returns field type)
             # Note: visit_MemberAccess evaluates the OBJECT and checks field existence.
             target_type = self.visit(node.target)
             
             if target_type != value_type:
                  raise Exception(f"Type Error: Cannot assign '{value_type}' to member of type '{target_type}'")
        else:
             raise Exception("Invalid assignment target")

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
        return 'i32' # Default

    def visit_BooleanLiteral(self, node):
        return 'bool'

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
        # Intrinsics
        if node.callee == 'print':
            for arg in node.args:
                self.visit(arg)
            return 'void'
        elif node.callee == 'memcpy':
            if len(node.args) != 3: raise Exception("memcpy requires 3 args")
            # Validation logic? dest and src should be pointers.
            dest_ty = self.visit(node.args[0])
            src_ty = self.visit(node.args[1])
            size_ty = self.visit(node.args[2])
            if size_ty != 'i32': raise Exception("memcpy size must be i32")
            return 'void'
            
        elif node.callee == 'malloc':
             if len(node.args) != 1: raise Exception("malloc takes 1 arg")
             size_type = self.visit(node.args[0])
             if size_type != 'i32': raise Exception("malloc expects i32")
             return 'u8*'
             
        elif node.callee == 'free':
             if len(node.args) != 1: raise Exception("free takes 1 arg")
             self.visit(node.args[0])
             return 'void'
             
        elif node.callee == 'realloc':
             if len(node.args) != 2: raise Exception("realloc takes 2 args")
             self.visit(node.args[0])
             self.visit(node.args[1])
             return 'u8*'
             
        elif node.callee == 'panic':
            if len(node.args) != 1: raise Exception("panic expects 1 arg")
            self.visit(node.args[0])
            return 'void'
            
        elif node.callee == 'assert':
            if len(node.args) != 2: raise Exception("assert expects 2 args")
            self.visit(node.args[0])
            self.visit(node.args[1])
            return 'void'
            
        elif node.callee == 'gpu::global_id':
             return 'i32'
             
        elif node.callee.startswith('cast<'):
             target_type = node.callee[5:-1]
             if len(node.args) != 1: raise Exception("cast takes 1 arg")
             self.visit(node.args[0])
             return target_type
             
        elif node.callee.startswith('sizeof<'):
             return 'i32'
             
        elif node.callee.startswith('ptr_offset<'):
             target_type = node.callee[11:-1]
             if len(node.args) != 2: raise Exception("ptr_offset expects 2 args")
             self.visit(node.args[0])
             self.visit(node.args[1])
             return f"{target_type}*"

        # Generic Instantiation if needed
        if '<' in node.callee:
             self.instantiate_generic_function(node.callee)
        
        # Standard Function Call
        if node.callee in self.functions:
             for arg in node.args:
                 self.visit(arg)
                 
             if node.callee in self.function_defs:
                 return self.function_defs[node.callee].return_type
             else:
                 return 'STD_MISSING'
                 
        elif node.callee in self.structs:
             for arg in node.args:
                 self.visit(arg)
                 
             if node.callee in self.function_defs:
                 return self.function_defs[node.callee].return_type
             else:
                 return 'STD_MISSING'
                 
        elif node.callee in self.structs:
            # Struct Instantiation
            fields = self.structs[node.callee]
            if len(node.args) != len(fields):
                raise Exception(f"Semantic Error: Struct '{node.callee}' expects {len(fields)} arguments, got {len(node.args)}")
            
            for arg in node.args:
                self.visit(arg)
            return node.callee
            
        # Enum Variant: Enum::Variant
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
            
            if payloads:
                 # Expecting variable binding
                 if not case.var_name:
                      pass
                      
                 if case.var_name:
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
        if node.value: self.visit(node.value)
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
