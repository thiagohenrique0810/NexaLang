from llvmlite import ir
from n_parser import StructDef, EnumDef, ImplDef, FunctionDef, VariableExpr, UnaryExpr, MemberAccess, MethodCall, FloatLiteral, IndexAccess, CharLiteral, ExternBlock

class CodeGen:
    def __init__(
        self,
        target: str = "native",
        emit_kernels_only: bool = False,
        spirv_env: str = "opencl",
        spirv_local_size: str = "1,1,1",
    ):
        self.module = ir.Module(name="nexalang_module")
        self.target = target
        self.emit_kernels_only = emit_kernels_only
        self.spirv_env = spirv_env
        self.spirv_local_size = spirv_local_size
        self._kernel_function_names = set()
        self._vulkan_kernel_arg_globals = {}  # (kernel_name, arg_name) -> ir.GlobalVariable
        self._vulkan_buffer_args = {}  # (kernel_name, arg_name) -> (data_gv, len_gv)
        self._current_function_name = None
        self._current_function_is_kernel = False
        # Set a more appropriate module triple for SPIR(-V) translation paths.
        # Note: actual .spv emission depends on external tooling (llvm-spirv).
        if self.target == "spirv":
            if self.spirv_env == "vulkan":
                self.module.triple = "spirv64-unknown-vulkan"
            else:
                self.module.triple = "spirv64-unknown-unknown"
        self.builder = None
        self.printf = None
        self.exit_func = None
        self.malloc = None
        self.free = None
        self.realloc = None
        self.memcpy = None
        self.fopen = None
        self.fseek = None
        self.ftell = None
        self.fread = None
        self.fclose = None
        self.fwrite = None

        # Core compiler state
        self.struct_types = {} # name -> ir.LiteralStructType
        self.struct_fields = {}
        self._current_generics = [] # name -> {field_name: index}
        self.enum_types = {} # name -> {variant: tag_id}
        self.enum_payloads = {} # name -> {variant: payload_type}
        self.enum_definitions = {} # name -> (ir_struct_type, payload_size)
        self.scopes = []
        
        self._declare_intrinsics()
        self.loop_stack = [] # Stack of (continue_block, break_block, scope_depth, label)

        # For Vulkan SPIR-V kernels-only, avoid emitting host/runtime helpers
        # (Arena/malloc/printf) because they pull in pointer ops that are not shader-friendly.
        if self.target == "spirv" and self.spirv_env == "vulkan" and self.emit_kernels_only:
            self._declare_gpu_state()
        else:
            self._declare_printf()
            self._declare_exit()
            self._declare_malloc_free()
            self._declare_memcpy()
            self._declare_fileio()
            self._declare_gpu_state()
            self._declare_arena()

    def _declare_fileio(self):
        # Minimal libc FILE* I/O for self-hosting bootstrap helpers.
        # Treat FILE* as i8* (opaque).
        void_ptr = ir.IntType(8).as_pointer()
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)

        # FILE* fopen(const char* path, const char* mode)
        self.fopen = ir.Function(self.module, ir.FunctionType(void_ptr, [void_ptr, void_ptr]), name="fopen")
        # int fseek(FILE* stream, long offset, int whence)
        self.fseek = ir.Function(self.module, ir.FunctionType(i32, [void_ptr, i64, i32]), name="fseek")
        # long ftell(FILE* stream)
        self.ftell = ir.Function(self.module, ir.FunctionType(i64, [void_ptr]), name="ftell")
        # size_t fread(void* ptr, size_t size, size_t nmemb, FILE* stream)
        self.fread = ir.Function(self.module, ir.FunctionType(i64, [void_ptr, i64, i64, void_ptr]), name="fread")
        # size_t fwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream)
        self.fwrite = ir.Function(self.module, ir.FunctionType(i64, [void_ptr, i64, i64, void_ptr]), name="fwrite")
        # int fclose(FILE* stream)
        self.fclose = ir.Function(self.module, ir.FunctionType(i32, [void_ptr]), name="fclose")

    def _get_or_declare_vulkan_kernel_arg_global(self, kernel_name: str, arg_name: str, arg_type_name: str) -> ir.GlobalVariable:
        key = (kernel_name, arg_name)
        if key in self._vulkan_kernel_arg_globals:
            return self._vulkan_kernel_arg_globals[key]

        llvm_ty = self.get_llvm_type(arg_type_name)
        # For Vulkan, use StorageBuffer addrspace(11) for Buffer<T> so pointers are logical under VariablePointersStorageBuffer.
        # (addrspace mapping confirmed by experiment with llc spirv64-unknown-vulkan)
        addrspace = 11 if (isinstance(arg_type_name, str) and arg_type_name.startswith("Buffer<")) else 1
        gv = ir.GlobalVariable(self.module, llvm_ty, name=f"__nexa_{kernel_name}_{arg_name}", addrspace=addrspace)
        gv.linkage = "external"
        self._vulkan_kernel_arg_globals[key] = gv
        return gv

    def _get_or_declare_vulkan_buffer_globals(self, kernel_name: str, arg_name: str, buf_type_name: str):
        """
        Vulkan logical pointers can't be loaded as values from memory. So for Buffer<T> we model:
        - data: external addrspace(11) global <T>   (StorageBuffer)
        - len:  external addrspace(12) global i32   (Uniform)
        """
        key = (kernel_name, arg_name)
        if key in self._vulkan_buffer_args:
            return self._vulkan_buffer_args[key]

        # Parse Buffer<T>
        inner = buf_type_name[len("Buffer<"):-1]
        elem_ty = self.get_llvm_type(inner)
        # Use [0 x T] in StorageBuffer so the backend emits a runtime array + ArrayStride.
        data_ty = ir.ArrayType(elem_ty, 0)
        data_gv = ir.GlobalVariable(self.module, data_ty, name=f"__nexa_{kernel_name}_{arg_name}_data", addrspace=11)
        data_gv.linkage = "external"

        len_gv = ir.GlobalVariable(self.module, ir.IntType(32), name=f"__nexa_{kernel_name}_{arg_name}_len", addrspace=12)
        len_gv.linkage = "external"

        self._vulkan_buffer_args[key] = (data_gv, len_gv)
        return data_gv, len_gv

    def _postprocess_spirv_vulkan_kernel_attributes(self, llvm_ir: str) -> str:
        """
        llvmlite does not currently support arbitrary key/value function attributes, but
        LLVM's SPIR-V Vulkan path expects at least:
        - "hlsl.shader"="compute"
        - "hlsl.numthreads"="X,Y,Z"
        We inject an attribute group into the textual LLVM IR and attach it to kernel functions.
        """
        if not self._kernel_function_names:
            return llvm_ir

        import re

        # Find next free attribute group id.
        max_attr = -1
        for m in re.finditer(r"^attributes\s+#(\d+)\s*=\s*\{", llvm_ir, flags=re.MULTILINE):
            max_attr = max(max_attr, int(m.group(1)))
        attr_id = max_attr + 1

        attr_block = f'attributes #{attr_id} = {{ "hlsl.shader"="compute" "hlsl.numthreads"="{self.spirv_local_size}" }}'

        # Attach attribute group to each kernel function definition line if it doesn't already have one.
        # Handles both @name and @"name" forms.
        lines = llvm_ir.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith("define "):
                continue
            # Skip if already has an attribute group like "#0"
            if re.search(r"\s#\d+\s*(\{|$)", line):
                continue
            for kname in self._kernel_function_names:
                if (f'@{kname}(' in line) or (f'@"{kname}"(' in line):
                    # llvmlite often prints the opening brace on the next line, so we attach
                    # the attribute group to the `define ...` line itself.
                    lines[i] = f"{line} #{attr_id}"
                    break

        out = "\n".join(lines)
        if not out.endswith("\n"):
            out += "\n"
        out += "\n" + attr_block + "\n"
        return out

    def _declare_gpu_state(self):
        # Mock GPU execution state for bootstrap runtime dispatch (CPU loop).
        # When target=spirv, we also provide a placeholder builtin global that
        # external SPIR-V translation tools can optionally map.
        i32 = ir.IntType(32)
        # Only create the CPU-mock global when we are not emitting kernels-only SPIR-V.
        if not (self.target == "spirv" and self.emit_kernels_only):
            gv = ir.GlobalVariable(self.module, i32, name="__gpu_global_id")
            gv.linkage = "internal"
            gv.initializer = ir.Constant(i32, 0)
            self.gpu_global_id = gv

        # SPIR-V builtin (LLVM SPIR-V backend convention):
        # Declare `__spirv_BuiltInGlobalInvocationId` as v3i32 in addrspace(5).
        # LLVM will decorate it as BuiltIn GlobalInvocationId and map to a suitable memory class.
        if self.target == "spirv":
            v3i32 = ir.VectorType(i32, 3)
            spirv_gv = ir.GlobalVariable(self.module, v3i32, name="__spirv_BuiltInGlobalInvocationId", addrspace=5)
            spirv_gv.linkage = "external"
            self.spirv_global_invocation_id = spirv_gv

    def _infer_type_name_from_llvm(self, ty):
        # Best-effort mapping for local scope bookkeeping (mainly for drop calls).
        # Keep small and conservative for bootstrap.
        if isinstance(ty, ir.IntType):
            if ty.width == 32:
                return "i32"
            if ty.width == 64:
                return "i64"
            if ty.width == 8:
                return "u8"
        if isinstance(ty, ir.PointerType):
            # Treat i8* as "string" in this bootstrap (also used for raw bytes).
            if isinstance(ty.pointee, ir.IntType) and ty.pointee.width == 8:
                return "string"
            # Fallback pointer
            return "void*"
        return "STD_MISSING"

    def _declare_intrinsics(self):
        # ... existing intrinsics ...
        void_ptr = ir.IntType(8).as_pointer()
        i32 = ir.IntType(32)
        i1 = ir.IntType(1)
        i64 = ir.IntType(64)

        # External Coroutine Hooks (only if not already there)
        if "__nexa_resume" not in self.module.globals:
            ir.Function(self.module, ir.FunctionType(ir.VoidType(), [void_ptr]), name="__nexa_resume")
        if "__nexa_is_done" not in self.module.globals:
            ir.Function(self.module, ir.FunctionType(i1, [void_ptr]), name="__nexa_is_done")
        if "__nexa_destroy" not in self.module.globals:
            ir.Function(self.module, ir.FunctionType(ir.VoidType(), [void_ptr]), name="__nexa_destroy")

        # LLVM Coroutine Intrinsics
        token_ty = ir.IntType(8).as_pointer() # Placeholder for TokenType
        self.coro_id = ir.Function(self.module, ir.FunctionType(token_ty, [i32, void_ptr, void_ptr, void_ptr]), name="llvm.coro.id")
        self.coro_size = ir.Function(self.module, ir.FunctionType(i32, []), name="llvm.coro.size.i32")
        self.coro_begin = ir.Function(self.module, ir.FunctionType(void_ptr, [token_ty, void_ptr]), name="llvm.coro.begin")
        self.coro_free = ir.Function(self.module, ir.FunctionType(void_ptr, [token_ty, void_ptr]), name="llvm.coro.free")
        self.coro_end = ir.Function(self.module, ir.FunctionType(i1, [void_ptr, i1]), name="llvm.coro.end")
        self.coro_suspend = ir.Function(self.module, ir.FunctionType(ir.IntType(8), [token_ty, i1]), name="llvm.coro.suspend")
        self.coro_resume = ir.Function(self.module, ir.FunctionType(ir.VoidType(), [void_ptr]), name="llvm.coro.resume")
        self.coro_destroy = ir.Function(self.module, ir.FunctionType(ir.VoidType(), [void_ptr]), name="llvm.coro.destroy")
        self.coro_done = ir.Function(self.module, ir.FunctionType(i1, [void_ptr]), name="llvm.coro.done")
        self.coro_promise = ir.Function(self.module, ir.FunctionType(void_ptr, [void_ptr, i32, i1]), name="llvm.coro.promise")

    def _declare_arena(self):
        # struct Arena { chunk: i8*, offset: i32, capacity: i32 }
        void_ptr = ir.IntType(8).as_pointer()
        i32 = ir.IntType(32)
        arena_ty = self.module.context.get_identified_type("Arena")
        arena_ty.set_body(void_ptr, i32, i32)
        self.struct_types['Arena'] = arena_ty
        self.struct_fields['Arena'] = {'chunk': 0, 'offset': 1, 'capacity': 2}
        self._define_arena_methods()

    def _define_arena_methods(self):
        void_ptr = ir.IntType(8).as_pointer()
        i32 = ir.IntType(32)
        arena_ty = self.struct_types['Arena']

        # 1. Arena_new() -> Arena
        new_ty = ir.FunctionType(arena_ty, [])
        new_func = ir.Function(self.module, new_ty, name="Arena_new")
        bb = new_func.append_basic_block("entry")
        b = ir.IRBuilder(bb)

        size = ir.Constant(i32, 4096) # Default 4KB
        chunk = b.call(self.malloc, [size])

        # Build struct
        val = ir.Constant(arena_ty, ir.Undefined)
        val = b.insert_value(val, chunk, 0)
        val = b.insert_value(val, ir.Constant(i32, 0), 1)
        val = b.insert_value(val, size, 2)
        b.ret(val)

        # 2. Arena_drop(Arena) -> void
        # Passed by Value or Pointer? Drop usually takes by Value if we consume it,
        # but here we might want ref. Auto-Drop passes by Value if it's not a pointer?
        # Codegen.emit_scope_drops currently loads the value `b.load(ptr)`.
        # So it passes the STRUCT VALUE (Aggregate).
        # LLVM handling of First Class Structs as args is fine.
        drop_ty = ir.FunctionType(ir.VoidType(), [arena_ty])
        drop_func = ir.Function(self.module, drop_ty, name="Arena_drop")
        bb = drop_func.append_basic_block("entry")
        b = ir.IRBuilder(bb)
        arg = drop_func.args[0]
        chunk = b.extract_value(arg, 0)
        b.call(self.free, [chunk])
        b.ret_void()

        # 3. Arena_alloc(Arena*, i32) -> i8*
        # Takes POINTER to Arena
        alloc_ty = ir.FunctionType(void_ptr, [arena_ty.as_pointer(), i32])
        alloc_func = ir.Function(self.module, alloc_ty, name="Arena_alloc")
        bb = alloc_func.append_basic_block("entry")
        b = ir.IRBuilder(bb)

        arena_ptr = alloc_func.args[0]
        req_size = alloc_func.args[1]

        # Load offset, capacity, chunk
        chunk_ptr = b.gep(arena_ptr, [ir.Constant(i32, 0), ir.Constant(i32, 0)])
        offset_ptr = b.gep(arena_ptr, [ir.Constant(i32, 0), ir.Constant(i32, 1)])
        cap_ptr = b.gep(arena_ptr, [ir.Constant(i32, 0), ir.Constant(i32, 2)])

        base_chunk = b.load(chunk_ptr)
        offset = b.load(offset_ptr)
        cap = b.load(cap_ptr)

        new_offset = b.add(offset, req_size)
        cond = b.icmp_unsigned(">", new_offset, cap)

        with b.if_then(cond):
            b.call(self.exit_func, [ir.Constant(i32, 1)])

        # Pointer arithmetic
        base_int = b.ptrtoint(base_chunk, ir.IntType(64))
        off_64 = b.zext(offset, ir.IntType(64))
        res_int = b.add(base_int, off_64)
        res_ptr = b.inttoptr(res_int, void_ptr)

        b.store(new_offset, offset_ptr)
        b.ret(res_ptr)

    def _declare_malloc_free(self):
        # void* malloc(i32)
        # void free(void*)

        # malloc
        void_ptr = ir.IntType(8).as_pointer()
        malloc_ty = ir.FunctionType(void_ptr, [ir.IntType(32)])
        self.malloc = ir.Function(self.module, malloc_ty, name="malloc")

        # free
        free_ty = ir.FunctionType(ir.VoidType(), [void_ptr])
        self.free = ir.Function(self.module, free_ty, name="free")

        # realloc
        realloc_ty = ir.FunctionType(void_ptr, [void_ptr, ir.IntType(32)])
        self.realloc = ir.Function(self.module, realloc_ty, name="realloc")

    def _declare_memcpy(self):
        # Declare llvm.memcpy.p0i8.p0i8.i32
        # void @llvm.memcpy.p0i8.p0i8.i32(i8* <dest>, i8* <src>, i32 <len>, i1 <isvolatile>)
        void_ptr = ir.IntType(8).as_pointer()
        bool_ty = ir.IntType(1)
        i32_ty = ir.IntType(32)

        fnty = ir.FunctionType(ir.VoidType(), [void_ptr, void_ptr, i32_ty, bool_ty])
        self.memcpy = ir.Function(self.module, fnty, name="llvm.memcpy.p0i8.p0i8.i32")

    def _declare_printf(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    def _declare_exit(self):
        exit_ty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)])
        self.exit_func = ir.Function(self.module, exit_ty, name="exit")

    def get_llvm_type(self, type_name):
        # Debug trace
        # print(f"DEBUG GET_LLVM_TYPE: {type_name} generics={self._current_generics}", flush=True)
        if type_name == 'i32':
            return ir.IntType(32)
        elif type_name == 'bool':
            return ir.IntType(1)
        elif type_name == 'char':
            # Represent as i32 (Unicode scalar) in the bootstrap
            return ir.IntType(32)
        elif type_name == 'i64':
            return ir.IntType(64)
        elif type_name == 'u64':
            return ir.IntType(64)
        elif type_name == 'f32':
            return ir.FloatType()
        elif type_name == 'u8':
            return ir.IntType(8)
        elif type_name == 'void':
            return ir.VoidType()
        elif type_name == 'string':
            return ir.IntType(8).as_pointer()
        elif type_name.startswith('&'):
            inner = type_name[1:]
            if inner.startswith("mut "):
                inner = inner[4:]
            return self.get_llvm_type(inner).as_pointer()
        elif type_name.endswith('*'):
            inner_type = type_name[:-1]
            # Handle *void (void*)
            if inner_type == 'void':
                return ir.IntType(8).as_pointer()
            return self.get_llvm_type(inner_type).as_pointer()
        elif type_name in self.enum_definitions:
            enum_ty, _ = self.enum_definitions[type_name]
            return enum_ty
        elif type_name in self.struct_types:
            return self.struct_types[type_name]

        # Handle generic parameters as placeholders (e.g. 'T')
        if type_name in self._current_generics:
             return ir.IntType(8) # Placeholder

        if "<" in type_name:
            base = type_name.split("<")[0]
            if base in self.struct_types:
                return self.struct_types[base]
            if base in self.enum_definitions:
                enum_ty, _ = self.enum_definitions[base]
                return enum_ty
            print(f"DEBUG ERASURE: {type_name} -> i8")
                
        if type_name.startswith('fn('):
            # Parse fn(i32, bool)->void
            # Use rpartition for robust splitting of return type
            main_part, arrow, ret_str = type_name.rpartition(')->')
            if not arrow: raise Exception(f"Invalid function type: {type_name}")
            params_str = main_part[3:] # Strip 'fn('
            
            param_types = []
            if params_str:
                depth = 0
                current = ""
                for c in params_str:
                    if c == '<' or c == '(': depth += 1
                    elif c == '>' or c == ')': depth -= 1
                    if c == ',' and depth == 0:
                        param_types.append(self.get_llvm_type(current.strip()))
                        current = ""
                    else:
                        current += c
                if current:
                    param_types.append(self.get_llvm_type(current.strip()))
            
            ret_type = self.get_llvm_type(ret_str.strip())
            # Use Fat Pointer representation: { function_ptr, environment_ptr }
            return ir.LiteralStructType([ir.IntType(8).as_pointer(), ir.IntType(8).as_pointer()])

        # Fallback
        raise Exception(f"CodeGen: Unknown type '{type_name}'")

    def visit_StructDef(self, node):
        if getattr(node, 'generics', None): return
        #     return # Skip monomorphized version in bootstrap; use erased base type instead
            
        # We don't skip generic definitions anymore; we register them as base types
        # so that monomorphized instances (type erasure in bootstrap) can find them.
        
        # We need a way to avoid failing on generic parameter types like 'T'
        # We need a way to avoid failing on generic parameter types like 'T'
        self._current_generics = [g[0] for g in node.generics]

        if node.name not in self.struct_types:
             # Generic instantiation or dynamic creation
             self.struct_types[node.name] = self.module.context.get_identified_type(node.name)
        
        struct_ty = self.struct_types[node.name]
        
        # Create LLVM Struct Type
        field_types = []
        is_spirv_buffer = self.target == "spirv" and isinstance(node.name, str) and node.name.startswith("Buffer<")
        for field_name, type_name in node.fields:
            # For SPIR-V, treat Buffer<T>.ptr as a storage pointer (addrspace depends on env).
            # - opencl: addrspace(1) -> CrossWorkgroup
            # - vulkan: addrspace(11) -> StorageBuffer
            if is_spirv_buffer and field_name == "ptr" and isinstance(type_name, str) and type_name.endswith("*"):
                inner = type_name[:-1]
                asid = 11 if (self.target == "spirv" and self.spirv_env == "vulkan") else 1
                field_types.append(self.get_llvm_type(inner).as_pointer(addrspace=asid))
            else:
                field_types.append(self.get_llvm_type(type_name))

        # Check if opaque/empty and set body
        # If it's already set (e.g. checked before?), avoid double setting if not allowed?
        # llvmlite set_body can be called once.
        if struct_ty.is_opaque:
             struct_ty.set_body(*field_types)

        # Map field names to indices
        self.struct_fields[node.name] = {name: i for i, (name, _) in enumerate(node.fields)}
        self._current_generics = []

    def visit_EnumDef(self, node):
        if getattr(node, 'generics', None): return
        #     return # Skip monomorphized version in bootstrap
            
        # if node.generics: return # ALLOW GENERICS (Type Erasure)
        
        self._current_generics = [g[0] for g in node.generics]

        # Representation: { i32 tag, [MaxPayloadSize x i8] data }
        # 1. Determine max payload size
        max_size = 0
        variant_tags = {}
        variant_payload_types = {}

        # Helper for recursive types: Register proper placeholder (not i8!)
        placeholder_padding = ir.ArrayType(ir.IntType(8), 256)
        placeholder_enum = ir.LiteralStructType([ir.IntType(32), placeholder_padding])
        self.enum_definitions[node.name] = (placeholder_enum, {})

        # For this phase, assume only i32 payloads for simplicity of size calc
        # i32 = 4 bytes

        for i, (vname, payloads) in enumerate(node.variants):
            variant_tags[vname] = i
            if payloads:
                # Map all payload types to LLVM types
                llvm_types = []
                current_size = 0
                for ptype in payloads:
                    lty = self.get_llvm_type(ptype)
                    llvm_types.append(lty)

                    # Rough size estimation (bootstrap limitation)
                    if isinstance(lty, ir.IntType):
                        current_size += lty.width // 8
                    elif isinstance(lty, ir.PointerType):
                        current_size += 8
                    else:
                        current_size += 64 # Struct/Array fallback

                max_size = max(max_size, current_size)

                if len(llvm_types) == 1:
                    variant_payload_types[vname] = llvm_types[0]
                else:
                    variant_payload_types[vname] = ir.LiteralStructType(llvm_types)
            else:
                variant_payload_types[vname] = None

        # Tag (i32) + Payload (Array of i8)
        # Use Fixed Size (Universal Enum)
        # max_size = max(max_size, current_size) # We ignore calculated max
        final_size = 256
        padding_ty = ir.ArrayType(ir.IntType(8), final_size)

        # Check if any variant exceeds limit?
        if max_size > final_size:
            print(f"WARNING: Enum {node.name} payload size {max_size} exceeds fixed limit {final_size}")
        enum_ty = ir.LiteralStructType([ir.IntType(32), padding_ty])

        self.enum_types[node.name] = variant_tags
        self.enum_payloads[node.name] = variant_payload_types
        self.enum_definitions[node.name] = (enum_ty, max_size)

    def visit_TraitDef(self, node):
        pass # Traits are compile-time only for now (static dispatch)

    def visit_ImplDef(self, node):
        if getattr(node, 'generics', None): return
        self._current_generics = [g[0] for g in node.generics]
        for method in node.methods:
            self.visit(method)
        self._current_generics = []

    def generate(self, ast):
        # Pre-Pass: Register Universal Enum Types to handle recursion
        enum_payload_size = 256
        placeholder_padding = ir.ArrayType(ir.IntType(8), enum_payload_size)
        placeholder_enum = ir.LiteralStructType([ir.IntType(32), placeholder_padding])

        for node in ast:
            if isinstance(node, EnumDef):
                self.enum_definitions[node.name] = (placeholder_enum, {})
            elif isinstance(node, StructDef):
                 if '<' not in node.name:
                     # Use IdentifiedStructType to support recursive/out-of-order types
                     self.struct_types[node.name] = self.module.context.get_identified_type(node.name)

        # Pass 1: Types (Structs, Enums)
        for node in ast:
            # No debug print
            if isinstance(node, (StructDef, EnumDef)):
                self.visit(node)
        # No debug print

        # Pass 2: Function Headers
        for node in ast:
            if isinstance(node, FunctionDef):
                if self.emit_kernels_only and not node.is_kernel:
                    continue
                self._declare_function(node)
            elif isinstance(node, ExternBlock):
                for func in node.functions:
                    self._declare_function(func)
            elif isinstance(node, ImplDef):
                self._current_generics = [g[0] for g in node.generics]
                for method in node.methods:
                    self._declare_function(method)
                self._current_generics = []

        # Pass 3: Bodies
        for node in ast:
            if not isinstance(node, (StructDef, EnumDef)):
                if isinstance(node, FunctionDef) and self.emit_kernels_only and not node.is_kernel:
                    continue
                self.visit(node)

        llvm_ir = str(self.module)
        if self.target == "spirv" and self.spirv_env == "vulkan":
            llvm_ir = self._postprocess_spirv_vulkan_kernel_attributes(llvm_ir)
        return llvm_ir

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def visit_UseStmt(self, node):
        pass

    def visit_TypeAlias(self, node):
        pass

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method")

    def visit_MacroCallExpr(self, node):
        if hasattr(node, 'expanded'):
            return self.visit(node.expanded)
        raise Exception(f"Macro {node.name}! was not expanded during semantic analysis")

    def visit_AwaitExpr(self, node):
        # 1. Evaluate the expression (returns handle/pointer to state)
        h = self.visit(node.value)
        
        # 2. Polling Loop
        res_type_name = getattr(node, 'type_name', 'void')
        res_ty = self.get_llvm_type(res_type_name)
        # struct { i1 done, T result }
        state_ty = ir.LiteralStructType([ir.IntType(1), res_ty])
        state = self.builder.bitcast(h, state_ty.as_pointer())
        
        cond_bb = self.builder.append_basic_block("await_cond")
        body_bb = self.builder.append_basic_block("await_poll")
        cont_bb = self.builder.append_basic_block("await_cont")
        
        self.builder.branch(cond_bb)
        self.builder.position_at_end(cond_bb)
        
        done_ptr = self.builder.gep(state, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        done = self.builder.load(done_ptr)
        self.builder.cbranch(done, cont_bb, body_bb)
        
        self.builder.position_at_end(body_bb)
        # Here we could yield to executor. For bootstrap, we just loop (busy wait).
        self.builder.branch(cond_bb)
        
        self.builder.position_at_end(cont_bb)
        # 3. Load result
        if res_type_name == 'void':
            return ir.Constant(ir.IntType(32), 0)
            
        res_ptr = self.builder.gep(state, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 1)])
        return self.builder.load(res_ptr)

    def visit_UnaryExpr(self, node):
        if node.op == '&':
            # Address Of: We need the address of the operand.
            # visit(operand) usually returns a value (load).
            # We need a method to get address.
            # HACK: If operand is VariableExpr, look up alloca.
            if isinstance(node.operand, VariableExpr):
                for scope in reversed(self.scopes):
                        if node.operand.name in scope:
                            ptr, _ = scope[node.operand.name]
                            return ptr
                raise Exception(f"Undefined variable for address of: {node.operand.name}")
            else:
                raise Exception("Address of (&) only supported for variables currently")

        elif node.op == '*':
            # Dereference
            ptr_val = self.visit(node.operand)
            return self.builder.load(ptr_val)
        elif node.op == '!':
            val = self.visit(node.operand)
            return self.builder.not_(val)
        elif node.op == '-':
            val = self.visit(node.operand)
            return self.builder.neg(val)
        else:
            raise Exception(f"Unsupported unary operator: {node.op}")

    def visit_MemberAccess(self, node):
        # ... existing implementation ...
        # Vulkan: Buffer<T> is lowered to interface globals, not an in-memory struct.
        if (
            self.target == "spirv"
            and self.spirv_env == "vulkan"
            and hasattr(node, "struct_type")
            and isinstance(node.struct_type, str)
            and node.struct_type.startswith("Buffer<")
        ):
            if not isinstance(node.object, VariableExpr):
                raise Exception("Vulkan Buffer<T> member access only supported on variables (e.g. buf.ptr / buf.len).")
            # Find buffer globals
            key = (self._current_function_name or "", node.object.name)
            if key not in self._vulkan_buffer_args:
                # In case it wasn't predeclared (defensive)
                self._get_or_declare_vulkan_buffer_globals(self._current_function_name or "", node.object.name, node.struct_type)
            data_gv, len_gv = self._vulkan_buffer_args[(self._current_function_name or "", node.object.name)]
            if node.member == "ptr":
                # Return the runtime array base pointer. Indexing will use gep(0, idx).
                return data_gv
            if node.member == "len":
                return self.builder.load(len_gv, name=f"{node.object.name}_len")
            raise Exception(f"Unknown Buffer member '{node.member}'")

        struct_val = self.visit(node.object)
        if not hasattr(node, 'struct_type'):
            raise Exception("CodeGen Error: MemberAccess node missing 'struct_type' annotation from Semantic phase.")
        struct_name = node.struct_type

        # Determine if we have value or pointer
        if isinstance(struct_val.type, ir.PointerType):
            # GEP + Load
            param_ptr = struct_val

            # Ensure param_ptr points to the actual struct type (handles type erasure fallback to i8*)
            actual_struct_ty = self.get_llvm_type(struct_name)
            if param_ptr.type.pointee != actual_struct_ty:
                param_ptr = self.builder.bitcast(param_ptr, actual_struct_ty.as_pointer())

            # Need to find field index
            field_idx = self.struct_fields[struct_name][node.member]

            ptr = self.builder.gep(param_ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), field_idx)])
            return self.builder.load(ptr)
        else:
            # Extract Value case
            # Ensure it's not type-erased i8 (LLVM IntType doesn't have fields)
            if isinstance(struct_val.type, ir.IntType):
                 # Typeless load or erasure: Spill to stack to bitcast and access fields
                 actual_struct_ty = self.get_llvm_type(struct_name)
                 temp_mem = self.builder.alloca(struct_val.type)
                 self.builder.store(struct_val, temp_mem)
                 # Bitcast to actual struct pointer
                 casted_ptr = self.builder.bitcast(temp_mem, actual_struct_ty.as_pointer())
                 field_idx = self.struct_fields[struct_name][node.member]
                 field_ptr = self.builder.gep(casted_ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), field_idx)])
                 return self.builder.load(field_ptr)
            
            field_index = self.struct_fields[struct_name][node.member]
            return self.builder.extract_value(struct_val, field_index)

    def visit_MethodCall(self, node):
        # Desugar obj.method(args) -> Type_method(obj, args)
        receiver_val = self.visit(node.receiver)
        
        # In modern semantic pass, node.method_name is already fully mangled
        func_name = node.method_name
        
        if func_name not in self.module.globals:
            # Fallback for old style (or if semantic didn't mangle for some reason)
            struct_type = node.struct_type
            legacy_name = f"{struct_type}_{node.method_name}"
            if legacy_name in self.module.globals:
                func_name = legacy_name
            else:
                raise Exception(f"CodeGen Error: Method '{func_name}' not found in module")

        func = self.module.globals[func_name]
        expected_param_type = func.function_type.args[0]

        receiver_arg = None
        if isinstance(expected_param_type, ir.PointerType) and not isinstance(receiver_val.type, ir.PointerType):
            # Method expects pointer but we have value.
            from n_parser import VariableExpr
            if isinstance(node.receiver, VariableExpr):
                # Look up variable's address in all scopes
                for scope in reversed(self.scopes):
                    if node.receiver.name in scope:
                        entry = scope[node.receiver.name]
                        if isinstance(entry, tuple) and (len(entry) == 2 or (len(entry) == 3 and entry[2] != "value")):
                            receiver_arg = entry[0] # The alloca
                        break

            if receiver_arg is None:
                # Fallback: create temporary
                temp = self.builder.alloca(receiver_val.type, name="method_self_tmp")
                self.builder.store(receiver_val, temp)
                receiver_arg = temp
        else:
            receiver_arg = receiver_val

        args = [receiver_arg] + [self.visit(arg) for arg in node.args]
        
        # Auto-cast arguments to match function signature (for type erasure)
        for i in range(len(args)):
            expected_type = func.function_type.args[i]
            actual_val = args[i]
            if actual_val.type != expected_type:
                if isinstance(actual_val.type, ir.PointerType) and isinstance(expected_type, ir.PointerType):
                    args[i] = self.builder.bitcast(actual_val, expected_type)
                elif isinstance(actual_val.type, ir.IntType) and isinstance(expected_type, ir.IntType):
                    if actual_val.type.width > expected_type.width:
                        args[i] = self.builder.trunc(actual_val, expected_type)
                    else:
                        args[i] = self.builder.zext(actual_val, expected_type)
                elif isinstance(actual_val.type, ir.LiteralStructType) and isinstance(expected_type, ir.LiteralStructType):
                    # We can't bitcast aggregate values directly. 
                    # Same hack as visit_VarDecl: store to temp, bitcast pointer, load.
                    tmp = self.builder.alloca(actual_val.type)
                    self.builder.store(actual_val, tmp)
                    tmp_cast = self.builder.bitcast(tmp, expected_type.as_pointer())
                    args[i] = self.builder.load(tmp_cast)
                else:
                    # Fallback bitcast
                    try:
                        args[i] = self.builder.bitcast(actual_val, expected_type)
                    except:
                        pass # Let it fail in call() if it must

        return self.builder.call(func, args)

    def visit_ArrayLiteral(self, node):
        # Create array value
        # Assume i32 elements for now since we parsed [i32:N] or inferred
        # We need to know the type!
        # SemanticAnalyzer verified types are consistent.
        # We can inspect the first element's LLVM type?
        val0 = self.visit(node.elements[0])
        elem_ty = val0.type
        size = len(node.elements)
        array_ty = ir.ArrayType(elem_ty, size)

        array_val = ir.Constant(array_ty, ir.Undefined)
        array_val = self.builder.insert_value(array_val, val0, 0)

        for i in range(1, size):
            val = self.visit(node.elements[i])
            array_val = self.builder.insert_value(array_val, val, i)

        return array_val

    def visit_IndexAccess(self, node):
        index_val = self.visit(node.index)

        # Fast path: If object is a variable, use its alloca + recorded type.
        if type(node.object).__name__ == 'VariableExpr':
            entry = None
            for scope in reversed(self.scopes):
                if node.object.name in scope:
                    entry = scope[node.object.name]
                    break
            if entry:
                if isinstance(entry, tuple) and len(entry) == 3:
                    var_ptr, type_name, tag = entry
                    if tag in ("value", "vulkan_buffer"):
                        # SSA pointer value: gep then load
                        if not isinstance(var_ptr.type, ir.PointerType):
                            raise Exception("Index access requires pointer/array base")
                        if isinstance(var_ptr.type.pointee, ir.ArrayType):
                            zero = ir.Constant(ir.IntType(32), 0)
                            elem_ptr = self.builder.gep(var_ptr, [zero, index_val])
                        else:
                            elem_ptr = self.builder.gep(var_ptr, [index_val])
                        return self.builder.load(elem_ptr)
                    # fallback to legacy behavior
                    var_ptr = var_ptr
                else:
                    var_ptr, type_name = entry

                # Slice<T> sugar: slice[i] -> *(slice.ptr + i)
                if isinstance(type_name, str) and type_name.startswith("Slice<"):
                    slice_val = self.builder.load(var_ptr, name=f"{node.object.name}_slice")
                    base_ptr = self.builder.extract_value(slice_val, 0, name="slice_ptr")
                    elem_ptr = self.builder.gep(base_ptr, [index_val])
                    return self.builder.load(elem_ptr)

                # Arrays: alloca [N x T] -> gep (0, idx)
                if isinstance(var_ptr.type.pointee, ir.ArrayType):
                    zero = ir.Constant(ir.IntType(32), 0)
                    elem_ptr = self.builder.gep(var_ptr, [zero, index_val])
                    return self.builder.load(elem_ptr)

                # Pointers: alloca T* -> load base ptr -> gep (idx)
                if isinstance(var_ptr.type.pointee, ir.PointerType) or (isinstance(type_name, str) and type_name.endswith('*')):
                    base_ptr = self.builder.load(var_ptr)
                    elem_ptr = self.builder.gep(base_ptr, [index_val])
                    return self.builder.load(elem_ptr)

        # Fallback: Evaluate object to value (loads it)
        obj_val = self.visit(node.object)
        
        # Pointer indexing: gep(ptr, idx) then load
        if isinstance(obj_val.type, ir.PointerType) or str(obj_val.type).endswith('*'):
            if isinstance(obj_val.type, ir.PointerType) and isinstance(obj_val.type.pointee, ir.ArrayType):
                zero = ir.Constant(ir.IntType(32), 0)
                elem_ptr = self.builder.gep(obj_val, [zero, index_val])
            else:
                elem_ptr = self.builder.gep(obj_val, [index_val])
            return self.builder.load(elem_ptr)

        # If index_val is Constant, we can use extract_value
        if isinstance(index_val, ir.Constant):
            # extract_value index must be python int
            idx = index_val.constant
            return self.builder.extract_value(obj_val, idx)

        # Runtime index on SSA value: Spill to stack
        temp_ptr = self.builder.alloca(obj_val.type)
        self.builder.store(obj_val, temp_ptr)
        zero = ir.Constant(ir.IntType(32), 0)
        try:
            if isinstance(obj_val.type, (ir.ArrayType, ir.LiteralStructType, ir.StructureType)):
                ptr = self.builder.gep(temp_ptr, [zero, index_val])
            else:
                ptr = self.builder.gep(temp_ptr, [index_val])
        except Exception as e:
            print(f"DEBUG GEP FAILED: type={obj_val.type} agg={isinstance(obj_val.type, (ir.ArrayType, ir.LiteralStructType, ir.StructureType))}")
            raise e
        return self.builder.load(ptr)

        func = ir.Function(self.module, func_ty, name=node.name)

        if node.is_kernel:
            func.calling_convention = 'spir_kernel'

        block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        # Scope dictionary: variable_name -> pointer
        self.scopes = [{}]

        for stmt in node.body:
            self.visit(stmt)

        self.builder.ret_void()

    def emit_scope_drops(self, scope):
        # Iterate in reverse order of declaration (LIFO)
        # Scope is dict, assuming insertion order preserved (Python 3.7+)
        for var_name, entry in reversed(list(scope.items())):
            if var_name.startswith('$') or not isinstance(entry, tuple) or len(entry) < 2:
                continue
            ptr, type_name = entry[0], entry[1]
            # Look for destructor: {TypeName}_drop(T) or {TypeName}_drop(&T)
            # Modern mangled name: {TypeName}_drop__args__SELF_PTR
            drop_func_name = f"{type_name}_drop"
            mangled_drop = f"{type_name}_drop__args__SELF_PTR"
            
            drop_func = None
            if mangled_drop in self.module.globals:
                drop_func = self.module.get_global(mangled_drop)
            elif drop_func_name in self.module.globals:
                drop_func = self.module.get_global(drop_func_name)
                
            if drop_func:
                # Load value
                val = self.builder.load(ptr)
                # Call drop
                self.builder.call(drop_func, [val])

    def visit_Block(self, node):
        self.scopes.append({})
        for stmt in node.statements:
            self.visit(stmt)

        # Auto-drop variables in this scope
        self.emit_scope_drops(self.scopes[-1])
        self.scopes.pop()

    def visit_ForStmt(self, node):
        if hasattr(node, 'is_iterator') and node.is_iterator:
            # Iterator Loop: for x in iterator { ... }
            
            # 1. Evaluate Iterator (once)
            iter_val = self.visit(node.start_expr)
            
            # We need a stable memory location for the iterator (passed as &mut self)
            func = self.builder.function
            entry_block = func.entry_basic_block
            
            # Create stack slot for iterator
            with self.builder.goto_block(entry_block):
                 if entry_block.instructions:
                      self.builder.position_before(entry_block.instructions[0])
                 iter_ptr = self.builder.alloca(iter_val.type, name="iter_tmp")
            
            self.builder.store(iter_val, iter_ptr)
            
            # 2. Logic Blocks
            cond_block = self.builder.append_basic_block(f"for_iter_cond_{node.var_name}")
            body_block = self.builder.append_basic_block(f"for_iter_body_{node.var_name}")
            end_block = self.builder.append_basic_block(f"for_iter_end_{node.var_name}")
            
            self.builder.branch(cond_block)
            
            # 3. Request Next Item
            self.builder.position_at_end(cond_block)
            
            # Call {Type}_next(&mut iter)
            base_type = node.iterator_type.split('<')[0]
            func_name = f"{base_type}_next"
            if func_name not in self.module.globals:
                 raise Exception(f"CodeGen: Method '{func_name}' not found")
                 
            func_def = self.module.globals[func_name]
            
            # Prepare arguments: &mut self
            # iter_ptr matches type?
            # func expects pointer to struct. iter_ptr is pointer to struct.
            # cast if necessary (e.g. if type erasure diff)
            arg0 = iter_ptr
            if arg0.type != func_def.function_type.args[0]:
                 arg0 = self.builder.bitcast(arg0, func_def.function_type.args[0])
            
            option_ret = self.builder.call(func_def, [arg0])
            
            # 4. Debox Option
            # Store option to stack to access fields
            opt_mem = self.builder.alloca(option_ret.type)
            self.builder.store(option_ret, opt_mem)
            
            # Extract Tag (field 0)
            # Ensure opt_mem points to a struct even if type was erased to i8
            if not isinstance(opt_mem.type.pointee, (ir.ArrayType, ir.LiteralStructType, ir.StructureType)):
                 # Deduce return type name (e.g. Option<T>)
                 opt_type_name = getattr(node, 'option_type', f"Option<{node.item_type}>")
                 actual_ty = self.get_llvm_type(opt_type_name)
                 opt_mem = self.builder.bitcast(opt_mem, actual_ty.as_pointer())

            tag_ptr = self.builder.gep(opt_mem, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
            tag = self.builder.load(tag_ptr)
            
            # Check if Some (0) or None (1)
            # Assuming Some is 0.
            cond = self.builder.icmp_signed("==", tag, ir.Constant(ir.IntType(32), 0))
            self.builder.cbranch(cond, body_block, end_block)
            
            # 5. Body
            self.builder.position_at_end(body_block)
            self.scopes.append({})
            self.loop_stack.append((cond_block, end_block, len(self.scopes) - 1, node.label))
            
            # Extract Value (field 1)
            payload_arr_ptr = self.builder.gep(opt_mem, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 1)])
            
            # Cast payload array to item value type
            # We need LLVM type for item_type
            item_llvm_type = self.get_llvm_type(node.item_type)
            payload_ptr = self.builder.bitcast(payload_arr_ptr, item_llvm_type.as_pointer())
            item_val = self.builder.load(payload_ptr)
            
            # Declare Loop Variable
            with self.builder.goto_block(entry_block):
                 if entry_block.instructions:
                     self.builder.position_before(entry_block.instructions[0])
                 var_ptr = self.builder.alloca(item_llvm_type, name=node.var_name)
                 
            self.builder.store(item_val, var_ptr)
            self.scopes[-1][node.var_name] = (var_ptr, node.item_type)
            
            for stmt in node.body:
                self.visit(stmt)
                
            self.emit_scope_drops(self.scopes[-1])
            self.scopes.pop()
            self.loop_stack.pop()
            self.builder.branch(cond_block)
            
            self.builder.position_at_end(end_block)
            
            return

        # Range Loop: for i in start..end { body }
        
        # 1. Initialize Loop Variable
        start_val = self.visit(node.start_expr)
        end_val = self.visit(node.end_expr)
        
        # Create alloca for loop var
        func = self.builder.function
        entry_block = func.entry_basic_block
        # Insert alloca at top of entry
        with self.builder.goto_block(entry_block):
             if entry_block.instructions:
                  self.builder.position_before(entry_block.instructions[0])
             loop_var_ptr = self.builder.alloca(ir.IntType(32), name=node.var_name)
             
        # Initialize
        self.builder.store(start_val, loop_var_ptr)
        
        # 2. Basic Blocks
        cond_block = self.builder.append_basic_block(f"for_cond_{node.var_name}")
        body_block = self.builder.append_basic_block(f"for_body_{node.var_name}")
        inc_block = self.builder.append_basic_block(f"for_inc_{node.var_name}")
        end_block = self.builder.append_basic_block(f"for_end_{node.var_name}")
        
        self.builder.branch(cond_block)
        
        # 3. Condition Block
        self.builder.position_at_end(cond_block)
        curr_val = self.builder.load(loop_var_ptr, name=f"{node.var_name}_val")
        
        if node.inclusive:
             cond = self.builder.icmp_signed("<=", curr_val, end_val, name="loop_cond")
        else:
             cond = self.builder.icmp_signed("<", curr_val, end_val, name="loop_cond")
             
        self.builder.cbranch(cond, body_block, end_block)
        
        # 4. Body Block
        self.builder.position_at_end(body_block)
        
        # Enter Scope
        self.scopes.append({})
        self.loop_stack.append((inc_block, end_block, len(self.scopes) - 1, node.label))
        
        # Register loop variable
        self.scopes[-1][node.var_name] = (loop_var_ptr, 'i32') # i32, no tag needed for primitives
        
        for stmt in node.body:
             self.visit(stmt)
             
        # Auto-drop (if we supported break/continue, we would need to handle drops there too)
        self.emit_scope_drops(self.scopes[-1])
        
        self.scopes.pop()
        self.loop_stack.pop()
        
        self.builder.branch(inc_block)
        
        # 5. Increment Block
        self.builder.position_at_end(inc_block)
        curr_val_2 = self.builder.load(loop_var_ptr)
        one = ir.Constant(ir.IntType(32), 1)
        next_val = self.builder.add(curr_val_2, one)
        self.builder.store(next_val, loop_var_ptr)
        self.builder.branch(cond_block)
        
        # 6. End Block
        self.builder.position_at_end(end_block)

    def visit_BreakStmt(self, node):
        if not self.loop_stack:
            raise Exception("Break statement outside of loop")
            
        target = self.loop_stack[-1]
        if node.label:
            found = False
            for entry in reversed(self.loop_stack):
                if entry[3] == node.label:
                    target = entry
                    found = True
                    break
            if not found:
                raise Exception(f"Break label '{node.label}' not found")
                
        _, break_block, loop_scope_depth, _ = target
        
        # Unwind scopes until loop scope
        current_depth = len(self.scopes) - 1
        while current_depth >= loop_scope_depth:
             self.emit_scope_drops(self.scopes[current_depth])
             current_depth -= 1
             
        self.builder.branch(break_block)

    def visit_ContinueStmt(self, node):
        if not self.loop_stack:
            raise Exception("Continue statement outside of loop")
            
        target = self.loop_stack[-1]
        if node.label:
            found = False
            for entry in reversed(self.loop_stack):
                if entry[3] == node.label:
                    target = entry
                    found = True
                    break
            if not found:
                raise Exception(f"Continue label '{node.label}' not found")
                
        continue_block, _, loop_scope_depth, _ = target
        
        # Unwind scopes until loop scope
        current_depth = len(self.scopes) - 1
        while current_depth >= loop_scope_depth:
             self.emit_scope_drops(self.scopes[current_depth])
             current_depth -= 1
             
        self.builder.branch(continue_block)
    def visit_BlockStmt(self, node):
        self.scopes.append({})
        for stmt in node.stmts:
             self.visit(stmt)
        # Auto-drop variables in this scope
        self.emit_scope_drops(self.scopes[-1])
        self.scopes.pop()

    def visit_VarDecl(self, node):
        # Determine type
        try:
            llvm_type = self.get_llvm_type(node.type_name)
        except Exception:
            # Handle array type [T:N] manually if not in get_llvm_type yet
            if node.type_name.startswith('['):
                content = node.type_name[1:-1]
                elem_type_str, size_str = content.split(':')
                elem_ty = self.get_llvm_type(elem_type_str)
                size = int(size_str)
                llvm_type = ir.ArrayType(elem_ty, size)
            else:
                raise

        # Evaluate initializer
        init_val = self.visit(node.initializer)

        # Auto-cast for string (array ptr -> i8*)
        if node.type_name == "string" and init_val.type != llvm_type:
            init_val = self.builder.bitcast(init_val, llvm_type)

        # SPIR-V: allow pointer address-space propagation from initializer.
        # Example: Buffer<T>.ptr may be addrspace(1) but source type is just `T*`.
        if (
            isinstance(llvm_type, ir.PointerType)
            and isinstance(init_val.type, ir.PointerType)
            and llvm_type.pointee == init_val.type.pointee
            and llvm_type.addrspace != init_val.type.addrspace
        ):
            llvm_type = init_val.type

        # Vulkan: avoid storing/loading pointer-typed SSA values (OpLoad result = pointer is invalid in Logical addressing).
        if (
            self.target == "spirv"
            and self.spirv_env == "vulkan"
            and isinstance(llvm_type, ir.PointerType)
        ):
            self.scopes[-1][node.name] = (init_val, node.type_name, "value")
            return

        # Alloca
        ptr = self.builder.alloca(llvm_type, name=node.name)
        # No debug print
        if init_val.type != llvm_type:
            # Struct erasure bitcast hack for bootstrap
            if isinstance(init_val.type, ir.LiteralStructType) and isinstance(llvm_type, ir.LiteralStructType):
                if len(init_val.type.elements) == len(llvm_type.elements):
                    # For aggregate values, we can't bitcast directly in LLVM without a pointer.
                    # We store to a temp and load as target type.
                    tmp = self.builder.alloca(init_val.type)
                    # Auto-bitcast for type erasure
                    if init_val.type != tmp.type.pointee:
                        tmp = self.builder.bitcast(tmp, init_val.type.as_pointer())
                    self.builder.store(init_val, tmp)
                    tmp_cast = self.builder.bitcast(tmp, llvm_type.as_pointer())
                    init_val = self.builder.load(tmp_cast)
        
        # Auto-bitcast for type erasure
        if init_val.type != ptr.type.pointee:
            ptr = self.builder.bitcast(ptr, init_val.type.as_pointer())
        
        self.builder.store(init_val, ptr)

        # Store in scope: (Pointer, TypeName)
        self.scopes[-1][node.name] = (ptr, node.type_name)

    def visit_Assignment(self, node):
        val = self.visit(node.value)

        if isinstance(node.target, VariableExpr):
            # Find stack allocation
            ptr = None
            entry = None
            for scope in reversed(self.scopes):
                    if node.target.name in scope:
                        entry = scope[node.target.name]
                        if isinstance(entry, tuple) and len(entry) == 3 and entry[2] == "value":
                            # SSA update (Vulkan pointer values, etc.)
                            scope[node.target.name] = (val, entry[1], "value")
                            return
                        ptr, _ = entry
                        break
            if not ptr: raise Exception(f"Undefined var {node.target.name}")
            if val.type != ptr.type.pointee:
                ptr = self.builder.bitcast(ptr, val.type.as_pointer())
            # Auto-bitcast for type erasure
            if val.type != ptr.type.pointee:
                ptr = self.builder.bitcast(ptr, val.type.as_pointer())
            self.builder.store(val, ptr)

        elif isinstance(node.target, MemberAccess):
            struct_val = self.visit(node.target.object)
            # print(f"DEBUG ASSIGN MEMBER: {struct_val.type}")
            # if not hasattr(node.target, 'struct_type'): raise Exception('Missing struct_type')
            if not isinstance(struct_val.type, ir.PointerType):
                obj_repr = repr(node.target.object) if hasattr(node.target, 'object') else 'unknown'
                raise Exception(f"Cannot assign to member '{node.target.member}' of Value-type struct (object={obj_repr}): {struct_val.type} (is_ptr={isinstance(struct_val.type, ir.PointerType)})")
            struct_name = node.target.struct_type
            field_idx = self.struct_fields[struct_name][node.target.member]
            if isinstance(struct_val.type, ir.PointerType):
                ptr = self.builder.gep(struct_val, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), field_idx)])
                if val.type != ptr.type.pointee:
                    ptr = self.builder.bitcast(ptr, val.type.as_pointer())
                # Auto-bitcast for type erasure
                if val.type != ptr.type.pointee:
                    ptr = self.builder.bitcast(ptr, val.type.as_pointer())
                self.builder.store(val, ptr)
            else:
                raise Exception('Cannot assign to member of Value-type struct')

        elif isinstance(node.target, UnaryExpr):
            if node.target.op == '*':
                # *ptr = val
                # Evaluate 'ptr' (the operand of *)
                ptr_val = self.visit(node.target.operand)
                if val.type != ptr_val.type.pointee:
                    ptr_val = self.builder.bitcast(ptr_val, val.type.as_pointer())
                # Auto-bitcast for type erasure
                if val.type != ptr_val.type.pointee:
                    ptr_val = self.builder.bitcast(ptr_val, val.type.as_pointer())
                self.builder.store(val, ptr_val)
            else:
                raise Exception("Invalid assignment target")

        elif isinstance(node.target, IndexAccess):
            # ptr[i] = val  OR  arr[i] = val
            index_val = self.visit(node.target.index)
            # Evaluate base (e.g. self.ptr)
            ptr_val = self.visit(node.target.object)
            
            # For bootstrap, always treat as pointer indexing if not an array.
            # If it's a pointer to array, GEP needs [0, i]. If it's a pointer to T, GEP needs [i].
            if isinstance(ptr_val.type.pointee, ir.ArrayType):
                ptr = self.builder.gep(ptr_val, [ir.Constant(ir.IntType(32), 0), index_val])
            else:
                ptr = self.builder.gep(ptr_val, [index_val])
                
            if val.type != ptr.type.pointee:
                ptr = self.builder.bitcast(ptr, val.type.as_pointer())
            # Auto-bitcast for type erasure
            if val.type != ptr.type.pointee:
                ptr = self.builder.bitcast(ptr, val.type.as_pointer())
            self.builder.store(val, ptr)

            # If object is a variable, prefer addressable access.
            if isinstance(node.target.object, VariableExpr):
                entry = None
                for scope in reversed(self.scopes):
                        if node.target.object.name in scope:
                            entry = scope[node.target.object.name]
                            break
                if not entry:
                        raise Exception(f"Undefined var {node.target.object.name}")

                if isinstance(entry, tuple) and len(entry) == 3:
                        obj_val, type_name, tag = entry
                        # SSA pointer value (Vulkan) or Vulkan Buffer data pointer.
                        if tag in ("value", "vulkan_buffer"):
                            if not isinstance(obj_val.type, ir.PointerType):
                                raise Exception("Index assignment requires pointer/array base")
                            if isinstance(obj_val.type.pointee, ir.ArrayType):
                                zero = ir.Constant(ir.IntType(32), 0)
                                elem_ptr = self.builder.gep(obj_val, [zero, index_val])
                            else:
                                elem_ptr = self.builder.gep(obj_val, [index_val])
                            # Auto-bitcast for type erasure
                            if val.type != elem_ptr.type.pointee:
                                elem_ptr = self.builder.bitcast(elem_ptr, val.type.as_pointer())
                            self.builder.store(val, elem_ptr)
                            return
                        # Fallback to old behavior if unknown tag
                        obj_ptr = obj_val
                else:
                        obj_ptr, type_name = entry

                # Array alloca: gep (0, idx)
                if isinstance(obj_ptr.type.pointee, ir.ArrayType):
                        zero = ir.Constant(ir.IntType(32), 0)
                        elem_ptr = self.builder.gep(obj_ptr, [zero, index_val])
                        # Auto-bitcast for type erasure
                        if val.type != elem_ptr.type.pointee:
                            elem_ptr = self.builder.bitcast(elem_ptr, val.type.as_pointer())
                        self.builder.store(val, elem_ptr)
                        return

                # Pointer alloca: load base ptr then gep (idx)
                base_ptr = self.builder.load(obj_ptr)
                elem_ptr = self.builder.gep(base_ptr, [index_val])
                # Auto-bitcast for type erasure
                if val.type != elem_ptr.type.pointee:
                    elem_ptr = self.builder.bitcast(elem_ptr, val.type.as_pointer())
                self.builder.store(val, elem_ptr)
                return

            # Fallback: if object evaluates to a pointer value
            obj_val = self.visit(node.target.object)
            if isinstance(obj_val.type, ir.PointerType):
                elem_ptr = self.builder.gep(obj_val, [index_val])
                # Auto-bitcast for type erasure
                if val.type != elem_ptr.type.pointee:
                    elem_ptr = self.builder.bitcast(elem_ptr, val.type.as_pointer())
                self.builder.store(val, elem_ptr)
                return

            raise Exception("Index assignment requires array or pointer base")

        elif isinstance(node.target, MemberAccess):
            # Assign to struct field: struct.field = val
            # We need address of the struct.
            struct_ptr = None

            # Case 1: struct is a variable
            if isinstance(node.target.object, VariableExpr):
                var_name = node.target.object.name
                for scope in reversed(self.scopes):
                        if var_name in scope:
                            struct_ptr, _ = scope[var_name]
                            break
                if not struct_ptr: raise Exception(f"Undefined var {var_name}")

            # Case 2: struct is dereferenced pointer (*ptr).field
            elif isinstance(node.target.object, UnaryExpr) and node.target.object.op == '*':
                # ptr expression
                struct_ptr = self.visit(node.target.object.operand)
            else:
                raise Exception("Assignment to field requires variable or pointer dereference")

            # Get field index
            # Semantic analyzer should have annotated struct type? 
            # Or we look it up.
            # MemberAccess node usually has .struct_type if we annotated it in Semantic
            if not hasattr(node.target, 'struct_type'):
                # Fallback: We need to know the type to find the index.
                # It's tricky to get if not annotated.
                # Assuming Semantic annotated it.
                # If not, we might fail.
                # Let's hope Semantic visit_MemberAccess does it.
                # CHECK Semantic logic later.
                # For now, raise or try?
                raise Exception("CodeGen: Missing struct_type on MemberAccess target. Semantic analysis incomplete.")

            struct_name = node.target.struct_type
            field_index = self.struct_fields[struct_name][node.target.member]

            # Ensure struct_ptr points to the actual struct type (handles type erasure fallback to i8*)
            actual_struct_ty = self.get_llvm_type(struct_name)
            if struct_ptr.type.pointee != actual_struct_ty:
                struct_ptr = self.builder.bitcast(struct_ptr, actual_struct_ty.as_pointer())

            # GEP to field
            zero = ir.Constant(ir.IntType(32), 0)
            idx = ir.Constant(ir.IntType(32), field_index)
            field_ptr = self.builder.gep(struct_ptr, [zero, idx])

            # Auto-bitcast for type erasure
            if val.type != field_ptr.type.pointee:
                field_ptr = self.builder.bitcast(field_ptr, val.type.as_pointer())
            self.builder.store(val, field_ptr)

    def visit_BinaryExpr(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        # Mapping token types to operations
        if node.op == 'PLUS':
            if isinstance(left.type, ir.PointerType):
                return self.builder.gep(left, [right], name="ptr_add")
            if isinstance(right.type, ir.PointerType):
                return self.builder.gep(right, [left], name="ptr_add")
            if left.type == ir.FloatType():
                return self.builder.fadd(left, right, name="faddtmp")
            return self.builder.add(left, right, name="addtmp")
        elif node.op == 'MINUS':
            if isinstance(left.type, ir.PointerType):
                # pointer - offset: gep with neg offset
                neg_right = self.builder.neg(right, name="neg_offset")
                return self.builder.gep(left, [neg_right], name="ptr_sub")
            if left.type == ir.FloatType():
                return self.builder.fsub(left, right, name="fsubtmp")
            return self.builder.sub(left, right, name="subtmp")
        elif node.op == 'STAR':
            if left.type == ir.FloatType():
                return self.builder.fmul(left, right, name="fmultmp")
            return self.builder.mul(left, right, name="multmp")
        elif node.op == 'SLASH':
            if left.type == ir.FloatType():
                return self.builder.fdiv(left, right, name="fdivtmp")
            return self.builder.sdiv(left, right, name="divtmp")
        elif node.op == 'PERCENT':
            if left.type == ir.FloatType():
                return self.builder.frem(left, right, name="fremtmp")
            return self.builder.srem(left, right, name="remtmp")
        elif node.op == 'AND':
            return self.builder.and_(left, right, name="andtmp")
        elif node.op == 'OR':
            return self.builder.or_(left, right, name="ortmp")
        elif node.op == 'EQEQ':
            if left.type == ir.FloatType():
                return self.builder.fcmp_ordered('==', left, right, name="feqtmp")
            return self.builder.icmp_signed('==', left, right, name="eqtmp")
        elif node.op == 'NEQ':
            if left.type == ir.FloatType():
                return self.builder.fcmp_ordered('!=', left, right, name="fneqtmp")
            # Handle pointer != 0 comparison
            if isinstance(left.type, ir.PointerType) and isinstance(right.type, ir.IntType):
                right = ir.Constant(left.type, None) # Convert 0 to null pointer
            return self.builder.icmp_signed('!=', left, right, name="neqtmp")
        elif node.op == 'LT':
            if left.type == ir.FloatType():
                return self.builder.fcmp_ordered('<', left, right, name="flttmp")
            return self.builder.icmp_signed('<', left, right, name="lttmp")
        elif node.op == 'GT':
            if left.type == ir.FloatType():
                return self.builder.fcmp_ordered('>', left, right, name="fgttmp")
            return self.builder.icmp_signed('>', left, right, name="gttmp")
        elif node.op == 'LTE':
            if left.type == ir.FloatType():
                return self.builder.fcmp_ordered('<=', left, right, name="fltettmp")
            return self.builder.icmp_signed('<=', left, right, name="ltetmp")
        elif node.op == 'GTE':
            if left.type == ir.FloatType():
                return self.builder.fcmp_ordered('>=', left, right, name="fgtetmp")
            return self.builder.icmp_signed('>=', left, right, name="gtetmp")
        else:
            raise Exception(f"Unknown operator: {node.op}")

    def visit_IfStmt(self, node):
        cond_val = self.visit(node.condition)

        # Ensure condition is a boolean (i1)
        # If it's not (e.g. i32), compare it to 0
        if cond_val.type != ir.IntType(1):
            cond_val = self.builder.icmp_signed('!=', cond_val, ir.Constant(cond_val.type, 0), name="ifcond")

        then_bb = self.builder.append_basic_block(name="then")
        else_bb = self.builder.append_basic_block(name="else")
        merge_bb = self.builder.append_basic_block(name="ifcont")

        self.builder.cbranch(cond_val, then_bb, else_bb)

        # Generate 'then' block
        self.builder.position_at_end(then_bb)
        for stmt in node.then_branch:
            self.visit(stmt)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)

        # Generate 'else' block
        self.builder.position_at_end(else_bb)
        if node.else_branch:
            for stmt in node.else_branch:
                self.visit(stmt)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)

        # Continue
        self.builder.position_at_end(merge_bb)

    def visit_WhileStmt(self, node):
        cond_bb = self.builder.append_basic_block(name="whilecond")
        loop_bb = self.builder.append_basic_block(name="whileloop")
        end_bb = self.builder.append_basic_block(name="whileend")

        # Jump to condition
        self.builder.branch(cond_bb)

        # Condition Block
        self.builder.position_at_end(cond_bb)
        cond_val = self.visit(node.condition)
        if cond_val.type != ir.IntType(1):
            cond_val = self.builder.icmp_signed('!=', cond_val, ir.Constant(cond_val.type, 0), name="loopcond")
        self.builder.cbranch(cond_val, loop_bb, end_bb)

        # Loop Block
        self.builder.position_at_end(loop_bb)
        
        self.scopes.append({})
        self.loop_stack.append((cond_bb, end_bb, len(self.scopes) - 1, node.label))
        
        for stmt in node.body:
            self.visit(stmt)
            
        self.emit_scope_drops(self.scopes[-1])
        self.scopes.pop()
        self.loop_stack.pop()
        
        if not self.builder.block.is_terminated:
            self.builder.branch(cond_bb) # Loop back to condition
            
        # End Block
        self.builder.position_at_end(end_bb)

    def visit_RegionStmt(self, node):
        # 1. New Scope for logic
        self.scopes.append({})

        # 2. Call Arena_new()
        new_func = self.module.get_global("Arena_new")
        val = self.builder.call(new_func, []) # Returns Arena struct

        # 3. Alloca and store
        arena_ty = self.struct_types['Arena']
        ptr = self.builder.alloca(arena_ty, name=node.name)
        self.builder.store(val, ptr)

        # 4. Register in Scope (with type name 'Arena')
        self.scopes[-1][node.name] = (ptr, 'Arena')

        # 5. Visit Body
        for stmt in node.body:
            self.visit(stmt)

        # 6. Auto-Drop
        self.emit_scope_drops(self.scopes[-1])

        self.scopes.pop()

    def _get_or_create_string(self, string_val, name="str"):
        if not string_val.endswith('\0'):
            string_val += '\0'
        
        # Check if already exists
        if hasattr(self, "_strings") and string_val in self._strings:
            return self.builder.bitcast(self._strings[string_val], ir.IntType(8).as_pointer())
        
        if not hasattr(self, "_strings"):
            self._strings = {}
            
        # Create global constant
        # Use bytearray without the explicit \0 if it was already handled by Python strings
        # Wait, the issue is how it's encoded.
        encoded = string_val.encode('utf-8')
        arr_ty = ir.ArrayType(ir.IntType(8), len(encoded))
        gv = ir.GlobalVariable(self.module, arr_ty, name=f"{name}.{len(self._strings)}")
        gv.global_constant = True
        gv.initializer = ir.Constant(arr_ty, bytearray(encoded))
        gv.linkage = 'internal'
        
        self._strings[string_val] = gv
        return self.builder.bitcast(gv, ir.IntType(8).as_pointer())

    def visit_StringLiteral(self, node, name="str", value_override=None):
        val = value_override if value_override is not None else node.value
        return self._get_or_create_string(val, name=name)

    def visit_IntegerLiteral(self, node):
        # Infer i32 for simplicity or u8/i64 if context?
        # For now return i32 constant
        return ir.Constant(ir.IntType(32), node.value)

    def visit_FloatLiteral(self, node):
        return ir.Constant(ir.FloatType(), node.value)

    def visit_CharLiteral(self, node):
        return ir.Constant(ir.IntType(32), node.value)

    def visit_BooleanLiteral(self, node):
        val = 1 if node.value else 0
        return ir.Constant(ir.IntType(1), val)

    def get_env_struct(self, lambda_node):
        struct_name = f"{lambda_node.lambda_name}_env"
        if struct_name in self.struct_types:
            return self.struct_types[struct_name]
        
        field_types = []
        field_names = []
        if not hasattr(lambda_node, 'captures'): return None

        for name in sorted(lambda_node.captures.keys()):
            type_name = lambda_node.captures[name]
            field_types.append(self.get_llvm_type(type_name))
            field_names.append(name)
            
        struct_ty = ir.LiteralStructType(field_types)
        self.struct_types[struct_name] = struct_ty
        self.struct_fields[struct_name] = {name: i for i, name in enumerate(field_names)}
        return struct_ty

    def visit_VariableExpr(self, node):
        # 1. Primary Lookup (Local Scopes)
        for scope in reversed(self.scopes):
            if node.name in scope:
                entry = scope[node.name]
                if isinstance(entry, tuple) and len(entry) == 3:
                    val_or_ptr, _, tag = entry
                    if tag == "value":
                        return val_or_ptr
                ptr, _ = entry
                return self.builder.load(ptr, name=node.name)

        # 2. Secondary Lookup (Closure Environment)
        # Search for the closest $env in scopes
        for scope in reversed(self.scopes):
            if '$env' in scope:
                env_ptr_raw, _ = scope['$env']
                lambda_node = scope.get('$env_lambda')
                if lambda_node and node.name in getattr(lambda_node, 'captures', {}):
                    struct_name = f"{lambda_node.lambda_name}_env"
                    struct_ty = self.struct_types[struct_name]
                    env_ptr = self.builder.bitcast(env_ptr_raw, struct_ty.as_pointer())
                    field_idx = self.struct_fields[struct_name][node.name]
                    ptr = self.builder.gep(env_ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), field_idx)])
                    return self.builder.load(ptr, name=node.name)

        # 3. Check for Enum Variant (Unit Variant)
        
        # Check for Enum Variant (Unit Variant)
        if '::' in node.name:
             parts = node.name.rsplit('::', 1)
             lhs = parts[0]; rhs = parts[1]
             base_lhs = lhs.split('<')[0]
             if base_lhs in self.enum_definitions:
                  enum_ty, max_size = self.enum_definitions[base_lhs]
                  if rhs in self.enum_types[base_lhs]:
                      tag = self.enum_types[base_lhs][rhs]
                      
                      # Create Enum Value
                      enum_val = ir.Constant(enum_ty, ir.Undefined)
                      tag_val = ir.Constant(ir.IntType(32), tag)
                      enum_val = self.builder.insert_value(enum_val, tag_val, 0)
                      return enum_val

        raise Exception(f"Ref to undefined variable: {node.name}")

    def visit_LambdaExpr(self, node):
        func = self.module.get_global(node.lambda_name)
        if not func:
             raise Exception(f"Lambda function '{node.lambda_name}' not found in LLVM module")
        
        # 1. Environment Packing
        if hasattr(node, 'captures') and node.captures:
             struct_ty = self.get_env_struct(node)
             # For simplicity in bootstrap, we use stack-based environment (alloca)
             # In a real compiler, escaped closures would need heap allocation (Arena)
             env_ptr = self.builder.alloca(struct_ty, name="closure_env")
             struct_name = f"{node.lambda_name}_env"
             for name in sorted(node.captures.keys()):
                 # Visit a temporary VariableExpr to get current value of the capture
                 val = self.visit(VariableExpr(name))
                 idx = self.struct_fields[struct_name][name]
                 field_ptr = self.builder.gep(env_ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), idx)])
                 self.builder.store(val, field_ptr)
             env_ptr_raw = self.builder.bitcast(env_ptr, ir.IntType(8).as_pointer())
        else:
             env_ptr_raw = ir.Constant(ir.IntType(8).as_pointer(), None)
             
        # 2. Pack Fat Pointer: { func_ptr, env_ptr }
        func_ptr_raw = self.builder.bitcast(func, ir.IntType(8).as_pointer())
        
        fat_ty = ir.LiteralStructType([ir.IntType(8).as_pointer(), ir.IntType(8).as_pointer()])
        fat_ptr = ir.Constant(fat_ty, ir.Undefined)
        fat_ptr = self.builder.insert_value(fat_ptr, func_ptr_raw, 0)
        fat_ptr = self.builder.insert_value(fat_ptr, env_ptr_raw, 1)
        return fat_ptr

    def _emit_closure_call(self, fat_ptr, callee_type_name, args):
        func_ptr_raw = self.builder.extract_value(fat_ptr, 0)
        env_ptr_raw = self.builder.extract_value(fat_ptr, 1)
        
        main_part, _, ret_str = callee_type_name.rpartition(')->')
        ret_type = self.get_llvm_type(ret_str.strip())
        params_str = main_part[3:]
        arg_types = [ir.IntType(8).as_pointer()]
        if params_str:
             # Very basic comma split; won't work for nested fn types in params but good for bootstrap
             for pt in params_str.split(','):
                  if pt.strip(): arg_types.append(self.get_llvm_type(pt.strip()))
        
        func_ty = ir.FunctionType(ret_type, arg_types)
        func_ptr = self.builder.bitcast(func_ptr_raw, func_ty.as_pointer())
        
        processed_args = [env_ptr_raw] + [self.visit(arg) for arg in args]
        return self.builder.call(func_ptr, processed_args)
    def visit_CallExpr(self, node):
        callee = node.callee
        if isinstance(callee, VariableExpr):
            callee = callee.name
            
        callee_name = callee
        
        # 1. Built-in generic functions (Intrinsics)
        if isinstance(callee_name, str) and callee_name.startswith('cast<'):
            target_ty_name = callee_name[5:].rstrip('>')
            target_ty = self.get_llvm_type(target_ty_name)
            val = self.visit(node.args[0])
            
            # Pointer to pointer
            if isinstance(val.type, ir.PointerType) and isinstance(target_ty, ir.PointerType):
                return self.builder.bitcast(val, target_ty)
            # Int to pointer
            if isinstance(val.type, ir.IntType) and isinstance(target_ty, ir.PointerType):
                return self.builder.inttoptr(val, target_ty)
            # Pointer to int
            if isinstance(val.type, ir.PointerType) and isinstance(target_ty, ir.IntType):
                return self.builder.ptrtoint(val, target_ty)
            # Int to int
            if isinstance(val.type, ir.IntType) and isinstance(target_ty, ir.IntType):
                if val.type.width < target_ty.width:
                    return self.builder.zext(val, target_ty)
                elif val.type.width > target_ty.width:
                    return self.builder.trunc(val, target_ty)
                return val
            return self.builder.bitcast(val, target_ty)

        elif isinstance(callee_name, str) and callee_name.startswith('sizeof<'):
            type_name = callee_name[7:].rstrip('>')
            llvm_ty = self.get_llvm_type(type_name)
            null_ptr = ir.Constant(llvm_ty.as_pointer(), None)
            gep = self.builder.gep(null_ptr, [ir.Constant(ir.IntType(32), 1)])
            return self.builder.ptrtoint(gep, ir.IntType(32))

        elif isinstance(callee_name, str) and callee_name.startswith('ptr_offset<'):
            ptr_val = self.visit(node.args[0])
            idx_val = self.visit(node.args[1])
            return self.builder.gep(ptr_val, [idx_val])

        # 2. Local variables / Function pointers
        if isinstance(callee_name, str):
            for scope in reversed(self.scopes):
                if callee_name in scope:
                    entry = scope[callee_name]
                    if isinstance(entry, tuple) and len(entry) >= 2:
                        ptr, ptype_name = entry
                        if isinstance(ptype_name, str) and ptype_name.startswith('fn('):
                            fat_ptr = self.builder.load(ptr)
                            return self._emit_closure_call(fat_ptr, ptype_name, node.args)

        # 3. Specific Intrinsics
        if callee_name == "print":
            val = self.visit(node.args[0])
            voidptr_ty = ir.IntType(8).as_pointer()
            arg_node = node.args[0] if node.args else None
            arg_type_name = getattr(arg_node, "type_name", None)
            int_like = False
            if arg_type_name == "char":
                fmt_str = self.visit_StringLiteral(None, name="fmt_c", value_override="%c\n\0")
                if val.type != ir.IntType(32): val = self.builder.zext(val, ir.IntType(32))
                int_like = True
            elif arg_type_name in ("bool", "u8", "i32"):
                fmt_str = self.visit_StringLiteral(None, name="fmt_d", value_override="%d\n\0")
                if val.type != ir.IntType(32): val = self.builder.zext(val, ir.IntType(32))
                int_like = True
            elif val.type == ir.FloatType():
                fmt_str = self.visit_StringLiteral(None, name="fmt_f", value_override="%f\n\0")
            else:
                fmt_str = self.visit_StringLiteral(None, name="fmt_s", value_override="%s\n\0")
            
            fmt_arg = self.builder.bitcast(fmt_str, voidptr_ty)
            if int_like: self.builder.call(self.printf, [fmt_arg, val])
            elif val.type == ir.FloatType(): self.builder.call(self.printf, [fmt_arg, self.builder.fpext(val, ir.DoubleType())])
            else:
                if isinstance(val.type, ir.PointerType): val_arg = self.builder.bitcast(val, voidptr_ty)
                else: val_arg = val
                self.builder.call(self.printf, [fmt_arg, val_arg])
            return None

        elif callee_name == "fprintf":
            file_ptr = self.visit(node.args[0])
            fmt_ptr = self.visit(node.args[1])
            
            # Coleta args variádicos
            call_args = [file_ptr, fmt_ptr]
            for i in range(2, len(node.args)):
                call_args.append(self.visit(node.args[i]))
            
            # Usa a função já declarada no módulo (pelo ExternBlock do Semantic)
            func = self.module.globals.get("fprintf")
            if not func:
                void_ptr = ir.IntType(8).as_pointer()
                fprintf_ty = ir.FunctionType(ir.IntType(32), [void_ptr, void_ptr], var_arg=True)
                func = ir.Function(self.module, fprintf_ty, name="fprintf")
            
            # Garante que os dois primeiros args fixos sejam compatíveis com a assinatura
            for i in range(min(len(call_args), len(func.function_type.args))):
                expected = func.function_type.args[i]
                actual = call_args[i]
                if actual.type != expected:
                    if isinstance(actual.type, ir.PointerType) and isinstance(expected, ir.IntType):
                        call_args[i] = self.builder.ptrtoint(actual, expected)
                    elif isinstance(actual.type, ir.IntType) and isinstance(expected, ir.PointerType):
                        call_args[i] = self.builder.inttoptr(actual, expected)
                    else:
                        call_args[i] = self.builder.bitcast(actual, expected)
            
            return self.builder.call(func, call_args)

        elif callee_name == "fopen":
            path_ptr = self.visit(node.args[0])
            mode_ptr = self.visit(node.args[1])
            void_ptr = ir.IntType(8).as_pointer()
            if path_ptr.type != void_ptr: path_ptr = self.builder.bitcast(path_ptr, void_ptr)
            if mode_ptr.type != void_ptr: mode_ptr = self.builder.bitcast(mode_ptr, void_ptr)
            return self.builder.call(self.fopen, [path_ptr, mode_ptr])

        elif callee_name == "fclose":
            file_ptr = self.visit(node.args[0])
            void_ptr = ir.IntType(8).as_pointer()
            if file_ptr.type != void_ptr: file_ptr = self.builder.bitcast(file_ptr, void_ptr)
            return self.builder.call(self.fclose, [file_ptr])

        elif callee_name == "gpu::global_id":
            if self.target == "spirv" and hasattr(self, "spirv_global_invocation_id"):
                gid = self.builder.load(self.spirv_global_invocation_id, name="spirv_gid")
                return self.builder.extract_element(gid, ir.Constant(ir.IntType(32), 0), name="spirv_gid_x")
            if not hasattr(self, "gpu_global_id") or self.gpu_global_id is None:
                self.gpu_global_id = ir.GlobalVariable(self.module, ir.IntType(32), name="__gpu_global_id_sim")
                self.gpu_global_id.initializer = ir.Constant(ir.IntType(32), 0)
                self.gpu_global_id.linkage = "internal"
            return self.builder.load(self.gpu_global_id, name="gpu_global_id")

        elif callee_name == "gpu::dispatch":
            if len(node.args) < 2: raise Exception("gpu::dispatch expects at least 2 args")
            fn_ref = node.args[0]
            if not isinstance(fn_ref, VariableExpr): raise Exception("gpu::dispatch first argument must be a function name")
            kernel_name = fn_ref.name
            threads_val = self.visit(node.args[1])
            gpu_args = [self.visit(node.args[i]) for i in range(2, len(node.args))]
            if self.target == "native":
                if "__nexa_gpu_dispatch" not in self.module.globals:
                    void_ptr = ir.IntType(8).as_pointer()
                    dispatch_ty = ir.FunctionType(ir.IntType(32), [void_ptr, ir.IntType(32), ir.IntType(32), void_ptr.as_pointer()])
                    ir.Function(self.module, dispatch_ty, name="__nexa_gpu_dispatch")
                dispatch_fn = self.module.globals["__nexa_gpu_dispatch"]
                k_name_const = self._get_or_create_string(kernel_name)
                i32 = ir.IntType(32)
                arg_count = len(gpu_args)
                args_array = self.builder.alloca(ir.IntType(8).as_pointer(), size=ir.Constant(i32, max(1, arg_count)), name="gpu_args_array")
                for i, arg in enumerate(gpu_args):
                    arg_ptr = self.builder.alloca(arg.type)
                    self.builder.store(arg, arg_ptr)
                    void_arg = self.builder.bitcast(arg_ptr, ir.IntType(8).as_pointer())
                    ptr_to_slot = self.builder.gep(args_array, [ir.Constant(i32, i)])
                    self.builder.store(void_arg, ptr_to_slot)
                self.builder.call(dispatch_fn, [k_name_const, threads_val, ir.Constant(i32, arg_count), args_array])
            return None

        # 4. Struct Instantiation
        if isinstance(callee_name, str) and (callee_name in self.struct_types or ('<' in callee_name and callee_name.split('<')[0] in self.struct_types)):
            struct_key = callee_name
            if '<' in struct_key and struct_key not in self.struct_types:
                 base_name = struct_key.split('<')[0]
                 if base_name in self.struct_types: struct_key = base_name
                 else:
                     struct_ty = self.module.context.get_identified_type(struct_key)
                     self.struct_types[struct_key] = struct_ty
            struct_ty = self.struct_types[struct_key]
            struct_val = ir.Constant(struct_ty, ir.Undefined)
            for i, arg in enumerate(node.args):
                arg_val = self.visit(arg)
                struct_val = self.builder.insert_value(struct_val, arg_val, i)
            return struct_val

        # 5. Enum Variant Instantiation
        if isinstance(node.callee, str) and '::' in node.callee:
            parts = node.callee.split('::', 1)
            enum_key = parts[0]
            if enum_key not in self.enum_definitions and '<' in enum_key:
                 enum_key = enum_key.split('<')[0]
            if enum_key in self.enum_definitions:
                enum_ty, max_size = self.enum_definitions[enum_key]
                tag = self.enum_types[enum_key][parts[1]]
                enum_val = ir.Constant(enum_ty, ir.Undefined)
                tag_val = ir.Constant(ir.IntType(32), tag)
                enum_val = self.builder.insert_value(enum_val, tag_val, 0)
                if node.args:
                    payload_val = self.visit(node.args[0])
                    enum_ptr = self.builder.alloca(enum_ty)
                    self.builder.store(enum_val, enum_ptr)
                    zero = ir.Constant(ir.IntType(32), 0)
                    one = ir.Constant(ir.IntType(32), 1)
                    data_ptr = self.builder.gep(enum_ptr, [zero, one])
                    payload_ptr = self.builder.bitcast(data_ptr, payload_val.type.as_pointer())
                    self.builder.store(payload_val, payload_ptr)
                    return self.builder.load(enum_ptr)
                return enum_val

        # 6. Built-in complex intrinsics
        if callee_name == "fs::read_file":
            path = self.visit(node.args[0])
            void_ptr = ir.IntType(8).as_pointer()
            i32 = ir.IntType(32)
            i64 = ir.IntType(64)
            mode_arr = self.visit_StringLiteral(None, name="fs_mode_rb", value_override="rb\0")
            mode = self.builder.bitcast(mode_arr, void_ptr)
            f = self.builder.call(self.fopen, [path, mode], name="f")
            is_null = self.builder.icmp_unsigned("==", f, ir.Constant(void_ptr, None))
            with self.builder.if_then(is_null): self.builder.call(self.exit_func, [ir.Constant(i32, 1)])
            self.builder.call(self.fseek, [f, ir.Constant(i64, 0), ir.Constant(i32, 2)])
            sz = self.builder.call(self.ftell, [f], name="file_size")
            self.builder.call(self.fseek, [f, ir.Constant(i64, 0), ir.Constant(i32, 0)])
            sz_i32 = self.builder.trunc(sz, i32)
            buf_void = self.builder.call(self.malloc, [self.builder.add(sz_i32, ir.Constant(i32, 1))])
            buf = self.builder.bitcast(buf_void, ir.IntType(8).as_pointer())
            self.builder.call(self.fread, [buf_void, ir.Constant(i64, 1), sz, f])
            self.builder.call(self.fclose, [f])
            buf_struct_ty = self.get_llvm_type("Buffer<u8>")
            bval = ir.Constant(buf_struct_ty, ir.Undefined)
            bval = self.builder.insert_value(bval, buf, 0)
            bval = self.builder.insert_value(bval, sz_i32, 1)
            bval = self.builder.insert_value(bval, sz_i32, 2)
            return bval

        elif callee_name in ("fs::write_file", "fs::append_file"):
            path = self.visit(node.args[0])
            data = self.visit(node.args[1])
            length = self.visit(node.args[2])
            mode = "wb\0" if callee_name == "fs::write_file" else "ab\0"
            mode_ptr = self.builder.bitcast(self.visit_StringLiteral(None, value_override=mode), ir.IntType(8).as_pointer())
            f = self.builder.call(self.fopen, [path, mode_ptr])
            self.builder.call(self.fwrite, [data, ir.Constant(ir.IntType(64), 1), self.builder.zext(length, ir.IntType(64)), f])
            self.builder.call(self.fclose, [f])
            return None

        elif callee_name == "memcpy":
            dest = self.visit(node.args[0])
            src = self.visit(node.args[1])
            size = self.visit(node.args[2])
            return self.builder.call(self.memcpy, [dest, src, size, ir.Constant(ir.IntType(1), 0)])

        # 7. Regular Function Calls
        callee_func_name = callee_name
        if isinstance(callee_func_name, str) and '::' in callee_func_name:
            parts = callee_func_name.split('::')
            struct_name = parts[0].split('<')[0]
            callee_func_name = f"{struct_name}_{parts[1]}"

        if isinstance(callee_func_name, str) and callee_func_name in self.module.globals:
            callee_func = self.module.globals[callee_func_name]
            processed_args = [self.visit(arg) for arg in node.args]
            for i in range(min(len(processed_args), len(callee_func.function_type.args))):
                if processed_args[i].type != callee_func.function_type.args[i]:
                    processed_args[i] = self.builder.bitcast(processed_args[i], callee_func.function_type.args[i])
            return self.builder.call(callee_func, processed_args)

        raise Exception(f"Unknown function call: {callee_name}")

    def visit_ReturnStmt(self, node):
        # Evaluate return value FIRST (before drops)
        ret_val = None
        if node.value:
            ret_val = self.visit(node.value)

        # Unwind scopes: Drop everything in current function scopes (LIFO)
        for scope in reversed(self.scopes):
            self.emit_scope_drops(scope)

        if not self.builder.block.is_terminated:
            if getattr(self._current_function_node, 'is_async', False):
                # Set done = true and store result in manual state
                for scope in reversed(self.scopes):
                    if '$async_state' in scope:
                        state = scope['$async_state']
                        done_ptr = self.builder.gep(state, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
                        self.builder.store(ir.Constant(ir.IntType(1), 1), done_ptr)
                        if ret_val:
                            res_ptr = self.builder.gep(state, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 1)])
                            self.builder.store(ret_val, res_ptr)
                        void_ptr = ir.IntType(8).as_pointer()
                        h = self.builder.bitcast(state, void_ptr)
                        self.builder.ret(h)
                        break
            elif ret_val:
                self.builder.ret(ret_val)
            else:
                self.builder.ret_void()

    def _declare_function(self, node):
        if node.generics: return

        # Determine function type
        arg_types = []
        if not (self.target == "spirv" and self.spirv_env == "vulkan" and node.is_kernel):
            for pname, ptype in node.params:
                arg_types.append(self.get_llvm_type(ptype))
        
        ret_type = ir.VoidType()
        if getattr(node, 'is_async', False):
            # Async functions return a coroutine handle (i8*)
            ret_type = ir.IntType(8).as_pointer()
        elif node.return_type != 'void':
            ret_type = self.get_llvm_type(node.return_type)

        # Add env pointer to lambda signatures
        if getattr(node, 'is_lambda', False):
            # Lambda functions always take environment as first argument (fat pointer convention)
            arg_types.insert(0, ir.IntType(8).as_pointer())

        func_ty = ir.FunctionType(ret_type, arg_types, var_arg=getattr(node, 'is_vararg', False))

        # Check if exists
        try:
            func = self.module.get_global(node.name)
        except (KeyError, AttributeError):
            func = ir.Function(self.module, func_ty, name=node.name)

        if node.is_kernel:
            func.calling_convention = 'spir_kernel'

        return func

    def visit_ExternBlock(self, node):
        pass # Headers already declared in Pass 2

    def visit_FunctionDef(self, node):
        if getattr(node, 'generics', None): return
        
        # If it's a declaration only (extern), stop here
        if node.body is None:
            return

        # Function already declared in Pass 2
        func = self.module.get_global(node.name)
        if self.target == "spirv" and self.spirv_env == "vulkan" and node.is_kernel:
            self._kernel_function_names.add(node.name)

        self._current_function_name = node.name
        self._current_function_node = node
        self._current_function_is_kernel = bool(node.is_kernel)
        self.current_lambda_node = getattr(node, 'lambda_node', None)
        if self.current_lambda_node:
             self.get_env_struct(self.current_lambda_node)

        entry_block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)

        # Create new scope
        self.scopes.append({})

        is_async = getattr(node, 'is_async', False)
        if is_async:
            # Manual Coroutine State
            res_ty = self.get_llvm_type(node.return_type)
            # struct { i1 done, T result }
            state_ty = ir.LiteralStructType([ir.IntType(1), res_ty])
            state_ptr = self.builder.call(self.malloc, [ir.Constant(ir.IntType(32), 16)]) # Simplified size
            state = self.builder.bitcast(state_ptr, state_ty.as_pointer())
            
            # Init state: done = false
            done_ptr = self.builder.gep(state, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
            self.builder.store(ir.Constant(ir.IntType(1), 0), done_ptr)
            
            self.scopes[-1]['$async_state'] = state

        # Register arguments in scope
        if self.target == "spirv" and self.spirv_env == "vulkan" and node.is_kernel:
            for pname, ptype in node.params:
                if isinstance(ptype, str) and ptype.startswith("Buffer<"):
                    data_gv, len_gv = self._get_or_declare_vulkan_buffer_globals(node.name, pname, ptype)
                    # Store as special entry; MemberAccess will resolve ptr/len.
                    self.scopes[-1][pname] = (data_gv, ptype, "vulkan_buffer")
                    self._vulkan_buffer_args[(node.name, pname)] = (data_gv, len_gv)
                else:
                    gv = self._get_or_declare_vulkan_kernel_arg_global(node.name, pname, ptype)
                    # Treat global like a pointer slot for VariableExpr loads.
                    self.scopes[-1][pname] = (gv, ptype)
        else:
            offset = 1 if getattr(node, 'is_lambda', False) else 0
            if offset == 1:
                 env_ptr = func.args[0]
                 env_ptr.name = "env"
                 # Store env ptr and the lambda node in scope so closures can find it
                 self.scopes[-1]['$env'] = (env_ptr, 'i8*')
                 self.scopes[-1]['$env_lambda'] = getattr(node, 'lambda_node', None)

            # Special case for main(argc, argv)
            is_main = node.name == "main" and not getattr(node, 'is_lambda', False)
            
            for i, (pname, ptype) in enumerate(node.params):
                arg_val = func.args[i + offset]
                arg_val.name = pname
                
                # Use recorded type if it's main args
                actual_ptype = ptype
                if is_main:
                    if i == 0: actual_ptype = "i32"
                    if i == 1: actual_ptype = "u8**"
                
                alloca = self.builder.alloca(self.get_llvm_type(actual_ptype), name=pname)
                self.builder.store(arg_val, alloca)

                # Store in scope: (Pointer, TypeName)
                self.scopes[-1][pname] = (alloca, actual_ptype)

        # Process body
        for stmt in node.body:
            if self.builder.block.is_terminated:
                break
            self.visit(stmt)

        # Add return void/undef if missing
        if not self.builder.block.is_terminated:
            if isinstance(func.function_type.return_type, ir.VoidType):
                self.builder.ret_void()
            elif isinstance(func.function_type.return_type, ir.IntType):
                self.builder.ret(ir.Constant(func.function_type.return_type, 0))
            elif isinstance(func.function_type.return_type, ir.PointerType):
                self.builder.ret(ir.Constant(func.function_type.return_type, None)) # Null ptr
            else:
                self.builder.unreachable()

        self.scopes.pop()
