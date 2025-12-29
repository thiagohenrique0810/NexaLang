import llvmlite.binding as llvm
import ctypes
import os
import sys

# Define types
c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_int32 = ctypes.c_int32
c_size_t = ctypes.c_size_t

# Buffer<T> struct layout
class Buffer(ctypes.Structure):
    _fields_ = [("ptr", c_void_p), ("len", c_int32), ("cap", c_int32)]

# --- Implementations ---

def fs_read_file(path_ptr):
    path = ctypes.string_at(path_ptr).decode('utf-8')
    # print(f"[JIT] Reading file: {path}")
    if not os.path.exists(path):
        print(f"Error: File not found: {path} (cwd: {os.getcwd()})")
        # Return empty buffer or null?
        return Buffer(None, 0, 0)
    
    with open(path, 'rb') as f:
        content = f.read()
    
    length = len(content)
    # Alloc memory that won't be GC'd immediately?
    # We use libc malloc so JIT code can free it?
    # If we use Python bytes, pointer might be invalid after return?
    # Yes. We MUST use malloc.
    
    libc = ctypes.CDLL(None) if os.name != 'nt' else ctypes.CDLL('msvcrt')
    buf_ptr = libc.malloc(length + 1)
    ctypes.memmove(buf_ptr, content, length)
    # Null terminate?
    ctypes.memset(buf_ptr + length, 0, 1)
    
    return Buffer(buf_ptr, length, length + 1)

def fs_write_file(path_ptr, data_ptr, length):
    path = ctypes.string_at(path_ptr).decode('utf-8')
    data = ctypes.string_at(data_ptr, length) # Read raw bytes
    # print(f"[JIT] Writing file: {path} ({length} bytes)")
    
    # Ensure dir exists
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        
    with open(path, 'wb') as f:
        f.write(data)

def fs_append_file(path_ptr, data_ptr, length):
    path = ctypes.string_at(path_ptr).decode('utf-8')
    data = ctypes.string_at(data_ptr, length)
    # print(f"[JIT] Appending to file: {path} ({length} bytes)")
    
    with open(path, 'ab') as f:
        f.write(data)

def my_print(s_ptr):
    s = ctypes.string_at(s_ptr).decode('utf-8')
    print(s)

# --- Types for Callbacks ---

# fs::read_file(i8*) -> Buffer
# Returning struct by value via ctypes callback is tricky. 
# LLVM ABI for struct return usually passes hidden pointer as first arg (sret).
# python JIT usually follows C ABI.
# If struct is small (16 bytes), on x64 Windows/Linux it matches register return?
# Windows x64: Structs 1,2,4,8 bytes return in RAX. Others by ref (hidden ptr).
# Buffer is 16 bytes.
# So it expects a pointer!
# I need to change `fs_read_file` signature to accept `sret` pointer!
# void fs_read_file(Buffer* out, char* path)

def fs_read_file_sret(out_ptr, path_ptr):
    # Call internal logic
    buf = fs_read_file(path_ptr)
    # Copy buf to out_ptr
    ctypes.memmove(out_ptr, ctypes.byref(buf), ctypes.sizeof(Buffer))

# --- JIT Engine ---

def run_jit(llvm_ir):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    
    # Create Module
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    
    # Create EE
    # We use MCJIT
    ee = llvm.create_mcjit_compiler(mod, target_machine)
    ee.finalize_object()
    
    # Register symbols
    # libc symbols usually available if process linked them?
    # Windows: msvcrt
    if os.name == 'nt':
        libc = ctypes.CDLL('msvcrt')
    else:
        libc = ctypes.CDLL(None)
    
    # We map "malloc" -> libc.malloc
    # On Windows, symbol might be "malloc".
    llvm.add_symbol("malloc", ctypes.cast(libc.malloc, c_void_p).value)
    llvm.add_symbol("realloc", ctypes.cast(libc.realloc, c_void_p).value)
    llvm.add_symbol("free", ctypes.cast(libc.free, c_void_p).value)
    llvm.add_symbol("memcpy", ctypes.cast(libc.memcpy, c_void_p).value)
    llvm.add_symbol("printf", ctypes.cast(libc.printf, c_void_p).value)
    
    # Custom symbols
    # Verify ABI for Buffer return
    # Assuming generic x64, >8 bytes returns via sret.
    # LLVM IR usually marks it explicitly: `define void @f(%Buffer* sret %0, ...)`
    # But checking generated IR is hard dynamically.
    # We'll assume sret for fs::read_file.
    
    FS_READ_PROTO = ctypes.CFUNCTYPE(None, c_void_p, c_void_p)
    fs_read_func = FS_READ_PROTO(fs_read_file_sret)
    llvm.add_symbol("fs::read_file", ctypes.cast(fs_read_func, c_void_p).value)
    
    FS_WRITE_PROTO = ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_int32)
    fs_write_func = FS_WRITE_PROTO(fs_write_file)
    llvm.add_symbol("fs::write_file", ctypes.cast(fs_write_func, c_void_p).value)
    
    FS_APPEND_PROTO = ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_int32)
    fs_append_func = FS_APPEND_PROTO(fs_append_file)
    llvm.add_symbol("fs::append_file", ctypes.cast(fs_append_func, c_void_p).value)
    
    # Run main
    # main prototype: i32 main()
    main_ptr = ee.get_function_address("main")
    if not main_ptr:
        print("Error: 'main' function not found in JIT module.")
        return 1
        
    main_func = ctypes.CFUNCTYPE(c_int32)(main_ptr)
    
    print("[JIT] Running main...")
    try:
        ret = main_func()
        print(f"[JIT] main returned: {ret}")
        return ret
    except OSError as e:
        print(f"[JIT] Runtime Exception: {e}")
        return 1
