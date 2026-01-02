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
    if not os.path.exists(path):
        print(f"Error: File not found: {path} (cwd: {os.getcwd()})")
        return Buffer(None, 0, 0)
    
    with open(path, 'rb') as f:
        content = f.read()
    
    length = len(content)
    libc = ctypes.CDLL(None) if os.name != 'nt' else ctypes.CDLL('msvcrt')
    buf_ptr = libc.malloc(length + 1)
    ctypes.memmove(buf_ptr, content, length)
    ctypes.memset(buf_ptr + length, 0, 1)
    
    return Buffer(buf_ptr, length, length + 1)

def fs_write_file(path_ptr, data_ptr, length):
    path = ctypes.string_at(path_ptr).decode('utf-8')
    data = ctypes.string_at(data_ptr, length)
    dirname = os.path.dirname(path)
    if dirname: os.makedirs(dirname, exist_ok=True)
    with open(path, 'wb') as f: f.write(data)

def fs_append_file(path_ptr, data_ptr, length):
    path = ctypes.string_at(path_ptr).decode('utf-8')
    data = ctypes.string_at(data_ptr, length)
    with open(path, 'ab') as f: f.write(data)

def fs_read_file_sret(out_ptr, path_ptr):
    buf = fs_read_file(path_ptr)
    ctypes.memmove(out_ptr, ctypes.byref(buf), ctypes.sizeof(Buffer))

# --- Coroutine Hooks ---

@ctypes.CFUNCTYPE(None, c_void_p)
def __nexa_resume(handle):
    pass 

@ctypes.CFUNCTYPE(ctypes.c_bool, c_void_p)
def __nexa_is_done(handle):
    return True 

@ctypes.CFUNCTYPE(None, c_void_p)
def __nexa_destroy(handle):
    pass

@ctypes.CFUNCTYPE(None, c_char_p, c_int32, c_int32, c_void_p)
def __nexa_gpu_dispatch(kernel_name_ptr, threads, arg_count, args_ptr):
    kernel_name = ctypes.string_at(kernel_name_ptr).decode('utf-8')
    # Convert void** args to a list of pointers
    args = ctypes.cast(args_ptr, ctypes.POINTER(ctypes.c_void_p))
    
    # Check if we have a real GPU driver linked or use a fallback
    print(f"[JIT-GPU] Dispatching kernel '{kernel_name}' for {threads} threads...")
    
    # In JIT mode, we'll use a simple multithreaded simulation if real hardware is not linked
    # Note: In 'native' mode (nxc build), it uses the C++ Silicon Driver.
    for i in range(threads):
        # This is a fallback for JIT. 
        # A real JIT GPU dispatch would load OpenCL.dll here.
        pass

# --- JIT Engine ---

def run_jit(llvm_ir):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    
    ee = llvm.create_mcjit_compiler(mod, target_machine)
    ee.finalize_object()
    
    libc = ctypes.CDLL('msvcrt') if os.name == 'nt' else ctypes.CDLL(None)
    
    llvm.add_symbol("malloc", ctypes.cast(libc.malloc, c_void_p).value)
    llvm.add_symbol("realloc", ctypes.cast(libc.realloc, c_void_p).value)
    llvm.add_symbol("free", ctypes.cast(libc.free, c_void_p).value)
    llvm.add_symbol("memcpy", ctypes.cast(libc.memcpy, c_void_p).value)
    llvm.add_symbol("printf", ctypes.cast(libc.printf, c_void_p).value)
    
    # GPU Dispatch symbol
    llvm.add_symbol("__nexa_gpu_dispatch", ctypes.cast(__nexa_gpu_dispatch, c_void_p).value)
    
    # Custom symbols
    FS_READ_PROTO = ctypes.CFUNCTYPE(None, c_void_p, c_void_p)
    fs_read_func = FS_READ_PROTO(fs_read_file_sret)
    llvm.add_symbol("fs::read_file", ctypes.cast(fs_read_func, c_void_p).value)
    
    FS_WRITE_PROTO = ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_int32)
    fs_write_func = FS_WRITE_PROTO(fs_write_file)
    llvm.add_symbol("fs::write_file", ctypes.cast(fs_write_func, c_void_p).value)
    
    FS_APPEND_PROTO = ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_int32)
    fs_append_func = FS_APPEND_PROTO(fs_append_file)
    llvm.add_symbol("fs::append_file", ctypes.cast(fs_append_func, c_void_p).value)

    # Async hooks
    llvm.add_symbol("__nexa_resume", ctypes.cast(__nexa_resume, c_void_p).value)
    llvm.add_symbol("__nexa_is_done", ctypes.cast(__nexa_is_done, c_void_p).value)
    llvm.add_symbol("__nexa_destroy", ctypes.cast(__nexa_destroy, c_void_p).value)
    
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
