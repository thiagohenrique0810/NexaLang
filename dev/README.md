# dev/ — Development & Patch Scripts

Historical scripts used during NexaLang compiler development. These were one-time patches applied to the bootstrap compiler during its evolution. They are preserved for reference but **should not be re-run**.

## Status

| Script | Purpose | Status |
|--------|---------|--------|
| `add_method_visitor.py` | Injected `visit_MethodCall` into codegen | ✅ Applied |
| `fix_method_call.py` | Fixed `&mut self` pointer coercion | ✅ Applied |
| `fix_method_lookup.py` | Fixed variable lookup in parent scopes | ✅ Applied |
| `refine_method_call.py` | Complete rewrite of method call visitor | ✅ Applied |
| `add_debug.py` | Debug prints for type mismatch diagnosis | ⚠️ Debug only |
| `fix_codegen_indent.py` | Normalized mixed indentation | ✅ Applied |
| `normalize_indent.py` | Global 4-space indentation fix | ✅ Applied |
| `fix_indent.py` | Binary rounding indentation fix | ✅ Applied |
| `fix_get_llvm_type.py` | Fixed generic struct type lookup | ✅ Applied |
| `patch_codegen.py` | Added `MemberAccess` assignment handling | ✅ Applied |
| `patch_codegen_generics.py` | Added `_current_generics` tracking | ✅ Applied |
| `patch_stores.py` | Auto-bitcast before stores | ✅ Applied |
| `fix_parse_fn.py` | Added return type parsing (`-> Type`) | ✅ Applied |
| `fix_register.py` | Registered struct methods in semantic analyzer | ✅ Applied |
| `create_test.py` | Generated `methods_test.nxl` example | ✅ Applied |
| `verify_semantic.py` | Validated `ForStmt` visitor exists | ✅ Applied |
| `temp_semantic.py` | Temporary/corrupted file | ❌ Obsolete |

## Lessons Learned

1. **Method call desugaring** required 5 iterations — signals need for better upfront design of method dispatch
2. **Pointer/reference handling** (`&self` vs `&mut self`) was a persistent source of bugs
3. **Indentation issues** showed the codebase needed an auto-formatter or linter early on
4. **Generic type resolution** needs careful tracking of `_current_generics` context through codegen

## Subdirectories

- `scratch/` — Experimental throwaway code
- `artifacts/` — Build output directory (gitignored)
