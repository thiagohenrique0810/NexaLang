# Self-hosting prototypes

These stages are historical experiments. The supported compiler is `../nxc`, backed by Python and LLVM. A successful compilation of a small input by one stage does not establish that the compiler can compile itself.

`stage5_full.nxl` is incomplete: function parsing assumes empty parameter lists, function registration and emission still hard-code `main`, struct fields lower to `i32`, and string globals are omitted. Token and symbol storage also use fixed capacities. Do not use this compiler for untrusted or large input.

Completion requires implementing the missing language subset, bounded storage, and a reproducible bootstrap test that builds successive compiler stages and compares their behavior/artifacts. No self-hosting completion claim is made by the current repository.
