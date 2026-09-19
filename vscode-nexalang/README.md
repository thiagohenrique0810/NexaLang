# NexaLang for VS Code

Syntax highlighting, snippets, diagnostics, hover, document symbols, completion,
and definition navigation for `.nxl` files. The language server uses Python 3 and
the compiler frontend. LLVM and `llvmlite` are not required for editor features.

## Install from a checkout

From `vscode-nexalang/`:

```sh
npm install
npm run package
code --install-extension nexalang-0.2.0.vsix
```

Packaging runs `prepare-server`, copying the LSP, its Python frontend modules, and
`std/` into the VSIX. These generated copies are ignored by git. An installed
extension works in another project without requiring the NexaLang compiler
repository in that workspace. Node dependencies are included by VSCE.

For local development, run `npm install` and `npm run prepare-server`, then launch
an Extension Development Host with this folder as its extension development path.
Run `npm run prepare-server` again after changing the Python frontend.

## Commands and configuration

The command palette includes **NexaLang: Build**, **NexaLang: Run**, and
**NexaLang: Test**. They save the active file and invoke the separately installed
`nxc` compiler. They remain available when the language server is disabled or
cannot start. Filenames are passed as process arguments, including names with
spaces or shell metacharacters.

- `nexalang.compilerPath`: compiler executable, default `nxc`.
- `nexalang.lsp.enabled`: enable the language server, default `true`.
- `nexalang.lsp.pythonPath`: Python interpreter, default `python3`.
- `nexalang.lsp.serverPath`: optional custom server script; otherwise use the bundle.

Diagnostics expand imported modules using the compiler's module resolver. Hover,
completion, and definition navigation currently use top-level symbols in the
active document. Completion is not yet type-directed; workspace-wide references,
rename, and imported-symbol navigation are not implemented.
