"""Shared module loading for the compiler and language server."""
from pathlib import Path
import json

from lexer import Lexer
from n_parser import (Parser, ModDecl, UseStmt, FunctionDef, StructDef,
                      EnumDef, TraitDef, ImplDef)

REPO_ROOT = Path(__file__).resolve().parent.parent


def mangle_ast(nodes, prefix):
    for node in nodes:
        if isinstance(node, (FunctionDef, StructDef, EnumDef, TraitDef)):
            node.module = f"{prefix}::{node.module}" if node.module else prefix
            node.name = f"{prefix}_{node.name}"
        elif isinstance(node, ImplDef):
            module = getattr(node, "module", "")
            node.module = f"{prefix}::{module}" if module else prefix
            node.struct_name = f"{prefix}_{node.struct_name}"
            for method in node.methods:
                method.module = node.module


def resolve_modules(ast, base_dir, loaded_modules=None):
    """Flatten declarations once per file/namespace, independent of the CWD.

    Standard modules are always loaded from the toolchain. Imports discovered
    inside a module retain their own namespace rather than being prefixed twice.
    """
    loaded = loaded_modules if loaded_modules is not None else set()
    active = set()
    base_dir = Path(base_dir).resolve()
    project_dir = next((directory for directory in (base_dir, *base_dir.parents)
                        if (directory / "nexa.json").is_file()), base_dir)

    def find_module(parts, directory):
        relative = Path(*parts)
        if parts[0] == "std":
            roots = [REPO_ROOT]
        else:
            roots = [directory, base_dir, project_dir, project_dir / "deps"]
        for root in roots:
            for candidate in (root / relative.with_suffix(".nxl"), root / relative / "mod.nxl"):
                if candidate.is_file():
                    return candidate.resolve()
            package = root / relative
            manifest = package / "nexa.json"
            if manifest.is_file():
                config = json.loads(manifest.read_text(encoding="utf-8"))
                entry = config.get("main")
                if not isinstance(entry, str):
                    raise ValueError(f"Missing package entry point: {manifest}")
                candidate = (package / entry).resolve()
                if not candidate.is_relative_to(package.resolve()):
                    raise ValueError(f"Package entry point escapes its directory: {manifest}")
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"Module not found: {'::'.join(parts)} (from {directory})")

    def load(path, namespace):
        key = (str(path), tuple(namespace))
        if path in active:
            raise ValueError(f"Cyclic module dependency: {path}")
        if key in loaded:
            return []
        active.add(path)
        try:
            nodes = Parser(Lexer(path.read_text(encoding="utf-8")).tokenize()).parse()
            resolved = resolve(nodes, path.parent, namespace)
            loaded.add(key)
            return resolved
        finally:
            active.remove(path)

    def resolve(nodes, directory, namespace):
        result = []
        for node in nodes:
            if isinstance(node, ModDecl):
                inner_namespace = namespace + [node.name]
                if node.body is not None:
                    result.extend(resolve(node.body, directory, inner_namespace))
                else:
                    path = find_module([node.name], directory)
                    result.extend(load(path, inner_namespace))
            elif isinstance(node, UseStmt):
                if node.path and node.path[0] == "std" and len(node.path) >= 2:
                    parts = node.path if node.is_glob else node.path[:-1]
                    if len(parts) >= 2:
                        result.extend(load(find_module(parts, directory), parts))
                result.append(node)
            else:
                for prefix in reversed(namespace):
                    mangle_ast([node], prefix)
                result.append(node)
        return result

    return resolve(ast, base_dir, [])
