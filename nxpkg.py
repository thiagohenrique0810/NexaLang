import os
import json
import shutil
import argparse
import hashlib
import time
import re

REGISTRY_DIR = os.path.expanduser("~/.nxpkg/registry")
CACHE_DIR = os.path.expanduser("~/.nxpkg/cache")
LOCK_FILE = "nexa-lock.json"

# ── Helpers ──────────────────────────────────────────────────────────────

def load_config():
    if not os.path.exists("nexa.json"):
        print("Error: No nexa.json found. Run 'nxpkg init <name>' first.")
        return None
    with open("nexa.json", "r") as f:
        return json.load(f)

def save_config(config):
    with open("nexa.json", "w") as f:
        json.dump(config, f, indent=4)

def load_lockfile():
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, "r") as f:
            return json.load(f)
    return {"packages": {}}

def save_lockfile(lock):
    with open(LOCK_FILE, "w") as f:
        json.dump(lock, f, indent=4)

def ensure_dirs():
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def hash_dir(path):
    """SHA-256 of all .nxl files in a directory (for integrity)."""
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for fn in sorted(files):
            if fn.endswith('.nxl'):
                fp = os.path.join(root, fn)
                with open(fp, 'rb') as f:
                    h.update(f.read())
    return h.hexdigest()

def parse_version(v):
    """Parse semver string to tuple (major, minor, patch)."""
    m = re.match(r'^(\d+)\.(\d+)\.(\d+)', v)
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

def version_satisfies(version, constraint):
    """Check if version satisfies a constraint like ^1.2.0 or >=1.0.0."""
    if not constraint or constraint == '*':
        return True
    if constraint.startswith('^'):
        # Caret: compatible with major version
        req = parse_version(constraint[1:])
        ver = parse_version(version)
        if req[0] == 0:
            return ver[0] == req[0] and ver[1] == req[1] and ver[2] >= req[2]
        return ver[0] == req[0] and (ver[1], ver[2]) >= (req[1], req[2])
    if constraint.startswith('>='):
        return parse_version(version) >= parse_version(constraint[2:])
    if constraint.startswith('='):
        return parse_version(version) == parse_version(constraint[1:])
    # Exact match or path
    return version == constraint or os.path.isdir(constraint)

# ── Commands ─────────────────────────────────────────────────────────────

def cmd_init(args):
    name = args.name
    template = getattr(args, 'template', 'default')

    config = {
        "name": name,
        "version": "0.1.0",
        "description": "",
        "author": "",
        "license": "MIT",
        "main": "src/main.nxl",
        "scripts": {
            "build": f"nxc build src/main.nxl",
            "test": f"nxc test src/main.nxl",
        },
        "dependencies": {},
        "dev_dependencies": {}
    }
    with open("nexa.json", "w") as f:
        json.dump(config, f, indent=4)

    os.makedirs("src", exist_ok=True)
    os.makedirs("tests", exist_ok=True)

    if template == 'lib':
        with open("src/lib.nxl", "w") as f:
            f.write(f'# {name} library\n\npub fn hello() -> i32 {{\n    print("{name} loaded");\n    return 0;\n}}\n')
        config["main"] = "src/lib.nxl"
        save_config(config)
    else:
        with open("src/main.nxl", "w") as f:
            f.write(f'fn main() -> i32 {{\n    print("Hello from {name}!");\n    return 0;\n}}\n')

    with open("tests/test_main.nxl", "w") as f:
        f.write(f'@[test]\nfn test_example() {{\n    assert!(1 + 1 == 2, "basic math works");\n}}\n')

    # Create .gitignore
    with open(".gitignore", "w") as f:
        f.write("artifacts/\n*.o\n*.ll\n*.exe\n")

    save_lockfile({"packages": {}})
    print(f"✓ Project '{name}' initialized.")


def cmd_add(args):
    config = load_config()
    if not config: return

    dep_path = args.path
    version = getattr(args, 'version', None)

    if os.path.isdir(dep_path):
        # Local dependency
        pkg_name = os.path.basename(os.path.abspath(dep_path))
        # Check for nexa.json in the dependency
        dep_config_path = os.path.join(dep_path, 'nexa.json')
        if os.path.exists(dep_config_path):
            with open(dep_config_path) as f:
                dep_config = json.load(f)
            pkg_name = dep_config.get('name', pkg_name)

        config["dependencies"][pkg_name] = dep_path
    else:
        # Treat as name + version
        pkg_name = dep_path
        config["dependencies"][pkg_name] = version or "^0.1.0"

    save_config(config)

    # Update lockfile
    lock = load_lockfile()
    lock["packages"][pkg_name] = {
        "version": version or "local",
        "source": dep_path,
        "integrity": hash_dir(dep_path) if os.path.isdir(dep_path) else ""
    }
    save_lockfile(lock)

    print(f"✓ Added '{pkg_name}'.")


def cmd_remove(args):
    config = load_config()
    if not config: return

    name = args.name
    removed = False

    if name in config.get("dependencies", {}):
        del config["dependencies"][name]
        removed = True
    if name in config.get("dev_dependencies", {}):
        del config["dev_dependencies"][name]
        removed = True

    if not removed:
        print(f"Error: Package '{name}' not found in dependencies.")
        return

    save_config(config)

    # Remove from lockfile
    lock = load_lockfile()
    lock["packages"].pop(name, None)
    save_lockfile(lock)

    # Remove from deps/ directory if present
    dep_dir = os.path.join("deps", name)
    if os.path.exists(dep_dir):
        shutil.rmtree(dep_dir)

    print(f"✓ Removed '{name}'.")


def cmd_install(args):
    """Install all dependencies listed in nexa.json."""
    config = load_config()
    if not config: return

    deps = config.get("dependencies", {})
    if not deps:
        print("No dependencies to install.")
        return

    os.makedirs("deps", exist_ok=True)
    lock = load_lockfile()

    installed = 0
    for name, source in deps.items():
        dest = os.path.join("deps", name)

        if os.path.isdir(source):
            # Local dependency: symlink or copy
            if os.path.exists(dest):
                if os.path.islink(dest):
                    os.unlink(dest)
                else:
                    shutil.rmtree(dest)
            os.symlink(os.path.abspath(source), dest)
            integrity = hash_dir(source)
        else:
            # Check local registry cache
            cached = os.path.join(CACHE_DIR, name)
            if os.path.exists(cached):
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.copytree(cached, dest)
                integrity = hash_dir(dest)
            else:
                print(f"  ⚠ Package '{name}' not found locally. Skipping.")
                continue

        lock["packages"][name] = {
            "version": source if not os.path.isdir(source) else "local",
            "source": source,
            "integrity": integrity
        }
        installed += 1
        print(f"  ✓ {name}")

    save_lockfile(lock)
    print(f"\n✓ Installed {installed} package(s).")


def cmd_update(args):
    """Re-install deps and update lockfile."""
    config = load_config()
    if not config: return

    name = getattr(args, 'name', None)
    deps = config.get("dependencies", {})
    lock = load_lockfile()

    if name:
        if name not in deps:
            print(f"Error: '{name}' not in dependencies.")
            return
        deps = {name: deps[name]}

    updated = 0
    for pkg, source in deps.items():
        dest = os.path.join("deps", pkg)
        if os.path.isdir(source):
            if os.path.exists(dest):
                if os.path.islink(dest):
                    os.unlink(dest)
                else:
                    shutil.rmtree(dest)
            os.symlink(os.path.abspath(source), dest)
            integrity = hash_dir(source)
            lock["packages"][pkg] = {
                "version": "local",
                "source": source,
                "integrity": integrity,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            updated += 1
            print(f"  ✓ Updated {pkg}")

    save_lockfile(lock)
    print(f"\n✓ Updated {updated} package(s).")


def cmd_publish(args):
    """Publish package to local registry (~/.nxpkg/registry/)."""
    ensure_dirs()
    config = load_config()
    if not config: return

    name = config["name"]
    version = config["version"]
    pkg_dir = os.path.join(REGISTRY_DIR, name, version)

    if os.path.exists(pkg_dir):
        if not getattr(args, 'force', False):
            print(f"Error: {name}@{version} already exists in registry. Use --force to overwrite.")
            return
        shutil.rmtree(pkg_dir)

    os.makedirs(pkg_dir, exist_ok=True)

    # Copy source files
    for item in os.listdir('.'):
        if item in ('.git', 'deps', 'dev', 'node_modules', '__pycache__'):
            continue
        src = os.path.join('.', item)
        dst = os.path.join(pkg_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Write registry metadata
    meta = {
        "name": name,
        "version": version,
        "description": config.get("description", ""),
        "author": config.get("author", ""),
        "published": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "integrity": hash_dir(pkg_dir)
    }
    with open(os.path.join(pkg_dir, ".nxpkg-meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    # Also cache it
    cache_dir = os.path.join(CACHE_DIR, name)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    shutil.copytree(pkg_dir, cache_dir)

    print(f"✓ Published {name}@{version} to local registry.")


def cmd_list(args):
    """List installed dependencies."""
    config = load_config()
    if not config: return

    deps = config.get("dependencies", {})
    dev_deps = config.get("dev_dependencies", {})

    if not deps and not dev_deps:
        print("No dependencies.")
        return

    lock = load_lockfile()

    print(f"{config['name']}@{config['version']}")
    for name, source in deps.items():
        lock_info = lock.get("packages", {}).get(name, {})
        ver = lock_info.get("version", "?")
        status = "✓" if os.path.exists(os.path.join("deps", name)) else "✗"
        print(f"  {status} {name} ({ver}) <- {source}")

    if dev_deps:
        print("\ndev dependencies:")
        for name, source in dev_deps.items():
            print(f"    {name} <- {source}")


def cmd_search(args):
    """Search local registry for packages."""
    ensure_dirs()
    query = args.query.lower()
    found = 0

    for pkg_name in sorted(os.listdir(REGISTRY_DIR)):
        if query in pkg_name.lower():
            pkg_path = os.path.join(REGISTRY_DIR, pkg_name)
            versions = sorted(os.listdir(pkg_path)) if os.path.isdir(pkg_path) else []
            latest = versions[-1] if versions else "?"
            # Read description
            meta_path = os.path.join(pkg_path, latest, ".nxpkg-meta.json")
            desc = ""
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    desc = json.load(f).get("description", "")
            print(f"  {pkg_name} ({latest}) - {desc}")
            found += 1

    if not found:
        print(f"No packages matching '{args.query}'.")
    else:
        print(f"\n{found} package(s) found.")


def cmd_info(args):
    """Show info about current project or a specific package."""
    if args.name:
        # Look up in registry
        pkg_path = os.path.join(REGISTRY_DIR, args.name)
        if not os.path.exists(pkg_path):
            print(f"Package '{args.name}' not found in registry.")
            return
        versions = sorted(os.listdir(pkg_path))
        for v in versions:
            meta_path = os.path.join(pkg_path, v, ".nxpkg-meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                print(f"  {meta['name']}@{v}")
                if meta.get('description'): print(f"    {meta['description']}")
                if meta.get('author'): print(f"    by {meta['author']}")
                print(f"    published: {meta.get('published', '?')}")
    else:
        config = load_config()
        if not config: return
        print(f"Name:    {config['name']}")
        print(f"Version: {config['version']}")
        if config.get('description'): print(f"Desc:    {config['description']}")
        if config.get('author'): print(f"Author:  {config['author']}")
        if config.get('license'): print(f"License: {config['license']}")
        deps = config.get('dependencies', {})
        print(f"Deps:    {len(deps)}")


def cmd_run(args):
    """Run a script defined in nexa.json."""
    config = load_config()
    if not config: return

    scripts = config.get("scripts", {})
    name = args.script

    if name not in scripts:
        print(f"Error: Script '{name}' not found. Available: {', '.join(scripts.keys())}")
        return

    cmd = scripts[name]
    print(f"$ {cmd}")
    os.system(cmd)


def cmd_clean(args):
    """Remove deps/ directory and lockfile."""
    removed = []
    if os.path.exists("deps"):
        shutil.rmtree("deps")
        removed.append("deps/")
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        removed.append(LOCK_FILE)
    if os.path.exists("artifacts"):
        shutil.rmtree("artifacts")
        removed.append("artifacts/")
    elif os.path.exists("dev/artifacts"):
        shutil.rmtree("dev/artifacts")
        removed.append("dev/artifacts/")

    if removed:
        print(f"✓ Cleaned: {', '.join(removed)}")
    else:
        print("Nothing to clean.")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="nxpkg",
        description="NexaLang Package Manager",
        epilog="Run 'nxpkg <command> -h' for more info on a command."
    )
    subparsers = parser.add_subparsers(dest="command")

    # init
    p_init = subparsers.add_parser("init", help="Initialize a new project")
    p_init.add_argument("name", help="Project name")
    p_init.add_argument("--template", choices=["default", "lib"], default="default", help="Project template")

    # add
    p_add = subparsers.add_parser("add", help="Add a dependency")
    p_add.add_argument("path", help="Package name or local path")
    p_add.add_argument("--version", help="Version constraint (e.g. ^1.0.0)")

    # remove
    p_remove = subparsers.add_parser("remove", help="Remove a dependency")
    p_remove.add_argument("name", help="Package name to remove")

    # install
    subparsers.add_parser("install", help="Install all dependencies from nexa.json")

    # update
    p_update = subparsers.add_parser("update", help="Update dependencies")
    p_update.add_argument("name", nargs="?", help="Specific package to update (optional)")

    # publish
    p_publish = subparsers.add_parser("publish", help="Publish package to local registry")
    p_publish.add_argument("--force", action="store_true", help="Overwrite existing version")

    # list
    subparsers.add_parser("list", help="List project dependencies")

    # search
    p_search = subparsers.add_parser("search", help="Search local registry")
    p_search.add_argument("query", help="Search query")

    # info
    p_info = subparsers.add_parser("info", help="Show package/project info")
    p_info.add_argument("name", nargs="?", help="Package name (omit for current project)")

    # run
    p_run = subparsers.add_parser("run", help="Run a script from nexa.json")
    p_run.add_argument("script", help="Script name")

    # clean
    subparsers.add_parser("clean", help="Remove deps/, lockfile, and build artifacts")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "add": cmd_add,
        "remove": cmd_remove,
        "install": cmd_install,
        "update": cmd_update,
        "publish": cmd_publish,
        "list": cmd_list,
        "search": cmd_search,
        "info": cmd_info,
        "run": cmd_run,
        "clean": cmd_clean,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
