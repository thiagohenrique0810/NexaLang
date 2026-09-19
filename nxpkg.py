import os
import json
import shutil
import argparse
import hashlib
import time
import re
import sys
import subprocess
import tempfile

REGISTRY_DIR = os.path.expanduser("~/.nxpkg/registry")
CACHE_DIR = os.path.expanduser("~/.nxpkg/cache")
LOCK_FILE = "nexa-lock.json"

# ── Helpers ──────────────────────────────────────────────────────────────

class PackageError(Exception):
    """A package operation failed without completing successfully."""


EXCLUDED = {'.git', 'deps', 'dev', 'node_modules', '__pycache__', 'artifacts', '.nxpkg-meta.json'}
NAME_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
VERSION_PATTERN = re.compile(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)')


def validate_name(name):
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise PackageError(f"Invalid package name: {name!r}. Use letters, numbers, '.', '_' or '-'.")
    return name


def read_json(path):
    with open(path, encoding='utf-8') as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise PackageError(f"Expected a JSON object in {path}")
    return value


def write_json(path, value):
    # Replace atomically, so interruption cannot leave half a manifest/lockfile.
    directory = os.path.dirname(os.path.abspath(path))
    fd, temp = tempfile.mkstemp(prefix='.nxpkg-', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=4)
            f.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_config():
    if not os.path.isfile('nexa.json'):
        raise PackageError("No nexa.json found. Run 'nxpkg init <name>' first.")
    config = read_json('nexa.json')
    validate_name(config.get('name'))
    for key in ('dependencies', 'dev_dependencies'):
        deps = config.get(key, {})
        if not isinstance(deps, dict):
            raise PackageError(f"{key} must be an object")
        for name, source in deps.items():
            validate_name(name)
            if not isinstance(source, str) or not source:
                raise PackageError(f"Invalid dependency source for {name!r}")
    return config


def save_config(config):
    write_json('nexa.json', config)


def load_lockfile():
    lock = read_json(LOCK_FILE) if os.path.exists(LOCK_FILE) else {'packages': {}}
    if not isinstance(lock.get('packages'), dict):
        raise PackageError('Invalid nexa-lock.json: packages must be an object')
    for name, entry in lock['packages'].items():
        validate_name(name)
        if not isinstance(entry, dict):
            raise PackageError(f"Invalid lock entry for {name!r}")
    return lock


def save_lockfile(lock):
    write_json(LOCK_FILE, lock)


def managed_root(root, create=False):
    root = os.path.abspath(root)
    if os.path.islink(root):
        raise PackageError(f"Refusing to manage a symlink directory: {root}")
    if create:
        os.makedirs(root, exist_ok=True)
    return root


def package_path(root, name, version=None):
    path = os.path.join(managed_root(root), validate_name(name))
    if os.path.islink(path):
        raise PackageError(f"Refusing symlink in package storage: {path}")
    if version is not None:
        parse_version(version)
        path = os.path.join(path, version)
        if os.path.islink(path):
            raise PackageError(f"Refusing symlink in package storage: {path}")
    return path


def dependency_path(name):
    return os.path.join(managed_root('deps'), validate_name(name))


def remove_entry(path):
    # Unlink symlinks, including broken links, without following their target.
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.exists(path):
        shutil.rmtree(path)


def ensure_dirs():
    managed_root(REGISTRY_DIR, create=True)
    managed_root(CACHE_DIR, create=True)


def hash_dir(path):
    """Hash relative names and all package file contents, excluding generated metadata."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
        for name in dirs + sorted(files):
            if name in EXCLUDED:
                continue
            fp = os.path.join(root, name)
            if os.path.islink(fp):
                raise PackageError(f"Package contents must not contain symlinks: {fp}")
            if not os.path.isfile(fp):
                continue
            relative = os.path.relpath(fp, path).replace(os.sep, '/').encode('utf-8')
            h.update(len(relative).to_bytes(8, 'big'))
            h.update(relative)
            h.update(os.path.getsize(fp).to_bytes(8, 'big'))
            with open(fp, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    h.update(chunk)
    return h.hexdigest()


def parse_version(version):
    """The supported version format is a stable major.minor.patch release."""
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise PackageError(f"Invalid version: {version!r}; expected major.minor.patch")
    return tuple(map(int, version.split('.')))


def version_satisfies(version, constraint):
    ver = parse_version(version)
    if constraint in ('', '*', None):
        return True
    prefix = next((p for p in ('>=', '^', '=') if constraint.startswith(p)), '')
    req = parse_version(constraint[len(prefix):])
    if prefix == '>=':
        return ver >= req
    if prefix == '^':
        upper = (req[0] + 1, 0, 0) if req[0] else ((0, req[1] + 1, 0) if req[1] else (0, 0, req[2] + 1))
        return req <= ver < upper
    return ver == req


def all_dependencies(config):
    deps = dict(config.get('dependencies', {}))
    for name, source in config.get('dev_dependencies', {}).items():
        if name in deps and deps[name] != source:
            raise PackageError(f"Conflicting dependency and dev_dependency: {name}")
        deps[name] = source
    return deps


def available_packages(name):
    packages = {}
    # A versioned cache and registry can coexist. The registry is authoritative.
    for root in (CACHE_DIR, REGISTRY_DIR):
        directory = package_path(root, name)
        if not os.path.isdir(directory):
            continue
        for version in os.listdir(directory):
            if VERSION_PATTERN.fullmatch(version):
                path = package_path(root, name, version)
                if os.path.isdir(path):
                    packages[version] = path
    return packages


def verify_package(path, name, version, expected=None):
    manifest = read_json(os.path.join(path, 'nexa.json'))
    if manifest.get('name') != name or manifest.get('version') != version:
        raise PackageError(f"Package metadata does not match {name}@{version}")
    integrity = hash_dir(path)
    metadata_path = os.path.join(path, '.nxpkg-meta.json')
    if os.path.isfile(metadata_path):
        metadata = read_json(metadata_path)
        if metadata.get('integrity') != integrity:
            raise PackageError(f"Integrity check failed for {name}@{version}")
    if expected and integrity != expected:
        raise PackageError(f"Integrity differs from lockfile for {name}@{version}")
    return integrity


def require_separate_trees(source, destination):
    source = os.path.realpath(source)
    destination = os.path.realpath(destination)
    try:
        common = os.path.commonpath([source, destination])
    except ValueError:
        return  # Different Windows drives cannot overlap.
    if common in (source, destination):
        raise PackageError(f"Package source and destination overlap: {source} -> {destination}")


def copy_package(source, dest):
    require_separate_trees(source, dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    stage = tempfile.mkdtemp(prefix='.nxpkg-', dir=os.path.dirname(dest))
    try:
        shutil.copytree(source, stage, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*EXCLUDED))
        # Preserve publisher metadata (excluded from integrity itself).
        meta = os.path.join(source, '.nxpkg-meta.json')
        if os.path.isfile(meta):
            shutil.copy2(meta, stage)
        remove_entry(dest)
        os.replace(stage, dest)
    finally:
        if os.path.exists(stage):
            shutil.rmtree(stage)


def install_dependencies(config, update=False, only=None):
    deps = all_dependencies(config)
    if only:
        validate_name(only)
        if only not in deps:
            raise PackageError(f"'{only}' not in dependencies")
        deps = {only: deps[only]}
    if not deps:
        print('No dependencies to install.')
        return
    managed_root('deps')  # Validate before any deletion or lockfile mutation.
    lock = load_lockfile()
    plan = []
    for name, source in deps.items():
        dest = dependency_path(name)
        if os.path.isdir(source):
            resolved = os.path.realpath(source)
            deps_root = os.path.realpath('deps')
            if os.path.commonpath([resolved, deps_root]) == deps_root or os.path.commonpath([deps_root, resolved]) == resolved:
                raise PackageError(f"Local dependency {name!r} overlaps the managed deps directory")
            integrity = hash_dir(resolved)
            plan.append((name, source, resolved, dest, 'local', integrity))
            continue
        # Validating here also distinguishes missing paths from valid constraints.
        version_satisfies('0.0.0', source)
        packages = available_packages(name)
        locked = lock['packages'].get(name)
        expected = None
        if not update and locked and locked.get('source') == source:
            version = locked.get('version')
            if not version_satisfies(version, source):
                raise PackageError(f"Locked version of {name!r} violates {source}; run nxpkg update")
            expected = locked.get('integrity')
            if not expected:
                raise PackageError(f"Lock entry for {name!r} has no integrity; run nxpkg update")
        else:
            versions = [v for v in packages if version_satisfies(v, source)]
            if not versions:
                raise PackageError(f"No local release of {name!r} satisfies {source}")
            version = max(versions, key=parse_version)
        if version not in packages:
            raise PackageError(f"Locked package {name}@{version} is not available locally")
        resolved = packages[version]
        integrity = verify_package(resolved, name, version, expected)
        plan.append((name, source, resolved, dest, version, integrity))
    # Resolve and verify every input before replacing installed packages.
    managed_root('deps', create=True)
    for name, source, resolved, dest, version, integrity in plan:
        if version == 'local':
            remove_entry(dest)
            os.symlink(resolved, dest)
        else:
            copy_package(resolved, dest)
            cache = package_path(CACHE_DIR, name, version)
            if os.path.realpath(resolved) != os.path.realpath(cache):
                copy_package(resolved, cache)
        lock['packages'][name] = {'version': version, 'source': source, 'integrity': integrity}
        print(f"  ✓ {name}@{version}")
    if not only:
        lock['packages'] = {name: entry for name, entry in lock['packages'].items() if name in deps}
    save_lockfile(lock)
    print(f"\n✓ Installed {len(plan)} package(s).")


# ── Commands ─────────────────────────────────────────────────────────────

def cmd_init(args):
    name = validate_name(args.name)
    template = getattr(args, 'template', 'default')
    main = 'src/lib.nxl' if template == 'lib' else 'src/main.nxl'
    for path in ('nexa.json', LOCK_FILE, main, 'tests/test_main.nxl'):
        if os.path.lexists(path):
            raise PackageError(f"Refusing to overwrite existing project file: {path}")
    for directory in ('src', 'tests'):
        managed_root(directory, create=True)
    config = {'name': name, 'version': '0.1.0', 'description': '', 'author': '', 'license': 'MIT',
              'main': main, 'scripts': {'build': f'nxc build {main}', 'test': 'nxc test tests/test_main.nxl'},
              'dependencies': {}, 'dev_dependencies': {}}
    save_config(config)
    with open(main, 'w', encoding='utf-8') as f:
        f.write(f'pub fn hello() -> i32 {{\n    return 0;\n}}\n' if template == 'lib' else
                f'fn main() -> i32 {{\n    print("Hello from {name}!");\n    return 0;\n}}\n')
    with open('tests/test_main.nxl', 'w', encoding='utf-8') as f:
        f.write('@[test]\nfn test_example() {\n    assert!(1 + 1 == 2, "basic math works");\n}\n')
    if not os.path.lexists('.gitignore'):
        with open('.gitignore', 'w', encoding='utf-8') as f:
            f.write('artifacts/\ndeps/\n*.o\n*.ll\n*.exe\n')
    save_lockfile({'packages': {}})
    print(f"✓ Project '{name}' initialized.")


def cmd_add(args):
    config = load_config()
    source = args.path
    if os.path.isdir(source):
        path = os.path.join(source, 'nexa.json')
        name = read_json(path).get('name') if os.path.isfile(path) else os.path.basename(os.path.abspath(source))
    else:
        name = source
        source = getattr(args, 'version', None) or '^0.1.0'
        version_satisfies('0.0.0', source)
    validate_name(name)
    lock = load_lockfile()
    config.setdefault('dependencies', {})[name] = source
    lock['packages'].pop(name, None)  # Install creates a resolved, verified entry.
    save_config(config)
    save_lockfile(lock)
    print(f"✓ Added '{name}'.")


def cmd_remove(args):
    config = load_config()
    name = validate_name(args.name)
    dest = dependency_path(name)
    if name not in all_dependencies(config):
        raise PackageError(f"Package '{name}' not found in dependencies")
    lock = load_lockfile()
    remove_entry(dest)
    for key in ('dependencies', 'dev_dependencies'):
        config.get(key, {}).pop(name, None)
    lock['packages'].pop(name, None)
    save_config(config)
    save_lockfile(lock)
    print(f"✓ Removed '{name}'.")


def cmd_install(args):
    install_dependencies(load_config())


def cmd_update(args):
    install_dependencies(load_config(), update=True, only=getattr(args, 'name', None))


def cmd_publish(args):
    config = load_config()
    name = validate_name(config.get('name'))
    version = config.get('version')
    parse_version(version)
    pkg_dir = package_path(REGISTRY_DIR, name, version)
    cache_dir = package_path(CACHE_DIR, name, version)
    # A project in HOME may contain ~/.nxpkg itself. Reject this before creating
    # staging directories that would otherwise recursively copy themselves.
    require_separate_trees('.', pkg_dir)
    require_separate_trees('.', cache_dir)
    require_separate_trees(pkg_dir, cache_dir)
    if os.path.exists(pkg_dir) and not getattr(args, 'force', False):
        raise PackageError(f"{name}@{version} already exists. Use --force to overwrite")
    integrity = hash_dir('.')  # Reject symlink contents before copying anything.
    ensure_dirs()
    copy_package('.', pkg_dir)
    meta = {'name': name, 'version': version, 'description': config.get('description', ''),
            'author': config.get('author', ''), 'published': time.strftime('%Y-%m-%dT%H:%M:%S'), 'integrity': integrity}
    write_json(os.path.join(pkg_dir, '.nxpkg-meta.json'), meta)
    copy_package(pkg_dir, cache_dir)
    print(f"✓ Published {name}@{version} to local registry.")


def cmd_list(args):
    config = load_config()
    lock = load_lockfile()
    print(f"{config['name']}@{config['version']}")
    for name, source in all_dependencies(config).items():
        entry = lock['packages'].get(name, {})
        status = '✓' if os.path.exists(dependency_path(name)) else '✗'
        print(f"  {status} {name} ({entry.get('version', '?')}) <- {source}")


def cmd_search(args):
    ensure_dirs()
    found = 0
    for name in sorted(os.listdir(REGISTRY_DIR)):
        if not NAME_PATTERN.fullmatch(name) or args.query.lower() not in name.lower():
            continue
        versions = available_packages(name)
        if versions:
            latest = max(versions, key=parse_version)
            config = read_json(os.path.join(versions[latest], 'nexa.json'))
            print(f"  {name} ({latest}) - {config.get('description', '')}")
            found += 1
    print(f"{found} package(s) found.")


def cmd_info(args):
    if args.name:
        name = validate_name(args.name)
        versions = available_packages(name)
        if not versions:
            raise PackageError(f"Package '{name}' not found in registry or cache")
        for version in sorted(versions, key=parse_version):
            print(f"  {name}@{version}")
    else:
        config = load_config()
        print(f"Name:    {config['name']}\nVersion: {config['version']}")
        for key in ('description', 'author', 'license'):
            if config.get(key):
                print(f"{key}: {config[key]}")
        print(f"Deps:    {len(all_dependencies(config))}")


def cmd_run(args):
    config = load_config()
    scripts = config.get('scripts', {})
    if args.script not in scripts:
        raise PackageError(f"Script '{args.script}' not found. Available: {', '.join(scripts)}")
    command = scripts[args.script]
    print(f'$ {command}', flush=True)
    return subprocess.call(command, shell=True)


def cmd_clean(args):
    for path in ('deps', LOCK_FILE, 'artifacts'):
        remove_entry(path)
    # Only traverse dev when it is a real project directory.
    if not os.path.islink('dev'):
        remove_entry(os.path.join('dev', 'artifacts'))
    print('✓ Cleaned package and build artifacts.')


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
        try:
            return commands[args.command](args) or 0
        except (PackageError, OSError, ValueError, KeyError, TypeError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
    else:
        parser.print_help()

if __name__ == "__main__":
    sys.exit(main())
