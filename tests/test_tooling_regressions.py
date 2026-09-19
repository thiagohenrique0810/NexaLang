"""Package manager and editor regressions; no external packages or network needed."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import nxpkg


def load_lsp():
    spec = importlib.util.spec_from_file_location('nexa_lsp_test', REPO / 'tools/lsp_server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackageRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nexa-packages-')
        self.root = Path(self.temp.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        self.previous_cwd = Path.cwd()
        os.chdir(self.project)
        self.registry = self.root / 'registry'
        self.cache = self.root / 'cache'
        self.patches = [patch.object(nxpkg, 'REGISTRY_DIR', str(self.registry)),
                        patch.object(nxpkg, 'CACHE_DIR', str(self.cache))]
        for mock in self.patches:
            mock.start()
        self.write_config()

    def tearDown(self):
        os.chdir(self.previous_cwd)
        for mock in reversed(self.patches):
            mock.stop()
        self.temp.cleanup()

    def write_config(self, deps=None, **extra):
        config = {'name': 'app', 'version': '0.1.0', 'dependencies': deps or {}}
        config.update(extra)
        (self.project / 'nexa.json').write_text(json.dumps(config))

    def publish(self, version, value=None):
        source = self.root / ('demo-source-' + version)
        source.mkdir(exist_ok=True)
        (source / 'nexa.json').write_text(json.dumps({'name': 'demo', 'version': version}))
        (source / 'main.nxl').write_text(f'fn value() -> i32 {{ return {value or 1}; }}')
        (source / 'data.json').write_text('{"payload": 1}')
        os.chdir(source)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                nxpkg.cmd_publish(SimpleNamespace(force=False))
        finally:
            os.chdir(self.project)

    def call(self, command, **args):
        with contextlib.redirect_stdout(io.StringIO()):
            return command(SimpleNamespace(**args))

    def installed_version(self):
        return json.loads((self.project / 'deps/demo/nexa.json').read_text())['version']

    def test_traversal_is_rejected_before_install_update_or_remove(self):
        victim = self.project / 'important'
        victim.mkdir()
        sentinel = victim / 'keep.nxl'
        sentinel.write_text('keep')
        source = self.root / 'local-library'
        source.mkdir()
        self.write_config({'../important': str(source)})
        for command, args in [(nxpkg.cmd_install, {}), (nxpkg.cmd_update, {'name': None}),
                              (nxpkg.cmd_remove, {'name': '../important'})]:
            with self.subTest(command=command.__name__):
                with self.assertRaises(nxpkg.PackageError):
                    self.call(command, **args)
                self.assertEqual(sentinel.read_text(), 'keep')
                self.assertFalse(victim.is_symlink())

    def test_symlinked_deps_directory_cannot_redirect_writes(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'demo').mkdir()
        (outside / 'demo/keep').write_text('keep')
        (self.project / 'deps').symlink_to(outside, target_is_directory=True)
        source = self.root / 'local-library'
        source.mkdir()
        self.write_config({'demo': str(source)})
        for command, args in [(nxpkg.cmd_install, {}), (nxpkg.cmd_update, {'name': None}),
                              (nxpkg.cmd_remove, {'name': 'demo'})]:
            with self.assertRaises(nxpkg.PackageError):
                self.call(command, **args)
        self.assertEqual((outside / 'demo/keep').read_text(), 'keep')

    def test_local_remove_unlinks_without_removing_source(self):
        source = self.root / 'local-library'
        source.mkdir()
        (source / 'main.nxl').write_text('fn main() {}')
        self.write_config({'demo': str(source)})
        self.call(nxpkg.cmd_install)
        self.call(nxpkg.cmd_remove, name='demo')
        self.assertTrue((source / 'main.nxl').exists())
        self.assertFalse(os.path.lexists(self.project / 'deps/demo'))

    def test_local_install_replaces_broken_symlink(self):
        source = self.root / 'local-library'
        source.mkdir()
        self.write_config({'demo': str(source)})
        (self.project / 'deps').mkdir()
        (self.project / 'deps/demo').symlink_to(self.root / 'missing')
        self.call(nxpkg.cmd_install)
        self.assertEqual((self.project / 'deps/demo').resolve(), source.resolve())

    def test_release_resolution_and_versioned_cache(self):
        for version in ('1.9.0', '1.10.0', '2.0.0'):
            self.publish(version)
        self.write_config({'demo': '^1.0.0'})
        self.call(nxpkg.cmd_install)
        self.assertEqual(self.installed_version(), '1.10.0')
        lock = nxpkg.load_lockfile()['packages']['demo']
        self.assertEqual(lock['version'], '1.10.0')
        for version in ('1.9.0', '1.10.0', '2.0.0'):
            self.assertTrue((self.cache / 'demo' / version / 'nexa.json').exists())

    def test_install_honors_lock_and_update_resolves_new_release(self):
        self.publish('1.0.0')
        self.write_config({'demo': '^1.0.0'})
        self.call(nxpkg.cmd_install)
        self.publish('1.1.0')
        self.call(nxpkg.cmd_install)
        self.assertEqual(self.installed_version(), '1.0.0')
        self.call(nxpkg.cmd_update, name='demo')
        self.assertEqual(self.installed_version(), '1.1.0')

    def test_registry_absent_uses_versioned_cache(self):
        self.publish('1.0.0')
        shutil.rmtree(self.registry)
        self.write_config({'demo': '=1.0.0'})
        self.call(nxpkg.cmd_install)
        self.assertEqual(self.installed_version(), '1.0.0')

    def test_tampered_non_source_data_fails_integrity_and_preserves_install(self):
        self.publish('1.0.0')
        self.write_config({'demo': '1.0.0'})
        self.call(nxpkg.cmd_install)
        original_lock = (self.project / 'nexa-lock.json').read_bytes()
        (self.registry / 'demo/1.0.0/data.json').write_text('{"payload": 999}')
        with self.assertRaisesRegex(nxpkg.PackageError, 'Integrity'):
            self.call(nxpkg.cmd_install)
        self.assertEqual((self.project / 'deps/demo/data.json').read_text(), '{"payload": 1}')
        self.assertEqual((self.project / 'nexa-lock.json').read_bytes(), original_lock)

    def test_lock_integrity_cannot_be_bypassed_by_rewriting_registry_metadata(self):
        self.publish('1.0.0')
        self.write_config({'demo': '1.0.0'})
        self.call(nxpkg.cmd_install)
        package = self.registry / 'demo/1.0.0'
        (package / 'data.json').write_text('changed')
        metadata = json.loads((package / '.nxpkg-meta.json').read_text())
        metadata['integrity'] = nxpkg.hash_dir(package)
        (package / '.nxpkg-meta.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(nxpkg.PackageError, 'lockfile'):
            self.call(nxpkg.cmd_install)

    def test_all_packages_are_resolved_before_installing_any(self):
        self.publish('1.0.0')
        self.write_config({'demo': '1.0.0', 'missing': '1.0.0'})
        with self.assertRaises(nxpkg.PackageError):
            self.call(nxpkg.cmd_install)
        self.assertFalse((self.project / 'deps/demo').exists())
        self.assertFalse((self.project / 'nexa-lock.json').exists())

    def test_dev_dependencies_are_installed(self):
        self.publish('1.0.0')
        self.write_config(dev_dependencies={'demo': '1.0.0'})
        self.call(nxpkg.cmd_install)
        self.assertEqual(self.installed_version(), '1.0.0')

    def test_hash_includes_filenames(self):
        (self.project / 'a.nxl').write_text('same')
        before = nxpkg.hash_dir(self.project)
        (self.project / 'a.nxl').rename(self.project / 'b.nxl')
        self.assertNotEqual(nxpkg.hash_dir(self.project), before)

    def test_semver_zero_major_and_invalid_versions(self):
        self.assertTrue(nxpkg.version_satisfies('0.2.9', '^0.2.3'))
        self.assertFalse(nxpkg.version_satisfies('0.3.0', '^0.2.3'))
        self.assertTrue(nxpkg.version_satisfies('0.0.3', '^0.0.3'))
        self.assertFalse(nxpkg.version_satisfies('0.0.4', '^0.0.3'))
        for bad in ('1.2', '../1.2.3', '1.2.3-trailing', '01.2.3'):
            with self.assertRaises(nxpkg.PackageError):
                nxpkg.parse_version(bad)

    def test_script_exit_status_reaches_cli(self):
        self.write_config(scripts={'fail': 'exit 7'})
        result = subprocess.run([sys.executable, '-B', str(REPO / 'nxpkg.py'), 'run', 'fail'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_invalid_dependency_exits_nonzero(self):
        # An invalid dependency tests the real CLI without touching user storage.
        self.write_config({'../invalid': '1.0.0'})
        result = subprocess.run([sys.executable, '-B', str(REPO / 'nxpkg.py'), 'install'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Invalid package name', result.stderr)

    def test_init_does_not_overwrite_an_existing_project(self):
        before = (self.project / 'nexa.json').read_bytes()
        with self.assertRaises(nxpkg.PackageError):
            self.call(nxpkg.cmd_init, name='another', template='default')
        self.assertEqual((self.project / 'nexa.json').read_bytes(), before)

    def test_publish_rejects_registry_or_cache_inside_project_before_copying(self):
        for setting in ('REGISTRY_DIR', 'CACHE_DIR'):
            nested = self.project / '.nxpkg' / setting.lower()
            with self.subTest(setting=setting), patch.object(nxpkg, setting, str(nested)):
                with self.assertRaisesRegex(nxpkg.PackageError, 'overlap'):
                    self.call(nxpkg.cmd_publish, force=False)
                self.assertFalse(nested.exists())
                self.assertFalse(self.registry.exists())
                self.assertFalse(self.cache.exists())

    def test_package_copy_cannot_replace_its_own_source_ancestor(self):
        source = self.project / 'nested-source'
        source.mkdir()
        sentinel = source / 'source.nxl'
        sentinel.write_text('keep')
        with self.assertRaisesRegex(nxpkg.PackageError, 'overlap'):
            nxpkg.copy_package(source, self.project)
        self.assertEqual(sentinel.read_text(), 'keep')


class LSPRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lsp = load_lsp()

    def setUp(self):
        self.server = self.lsp.NexaLSP()
        self.notifications = []
        self.sender = patch.object(self.lsp, 'send_notification', lambda method, params: self.notifications.append((method, params)))
        self.sender.start()
        self.addCleanup(self.sender.stop)

    def test_common_semantic_errors_are_reported(self):
        for body, message in [('if (1) { return 0; }', 'condition must be bool'),
                              ('break;', 'Break outside loop')]:
            with self.subTest(body=body):
                uri = 'file:///tmp/lsp-check.nxl'
                self.server.analyze_document(uri, f'fn main() -> i32 {{ {body} return 0; }}')
                self.assertTrue(any(message in d['message'] for d in self.server.diagnostics[uri]),
                                self.server.diagnostics[uri])

    def test_eof_terminates_server(self):
        process = subprocess.run([sys.executable, '-B', str(REPO / 'tools/lsp_server.py')],
                                 input=b'', capture_output=True, timeout=3)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_imported_module_is_analyzed_from_uri_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='nexa lsp ') as tmp:
            root = Path(tmp)
            (root / 'helper.nxl').write_text('pub fn value() -> i32 { return 42; }')
            uri = (root / 'main.nxl').as_uri()
            self.server.analyze_document(uri, 'mod helper; fn main() -> i32 { return helper::value(); }')
            errors = [d for d in self.server.diagnostics[uri] if d['severity'] == 1]
            self.assertEqual(errors, [])
            self.assertEqual(Path(self.server._uri_to_path(uri)), root / 'main.nxl')

    def test_invalid_edit_clears_previous_symbols(self):
        uri = 'file:///tmp/lsp-check.nxl'
        self.server.analyze_document(uri, 'fn original() -> i32 { return 0; }')
        self.assertTrue(self.server.symbols[uri])
        self.server.analyze_document(uri, 'fn (')
        self.assertFalse(self.server.symbols.get(uri))
        self.assertFalse(self.server.ast_cache.get(uri))
        self.assertTrue(self.server.diagnostics[uri])

    def test_changed_import_refreshes_open_document_diagnostics(self):
        with tempfile.TemporaryDirectory(prefix='nexa-import-') as tmp:
            root = Path(tmp)
            helper = root / 'helper.nxl'
            helper.write_text('pub fn value() -> i32 { return 42; }')
            uri = (root / 'main.nxl').as_uri()
            self.server.analyze_document(uri, 'mod helper; fn main() -> i32 { return helper::value(); }')
            self.assertFalse([d for d in self.server.diagnostics[uri] if d['severity'] == 1])
            helper.write_text('pub fn value() -> i32 { break; return 42; }')
            self.server.handle({'method': 'workspace/didChangeWatchedFiles',
                                'params': {'changes': [{'uri': helper.as_uri(), 'type': 2}]}})
            self.assertTrue([d for d in self.server.diagnostics[uri] if d['severity'] == 1])

    def test_push_diagnostics_does_not_advertise_unsupported_pull(self):
        responses = []
        with patch.object(self.lsp, 'send_response', lambda req_id, result: responses.append(result)):
            self.server.handle({'id': 1, 'method': 'initialize'})
        self.assertNotIn('diagnosticProvider', responses[0]['capabilities'])


@unittest.skipUnless(shutil.which('node'), 'Node is needed to exercise extension activation')
class ExtensionRegressionTests(unittest.TestCase):
    def test_commands_register_without_lsp_and_keep_filename_as_argument(self):
        script = r'''
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const registered = new Map();
const filename = '/tmp/source $(not-a-command).nxl';
let executed;
const vscode = {
  workspace: {
    getConfiguration: () => ({get: (key, fallback) => key === 'lsp.enabled' ? false : fallback}),
    getWorkspaceFolder: () => undefined
  },
  commands: {registerCommand: (name, callback) => {registered.set(name, callback); return {}; }},
  window: {activeTextEditor: {document: {languageId: 'nexalang', isDirty: false, uri: {scheme: 'file', fsPath: filename}}}},
  ProcessExecution: class {constructor(command, args, options) {Object.assign(this, {command, args, options});}},
  Task: class {constructor(definition, scope, name, source, execution) {this.execution = execution;}},
  TaskScope: {Workspace: 1},
  tasks: {executeTask: async task => {executed = task;}}
};
const sandbox = {module: {exports: {}}, require: name => name === 'vscode' ? vscode : name === 'vscode-languageclient/node' ? {} : require(name), __dirname: process.argv[1]};
vm.runInNewContext(fs.readFileSync(process.argv[1] + '/extension.js', 'utf8'), sandbox);
(async () => {
  await sandbox.module.exports.activate({subscriptions: []});
  assert.deepStrictEqual([...registered.keys()], ['nexalang.build', 'nexalang.run', 'nexalang.test']);
  await registered.get('nexalang.build')();
  assert.strictEqual(executed.execution.command, 'nxc');
  assert.strictEqual(executed.execution.args[0], 'build');
  assert.strictEqual(executed.execution.args[1], filename);
})().catch(error => {console.error(error); process.exit(1);});
'''
        result = subprocess.run(['node', '-e', script, str(REPO / 'vscode-nexalang')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_packaged_server_runs_outside_checkout(self):
        with tempfile.TemporaryDirectory(prefix='nexa-extension-') as tmp:
            root = Path(tmp)
            extension = root / 'vscode-nexalang'
            (extension / 'scripts').mkdir(parents=True)
            shutil.copy2(REPO / 'vscode-nexalang/scripts/prepare-server.js', extension / 'scripts')
            for relative in ['tools/lsp_server.py', 'bootstrap/lexer.py', 'bootstrap/n_parser.py',
                             'bootstrap/semantic.py', 'bootstrap/errors.py', 'bootstrap/modules.py']:
                target = root / relative
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(REPO / relative, target)
            for source in (REPO / 'std').rglob('*.nxl'):
                target = root / source.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            prepared = subprocess.run(['node', str(extension / 'scripts/prepare-server.js')], capture_output=True, text=True)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            # The runtime bundle must survive removal of all original source.
            for directory in ('bootstrap', 'tools', 'std'):
                shutil.rmtree(root / directory)
            message = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'}).encode()
            framed = f'Content-Length: {len(message)}\r\n\r\n'.encode() + message
            result = subprocess.run([sys.executable, '-B', str(extension / 'server/tools/lsp_server.py')],
                                    input=framed, capture_output=True, cwd=root, timeout=3)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(b'"hoverProvider": true', result.stdout)
            self.assertEqual(result.stderr, b'')


if __name__ == '__main__':
    unittest.main()
