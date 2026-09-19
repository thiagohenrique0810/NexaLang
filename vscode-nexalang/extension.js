const { LanguageClient, TransportKind } = require('vscode-languageclient/node');
const vscode = require('vscode');
const path = require('path');
const fs = require('fs');

let client;

function registerCompilerCommands(context) {
    for (const action of ['build', 'run', 'test']) {
        context.subscriptions.push(vscode.commands.registerCommand(`nexalang.${action}`, async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'nexalang' || editor.document.uri.scheme !== 'file') {
                vscode.window.showErrorMessage('Open a saved .nxl file first.');
                return;
            }
            if (editor.document.isDirty && !(await editor.document.save())) {
                return;
            }
            const file = editor.document.uri.fsPath;
            const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
            const config = vscode.workspace.getConfiguration('nexalang', editor.document.uri);
            const executable = config.get('compilerPath', 'nxc');
            // Pass arguments directly, so filenames never become shell commands.
            const execution = new vscode.ProcessExecution(executable, [action, file], {
                cwd: folder ? folder.uri.fsPath : path.dirname(file)
            });
            const task = new vscode.Task(
                { type: 'nexalang', action }, folder || vscode.TaskScope.Workspace,
                `NexaLang ${action}`, 'nexalang', execution, []
            );
            await vscode.tasks.executeTask(task);
        }));
    }
}

async function activate(context) {
    registerCompilerCommands(context);
    const config = vscode.workspace.getConfiguration('nexalang');
    if (!config.get('lsp.enabled', true)) {
        return;
    }

    const configuredServer = config.get('lsp.serverPath', '');
    const candidates = configuredServer ? [configuredServer] : [
        path.join(__dirname, 'server', 'tools', 'lsp_server.py'),
        path.join(__dirname, '..', 'tools', 'lsp_server.py')
    ];
    const serverScript = candidates.find(candidate => fs.existsSync(candidate));
    if (!serverScript) {
        vscode.window.showWarningMessage('NexaLang LSP server not found. Run npm run prepare-server when developing the extension.');
        return;
    }

    const serverOptions = {
        command: config.get('lsp.pythonPath', 'python3'),
        args: [serverScript],
        transport: TransportKind.stdio
    };
    const watcher = vscode.workspace.createFileSystemWatcher('**/*.nxl');
    context.subscriptions.push(watcher);
    client = new LanguageClient('nexalang', 'NexaLang Language Server', serverOptions, {
        documentSelector: [{ scheme: 'file', language: 'nexalang' }],
        synchronize: { fileEvents: watcher }
    });
    context.subscriptions.push({ dispose: () => { if (client) { void client.stop(); } } });
    try {
        await client.start();
    } catch (error) {
        vscode.window.showErrorMessage(`NexaLang LSP failed to start: ${error.message}`);
    }
}

function deactivate() {
    return client ? client.stop() : undefined;
}

module.exports = { activate, deactivate };
