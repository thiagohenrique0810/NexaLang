const { LanguageClient, TransportKind } = require('vscode-languageclient/node');
const vscode = require('vscode');
const path = require('path');

let client;

function activate(context) {
    const config = vscode.workspace.getConfiguration('nexalang');
    const lspEnabled = config.get('lsp.enabled', true);

    if (!lspEnabled) {
        return;
    }

    const pythonPath = config.get('lsp.pythonPath', 'python3');

    // Find the LSP server script
    // Try several locations: extension dir's parent (in-tree), or workspace
    let serverScript = null;
    const candidates = [
        path.join(__dirname, '..', 'tools', 'lsp_server.py'),
        path.join(__dirname, 'tools', 'lsp_server.py'),
    ];

    // Also check workspace folders
    if (vscode.workspace.workspaceFolders) {
        for (const folder of vscode.workspace.workspaceFolders) {
            candidates.push(path.join(folder.uri.fsPath, 'tools', 'lsp_server.py'));
        }
    }

    for (const candidate of candidates) {
        try {
            const fs = require('fs');
            if (fs.existsSync(candidate)) {
                serverScript = candidate;
                break;
            }
        } catch (e) {
            // ignore
        }
    }

    if (!serverScript) {
        vscode.window.showWarningMessage(
            'NexaLang LSP server not found. Diagnostics and hover disabled.'
        );
        return;
    }

    const serverOptions = {
        command: pythonPath,
        args: [serverScript],
        transport: TransportKind.stdio
    };

    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'nexalang' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.nxl')
        }
    };

    client = new LanguageClient(
        'nexalang',
        'NexaLang Language Server',
        serverOptions,
        clientOptions
    );

    client.start();

    context.subscriptions.push({
        dispose: () => {
            if (client) {
                client.stop();
            }
        }
    });

    // Register build command
    context.subscriptions.push(
        vscode.commands.registerCommand('nexalang.build', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'nexalang') {
                vscode.window.showErrorMessage('Open a .nxl file first.');
                return;
            }

            const file = editor.document.uri.fsPath;
            const terminal = vscode.window.createTerminal('NexaLang Build');
            terminal.show();
            terminal.sendText(`nxc build "${file}"`);
        })
    );

    // Register run command
    context.subscriptions.push(
        vscode.commands.registerCommand('nexalang.run', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'nexalang') {
                vscode.window.showErrorMessage('Open a .nxl file first.');
                return;
            }

            const file = editor.document.uri.fsPath;
            const terminal = vscode.window.createTerminal('NexaLang Run');
            terminal.show();
            terminal.sendText(`nxc run "${file}"`);
        })
    );

    // Register test command
    context.subscriptions.push(
        vscode.commands.registerCommand('nexalang.test', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'nexalang') {
                vscode.window.showErrorMessage('Open a .nxl file first.');
                return;
            }

            const file = editor.document.uri.fsPath;
            const terminal = vscode.window.createTerminal('NexaLang Test');
            terminal.show();
            terminal.sendText(`nxc test "${file}"`);
        })
    );
}

function deactivate() {
    if (client) {
        return client.stop();
    }
}

module.exports = { activate, deactivate };
