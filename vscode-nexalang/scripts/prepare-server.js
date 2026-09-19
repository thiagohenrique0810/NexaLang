// Bundle source from the repository at packaging time, without keeping copies in git.
const fs = require('fs');
const path = require('path');
const repo = path.resolve(__dirname, '..', '..');
const destination = path.resolve(__dirname, '..', 'server');
const files = [
    'tools/lsp_server.py',
    'bootstrap/lexer.py', 'bootstrap/n_parser.py', 'bootstrap/semantic.py',
    'bootstrap/errors.py', 'bootstrap/modules.py'
];
for (const file of files) {
    if (!fs.existsSync(path.join(repo, file))) {
        throw new Error(`Missing source ${file}; package the extension from a NexaLang checkout.`);
    }
}
fs.rmSync(destination, { recursive: true, force: true });
for (const file of files) {
    const target = path.join(destination, file);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.copyFileSync(path.join(repo, file), target);
}
fs.cpSync(path.join(repo, 'std'), path.join(destination, 'std'), {
    recursive: true,
    filter: source => fs.statSync(source).isDirectory() || source.endsWith('.nxl')
});
console.log('Prepared NexaLang LSP, bootstrap frontend, and standard library.');
