// Sistema de highlight de sintaxe para NexaLang
function highlightCode() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(block => {
        // Pega o texto puro
        let code = block.textContent || block.innerText || '';
        
        // Escapa HTML
        code = code.replace(/&/g, '&amp;')
                  .replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;');
        
        // Processa em ordem de prioridade
        
        // 1. Comentários (máxima prioridade)
        code = code.replace(/(#.*$)/gm, '<span class="comment">$1</span>');
        
        // 2. Strings duplas
        code = code.replace(/"([^"]*)"/g, '<span class="string">"$1"</span>');
        
        // 3. Strings simples (chars)
        code = code.replace(/'([^']*)'/g, '<span class="string">\'$1\'</span>');
        
        // 4. Números
        code = code.replace(/\b(\d+\.?\d*)\b/g, '<span class="number">$1</span>');
        
        // 5. Keywords
        const keywords = ['fn', 'let', 'mut', 'return', 'if', 'else', 'while', 'match', 
                         'struct', 'enum', 'impl', 'self', 'kernel', 'region', 'in', 
                         'unsafe', 'true', 'false', 'as'];
        keywords.forEach(kw => {
            const regex = new RegExp(`\\b${kw}\\b`, 'g');
            code = code.replace(regex, '<span class="keyword">$&</span>');
        });
        
        // 6. Tipos primitivos
        const types = ['i32', 'i64', 'u8', 'bool', 'f32', 'char', 'string', 'void'];
        types.forEach(t => {
            const regex = new RegExp(`\\b${t}\\b`, 'g');
            code = code.replace(regex, '<span class="type">$&</span>');
        });
        
        // 7. Funções built-in especiais
        code = code.replace(/\b(cast|sizeof|ptr_offset)\s*&lt;/g, '<span class="function">$1</span>&lt;');
        
        // 8. Funções built-in normais
        code = code.replace(/\b(print|panic|assert|malloc|free|realloc|memcpy|slice_from_array|gpu::\w+)\s*\(/g, 
            '<span class="function">$1</span>(');
        
        // 9. Nomes de função
        code = code.replace(/\b([a-z_][a-z0-9_]*)\s*(\(|::)/g, (m, name, punct) => {
            if (!keywords.includes(name) && !types.includes(name) && 
                name !== 'cast' && name !== 'sizeof' && name !== 'ptr_offset') {
                return `<span class="function">${name}</span>${punct}`;
            }
            return m;
        });
        
        // 10. Tipos (PascalCase)
        code = code.replace(/\b([A-Z][a-zA-Z0-9_]*)\b/g, (m, name) => {
            if (!types.includes(name.toLowerCase())) {
                return `<span class="type">${name}</span>`;
            }
            return m;
        });
        
        // 11. Operadores
        code = code.replace(/([+\-*/=<>!&|:;,.(){}[\]])/g, '<span class="operator">$1</span>');
        
        block.innerHTML = code;
    });
}

// Aplica highlight quando a página carrega
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', highlightCode);
} else {
    highlightCode();
}
