// Sistema de highlight de sintaxe para NexaLang
window.highlightNexaLangProcessed = window.highlightNexaLangProcessed || new WeakSet();

window.highlightNexaLang = function () {
    'use strict';

    // Use o conjunto global para evitar reprocessamento e loops infinitos
    const processed = window.highlightNexaLangProcessed;

    // Função simples de escape HTML
    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    const blocks = document.querySelectorAll('pre code');
    console.log('Blocos encontrados:', blocks.length);

    blocks.forEach(function (block, index) {
        if (processed.has(block)) {
            return;
        }
        processed.add(block);

        let text = block.textContent || block.innerText || '';
        if (!text.trim()) {
            return;
        }

        // 1. Escapa HTML básico.
        // Nota: Não escapamos " ou ' aqui para facilitar a regex de strings depois.
        // Assumimos que o texto fonte não tem entidades HTML codificadas manualmente exceto as que queremos exibir.
        let code = escapeHtml(text);

        // Armazena itens protegidos (Strings e Comentários)
        const protectedItems = [];

        // Regex unificado para extrair Strings e Comentários ANTES de qualquer outra coisa.
        // Ordem importa: "..." tem precedência sobre # dentro da string.
        code = code.replace(/(".*?"|'.*?'|#.*$)/gm, function (match) {
            protectedItems.push(match);
            return '___NXL_PROTECTED_' + (protectedItems.length - 1) + '___';
        });

        // Armazena Operadores para evitar colisão com tags HTML que vamos gerar (ex: class="...", <span...>)
        const protectedOps = [];
        function protectOp(op) {
            protectedOps.push(op);
            return '___NXL_OP_' + (protectedOps.length - 1) + '___';
        }

        // 2a. Protege operadores que são entidades HTML agora (&lt;, &gt;, &amp;)
        code = code.replace(/&lt;/g, protectOp('&lt;'));
        code = code.replace(/&gt;/g, protectOp('&gt;'));
        code = code.replace(/&amp;/g, protectOp('&amp;'));

        // 2b. Protege outros operadores literais
        // Nota: Excluímos < > & pois já foram tratados acima.
        // Excluímos " ' pois já foram tratados em strings.
        code = code.replace(/([+\-*/=!:;,.(){}[\]])/g, function (match) {
            return protectOp(match);
        });

        // --- AGORA O CÓDIGO ESTÁ LIMPO DE CARACTERES PERIGOSOS ---
        // Podemos inserir tags HTML (<span class="...">) sem medo de quebrar atributos ou tags.

        // 3. Keywords
        const keywords = ['return', 'while', 'match', 'struct', 'enum', 'impl', 'kernel', 'region', 'unsafe',
            'fn', 'let', 'mut', 'if', 'else', 'self', 'in', 'as', 'true', 'false'];

        keywords.forEach(function (kw) {
            const regex = new RegExp('\\b' + kw + '\\b', 'g');
            code = code.replace(regex, '<span class="keyword">' + kw + '</span>');
        });

        // 4. Tipos
        const types = ['string', 'void', 'i32', 'i64', 'bool', 'f32', 'char', 'u8', 'Buffer', 'Slice', 'Vec', 'Option', 'Result'];
        types.forEach(function (t) {
            const regex = new RegExp('\\b' + t + '\\b', 'g');
            code = code.replace(regex, '<span class="type">' + t + '</span>');
        });

        // 5. Números
        code = code.replace(/\b(\d+\.?\d*)\b/g, '<span class="number">$1</span>');

        // 6. Funções built-in
        const builtins = ['print', 'panic', 'assert', 'malloc', 'free', 'memcpy', 'sizeof', 'cast', 'ptr_offset', 'gpu::dispatch', 'gpu::global_id'];
        builtins.forEach(function (func) {
            // Escapa ::
            const escaped = func.replace('::', '::');
            // O lookahead deve procurar por ( ou placeholder de operador ( ou <
            // Como protejemos ( e <, agora eles são ___NXL_OP_X___.
            // Simplificação: apenas destaca o nome se for built-in.
            const regex = new RegExp('\\b' + escaped + '\\b', 'g');
            code = code.replace(regex, '<span class="function">' + func + '</span>');
        });

        // 7. Genéricos de Funções e Chamadas (Ex: foo(...))
        // Procura identificadores seguidos de (que agora é um placeholder de OP)
        code = code.replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*___NXL_OP_)/g, function (match) {
            // Se já foi transformado em span (keyword/type/builtin), ignora o match parcial se houver (mas \b protege)
            // Se for placeholder, ignora.
            if (match.startsWith('___')) return match;
            if (keywords.includes(match) || types.includes(match) || builtins.includes(match)) return match;
            // span e class são palavras perigosas se geradas, mas aqui 'match' é texto original.
            return '<span class="function">' + match + '</span>';
        });

        // 8. Tipos PascalCase (Ex: Vector3)
        code = code.replace(/\b([A-Z][a-zA-Z0-9_]*)\b/g, function (match) {
            if (match.startsWith('___')) return match; // Ignora placeholders
            if (keywords.includes(match) || types.includes(match)) return match;
            return '<span class="type">' + match + '</span>';
        });

        // --- RESTAURAÇÃO ---

        // 9. Restaura Operadores
        code = code.replace(/___NXL_OP_(\d+)___/g, function (match, i) {
            return '<span class="operator">' + protectedOps[i] + '</span>';
        });

        // 10. Restaura Strings e Comentários (por último para que nada dentro deles seja destacado)
        code = code.replace(/___NXL_PROTECTED_(\d+)___/g, function (match, i) {
            const item = protectedItems[i];
            if (item.trim().startsWith('#')) {
                return '<span class="comment">' + item + '</span>';
            } else {
                return '<span class="string">' + item + '</span>';
            }
        });

        block.innerHTML = code;
        console.log('Bloco', index, 'processado com sucesso');
    });

    console.log('Highlight concluído');
};

// Auto-executa
(function () {
    function init() {
        if (typeof window.highlightNexaLang === 'function') {
            window.highlightNexaLang();
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    window.addEventListener('load', function () {
        setTimeout(init, 50);
    });

    setTimeout(init, 100);
    setTimeout(init, 500);
})();
