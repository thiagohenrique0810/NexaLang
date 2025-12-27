# Documentação NexaLang

Esta pasta contém a documentação completa da linguagem NexaLang em formato HTML.

## Acesso à Documentação

Abra o arquivo `index.html` em seu navegador para visualizar a documentação completa.

### Visualização Local

```bash
# No Windows
start docs/index.html

# No Linux/Mac
open docs/index.html
# ou
xdg-open docs/index.html
```

### Servidor Local (Recomendado)

Para melhor experiência, você pode servir a documentação através de um servidor HTTP local:

```bash
# Python 3
python -m http.server 8000

# Node.js (com http-server)
npx http-server docs -p 8000

# PHP
php -S localhost:8000 -t docs
```

Depois acesse: `http://localhost:8000/index.html`

## Conteúdo

A documentação inclui:

- ✅ Introdução à linguagem
- ✅ Guia de instalação
- ✅ Sintaxe básica
- ✅ Tipos de dados
- ✅ Funções
- ✅ Structs e métodos
- ✅ Enums e pattern matching
- ✅ Generics
- ✅ Ownership & Borrowing
- ✅ Regions (Arenas)
- ✅ GPU Kernels
- ✅ Controle de fluxo
- ✅ Arrays & Slices
- ✅ Ponteiros
- ✅ Standard Library
- ✅ Ferramentas
- ✅ Exemplos práticos

## Atualizações

A documentação é atualizada conforme a linguagem evolui. Para contribuir, edite o arquivo `index.html` e faça um pull request.

