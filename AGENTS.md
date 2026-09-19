# Continuidade do desenvolvimento

Para trabalho no blueprint de IA/512 MB, leia primeiro
`docs/BLUEPRINT_512MB_CHECKLIST.md` e `docs/BLUEPRINT_512MB_AJUSTES.md`.
O checklist registra o marco atual, comandos de validação e a próxima tarefa.

Antes de iniciar cada módulo, consulte `docs/PLANOS_IMPLEMENTACAO_INDICE.md`:
ele mapeia os PDFs de referência para IDs do checklist, páginas/seções,
dependências e ajustes de engenharia. Leia as referências indicadas para o item;
exemplos de API nos PDFs não significam suporte implementado.

- Atualize o checkpoint e os resultados de testes ao concluir um incremento.
- Marque itens concluídos somente com implementação e evidência; um protótipo CPU
  não comprova funcionamento ou desempenho de GPU.
- Preserve alterações anteriores do usuário e de outros agentes no diretório.
- A entrada suportada da linguagem continua `nxc`/`nx.py`/`bootstrap/`.
  O novo pipeline de modelos em `compiler/` é independente do bootstrap da linguagem.
- Use `artifacts/` para relatórios, bibliotecas nativas e pesos gerados; não versione
  binários de plataforma ou modelos baixados.
