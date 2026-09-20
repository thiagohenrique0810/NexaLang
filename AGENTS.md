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

## Trabalho paralelo: fronteiras que ninguém negocia sozinho

Estas regras existem porque propostas independentes colidem em imports e links,
não em engenharia. Valem sempre que mais de um agente trabalha no repositório.

1. **Sem novos re-exports.** Módulos novos em `compiler/` e `compiler/planner/`
   são importados por caminho completo. `compiler/__init__.py` já reexporta só
   `hardware_profile`, `model_ir` e `planner`: `precision_map`, `calibration` e
   `kv_plan` nunca entraram e não devem entrar.
2. **Ninguém edita os três índices.** `docs/README.md`,
   `docs/PLANOS_IMPLEMENTACAO_INDICE.md` e `readme.md` são atualizados pelo
   integrador num commit só, depois que o trabalho mescla.
3. **Ninguém edita `.github/workflows/ci.yml`.** O passo
   `python -m unittest discover -s tests -p "test_*regressions.py"` já recolhe
   qualquer arquivo de regressão novo.
4. **`docs/BLUEPRINT_512MB_CHECKLIST.md` é do integrador.** Um agente registra o
   que fez no retorno, não no checklist: editá-lo em paralelo garante conflito.
5. **Nomes adjudicados antes de começar.** Um caminho novo pertence a um dono só,
   declarado antes de qualquer arquivo existir.
