# Organização dos Scripts — TCC

Os scripts foram reorganizados por função. Pastas numeradas para leitura na ordem do fluxo da pesquisa.

| Pasta | Conteúdo | Roda? |
|---|---|---|
| **1_Geradores_Dados_Sinteticos/** | Geradores das técnicas SEM bug: `gerador_sinonimos_p6.py` (Sinônimos) e `gerador_top_palavras_SLAF_p8.py` (SLAF). Constroem comentários "do zero". | ✅ local |
| **2_Classificador_SVM/** | `svm_baseline_local.py` (baseline só com reais), `code_base_COLAB.py` e `code_sint_prompts1a5_COLAB.py` (versões originais do Google Colab — prompts 1–5 via LLM). | ✅ baseline / ☁️ Colab |
| **3_Analise_e_Graficos/** | Análise léxica (`analise_palavras.py`) e geração de gráficos. ⚠️ `gerar_graficos_finais.py` tem os números ANTIGOS (vazados) hardcoded — ver nota abaixo. | ✅ local |
| **4_CODIGO_COM_ERRO/** | Código que continha o **vazamento de teste**, com comentários marcando exatamente onde estão os erros (`>>> ERRO Nº 1/2 <<<`). | ✅ (didático) |
| **5_CODIGO_REFEITO/** | Versões **corrigidas**: gerador de Rewriting sem vazamento + avaliação unificada que reproduz a tabela corrigida do TCC. | ✅ local |
| **6_Notebooks/** | Notebooks originais (`.ipynb`). | ☁️ Colab |
| **9_Auxiliares/** | Utilitários e scripts de rascunho (`scratch_*`, `parse_nb.py`) usados durante a escrita — não fazem parte do experimento. | — |

## O erro central, em uma frase
O gerador da técnica **Rewriting** sorteava as "sementes" do **corpus inteiro**, incluindo os 20% de teste. Como o Rewriting preserva ~90% do texto, quase-duplicatas do teste entravam no treino → **vazamento de teste (data leakage)** → o resultado de 0,82 era inflado. Corrigido (sementes só do treino), nenhuma técnica supera o baseline.

## Como reproduzir os resultados corretos
```
python 5_CODIGO_REFEITO/gerador_rewriting_CORRIGIDO.py     # gera prompt_7_rewriting_CORRIGIDO.csv
python 5_CODIGO_REFEITO/avaliacao_svm_CORRIGIDO.py         # imprime a tabela corrigida
```

## Observações
- Todos os scripts locais resolvem a raiz do projeto automaticamente (função `_raiz_projeto`), então rodam de qualquer pasta.
- `3_Analise_e_Graficos/gerar_graficos_finais.py` ainda contém os números vazados; os gráficos corrigidos já estão em `4_Resultados_e_Graficos/*_CORRIGIDO.png` e embutidos no `enrricoVs02_corrigido.docx`.
- `gerar_graficos.py` salva em caminho `TCC_google_drive/` (era Colab) — mantido como referência histórica.
