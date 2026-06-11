# 5_CODIGO_REFEITO — versões corrigidas

Correção dos dois erros documentados em `../4_CODIGO_COM_ERRO/`.

## Arquivos
- **`gerador_rewriting_CORRIGIDO.py`** — gera o Rewriting (p7) **sem vazamento**: faz o
  `train_test_split` ANTES e usa como semente apenas a partição de **treino (80%)**.
  Salva `2_Dados/Sinteticos/prompt_7_rewriting_CORRIGIDO.csv`.
- **`avaliacao_svm_CORRIGIDO.py`** — pipeline unificado que avalia baseline + as 3 técnicas.
  Usa **uma única divisão treino/teste fixada por índice** nos dois cenários (4 classes e
  3 classes/fusão); para a fusão apenas relabela as mesmas instâncias (corrige o Erro Nº 2).

## Como rodar
```
python gerador_rewriting_CORRIGIDO.py
python avaliacao_svm_CORRIGIDO.py
```

## Resultado esperado (tabela corrigida do TCC)

| Técnica | Macro F1 (4 classes) | Macro F1 (3 classes / fusão) |
|---|---|---|
| **Baseline** (só reais) | **0,685** | **0,767** |
| Sinônimos (p6) | 0,682 | 0,748 |
| SLAF / Controle (p8) | 0,643 | 0,720 |
| **Rewriting corrigido** | **0,654** | **0,755** |

**Conclusão:** com o vazamento removido, **nenhuma** das três técnicas supera o baseline.
O único ganho legítimo vem da **fusão de classes** (0,685 → 0,767), sem dados sintéticos.
(Para referência, o Rewriting *com* vazamento dava 0,82 — número que não media generalização.)
