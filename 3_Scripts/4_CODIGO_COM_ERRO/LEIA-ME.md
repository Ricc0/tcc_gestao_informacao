# 4_CODIGO_COM_ERRO — onde estão os erros

Esta pasta guarda o código **como foi rodado originalmente**, com os bugs preservados e
**comentados inline** (procure por `>>> ERRO`). A versão corrigida está em `../5_CODIGO_REFEITO/`.

## ERRO Nº 1 — Vazamento de teste (data leakage) — o mais grave
**Arquivo:** `gerador_rewriting_COM_ERRO.py`

A técnica Rewriting parte de um comentário real ("semente") e troca ~12 termos por gírias,
preservando ~90% do texto. O gerador sorteava as sementes do **corpus completo** (`df_real`
inteiro), **sem remover os 20% de teste**. Assim, comentários de teste eram reescritos para
dentro do treino, com o rótulo certo — o modelo passava a "decorar" o teste.

- **Onde:** linhas marcadas com `>>> ERRO Nº 1 <<<` (extração das listas de sementes e os
  `random.choice(...)` dentro dos loops).
- **Impacto medido:** ~21% das sementes vinham do teste. Macro F1 do Rewriting subia de
  forma artificial para **0,82** (e Neutro para 0,80).
- **Correção:** dividir treino/teste ANTES e usar só o TREINO como semente
  (ver `gerador_rewriting_CORRIGIDO.py`). Com isso o Rewriting cai para **0,65** (4 classes).

## ERRO Nº 2 — Split refeito no cenário de fusão (metodológico)
**Arquivos:** `comparar_prompts_COM_ERRO.py`, `svm_teste_sinonimos_COM_ERRO.py`

No teste de 3 classes (fusão de Neutro + Indeterminado), o `train_test_split` era **refeito**
sobre os rótulos já fundidos. Como a estratificação muda, o **conjunto de teste de 3 classes
fica diferente do de 4 classes**, e os dois cenários deixam de ser comparáveis sobre as mesmas
instâncias.

- **Onde:** linhas marcadas com `>>> ERRO Nº 2 <<<`.
- **Correção:** dividir UMA única vez (fixando por índice) e apenas RELABELAR as mesmas
  instâncias para a fusão (ver `../5_CODIGO_REFEITO/avaliacao_svm_CORRIGIDO.py`).

> Observação: os scripts `comparar_*` e `svm_teste_*` não têm bug para Sinônimos (p6) e SLAF
> (p8) — essas técnicas não usam sementes reais. O Erro Nº 1 afeta apenas o Rewriting (p7).
