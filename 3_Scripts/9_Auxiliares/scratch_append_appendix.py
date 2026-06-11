import os

file_path = r"C:\Users\Henricco Santos\OneDrive - exedconsulting\Documentos\TCC\5_Documentacao_TCC\tcc_henricco.md"

appendix_content = """
---

7. Apêndice: Descrição e Análise Descritiva dos Gráficos

Para fins de acessibilidade e detalhamento técnico, esta seção descreve textualmente o conteúdo visual e os dados quantitativos expressos nos gráficos gerados ao longo da pesquisa.

**Gráfico 1: Volume de Dados Sintéticos Gerados por Lote**
*   **Descrição Visual:** Trata-se de um gráfico de barras simples contendo três colunas verticais. O eixo X (horizontal) lista as três abordagens metodológicas de geração: "Sinônimos", "Rewriting" e "Top Palavras". O eixo Y (vertical) representa o número absoluto de comentários gerados, escalado de 0 a 1800.
*   **Dados e Análise:** As três barras ("Sinônimos" em azul, "Rewriting" em verde e "Top Palavras" em laranja) atingem exatamente a mesma altura. O valor "1600" está grafado sobre cada uma delas. Este gráfico comprova visualmente a igualdade volumétrica do experimento: todas as técnicas foram testadas com rigorosamente 1.600 instâncias sintéticas, eliminando a variável de "quantidade" de dados e isolando a "qualidade semântica" como fator de avaliação.

**Gráfico 2: Evolução do F1-Score do Modelo SVM (Cenário 4 Classes)**
*   **Descrição Visual:** Gráfico de barras agrupadas demonstrando o desempenho preditivo sobre as quatro classes de sentimento originais (eixo X: "Contra", "A favor", "Indeterminado", "Neutro"). Para cada classe, há um agrupamento de quatro barras, representando os cenários de treinamento (Baseline em cinza, Sinônimos em azul, Rewriting em verde, Top Palavras em laranja). O eixo Y mede a Medida-F (0.0 a 1.0).
*   **Dados e Análise:** 
    *   Nas classes polarizadas (Contra e A Favor), as barras mantêm-se altas e estáveis em todos os cenários (oscilando entre 0.75 e 0.88). 
    *   A anomalia visual ocorre nas classes minoritárias e ambíguas. Na classe "Neutro", a barra verde (*Rewriting*) salta verticalmente, atingindo a marca de 0.80, enquanto as barras cinza (Baseline, 0.50), azul (Sinônimos, 0.50) e laranja (Top Palavras, 0.47) permanecem estagnadas abaixo da linha de 0.55.
    *   O gráfico evidencia de forma incontestável que apenas a técnica de reescrita (*Rewriting*) conseguiu destravar o aprendizado nas classes com ruído semântico severo.

**Gráfico 3: Impacto Global: Média Macro F1 por Cenário (Comparativo 3 vs 4 Classes)**
*   **Descrição Visual:** Gráfico de barras duplas agrupadas por cenário metodológico (eixo X: "Baseline", "Sinônimos", "Top Palavras", "Rewriting"). Cada cenário possui duas colunas emparelhadas: uma barra azul escura representando o desempenho no cenário original de 4 classes, e uma barra roxa representando o desempenho após a fusão para 3 classes. O eixo Y exibe a Média Macro F1, ampliado entre 0.5 e 1.0.
*   **Dados e Análise:**
    *   **Baseline:** 4 Classes (0.69) vs 3 Classes (0.76).
    *   **Sinônimos:** 4 Classes (0.68) vs 3 Classes (0.70).
    *   **Top Palavras:** 4 Classes (0.65) vs 3 Classes (0.71).
    *   **Rewriting:** 4 Classes (0.82) vs 3 Classes (0.83).
    *   O gráfico demonstra duas conclusões: (1) A barra roxa (3 classes) é invariavelmente superior à barra azul (4 classes) em todos os agrupamentos, comprovando que a fusão estrutural (agrupamento de Neutro e Indeterminado) sempre reduz o ruído. (2) As colunas do *Rewriting* destacam-se isoladamente no topo do gráfico, superando a marca de 0.82 em ambos os cenários e provando ser a única abordagem sintética que não sofre de *Data Drift* (visto que "Sinônimos" e "Top Palavras" performam pior que o próprio Baseline).
"""

with open(file_path, "a", encoding="utf-8") as f:
    f.write(appendix_content)

print("Apêndice de Gráficos adicionado com sucesso ao final do documento!")
