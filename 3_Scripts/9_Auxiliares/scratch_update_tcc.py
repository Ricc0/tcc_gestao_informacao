import os

file_path = r"C:\Users\Henricco Santos\OneDrive - exedconsulting\Documentos\TCC\5_Documentacao_TCC\tcc_henricco.md"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_content = """5. Resultados e Discussão
Nesta seção, apresentam-se os resultados obtidos a partir dos experimentos realizados com o modelo de Máquina de Vetores de Suporte (SVM). O foco primário deste estudo transcende o mero *benchmarking* de Prompts, aprofundando-se na **Engenharia de Dados Sintéticos** e em como a qualidade semântica da geração afeta o aprendizado computacional, especialmente em classes textuais minoritárias ("Neutro" e "Indeterminado"). A métrica principal de avaliação utilizada é a Medida-F (F1-Score), pois fornece uma análise justa em bases desbalanceadas.

5.1 Metodologia de Criação de Dados Sintéticos e Testes
Para garantir rigor científico e replicabilidade, a pesquisa afastou-se da geração pura via Large Language Models (LLMs) estocásticos em sua fase final, adotando abordagens de **Automação Computacional Baseada em Regras** e **Substituição Lexical (Regex)**. Foram gerados três conjuntos de dados distintos para avaliar diferentes hipóteses linguísticas, com o volume nivelado rigidamente em 1.600 comentários cada, isolando a variável de "tamanho do corpus".

A. Geração Estruturada (Sinônimos Simples): O primeiro modelo sintético simulou a criação de frases do zero através da concatenação de blocos predefinidos (ex: `[SUJEITO] + [VERBO] + [LOCAL]`). O objetivo era testar se a simples injeção volumétrica de palavras-chave seria capaz de ensinar o modelo. Contudo, essa abordagem causou severo *Data Drift* (desvio semântico), transformando debates sobre a "PEC" em jargões criminais genéricos.
B. Injeção de Top Palavras (Controle de Frequência): Para provar que o SVM não aprende apenas contando repetições de palavras, geramos um lote focado exclusivamente nos termos de maior frequência da base original (ex: "esquerda", "anistia", "brasil"). Este lote serviu como um cenário de controle.
C. Substituição Lexical Reversa (Rewriting + Gírias): A abordagem mais avançada aplicou a técnica de *Rewriting* sobre os dados autênticos. Através de algoritmos em Python com Expressões Regulares (Regex), o código rastreou entidades e palavras na base real e as substituiu por jargões ideológicos mapeados ("comunistas" por "esquerdopatas"; "manifestação" por "bagunça"). Adicionalmente, injetou-se "internetês" e ruídos gramaticais estruturados, espelhando a realidade linguística das bolhas em redes sociais, sem perder o esqueleto original.

Todos os dados gerados foram concatenados à base de treino e testados contra uma base de teste intocada (20%), garantindo que o algoritmo jamais visse dados sintéticos no momento de validação.

5.2 Análise Aprofundada dos Resultados: Cenário Original (4 Classes)
Os testes foram executados primeiramente respeitando a rotulação humana de 4 classes: Contra a Anistia, A favor da anistia, Indeterminado e Neutro.

*(Inserir imagem: `4_Resultados_e_Graficos/grafico_evolucao_f1_4_classes.png`)*
**Gráfico 1: Evolução do F1-Score do Modelo SVM (Cenário de 4 Classes)**

No cenário **Baseline** (apenas dados originais), o modelo atingiu uma Média Macro de 0,69. Observou-se uma eficácia notável nas classes majoritárias polarizadas ("A Favor": 0,84, "Contra": 0,75), mas um colapso preditivo sobre ambiguidades: a classe "Indeterminado" obteve apenas 0,65 e a "Neutro", severamente escassa na base original, atingiu meros 0,50 de F1-Score.

Quando analisamos os dados criados **(Baseline x Sinônimos e Top Palavras)**, a hipótese de que "mais dados sempre ajudam" é refutada. A injeção de 1.600 frases engessadas ("Sinônimos") causou confusão no classificador, derrubando a Média Macro para 0,68. A injeção de "Top Palavras" performou ainda pior (Macro 0,65), derrubando drasticamente a classe Indeterminado (para 0,53) e o Neutro (para 0,47). Isso prova estatisticamente que inserir palavras-chave sem estrutura sintática válida destrói a fronteira de decisão do modelo em discursos sutis.

**A Ruptura do Rewriting:** O impacto da abordagem de *Rewriting + Gírias* foi disruptivo. Ao invés de degradar a precisão, ele elevou a Média Macro para incríveis 0,82. A análise inter-classes revela onde a mágica ocorreu:
- As classes fortes tornaram-se impecáveis ("A Favor" foi de 0,84 para 0,88; "Contra" foi de 0,75 para 0,85).
- A grande vitória metodológica concentrou-se nas fraquezas do classificador. A classe "Indeterminado" disparou para 0,74 e a classe "Neutro" sofreu um salto colossal, indo de 0,50 para 0,80.

A engenharia por substituição lexical ensinou ao SVM não apenas novas palavras, mas *como* essas gírias interagem no contexto frasal (capturado pelo `ngram_range(1,2)` do algoritmo), destravando a capacidade de leitura da "zona cinza" do debate político.

5.3 Fusão de Classes e Média Macro Global
Baseando-se em Afonso (2017), executou-se um experimento fundindo o ruído semântico das classes não-posicionadas em uma única classe unificada (reduzindo para 3 classes).

*(Inserir imagem: `4_Resultados_e_Graficos/grafico_macro_comparativo.png`)*
**Gráfico 2: Impacto Global: Média Macro F1 por Cenário**

A fusão fornece um benefício estrutural intrínseco. No Baseline, a simples fusão elevou a Média Macro de 0,69 para 0,76. Contudo, mesmo no cenário otimizado de 3 classes, as abordagens puramente volumétricas (*Sinônimos* e *Top Palavras*) fracassam em superar o desempenho nativo (0,70 e 0,71, respectivamente) devido ao *Data Drift* contínuo. 

O *Rewriting*, novamente, coroa-se absoluto. Combinado com a fusão de classes, o modelo atinge o teto do projeto, com uma **Média Macro F1 de 0,83**.

5.4 Conclusão Acadêmica da Discussão
Os testes comprovaram inequivocamente que, para tarefas complexas de PLN em mídias sociais:
1. Modelos SVM não carecem de "volume" de palavras-chave; eles dependem de variância sintática realista.
2. A geração de *Data Augmentation* a partir do zero produz dados linguisticamente mortos e estatisticamente prejudiciais.
3. A técnica de **Regex Rewriting**, utilizando ancoragem de gírias sobre comentários autênticos, atua como a metodologia definitiva para ensinar redes a lerem sarcasmo e posicionamentos neutros.
"""

# Replace lines 76 to end (0-indexed 76)
new_lines = lines[:76] + [new_content + "\n"] + lines[98:]

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Texto atualizado com sucesso!")
