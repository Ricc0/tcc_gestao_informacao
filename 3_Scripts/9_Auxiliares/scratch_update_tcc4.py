import os

file_path = r"C:\Users\Henricco Santos\OneDrive - exedconsulting\Documentos\TCC\5_Documentacao_TCC\tcc_henricco.md"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_content = """**A. Geração Estruturada (Sinônimos Simples)**
O primeiro modelo sintético simulou a criação de sentenças a partir do zero utilizando automação algorítmica. Um script Python foi programado para concatenar blocos estruturais predefinidos (por exemplo: sujeito, verbo, local e adjetivo), com o fito de representar as quatro classes ideológicas. 

* **Exemplo de Lógica Estrutural (Classe "Contra a Anistia"):**
  ```python
  sujeitos_contra = ["criminosos", "infratores", "indivíduos"]
  verbos_contra = ["devem cumprir pena", "precisam ser julgados"]
  ```

* **Exemplo de Lógica Estrutural (Classe "Indeterminado"):**
  ```python
  sujeitos_indet = ["essa situação", "o cenário atual"]
  verbos_indet = ["é muito confusa", "não me permite opinar"]
  ```

O algoritmo selecionava aleatoriamente elementos dessas listas para compor o corpus sintético. O objetivo consistia em testar se a injeção mecânica e volumétrica de vocabulário específico seria suficiente para parametrizar o modelo em cada posicionamento. Contudo, essa rigidez estrutural resultou em um grave *Data Drift* (desvio semântico). Ao forçar estruturas sintáticas engessadas e repetitivas, o texto perdeu a conexão com as entidades focais do debate real (como "PEC", "Congresso" ou "Constituição"), transformando as instâncias geradas em frases genéricas desvinculadas do contexto sociopolítico original.

**B. Injeção de Termos de Alta Frequência (Cenário de Controle)**
A fim de corroborar a hipótese de que o algoritmo SVM extrai atributos discriminativos da estrutura sintática — e não estritamente da contagem isolada de lexemas —, desenvolveu-se um cenário de controle empírico. O procedimento metodológico consistiu em analisar estatisticamente o corpus autêntico e isolar os termos de maior ocorrência para cada respectiva classe (por exemplo: "esquerda" com 708 ocorrências, "anistia" com 465, e "processo legal" com 210).

* **Exemplo de Geração Desestruturada:**
  > "esquerda anistia brasil povo direita pec constitucional"

A título de exemplificação, o algoritmo gerou sentenças agrupando esses termos de alta frequência em blocos textuais desprovidos de coesão gramatical, adicionando conectivos genéricos de forma aleatória. O propósito deste lote de dados consistiu em isolar o peso estatístico do vocabulário. Se o modelo SVM atingisse alta eficácia preditiva neste cenário, evidenciaria-se que seu funcionamento restringe-se a uma abordagem rudimentar de contagem de palavras (*Bag of Words*), ignorando a complexidade linguística.

**C. Substituição Lexical Reversa (Rewriting e Modulação de Ruído)**
A abordagem mais refinada deste estudo abdicou da criação estéril de textos para empregar a técnica de *Rewriting* (reescrita) diretamente sobre as instâncias autênticas. Um algoritmo fundamentado em Expressões Regulares (Regex) processou cada comentário real, identificando âncoras nominais e substituindo-as por equivalentes lexicais mapeados em um dicionário focado na variação linguística da internet, respeitando as nuances de cada uma das quatro classes.

* **Mapeamento Computacional (Exemplos de Regex por Classe):**
  ```python
  # Classe: Contra a Anistia
  r'\\bmanifestantes?\\b': ['baderneiros', 'vândalos']
  
  # Classe: A Favor da Anistia
  r'\\bprisão\\b': ['injustiça', 'perseguição política']
  
  # Classe: Neutro
  r'\\bnotícia\\b': ['informação atualizada', 'reportagem']
  
  # Classe: Indeterminado
  r'\\bnão sei\\b': ['difícil compreender', 'complexo opinar']
  ```

* **Exemplificação Prática (Pré e Pós-Processamento):**
  * *Dado Autêntico (Contra a Anistia):* "Esses manifestantes devem arcar com as consequências de seus atos."
  * *Sintético Gerado:* "Francamente, esses baderneiros devem arcar com as consequências de seus atos. [RUÍDO_INJETADO]"
  
  * *Dado Autêntico (Neutro):* "O portal publicou uma notícia sobre o andamento do projeto."
  * *Sintético Gerado:* "O portal publicou uma informação atualizada sobre o andamento do projeto."

Em contraste com as abordagens anteriores, a técnica de *Rewriting* garantiu a preservação da estrutura sintática original — mantendo as conjunções, advérbios e a fluidez argumentativa humana intactas. A engenharia limitou-se à injeção controlada de polarização léxica e ruídos informais característicos das mídias sociais. Essa metodologia sustentou o foco na entidade alvo do discurso (a "PEC"), fornecendo ao classificador um recurso de aumento de dados (*Data Augmentation*) dotado de alta verossimilhança.
"""

# We are replacing from the line starting with "**A. Geração Estruturada" up to the line starting with "5.2 Análise Aprofundada"
start_idx = -1
for i, line in enumerate(lines):
    if line.startswith("**A. Geração Estruturada"):
        start_idx = i
        break

end_idx = -1
if start_idx != -1:
    for i in range(start_idx, len(lines)):
        if line.startswith("5.2 Análise Aprofundada dos Resultados"):
            end_idx = i
            break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [new_content + "\n\n"] + lines[end_idx:]
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Sucesso!")
else:
    print(f"Indices não encontrados: start_idx={start_idx}, end_idx={end_idx}")
