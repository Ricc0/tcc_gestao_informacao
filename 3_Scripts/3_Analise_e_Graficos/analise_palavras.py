import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('stopwords', quiet=True)
pt_stopwords = stopwords.words('portuguese')

# Adicionar palavras comuns irrelevantes que possam aparecer
pt_stopwords.extend([
    'q', 'pra', 'pro', 'vc', 'vcs', 'ta', 'tá', 'to', 'tô', 'aí', 'lá', 'vai', 'ser', 
    'sobre', 'tudo', 'nada', 'pois', 'então', 'assim', 'aqui', 'ter', 'todos', 'todas', 
    'vão', 'olha', 'bom', 'boa', 'trás', 'devem', 'merecem', 'pagar', 'lugar', 'complicado', 
    'gente', 'pessoas', 'sim', 'não', 'nao', 'cada', 'algum', 'alguma', 'nenhum', 'outros',
    'outras', 'outro', 'outra', 'quem', 'qual', 'quais', 'onde', 'porque', 'porquê', 'qualquer',
    'mano', 'véi', 'vei', 'mds', 'cara', 'kkkkk', 'kkkk', 'kkk', 'kk', 'rsrs', 'rs', 'mt', 'mto', 'slá', 'sla', 'tbm', 'vdd'
])

current_dir = os.path.dirname(__file__)
# === Raiz do projeto (resolve a pasta TCC a partir de qualquer subpasta) ===
def _raiz_projeto():
    p = os.path.dirname(os.path.abspath(__file__))
    while p != os.path.dirname(p) and not os.path.isdir(os.path.join(p, '2_Dados')):
        p = os.path.dirname(p)
    return p
RAIZ = _raiz_projeto()
DIR_ORIGINAIS  = os.path.join(RAIZ, '2_Dados', 'Originais')
DIR_SINTETICOS = os.path.join(RAIZ, '2_Dados', 'Sinteticos')
DIR_GRAFICOS   = os.path.join(RAIZ, '4_Resultados_e_Graficos')


files = {
    'Original': os.path.join(DIR_ORIGINAIS, 'Atos_esquerda_pec_classificado.csv'),
    'Sinônimos Simples': os.path.join(DIR_SINTETICOS, 'prompt_6_sinonimos.csv'),
    'Rewriting + Gírias': os.path.join(DIR_SINTETICOS, 'prompt_7_rewriting.csv')
}

def clean_text(text):
    if not isinstance(text, str):
        return []
    # Remover pontuação e transformar em minúsculo
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenizar (separar por espaço)
    words = text.split()
    # Remover stopwords
    words = [w for w in words if w not in pt_stopwords and len(w) > 2]
    return words

plt.figure(figsize=(18, 6))

for i, (label, filename) in enumerate(files.items(), 1):
    df = pd.read_csv(filename)  # filename ja e caminho absoluto
    
    # Extrair palavras
    all_words = []
    for comment in df['comentario'].dropna():
        all_words.extend(clean_text(comment))
        
    # Contar frequências
    word_counts = Counter(all_words)
    top_20 = word_counts.most_common(20)
    
    # Preparar para plotar
    words, counts = zip(*top_20)
    
    plt.subplot(1, 3, i)
    plt.barh(list(reversed(words)), list(reversed(counts)), color='skyblue')
    plt.title(f'Top 20 Palavras - {label}')
    plt.xlabel('Frequência')
    plt.ylabel('Palavras')

plt.tight_layout()
output_file = os.path.join(DIR_GRAFICOS, 'frequencia_palavras_top20.png')
plt.savefig(output_file, dpi=300)
print(f"Gráfico gerado em: {output_file}")

# Também printar no console
for label, filename in files.items():
    df = pd.read_csv(filename)  # filename ja e caminho absoluto
    all_words = []
    for comment in df['comentario'].dropna():
        all_words.extend(clean_text(comment))
    word_counts = Counter(all_words)
    print(f"\n--- {label} ---")
    for w, c in word_counts.most_common(20):
        print(f"{w}: {c}")

