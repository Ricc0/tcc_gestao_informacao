import pandas as pd
import string
import collections
import nltk
import os
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

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

# Configurações estéticas globais
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = False

def gerar_grafico_comparativo_limpo():
    nltk.download('stopwords', quiet=True)
    pt_stopwords = set(stopwords.words('portuguese'))
    
    extra_stopwords = {
        'pra', 'tá', 'ter', 'ir', 'tudo', 'vai', 'sobre', 'ser', 'fazer', 'lá', 'aqui', 
        'vou', 'quer', 'querem', 'esta', 'essa', 'esse', 'deve', 'dar', 'dizer', 'ver', 
        'ficar', 'pode', 'assim', 'porque', 'só', 'outra', 'outro', 'outros', 'outras', 
        'bem', 'agora', 'então', 'todos', 'gente', 'nada', 'mas', 'como', 'mais', 'também', 
        'onde', 'quando', 'quem', 'ele', 'ela', 'eles', 'elas', 'uma', 'uns', 'umas', 
        'comentário', 'vídeo', 'canal', 'vídeos', 'comentários', 'contra', 'favor', 
        'anistia', 'pec', 'mano', 'véi', 'cara', 'olha'
    }
    stop_all = pt_stopwords.union(extra_stopwords)
    
    datasets = {
        'Original (Real)': os.path.join(DIR_ORIGINAIS, "Atos_esquerda_pec_classificado.csv"),
        'Sinônimos': os.path.join(DIR_SINTETICOS, "prompt_6_sinonimos.csv"),
        'Controle (SLAF)': os.path.join(DIR_SINTETICOS, "prompt_8_top_palavras.csv"),
        'Rewriting (Regex)': os.path.join(DIR_SINTETICOS, "prompt_7_rewriting.csv")
    }
    
    dfs = {}
    for name, path in datasets.items():
        df = pd.read_csv(path, encoding='utf-8')
        df = df.dropna(subset=['teor_classificado', 'comentario'])
        dfs[name] = df
        
    # Filtrar apenas as classes que sofreram data augmentation (incrementadas)
    augmented_classes = ['1 - Contra a Anistia', '3 - Indeterminado', '4 - Neutro']
    
    # 1. Extrair as top 100 palavras do Corpus Original nessas 3 classes
    df_orig_aug = dfs['Original (Real)'][dfs['Original (Real)']['teor_classificado'].isin(augmented_classes)]
    orig_text = ' '.join(df_orig_aug['comentario'].astype(str).tolist()).lower()
    text_clean = "".join([ch if ch not in string.punctuation else " " for ch in orig_text])
    tokens = [w for w in text_clean.split() if w not in stop_all and len(w) > 2]
    # Remover direita/esquerda da classe neutra/indeterminado
    tokens = [w for w in tokens if w not in ['direita', 'esquerda']]
    
    top_100_orig = set([word for word, _ in collections.Counter(tokens).most_common(100)])
    
    # 2. Obter o vocabulário mais frequente de cada base sintética (top 100 de cada)
    overlap_results = {}
    for name in ['Sinônimos', 'Controle (SLAF)', 'Rewriting (Regex)']:
        df_c = dfs[name]
        df_text = ' '.join(df_c['comentario'].astype(str).tolist()).lower()
        df_clean = "".join([ch if ch not in string.punctuation else " " for ch in df_text])
        df_tokens = [w for w in df_clean.split() if w not in stop_all and len(w) > 2]
        df_tokens = [w for w in df_tokens if w not in ['direita', 'esquerda']]
        
        top_100_synthetic = set([word for word, _ in collections.Counter(df_tokens).most_common(100)])
        
        # Calcular interseção
        overlap = len(top_100_orig.intersection(top_100_synthetic))
        overlap_results[name] = overlap

    # 3. Plotar o gráfico limpo e elegante
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#f8f9fa')
    ax.set_facecolor('#ffffff')
    
    # Ordem das barras
    categories = ['Sinônimos Simples', 'Controle (SLAF)', 'Rewriting (Regex)']
    valores = [overlap_results['Sinônimos'], overlap_results['Controle (SLAF)'], overlap_results['Rewriting (Regex)']]
    
    cores = ['#3498db', '#e67e22', '#2ecc71'] # Azul, Laranja, Verde Esmeralda
    
    bars = ax.barh(categories, valores, color=cores, edgecolor='white', height=0.5, alpha=0.9)
    
    # Estilização
    ax.set_title('Índice de Fidelidade Vocabular e Preservação Léxica\n'
                 'Percentual de palavras-âncora reais preservadas no vocabulário sintético (Top 100 termos reais)',
                 fontsize=12, fontweight='bold', pad=15, color='#2c3e50', loc='center')
    
    ax.set_xlim(0, 100)
    ax.set_xlabel('Porcentagem de Preservação Vocabular (%)', fontsize=10, labelpad=10, color='#2c3e50')
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#bdc3c7')
    
    # Adicionar os valores nas barras
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{width}%', 
                va='center', ha='left', fontsize=11, fontweight='bold', color='#2c3e50')
        
    plt.tight_layout()
    
    # Salvar
    output_path = os.path.join(DIR_GRAFICOS, 'grafico_comparativo_lexico.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    
    print(f"Novo gráfico simplificado sem sobreposições gerado com sucesso em: {output_path}")

if __name__ == "__main__":
    gerar_grafico_comparativo_limpo()
