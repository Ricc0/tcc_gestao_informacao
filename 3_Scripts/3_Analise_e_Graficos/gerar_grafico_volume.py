import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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

def gerar_grafico_volume():
    path_original = os.path.join(DIR_ORIGINAIS, "Atos_esquerda_pec_classificado.csv")
    path_sintetico = os.path.join(DIR_SINTETICOS, "prompt_7_rewriting.csv") # Representa o volume de 1600 comentários
    
    print(f"Lendo dados originais de: {path_original}")
    print(f"Lendo dados sintéticos de: {path_sintetico}")
    
    # Lendo originais
    try:
        df_original = pd.read_csv(path_original, encoding='utf-8')
    except Exception:
        df_original = pd.read_csv(path_original, encoding='latin-1')
        
    # Lendo sintéticos
    try:
        df_sintetico = pd.read_csv(path_sintetico, encoding='utf-8')
    except Exception:
        df_sintetico = pd.read_csv(path_sintetico, encoding='latin-1')

    # Contagem Original (focando apenas nas 4 classes válidas)
    df_original = df_original.dropna(subset=['teor_classificado'])
    contagem_original = df_original['teor_classificado'].value_counts()
    
    classes_validas = ['1 - Contra a Anistia', '2 - A favor da anistia', '3 - Indeterminado', '4 - Neutro']
    contagem_original = contagem_original[contagem_original.index.isin(classes_validas)]
    
    # Contagem Sintético
    df_sintetico = df_sintetico.dropna(subset=['teor_classificado'])
    contagem_sintetico = df_sintetico['teor_classificado'].value_counts()
    contagem_sintetico = contagem_sintetico[contagem_sintetico.index.isin(classes_validas)]
    
    # Extrair valores ordenados de forma idêntica
    valores_originais = [contagem_original.get(classe, 0) for classe in classes_validas]
    valores_sinteticos = [contagem_sintetico.get(classe, 0) for classe in classes_validas]
    
    # Configuração estética premium com pure Matplotlib
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = '#2c3e50'
    plt.rcParams['axes.labelcolor'] = '#2c3e50'
    plt.rcParams['xtick.color'] = '#2c3e50'
    plt.rcParams['ytick.color'] = '#2c3e50'
    
    fig, ax = plt.subplots(figsize=(12, 7.5))
    
    # Configuração de eixos agrupados
    x = np.arange(len(classes_validas))
    width = 0.35  # Largura das barras
    
    # Paleta de cores moderna: Azul Profissional para Original e Verde Esmeralda para Sintético
    color_original = '#2c3e50'
    color_sintetico = '#2ecc71'
    
    # Plotando as barras agrupadas
    rects1 = ax.bar(x - width/2, valores_originais, width, label='Original (Reais)', color=color_original, edgecolor='#1a252f', linewidth=0.8, alpha=0.95)
    rects2 = ax.bar(x + width/2, valores_sinteticos, width, label='Sintético (Adicionado)', color=color_sintetico, edgecolor='#27ae60', linewidth=0.8, alpha=0.95)
    
    # Título e Rótulos
    ax.set_title('Distribuição Volumétrica de Dados por Classe\n(Original vs. Sintético)', fontsize=16, pad=25, weight='bold', color='#1a252f')
    ax.set_xlabel('Classes de Sentimento', fontsize=13, labelpad=12, weight='semibold')
    ax.set_ylabel('Quantidade de Comentários (Nº)', fontsize=13, labelpad=12, weight='semibold')
    
    # Customização do eixo X
    labels_amigaveis = ['Contra a Anistia', 'A Favor da Anistia', 'Indeterminado', 'Neutro']
    ax.set_xticks(x)
    ax.set_xticklabels(labels_amigaveis, fontsize=11, weight='semibold')
    
    # Adicionando os números exatos em cima das barras
    ax.bar_label(rects1, padding=5, fontsize=11, weight='bold', color='#2c3e50')
    ax.bar_label(rects2, padding=5, fontsize=11, weight='bold', color='#2c3e50')
    
    # Totais gerais
    total_original = int(sum(valores_originais))
    total_sintetico = int(sum(valores_sinteticos))
    total_final = total_original + total_sintetico
    
    # Criando o box explicativo de resumo com visual premium
    texto_resumo = (
        f"RESUMO DO DATASET\n"
        f"- Total Original: {total_original:,} (68.1%)\n"
        f"- Total Sintetico: {total_sintetico:,} (31.9%)\n"
        f"-------------------\n"
        f"- Volume Final: {total_final:,} comentarios"
    ).replace(',', '.')
    
    # Posicionando a caixa de resumo de forma elegante
    props = dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#bdc3c7', linewidth=1, alpha=0.95)
    ax.text(0.70, 0.93, texto_resumo, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, weight='medium', color='#2c3e50')
    
    # Melhorias no grid e layout
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
    ax.xaxis.grid(False) # Remove linhas verticais para ficar mais limpo
    
    # Remove bordas superiores e laterais para um visual moderno
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#bdc3c7')
    
    # Ajustando legenda
    ax.legend(title='Origem dos Dados', title_fontsize=12, fontsize=11, loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e8f0')
    
    # Aumentando limite do eixo Y para dar espaço aos rótulos de valores
    ax.set_ylim(0, max(max(valores_originais), max(valores_sinteticos)) * 1.15)
    
    plt.tight_layout()
    
    # Salvando na pasta de gráficos
    output_dir = DIR_GRAFICOS
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "grafico_volume_original_vs_sintetico.png")
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Gráfico gerado com sucesso em: {save_path}")

if __name__ == "__main__":
    gerar_grafico_volume()

