import matplotlib.pyplot as plt
import os

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


# Configurações estéticas globais
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# Eixos de cores curados e harmoniosos
cores = {
    'Contra': '#2c3e50',       # Navy Blue
    'A Favor': '#c0392b',      # Crimson Red
    'Indeterminado': '#8e44ad',# Plum Purple
    'Neutro': '#7f8c8d'        # Slate Gray
}

# Dados empíricos refinados da análise léxica
# Para Neutro e Indeterminado, 'direita' e 'esquerda' foram excluídos conforme solicitado
dados = {
    '1 - Contra a Anistia': {
        'palavras': ['esquerda', 'povo', 'direita', 'brasil', 'bandidagem', 'bandeira', 'blindagem', 'brasileiro', 'sim', 'políticos'][::-1],
        'freqs': [199, 166, 164, 154, 142, 120, 89, 84, 58, 45][::-1],
        'cor': cores['Contra']
    },
    '2 - A Favor da Anistia': {
        'palavras': ['esquerda', 'bandeira', 'brasil', 'povo', 'direita', 'pessoas', 'artistas', 'vermelho', 'show', 'dinheiro'][::-1],
        'freqs': [322, 206, 172, 143, 141, 135, 120, 114, 99, 89][::-1],
        'cor': cores['A Favor']
    },
    '3 - Indeterminado': {
        'palavras': ['bandeira', 'brasil', 'povo', 'sim', 'patetico', 'brasileira', 'blindagem', 'manifestação', 'cnn', 'eua'][::-1],
        'freqs': [69, 63, 45, 38, 34, 26, 24, 23, 23, 19][::-1],
        'cor': cores['Indeterminado']
    },
    '4 - Neutro': {
        'palavras': ['povo', 'brasileiro', 'ato', 'brasileiros', 'manifestação', 'defende', 'políticos', 'brasil', 'cnn', 'sim'][::-1],
        'freqs': [53, 32, 29, 22, 21, 18, 16, 16, 15, 14][::-1],
        'cor': cores['Neutro']
    }
}

fig, axs = plt.subplots(2, 2, figsize=(15, 11))
axs = axs.ravel()

for i, (titulo, info) in enumerate(dados.items()):
    ax = axs[i]
    bars = ax.barh(info['palavras'], info['freqs'], color=info['cor'], edgecolor='black', alpha=0.9, height=0.6)
    
    # Customizações por subplot
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=10, color='#2c3e50')
    ax.set_xlabel('Frequência Absoluta', fontsize=10, labelpad=5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Rótulo de valores no final de cada barra horizontal
    max_val = max(info['freqs'])
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max_val * 0.015), bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                ha='left', va='center', fontsize=9, color='#2c3e50', fontweight='bold')
                
    # Margem do eixo X para os valores não colidirem
    ax.set_xlim(0, max_val * 1.15)

plt.suptitle('Termos Críticos de Maior Recorrência Absoluta por Classe (Corpus Original)', 
             fontsize=16, fontweight='bold', y=0.97, color='#2c3e50')
plt.tight_layout(rect=[0, 0, 1, 0.94])

# Criar pasta se não existir e salvar
output_path = os.path.join(DIR_GRAFICOS, 'grafico_frequencia_lexica.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Gráfico de frequências léxicas gerado com sucesso em: {output_path}")
