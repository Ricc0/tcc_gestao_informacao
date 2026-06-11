import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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


# ---------------------------------------------------------------------
# GRÁFICO 1: Volume de Dados Sintéticos
# ---------------------------------------------------------------------
dados_volume = {
    'Abordagem': ['Sinônimos', 'Rewriting', 'Controle'],
    'Quantidade de Comentários': [1600, 1600, 1600]
}

df_vol = pd.DataFrame(dados_volume)

plt.figure(figsize=(8, 6))
bars = plt.bar(df_vol['Abordagem'], df_vol['Quantidade de Comentários'], color=['#3498db', '#2ecc71', '#e67e22'])
plt.title('Volume de Dados Sintéticos Gerados por Lote', fontsize=14, pad=15)
plt.ylabel('Nº de Comentários')
plt.ylim(0, 1800)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 30, int(yval), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(DIR_GRAFICOS, 'grafico_volume_dados.png'), dpi=300)
plt.close()

# ---------------------------------------------------------------------
# GRÁFICO 2: Evolução do F1-Score (4 Classes)
# ---------------------------------------------------------------------
dados_f1 = {
    'Classe': ['Contra', 'A favor', 'Indeterminado', 'Neutro'],
    'Baseline (Reais)': [0.75, 0.84, 0.65, 0.50],
    'Sinônimos': [0.76, 0.83, 0.63, 0.50],
    'Rewriting + Gírias': [0.85, 0.88, 0.74, 0.80],
    'Controle': [0.76, 0.82, 0.53, 0.47]
}

df_f1 = pd.DataFrame(dados_f1)

x = np.arange(len(df_f1['Classe']))
width = 0.2

plt.figure(figsize=(14, 7))

rects1 = plt.bar(x - width*1.5, df_f1['Baseline (Reais)'], width, label='Baseline', color='#95a5a6')
rects2 = plt.bar(x - width*0.5, df_f1['Sinônimos'], width, label='Sinônimos', color='#3498db')
rects3 = plt.bar(x + width*0.5, df_f1['Rewriting + Gírias'], width, label='Rewriting', color='#2ecc71')
rects4 = plt.bar(x + width*1.5, df_f1['Controle'], width, label='Controle', color='#e67e22')

plt.title('Evolução do F1-Score do Modelo SVM (Cenário 4 Classes)', fontsize=15, pad=20)
plt.ylabel('Medida-F (F1-Score)', fontsize=12)
plt.xlabel('Classes de Sentimento', fontsize=12)
plt.xticks(x, df_f1['Classe'], fontsize=11)
plt.ylim(0, 1.0)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(DIR_GRAFICOS, 'grafico_evolucao_f1_4_classes.png'), dpi=300)
plt.close()

# ---------------------------------------------------------------------
# GRÁFICO 3: Média Macro Comparativa (3 vs 4 Classes)
# ---------------------------------------------------------------------
dados_macro = {
    'Abordagem': ['Baseline', 'Sinônimos', 'Controle', 'Rewriting'],
    'Média Macro (4 Classes)': [0.69, 0.68, 0.65, 0.82],
    'Média Macro (3 Classes)': [0.76, 0.70, 0.71, 0.83]
}

df_macro = pd.DataFrame(dados_macro)

x_macro = np.arange(len(df_macro['Abordagem']))
width_macro = 0.35

plt.figure(figsize=(10, 6))

rects_4c = plt.bar(x_macro - width_macro/2, df_macro['Média Macro (4 Classes)'], width_macro, label='4 Classes', color='#34495e')
rects_3c = plt.bar(x_macro + width_macro/2, df_macro['Média Macro (3 Classes)'], width_macro, label='3 Classes (Fusão)', color='#9b59b6')

plt.title('Impacto Global: Média Macro F1 por Cenário', fontsize=15, pad=20)
plt.ylabel('Média Macro F1', fontsize=12)
plt.xticks(x_macro, df_macro['Abordagem'], fontsize=11)
plt.ylim(0.5, 1.0)
plt.legend(loc='upper left')

autolabel(rects_4c)
autolabel(rects_3c)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_GRAFICOS, 'grafico_macro_comparativo.png'), dpi=300)
plt.close()

print("Novos gráficos gerados com sucesso e salvos na pasta '4_Resultados_e_Graficos'.")
