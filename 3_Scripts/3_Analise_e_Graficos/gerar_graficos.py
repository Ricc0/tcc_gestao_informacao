import matplotlib.pyplot as plt
import numpy as np
import os

# Configurações de estilo
plt.style.use('ggplot')
cores = ['#2b8cbe', '#fdbb84']

# GRÁFICO 1: Comparação de F1-Score (Base 4 classes vs Fusão 3 classes)
fig, ax = plt.subplots(figsize=(14, 7))

labels = ['Média Macro', 'Média Ponderada', 'A Favor', 'Contra', 'Indeterminado', 'Neutro', 'Não-Posicionados\n(Fusão)']
# Valores baseados no relatório de classificação (4 classes vs 3 classes)
base_4_classes = [0.68, 0.76, 0.84, 0.77, 0.65, 0.48, 0]  # 0 indica que a classe não existia ou foi fundida
fusao_3_classes = [0.76, 0.78, 0.83, 0.75, 0, 0, 0.69]    # 0 indica que a classe deixou de existir (foi fundida)

x = np.arange(len(labels))
width = 0.35

rects1 = ax.bar(x - width/2, base_4_classes, width, label='Modelo Base (4 Classes)', color=cores[0])
rects2 = ax.bar(x + width/2, fusao_3_classes, width, label='Fusão (3 Classes)', color=cores[1])

ax.set_ylabel('F1-Score')
ax.set_title('Impacto Global da Fusão de Classes no F1-Score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)
ax.legend()

# Adiciona os valores nas barras
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('TCC_google_drive/grafico_fusao_classes.png', dpi=300)
plt.close()

# GRÁFICO 2: Evolução das classes minoritárias
fig, ax = plt.subplots(figsize=(10, 6))

labels_2 = ['Neutro', 'Indeterminado', 'Classe Fundida\n(Não-Posicionados)']

# Calculando o "Antes" da Classe Fundida (Média ponderada do F1 no modelo de 4 classes)
# Neutro: 208 amostras, F1 = 0.48
# Indeterminado: 667 amostras, F1 = 0.65
# Media Ponderada = (208*0.48 + 667*0.65) / (208+667) = 0.61

f1_antes = [0.48, 0.65, 0.61]  
f1_depois = [0.51, 0.65, 0.69] # 0.51 (Sintetico), 0.65 (Sintetico), 0.69 (Modelo Fundido 3 Classes)

x2 = np.arange(len(labels_2))

rects3 = ax.bar(x2 - width/2, f1_antes, width, label='Abordagem Padrão (Base 4 Classes)', color='#756bb1')
rects4 = ax.bar(x2 + width/2, f1_depois, width, label='Melhoria (Com Sintéticos / Fusão)', color='#31a354')

ax.set_ylabel('F1-Score')
ax.set_title('Evolução do F1-Score das Classes Minoritárias')
ax.set_xticks(x2)
ax.set_xticklabels(labels_2)
ax.set_ylim(0, 1.0)
ax.legend()

autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
plt.savefig('TCC_google_drive/grafico_evolucao_minoritarias.png', dpi=300)
plt.close()

print("Gráficos gerados com sucesso na pasta TCC_google_drive!")
