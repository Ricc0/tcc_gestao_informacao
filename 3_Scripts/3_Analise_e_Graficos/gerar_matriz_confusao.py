# -*- coding: utf-8 -*-
# =====================================================================================
#  MATRIZ DE CONFUSÃO E MÉTRICAS POR CLASSE — MODELO BASE (4 classes)
# -------------------------------------------------------------------------------------
#  Treina o classificador base (SVM linear sobre TF-IDF, apenas dados reais) na mesma
#  partição 80/20 estratificada usada no TCC e gera:
#    - o relatório de Precisão/Revocação/F1/suporte por classe (Tabela 1 do TCC);
#    - a matriz de confusão (Figura 2 do TCC), salva como PNG.
#  A matriz evidencia a confusão entre Neutro e Indeterminado, que fundamenta a
#  fusão de classes discutida na Seção 5.4.
#
#  Saída: 4_Resultados_e_Graficos/matriz_confusao_baseline.png
# =====================================================================================
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
SW = stopwords.words('portuguese')

# Raiz do projeto (resolve a pasta TCC a partir de qualquer subpasta)
def _raiz_projeto():
    p = os.path.dirname(os.path.abspath(__file__))
    while p != os.path.dirname(p) and not os.path.isdir(os.path.join(p, '2_Dados')):
        p = os.path.dirname(p)
    return p
RAIZ = _raiz_projeto()
DIR_ORIGINAIS = os.path.join(RAIZ, '2_Dados', 'Originais')
DIR_GRAFICOS = os.path.join(RAIZ, '4_Resultados_e_Graficos')

df = pd.read_csv(os.path.join(DIR_ORIGINAIS, 'Atos_esquerda_pec_classificado.csv')).dropna(subset=['teor_classificado'])
X = df['comentario'].astype(str)
y = df['teor_classificado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vec = TfidfVectorizer(stop_words=SW, max_features=5000, ngram_range=(1, 2))
modelo = SVC(kernel='linear', class_weight='balanced', random_state=42)
modelo.fit(vec.fit_transform(X_train), y_train)
pred = modelo.predict(vec.transform(X_test))

labels = ['1 - Contra a Anistia', '2 - A favor da anistia', '3 - Indeterminado', '4 - Neutro']
nomes = ['Contra', 'A favor', 'Indeterminado', 'Neutro']

print("=== Métricas por classe (modelo base, 4 classes) ===")
print(classification_report(y_test, pred, labels=labels, target_names=nomes, zero_division=0))

cm = confusion_matrix(y_test, pred, labels=labels)

plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar(fraction=0.046, pad=0.04)
plt.xticks(range(4), nomes, rotation=30, ha='right')
plt.yticks(range(4), nomes)
limiar = cm.max() / 2
for i in range(4):
    for j in range(4):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > limiar else 'black', fontsize=12, fontweight='bold')
plt.ylabel('Classe real', fontsize=11)
plt.xlabel('Classe predita', fontsize=11)
plt.title('Matriz de Confusão — Modelo Base (4 Classes)', fontsize=12, pad=12)
plt.tight_layout()

saida = os.path.join(DIR_GRAFICOS, 'matriz_confusao_baseline.png')
plt.savefig(saida, dpi=200)
plt.close()
print(f"\nMatriz de confusão salva em: {saida}")
