# -*- coding: utf-8 -*-
# =====================================================================================
#  TESTE SVM (Sinônimos - prompt_6)  ***VERSÃO COM ERRO***
# -------------------------------------------------------------------------------------
#  Este script testa a técnica Sinônimos (p6), que NÃO tem vazamento. Mas ele apresenta:
#  >>> ERRO Nº 2 (metodológico): no teste de 3 classes (fusão), o train_test_split é
#      REFEITO sobre os rótulos fundidos -> conjunto de teste DIFERENTE do de 4 classes.
#      O correto é dividir UMA única vez e apenas relabelar para a fusão.
#  (Observação: a versão original também lia os CSV do próprio diretório; aqui o caminho
#   foi corrigido para a pasta 2_Dados para que o script rode.)
#  >>> CORREÇÃO COMPLETA EM: ../5_CODIGO_REFEITO/avaliacao_svm_CORRIGIDO.py
# =====================================================================================
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords

# Garantir que stopwords estão baixadas
nltk.download('stopwords', quiet=True)
pt_stopwords = stopwords.words('portuguese')

def _raiz_projeto():
    p = os.path.dirname(os.path.abspath(__file__))
    while p != os.path.dirname(p) and not os.path.isdir(os.path.join(p, '2_Dados')):
        p = os.path.dirname(p)
    return p
RAIZ = _raiz_projeto()
base_file = os.path.join(RAIZ, "2_Dados", "Originais", "Atos_esquerda_pec_classificado.csv")
sintetico_file = os.path.join(RAIZ, "2_Dados", "Sinteticos", "prompt_6_sinonimos.csv")

# Carregar dados
df = pd.read_csv(base_file)
df_sintetico = pd.read_csv(sintetico_file)

# 1. PREPARAÇÃO DOS DADOS REAIS
df_work = df.copy()
df_work = df_work.dropna(subset=['teor_classificado'])

# 2. PREPARAÇÃO DOS DADOS SINTÉTICOS
df_sintetico_work = df_sintetico.copy()
df_sintetico_work = df_sintetico_work.dropna(subset=['teor_classificado'])

# --- TESTE 1: 4 CLASSES ---
print("=============================================")
print("          TESTE SVM (4 CLASSES)              ")
print("=============================================")

X_real = df_work['comentario'].astype(str)
y_real = df_work['teor_classificado']

X_train_real, X_test, y_train_real, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

X_sintetico = df_sintetico_work['comentario'].astype(str)
y_sintetico = df_sintetico_work['teor_classificado']

X_train_final = pd.concat([X_train_real, X_sintetico])
y_train_final = pd.concat([y_train_real, y_sintetico])

print(f"Total de dados reais para teste: {len(X_test)}")
print(f"Total de dados para treino (Reais + Sintéticos): {len(X_train_final)}")
print("\nDistribuição das classes no Treino:\n", y_train_final.value_counts())

vetorizador = TfidfVectorizer(
    stop_words=pt_stopwords,
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vetorizador.fit_transform(X_train_final)
X_test_vec = vetorizador.transform(X_test)

svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_vec, y_train_final)

previsoes = svm_model.predict(X_test_vec)
print("\n--- RELATÓRIO DE CLASSIFICAÇÃO (4 CLASSES) ---")
print(classification_report(y_test, previsoes))


# --- TESTE 2: FUSÃO 3 CLASSES ---
print("\n\n=============================================")
print("       TESTE SVM COM FUSÃO (3 CLASSES)       ")
print("=============================================")

df_fusion_real = df_work.copy()
df_fusion_sint = df_sintetico_work.copy()

# Fundir Neutro em Indeterminado (Criando a classe Não-Posicionados)
df_fusion_real['teor_classificado'] = df_fusion_real['teor_classificado'].replace({
    '3 - Indeterminado': '4 - Neutro'
})

df_fusion_sint['teor_classificado'] = df_fusion_sint['teor_classificado'].replace({
    '3 - Indeterminado': '4 - Neutro'
})

X_real_fus = df_fusion_real['comentario'].astype(str)
y_real_fus = df_fusion_real['teor_classificado']

# >>> ERRO Nº 2: split refeito sobre rótulos fundidos -> teste diferente do de 4 classes
X_train_real_fus, X_test_fus, y_train_real_fus, y_test_fus = train_test_split(
    X_real_fus, y_real_fus, test_size=0.2, random_state=42, stratify=y_real_fus
)

X_sintetico_fus = df_fusion_sint['comentario'].astype(str)
y_sintetico_fus = df_fusion_sint['teor_classificado']

X_train_final_fus = pd.concat([X_train_real_fus, X_sintetico_fus])
y_train_final_fus = pd.concat([y_train_real_fus, y_sintetico_fus])

print(f"Total de dados reais para teste (Fusão): {len(X_test_fus)}")
print(f"Total de dados para treino (Reais + Sintéticos - Fusão): {len(X_train_final_fus)}")
print("\nDistribuição das classes no Treino:\n", y_train_final_fus.value_counts())

vetorizador_fus = TfidfVectorizer(
    stop_words=pt_stopwords,
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec_fus = vetorizador_fus.fit_transform(X_train_final_fus)
X_test_vec_fus = vetorizador_fus.transform(X_test_fus)

svm_model_fus = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model_fus.fit(X_train_vec_fus, y_train_final_fus)

previsoes_fus = svm_model_fus.predict(X_test_vec_fus)
print("\n--- RELATÓRIO DE CLASSIFICAÇÃO (3 CLASSES) ---")
print(classification_report(y_test_fus, previsoes_fus))
