# -*- coding: utf-8 -*-
# =====================================================================================
#  COMPARAÇÃO DE PROMPTS 6/7/8  ***VERSÃO COM ERRO***
# -------------------------------------------------------------------------------------
#  Este script avalia as 3 técnicas no SVM. Ele NÃO tem bug de split em si para as
#  técnicas Sinônimos (p6) e SLAF (p8). Os problemas estão em:
#
#  >>> ERRO Nº 1 (herdado): carrega 'prompt_7_rewriting.csv', que foi gerado COM
#      VAZAMENTO (ver gerador_rewriting_COM_ERRO.py). Por isso o Rewriting aparenta
#      Macro F1 ~0,82 — número inflado por data leakage, não por generalização real.
#
#  >>> ERRO Nº 2 (metodológico): no teste de 3 classes (fusão), o train_test_split é
#      REFEITO sobre os rótulos já fundidos (random_state=42 sobre y_real_fus). Isso
#      gera um CONJUNTO DE TESTE DIFERENTE do usado no cenário de 4 classes, de modo
#      que os dois cenários não são estritamente comparáveis sobre as mesmas instâncias.
#
#  >>> CORREÇÃO COMPLETA EM: ../5_CODIGO_REFEITO/avaliacao_svm_CORRIGIDO.py
# =====================================================================================
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
pt_stopwords = stopwords.words('portuguese')

base_file = r"C:\Users\Henricco Santos\OneDrive - exedconsulting\Documentos\TCC\2_Dados\Originais\Atos_esquerda_pec_classificado.csv"
prompts = {
    "Prompt 6 (Sinônimos)": r"C:\Users\Henricco Santos\OneDrive - exedconsulting\Documentos\TCC\2_Dados\Sinteticos\prompt_6_sinonimos.csv",
    # >>> ERRO Nº 1: este CSV foi gerado com vazamento de teste (sementes do corpus inteiro)
    "Prompt 7 (Rewriting)": r"C:\Users\Henricco Santos\OneDrive - exedconsulting\Documentos\TCC\2_Dados\Sinteticos\prompt_7_rewriting.csv",
    "Prompt 8 (Top Palavras)": r"C:\Users\Henricco Santos\OneDrive - exedconsulting\Documentos\TCC\2_Dados\Sinteticos\prompt_8_top_palavras.csv"
}

df_real = pd.read_csv(base_file)
df_real = df_real.dropna(subset=['teor_classificado'])
X_real = df_real['comentario'].astype(str)
y_real = df_real['teor_classificado']

# Fusao 3 classes
df_real_fus = df_real.copy()
df_real_fus['teor_classificado'] = df_real_fus['teor_classificado'].replace({'3 - Indeterminado': '4 - Neutro'})
y_real_fus = df_real_fus['teor_classificado']

results = []

for prompt_name, file_path in prompts.items():
    print(f"\n{'='*50}\nTESTANDO: {prompt_name}\n{'='*50}")
    df_sint = pd.read_csv(file_path)
    df_sint = df_sint.dropna(subset=['teor_classificado'])
    
    # --- TESTE 4 CLASSES ---
    X_train_real, X_test, y_train_real, y_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )
    
    X_sint = df_sint['comentario'].astype(str)
    y_sint = df_sint['teor_classificado']
    
    X_train_final = pd.concat([X_train_real, X_sint])
    y_train_final = pd.concat([y_train_real, y_sint])
    
    vetorizador = TfidfVectorizer(stop_words=pt_stopwords, max_features=5000, ngram_range=(1, 2))
    X_train_vec = vetorizador.fit_transform(X_train_final)
    X_test_vec = vetorizador.transform(X_test)
    
    svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    svm_model.fit(X_train_vec, y_train_final)
    previsoes = svm_model.predict(X_test_vec)
    
    print("\n--- RELATÓRIO 4 CLASSES ---")
    print(classification_report(y_test, previsoes))
    
    # --- TESTE 3 CLASSES (FUSÃO) ---
    df_sint_fus = df_sint.copy()
    df_sint_fus['teor_classificado'] = df_sint_fus['teor_classificado'].replace({'3 - Indeterminado': '4 - Neutro'})
    y_sint_fus = df_sint_fus['teor_classificado']
    
    # >>> ERRO Nº 2: split REFEITO sobre rótulos fundidos -> conjunto de teste diferente
    #     do cenário de 4 classes. O correto é dividir UMA vez e só relabelar para a fusão.
    X_train_real_fus, X_test_fus, y_train_real_fus, y_test_fus = train_test_split(
        X_real, y_real_fus, test_size=0.2, random_state=42, stratify=y_real_fus
    )
    
    X_train_final_fus = pd.concat([X_train_real_fus, X_sint])
    y_train_final_fus = pd.concat([y_train_real_fus, y_sint_fus])
    
    vetorizador_fus = TfidfVectorizer(stop_words=pt_stopwords, max_features=5000, ngram_range=(1, 2))
    X_train_vec_fus = vetorizador_fus.fit_transform(X_train_final_fus)
    X_test_vec_fus = vetorizador_fus.transform(X_test_fus)
    
    svm_model_fus = SVC(kernel='linear', class_weight='balanced', random_state=42)
    svm_model_fus.fit(X_train_vec_fus, y_train_final_fus)
    previsoes_fus = svm_model_fus.predict(X_test_vec_fus)
    
    print("\n--- RELATÓRIO 3 CLASSES (FUSÃO) ---")
    print(classification_report(y_test_fus, previsoes_fus))
