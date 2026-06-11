import pandas as pd

from google.colab import drive
drive.mount('/content/drive')
#---
df = pd.read_csv("/content/drive/MyDrive/UFG/2025/TCC/Atos_esquerda_pec_classificado.csv")
df.info()
df_sintetico = pd.read_csv("/content/drive/MyDrive/UFG/2025/TCC/prompt_1.csv")
df_sintetico.info( )
df_sintetico_2 = pd.read_csv("/content/drive/MyDrive/UFG/2025/TCC/prompt_2.csv")
df_sintetico_2.info( )
df_sintetico_3 = pd.read_csv("/content/drive/MyDrive/UFG/2025/TCC/prompt_3.csv")
df_sintetico_3.info( )
df_sintetico_4 = pd.read_csv("/content/drive/MyDrive/UFG/2025/TCC/prompt_4.csv")
df_sintetico_4.info( )
df_sintetico_5 = pd.read_csv("/content/drive/MyDrive/UFG/2025/TCC/prompt_5.csv")
df_sintetico_5.info( )
#---
df['teor_classificado'].value_counts()
#---
# ==============================================================================
# CÓDIGO SVM - PROMPT1 - GEMINI PRO - 204 comentários(todos contra a anistia)
# ==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords

# Garantir que stopwords estão baixadas
nltk.download('stopwords', quiet=True)
pt_stopwords = stopwords.words('portuguese')

# 1. PREPARAÇÃO DOS DADOS REAIS
df_work = df.copy()

# Limpeza: Remover linhas onde não tem classificação
df_work = df_work.dropna(subset=['teor_classificado'])

# 2. PREPARAÇÃO DOS DADOS SINTÉTICOS
# Presume-se que df_sintetico já foi carregado anteriormente no seu ambiente
df_sintetico_work = df_sintetico.copy()
df_sintetico_work = df_sintetico_work.dropna(subset=['teor_classificado'])

# 3. DIVISÃO TREINO E TESTE (APENAS DADOS REAIS)
# Separamos 80% para treino e 20% estritamente para teste usando apenas o comentário principal
X_real = df_work['comentario'].astype(str)
y_real = df_work['teor_classificado']

X_train_real, X_test, y_train_real, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

# 4. INCORPORAÇÃO DOS DADOS SINTÉTICOS AO CONJUNTO DE TREINAMENTO
X_sintetico = df_sintetico_work['comentario'].astype(str)
y_sintetico = df_sintetico_work['teor_classificado']

X_train_final = pd.concat([X_train_real, X_sintetico])
y_train_final = pd.concat([y_train_real, y_sintetico])

print(f"Total de dados reais para teste: {len(X_test)}")
print(f"Total de dados para treino (Reais + Sintéticos): {len(X_train_final)}")
print("\nDistribuição das classes no Treino:\n", y_train_final.value_counts())

# 5. VETORIZAÇÃO
vetorizador = TfidfVectorizer(
    stop_words=pt_stopwords,
    max_features=5000,
    ngram_range=(1, 2)
)

# Ajustar o vetorizador apenas nos dados de treino para evitar vazamento de dados
X_train_vec = vetorizador.fit_transform(X_train_final)
X_test_vec = vetorizador.transform(X_test)

# 6. TREINAMENTO (SVM COM PESOS BALANCEADOS)
print("\nTreinando SVM... (Aguarde)")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_vec, y_train_final)

# 7. RESULTADOS
previsoes = svm_model.predict(X_test_vec)

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
print(classification_report(y_test, previsoes))

# Matriz de Confusão Visual
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, previsoes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_model.classes_,
            yticklabels=svm_model.classes_)
plt.title('Matriz de Confusão SVM')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.show()
#---
# ==============================================================================
# CÓDIGO SVM - PROMPT 2 - GEMINI PRO - 204 comentários(todos contra a anistia) + dados originais
# ==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords

# Garantir que stopwords estão baixadas
nltk.download('stopwords', quiet=True)
pt_stopwords = stopwords.words('portuguese')

# 1. PREPARAÇÃO DOS DADOS REAIS
df_work = df.copy()

# Limpeza: Remover linhas onde não tem classificação
df_work = df_work.dropna(subset=['teor_classificado'])

# 2. PREPARAÇÃO DOS DADOS SINTÉTICOS
# Presume-se que df_sintetico já foi carregado anteriormente no seu ambiente
df_sintetico_work = df_sintetico_2.copy()
df_sintetico_work = df_sintetico_work.dropna(subset=['teor_classificado'])

# 3. DIVISÃO TREINO E TESTE (APENAS DADOS REAIS)
# Separamos 80% para treino e 20% estritamente para teste usando apenas o comentário principal
X_real = df_work['comentario'].astype(str)
y_real = df_work['teor_classificado']

X_train_real, X_test, y_train_real, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

# 4. INCORPORAÇÃO DOS DADOS SINTÉTICOS AO CONJUNTO DE TREINAMENTO
X_sintetico = df_sintetico_work['comentário'].astype(str)
y_sintetico = df_sintetico_work['teor_classificado']

X_train_final = pd.concat([X_train_real, X_sintetico])
y_train_final = pd.concat([y_train_real, y_sintetico])

print(f"Total de dados reais para teste: {len(X_test)}")
print(f"Total de dados para treino (Reais + Sintéticos): {len(X_train_final)}")
print("\nDistribuição das classes no Treino:\n", y_train_final.value_counts())

# 5. VETORIZAÇÃO
vetorizador = TfidfVectorizer(
    stop_words=pt_stopwords,
    max_features=5000,
    ngram_range=(1, 2)
)

# Ajustar o vetorizador apenas nos dados de treino para evitar vazamento de dados
X_train_vec = vetorizador.fit_transform(X_train_final)
X_test_vec = vetorizador.transform(X_test)

# 6. TREINAMENTO (SVM COM PESOS BALANCEADOS)
print("\nTreinando SVM... (Aguarde)")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_vec, y_train_final)

# 7. RESULTADOS
previsoes = svm_model.predict(X_test_vec)

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
print(classification_report(y_test, previsoes))

# Matriz de Confusão Visual
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, previsoes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_model.classes_,
            yticklabels=svm_model.classes_)
plt.title('Matriz de Confusão SVM')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.show()
#---
# ==============================================================================
# CÓDIGO SVM - PROMPT 3 -GEMINI PRO - 204 comentários(todas as classificações)
# ==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords

# Garantir que stopwords estão baixadas
nltk.download('stopwords', quiet=True)
pt_stopwords = stopwords.words('portuguese')

# 1. PREPARAÇÃO DOS DADOS REAIS
df_work = df.copy()

# Limpeza: Remover linhas onde não tem classificação
df_work = df_work.dropna(subset=['teor_classificado'])

# 2. PREPARAÇÃO DOS DADOS SINTÉTICOS
# Presume-se que df_sintetico já foi carregado anteriormente no seu ambiente
df_sintetico_work = df_sintetico_3.copy()
df_sintetico_work = df_sintetico_work.dropna(subset=['teor_classificado'])

# 3. DIVISÃO TREINO E TESTE (APENAS DADOS REAIS)
# Separamos 80% para treino e 20% estritamente para teste usando apenas o comentário principal
X_real = df_work['comentario'].astype(str)
y_real = df_work['teor_classificado']

X_train_real, X_test, y_train_real, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

# 4. INCORPORAÇÃO DOS DADOS SINTÉTICOS AO CONJUNTO DE TREINAMENTO
X_sintetico = df_sintetico_work['comentario'].astype(str)
y_sintetico = df_sintetico_work['teor_classificado']

X_train_final = pd.concat([X_train_real, X_sintetico])
y_train_final = pd.concat([y_train_real, y_sintetico])

print(f"Total de dados reais para teste: {len(X_test)}")
print(f"Total de dados para treino (Reais + Sintéticos): {len(X_train_final)}")
print("\nDistribuição das classes no Treino:\n", y_train_final.value_counts())

# 5. VETORIZAÇÃO
vetorizador = TfidfVectorizer(
    stop_words=pt_stopwords,
    max_features=5000,
    ngram_range=(1, 2)
)

# Ajustar o vetorizador apenas nos dados de treino para evitar vazamento de dados
X_train_vec = vetorizador.fit_transform(X_train_final)
X_test_vec = vetorizador.transform(X_test)

# 6. TREINAMENTO (SVM COM PESOS BALANCEADOS)
print("\nTreinando SVM... (Aguarde)")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_vec, y_train_final)

# 7. RESULTADOS
previsoes = svm_model.predict(X_test_vec)

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
print(classification_report(y_test, previsoes))

# Matriz de Confusão Visual
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, previsoes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_model.classes_,
            yticklabels=svm_model.classes_)
plt.title('Matriz de Confusão SVM')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.show()
#---
# ==============================================================================
# CÓDIGO SVM-PROMPT 4-GEMINI PRO-204 comentários(todas as classificações) + csv base
# ==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords

# Garantir que stopwords estão baixadas
nltk.download('stopwords', quiet=True)
pt_stopwords = stopwords.words('portuguese')

# 1. PREPARAÇÃO DOS DADOS REAIS
df_work = df.copy()

# Limpeza: Remover linhas onde não tem classificação
df_work = df_work.dropna(subset=['teor_classificado'])

# 2. PREPARAÇÃO DOS DADOS SINTÉTICOS
# Presume-se que df_sintetico já foi carregado anteriormente no seu ambiente
df_sintetico_work = df_sintetico_4.copy()
df_sintetico_work = df_sintetico_work.dropna(subset=['teor_classificado'])

# 3. DIVISÃO TREINO E TESTE (APENAS DADOS REAIS)
# Separamos 80% para treino e 20% estritamente para teste usando apenas o comentário principal
X_real = df_work['comentario'].astype(str)
y_real = df_work['teor_classificado']

X_train_real, X_test, y_train_real, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

# 4. INCORPORAÇÃO DOS DADOS SINTÉTICOS AO CONJUNTO DE TREINAMENTO
X_sintetico = df_sintetico_work['comentario'].astype(str)
y_sintetico = df_sintetico_work['teor_classificado']

X_train_final = pd.concat([X_train_real, X_sintetico])
y_train_final = pd.concat([y_train_real, y_sintetico])

print(f"Total de dados reais para teste: {len(X_test)}")
print(f"Total de dados para treino (Reais + Sintéticos): {len(X_train_final)}")
print("\nDistribuição das classes no Treino:\n", y_train_final.value_counts())

# 5. VETORIZAÇÃO
vetorizador = TfidfVectorizer(
    stop_words=pt_stopwords,
    max_features=5000,
    ngram_range=(1, 2)
)

# Ajustar o vetorizador apenas nos dados de treino para evitar vazamento de dados
X_train_vec = vetorizador.fit_transform(X_train_final)
X_test_vec = vetorizador.transform(X_test)

# 6. TREINAMENTO (SVM COM PESOS BALANCEADOS)
print("\nTreinando SVM... (Aguarde)")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_vec, y_train_final)

# 7. RESULTADOS
previsoes = svm_model.predict(X_test_vec)

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
print(classification_report(y_test, previsoes))

# Matriz de Confusão Visual
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, previsoes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_model.classes_,
            yticklabels=svm_model.classes_)
plt.title('Matriz de Confusão SVM')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.show()
#---
# ==============================================================================
# CÓDIGO SVM - PROMPT 5 - chatgpt - 204 comentários(todas as classificações) + csv base

# ==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords

# Garantir que stopwords estão baixadas
nltk.download('stopwords', quiet=True)
pt_stopwords = stopwords.words('portuguese')

# 1. PREPARAÇÃO DOS DADOS REAIS
df_work = df.copy()

# Limpeza: Remover linhas onde não tem classificação
df_work = df_work.dropna(subset=['teor_classificado'])

# 2. PREPARAÇÃO DOS DADOS SINTÉTICOS
# Presume-se que df_sintetico já foi carregado anteriormente no seu ambiente
df_sintetico_work = df_sintetico_5.copy()
df_sintetico_work = df_sintetico_work.dropna(subset=['teor_classificado'])

# 3. DIVISÃO TREINO E TESTE (APENAS DADOS REAIS)
# Separamos 80% para treino e 20% estritamente para teste usando apenas o comentário principal
X_real = df_work['comentario'].astype(str)
y_real = df_work['teor_classificado']

X_train_real, X_test, y_train_real, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

# 4. INCORPORAÇÃO DOS DADOS SINTÉTICOS AO CONJUNTO DE TREINAMENTO
X_sintetico = df_sintetico_work['comentario'].astype(str)
y_sintetico = df_sintetico_work['teor_classificado']

X_train_final = pd.concat([X_train_real, X_sintetico])
y_train_final = pd.concat([y_train_real, y_sintetico])

print(f"Total de dados reais para teste: {len(X_test)}")
print(f"Total de dados para treino (Reais + Sintéticos): {len(X_train_final)}")
print("\nDistribuição das classes no Treino:\n", y_train_final.value_counts())

# 5. VETORIZAÇÃO
vetorizador = TfidfVectorizer(
    stop_words=pt_stopwords,
    max_features=5000,
    ngram_range=(1, 2)
)

# Ajustar o vetorizador apenas nos dados de treino para evitar vazamento de dados
X_train_vec = vetorizador.fit_transform(X_train_final)
X_test_vec = vetorizador.transform(X_test)

# 6. TREINAMENTO (SVM COM PESOS BALANCEADOS)
print("\nTreinando SVM... (Aguarde)")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_vec, y_train_final)

# 7. RESULTADOS
previsoes = svm_model.predict(X_test_vec)

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
print(classification_report(y_test, previsoes))

# Matriz de Confusão Visual
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_test, previsoes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_model.classes_,
            yticklabels=svm_model.classes_)
plt.title('Matriz de Confusão SVM')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.show()
#---
