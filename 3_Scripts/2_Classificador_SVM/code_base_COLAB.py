import pandas as pd

from google.colab import drive
drive.mount('/content/drive')
#---
df = pd.read_csv("/content/drive/MyDrive/UFG/2025/TCC/Atos_esquerda_pec_classificado.csv")
df.info()
#---
df['teor_classificado'].value_counts()
#---
df_cleaned = df.copy()

# Ensure 'teor' column is string type before applying string methods and replacements
df_cleaned['teor'] = df_cleaned['teor'].astype(str)

# Standardize existing string values (lowercase and strip whitespace)
df_cleaned['teor'] = df_cleaned['teor'].str.lower().str.strip()

# Define the mapping for replacement
# Note: "contra a anistia" is already handled by .str.lower().str.strip()
# but can be explicitly included if there were other casing variations
replacement_map = {
    '2': 'favoravel a anistia',
    '1': 'contra a anistia',
    '3': 'indeterminado',
    '4': 'neutro',
    'contra a anistia': 'contra a anistia' # Explicitly ensure this value remains consistent
}

# Apply the replacements
df_cleaned['teor'] = df_cleaned['teor'].replace(replacement_map)

print("New distribution of 'teor' after cleaning and reformatting:")
print(df_cleaned['teor'].value_counts())
#---
# ==============================================================================
# CÓDIGO SVM - ADAPTADO PARA LER 'teor_classificado'
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

# 1. PREPARAÇÃO (TRABALHANDO NUMA CÓPIA PARA NÃO ALTERAR O DF ORIGINAL)
df_work = df.copy()

# Limpeza: Remover linhas onde não tem classificação (teor_classificado é nulo)
df_work = df_work.dropna(subset=['teor_classificado'])

# Feature Engineering:
df_work['respondendo outro comentario'] = df_work['respondendo outro comentario'].fillna('')
df_work['texto_completo'] = df_work['comentario'].astype(str) + " " + df_work['respondendo outro comentario'].astype(str)

print(f"Total de dados válidos para treino: {len(df_work)}")
print("Distribuição das classes:\n", df_work['teor_classificado'].value_counts())

# 2. VETORIZAÇÃO
vetorizador = TfidfVectorizer(
    stop_words=pt_stopwords,
    max_features=5000,
    ngram_range=(1, 2) # Pega palavras sozinhas e pares
)

X = vetorizador.fit_transform(df_work['texto_completo'])
y = df_work['teor_classificado']

# 3. DIVISÃO TREINO E TESTE (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TREINAMENTO (SVM COM PESOS BALANCEADOS)
print("\nTreinando SVM... (Aguarde)")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)

# 5. RESULTADOS
previsoes = svm_model.predict(X_test)

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
# SVM COM FUSÃO DE CLASSES (3 CLASSES)
# ==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. PREPARAÇÃO E FUSÃO
# Vamos trabalhar numa cópia limpa
df_fusion = df.copy()

# Limpeza básica
df_fusion = df_fusion.dropna(subset=['teor_classificado'])
#df_fusion['teor_classificado'] = df_fusion['teor_classificado'].str.lower().str.strip()

# --- AQUI ACONTECE A MÁGICA DA FUSÃO ---
# Tudo que for 'neutro' vira 'indeterminado' (ou vice-versa)
# Estamos criando uma super-classe de "Não-Posicionados"
df_fusion['teor_classificado'] = df_fusion['teor_classificado'].replace({
    '3 - Indeterminado': '4 - Neutro'            # Funde Neutro em Indeterminado
    # Se quiser chamar de outro nome, pode mudar, mas mantenha consistente
})
# Nota: Agora teremos apenas: 'favoravel a anistia', 'contra a anistia', 'indeterminado'

# Contexto (Comentário + Resposta)
df_fusion['respondendo outro comentario'] = df_fusion['respondendo outro comentario'].fillna('')
df_fusion['texto_completo'] = df_fusion['comentario'].astype(str) + " " + df_fusion['respondendo outro comentario'].astype(str)

print("--- NOVA DISTRIBUIÇÃO (3 CLASSES) ---")
print(df_fusion['teor_classificado'].value_counts())

# 2. VETORIZAÇÃO
vetorizador = TfidfVectorizer(
    stop_words=pt_stopwords,
    max_features=5000,
    ngram_range=(1, 2)
)

X = vetorizador.fit_transform(df_fusion['texto_completo'])
y = df_fusion['teor_classificado']

# 3. DIVISÃO TREINO E TESTE
# Stratify garante que a proporção das 3 classes se mantenha
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TREINAMENTO
print("\nTreinando SVM com 3 classes... (Aguarde)")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)

# 5. RESULTADOS
previsoes = svm_model.predict(X_test)

print("\n--- RELATÓRIO FINAL (FUSÃO) ---")
print(classification_report(y_test, previsoes))

# Matriz 3x3
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, previsoes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=svm_model.classes_,
            yticklabels=svm_model.classes_)
plt.title('Matriz de Confusão (3 Classes)')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.show()
#---
# ==============================================================================
# CÓDIGO SVM - ADAPTADO PARA LER 'teor_classificado' (APENAS COMENTÁRIO)
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

# 1. PREPARAÇÃO (TRABALHANDO NUMA CÓPIA PARA NÃO ALTERAR O DF ORIGINAL)
df_work = df.copy()

# Limpeza: Remover linhas onde não tem classificação (teor_classificado é nulo)
df_work = df_work.dropna(subset=['teor_classificado'])

print(f"Total de dados válidos para treino: {len(df_work)}")
print("Distribuição das classes:\n", df_work['teor_classificado'].value_counts())

# 2. VETORIZAÇÃO
vetorizador = TfidfVectorizer(
    stop_words=pt_stopwords,
    max_features=5000,
    ngram_range=(1, 2) # Pega palavras sozinhas e pares
)

# Utilizando estritamente a coluna 'comentario'
X = vetorizador.fit_transform(df_work['comentario'].astype(str))
y = df_work['teor_classificado']

# 3. DIVISÃO TREINO E TESTE (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TREINAMENTO (SVM COM PESOS BALANCEADOS)
print("\nTreinando SVM... (Aguarde)")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)

# 5. RESULTADOS
previsoes = svm_model.predict(X_test)

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
