import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
pt_stopwords = stopwords.words('portuguese')

base_file = r"C:\Users\Henricco Santos\OneDrive - exedconsulting\Documentos\TCC\2_Dados\Originais\Atos_esquerda_pec_classificado.csv"
df_real = pd.read_csv(base_file)
df_real = df_real.dropna(subset=['teor_classificado'])
X_real = df_real['comentario'].astype(str)
y_real = df_real['teor_classificado']

# --- TESTE 4 CLASSES ---
X_train, X_test, y_train, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

vetorizador = TfidfVectorizer(stop_words=pt_stopwords, max_features=5000, ngram_range=(1, 2))
X_train_vec = vetorizador.fit_transform(X_train)
X_test_vec = vetorizador.transform(X_test)

svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_vec, y_train)
previsoes = svm_model.predict(X_test_vec)

print("\n--- BASELINE: RELATORIO 4 CLASSES ---")
print(classification_report(y_test, previsoes))

# --- TESTE 3 CLASSES (FUSAO) ---
df_real_fus = df_real.copy()
df_real_fus['teor_classificado'] = df_real_fus['teor_classificado'].replace({'3 - Indeterminado': '4 - Neutro'})
y_real_fus = df_real_fus['teor_classificado']

X_train_fus, X_test_fus, y_train_fus, y_test_fus = train_test_split(
    X_real, y_real_fus, test_size=0.2, random_state=42, stratify=y_real_fus
)

vetorizador_fus = TfidfVectorizer(stop_words=pt_stopwords, max_features=5000, ngram_range=(1, 2))
X_train_vec_fus = vetorizador_fus.fit_transform(X_train_fus)
X_test_vec_fus = vetorizador_fus.transform(X_test_fus)

svm_model_fus = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model_fus.fit(X_train_vec_fus, y_train_fus)
previsoes_fus = svm_model_fus.predict(X_test_vec_fus)

print("\n--- BASELINE: RELATORIO 3 CLASSES (FUSAO) ---")
print(classification_report(y_test_fus, previsoes_fus))
