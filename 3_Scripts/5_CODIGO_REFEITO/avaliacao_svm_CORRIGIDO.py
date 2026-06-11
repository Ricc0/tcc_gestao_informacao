# -*- coding: utf-8 -*-
# =====================================================================================
#  AVALIAÇÃO SVM  ***VERSÃO CORRIGIDA E UNIFICADA***
# -------------------------------------------------------------------------------------
#  Corrige os dois erros das versões em ../4_CODIGO_COM_ERRO/:
#   - ERRO Nº 1 (vazamento): o Rewriting é regenerado AQUI com sementes só do treino.
#   - ERRO Nº 2 (split refeito): UMA única divisão treino/teste, fixada por índice, é
#     usada nos DOIS cenários (4 classes e 3 classes/fusão). Para a fusão, apenas
#     RELABELAMOS as mesmas instâncias — nunca refazemos o split.
#
#  Reproduz a tabela corrigida do TCC:
#   - Baseline 4cls 0,69 / 3cls 0,77
#   - Sinônimos / SLAF / Rewriting NÃO superam o baseline (nenhuma técnica ajuda)
#   - Único ganho legítimo: a fusão de classes (0,69 -> 0,77), sem dados sintéticos.
# =====================================================================================
import pandas as pd, random, re, numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
sw = stopwords.words('portuguese')

SEED = 42

def _raiz_projeto():
    p = os.path.dirname(os.path.abspath(__file__))
    while p != os.path.dirname(p) and not os.path.isdir(os.path.join(p, '2_Dados')):
        p = os.path.dirname(p)
    return p
RAIZ = _raiz_projeto()
DIR_ORIGINAIS  = os.path.join(RAIZ, '2_Dados', 'Originais')
DIR_SINTETICOS = os.path.join(RAIZ, '2_Dados', 'Sinteticos')

# ---------- Dados ----------
df = pd.read_csv(os.path.join(DIR_ORIGINAIS, 'Atos_esquerda_pec_classificado.csv')) \
       .dropna(subset=['teor_classificado']).reset_index(drop=True)
sin  = pd.read_csv(os.path.join(DIR_SINTETICOS, 'prompt_6_sinonimos.csv')).dropna(subset=['teor_classificado'])
slaf = pd.read_csv(os.path.join(DIR_SINTETICOS, 'prompt_8_top_palavras.csv')).dropna(subset=['teor_classificado'])

# ---------- DIVISÃO ÚNICA E FIXA (por índice) — usada nos 2 cenários ----------
idx = df.index.values
tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=df['teor_classificado'])
df_train = df.loc[tr_idx].reset_index(drop=True)   # 80% treino (fixo)
df_test  = df.loc[te_idx].reset_index(drop=True)   # 20% teste  (fixo, idêntico nos 2 cenários)

# ---------- Rewriting corrigido: sementes só do treino ----------
def apply_rewrite(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    reps = {
        r'\bcomunistas?\b': ['esquerdopatas','petistas','pão com mortadela','turma do L'],
        r'\besquerda\b': ['esquerdalha','canhotos','turma da mortadela'],
        r'\bmanifestação\b': ['bagunça','baderneiros','festa','passeata'],
        r'\bato\b': ['bagunça','baderneiros'],
        r'\babsurdo\b': ['fala sério','mt absurdo','inaceitável véi','slá mano mt absurdo'],
        r'\bbandidos?\b': ['vagabundos','corruptos','meliantes','ladrões'],
        r'\bpolíticos?\b': ['bandidos de terno','engravatados','corruptos'],
        r'\bdinheiro\b': ['grana','grana nossa','impostos','nosso suor'],
        r'\bconcordo\b': ['tbm acho','isso aí','disse tudo','falou pouco mas falou bonito'],
        r'\bnão sei\b': ['slá','tô perdido','sei lá mano','difícil dizer'],
        r'\blegal\b': ['daora','top','mto bom'],
        r'\bverdade\b': ['vdd','papo reto','isso aí','fato']}
    for p, r in reps.items(): text = re.sub(p, random.choice(r), text)
    pre = ["mano, ","véi, ","mds... ","cara, na boa, ","olha, ",""]
    suf = [" kkkkk"," rsrs"," 😂"," 🤡","..."," !!",""]
    text = random.choice(pre) + text + random.choice(suf)
    if random.random() < 0.2: text = text.upper()
    return text

def gera_rewrite_treino():
    random.seed(SEED); np.random.seed(SEED)
    rows = []
    for cls, n in [('1 - Contra a Anistia',800),('3 - Indeterminado',400),('4 - Neutro',400)]:
        seeds = df_train[df_train['teor_classificado']==cls]['comentario'].astype(str).tolist()
        for _ in range(n):
            rows.append({'comentario': apply_rewrite(random.choice(seeds)), 'teor_classificado': cls})
    return pd.DataFrame(rows)

rew = gera_rewrite_treino()

# ---------- Avaliação ----------
def fuse(s):  # funde Indeterminado em Neutro -> "Não-Posicionados"
    return s.replace({'3 - Indeterminado': '4 - Neutro'})

def avaliar(df_synth, fusion):
    Xtr = df_train['comentario'].astype(str).copy(); ytr = df_train['teor_classificado'].copy()
    Xte = df_test['comentario'].astype(str).copy();  yte = df_test['teor_classificado'].copy()
    if fusion: ytr, yte = fuse(ytr), fuse(yte)   # só relabela; NÃO refaz o split
    if df_synth is not None:
        ys = fuse(df_synth['teor_classificado']) if fusion else df_synth['teor_classificado']
        Xtr = pd.concat([Xtr, df_synth['comentario'].astype(str)]); ytr = pd.concat([ytr, ys])
    vec = TfidfVectorizer(stop_words=sw, max_features=5000, ngram_range=(1,2))
    m = SVC(kernel='linear', class_weight='balanced', random_state=SEED)
    m.fit(vec.fit_transform(Xtr), ytr)
    pred = m.predict(vec.transform(Xte))
    return f1_score(yte, pred, average='macro'), classification_report(yte, pred)

print("="*70)
print("TABELA CORRIGIDA — divisão única fixa (teste idêntico nos 2 cenários)")
print("="*70)
for nome, synth in [('Baseline (só reais)', None), ('Sinônimos (p6)', sin),
                    ('SLAF/Controle (p8)', slaf), ('Rewriting CORRIGIDO', rew)]:
    macro4, _ = avaliar(synth, fusion=False)
    macro3, _ = avaliar(synth, fusion=True)
    print(f"{nome:24s}  Macro F1: 4 classes = {macro4:.3f}  |  3 classes (fusão) = {macro3:.3f}")

print("\n--- Relatório detalhado: Baseline (4 classes) ---")
print(avaliar(None, fusion=False)[1])
print("--- Relatório detalhado: Rewriting corrigido (4 classes) ---")
print(avaliar(rew, fusion=False)[1])
