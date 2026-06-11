# -*- coding: utf-8 -*-
# =====================================================================================
#  VALIDAÇÃO CRUZADA + TESTE DE SIGNIFICÂNCIA  (responde à crítica metodológica C2)
# -------------------------------------------------------------------------------------
#  Em vez de uma única partição treino/teste, usa RepeatedStratifiedKFold
#  (5 folds × 6 repetições = 30 medições, mantendo ~80/20 estratificado).
#  Para cada fold:
#    - Baseline, Sinônimos (p6), SLAF (p8) e Rewriting são avaliados no MESMO teste.
#    - O Rewriting é REGENERADO a partir do treino daquele fold (sem vazamento).
#  Reporta média ± desvio do Macro F1 (e do F1 do Neutro) e testes pareados
#  (Wilcoxon e t de Student) de cada técnica CONTRA o baseline.
# =====================================================================================
import pandas as pd, numpy as np, random, re, os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon, ttest_rel
import nltk; nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
SW = stopwords.words('portuguese')

SEED = 42
N_SPLITS, N_REPEATS = 5, 6   # 30 medições

def _raiz():
    p = os.path.dirname(os.path.abspath(__file__))
    while p != os.path.dirname(p) and not os.path.isdir(os.path.join(p, '2_Dados')):
        p = os.path.dirname(p)
    return p
RAIZ = _raiz()
DO = os.path.join(RAIZ, '2_Dados', 'Originais')
DS = os.path.join(RAIZ, '2_Dados', 'Sinteticos')

df  = pd.read_csv(os.path.join(DO,'Atos_esquerda_pec_classificado.csv')).dropna(subset=['teor_classificado']).reset_index(drop=True)
sin = pd.read_csv(os.path.join(DS,'prompt_6_sinonimos.csv')).dropna(subset=['teor_classificado'])
slaf= pd.read_csv(os.path.join(DS,'prompt_8_top_palavras.csv')).dropna(subset=['teor_classificado'])

def apply_rewrite(text):
    if not isinstance(text,str): return ""
    text=text.lower()
    reps={r'\bcomunistas?\b':['esquerdopatas','petistas','pão com mortadela','turma do L'],
        r'\besquerda\b':['esquerdalha','canhotos','turma da mortadela'],r'\bmanifestação\b':['bagunça','baderneiros','festa','passeata'],
        r'\bato\b':['bagunça','baderneiros'],r'\babsurdo\b':['fala sério','mt absurdo','inaceitável véi','slá mano mt absurdo'],
        r'\bbandidos?\b':['vagabundos','corruptos','meliantes','ladrões'],r'\bpolíticos?\b':['bandidos de terno','engravatados','corruptos'],
        r'\bdinheiro\b':['grana','grana nossa','impostos','nosso suor'],r'\bconcordo\b':['tbm acho','isso aí','disse tudo','falou pouco mas falou bonito'],
        r'\bnão sei\b':['slá','tô perdido','sei lá mano','difícil dizer'],r'\blegal\b':['daora','top','mto bom'],r'\bverdade\b':['vdd','papo reto','isso aí','fato']}
    for p,r in reps.items(): text=re.sub(p,random.choice(r),text)
    pre=["mano, ","véi, ","mds... ","cara, na boa, ","olha, ",""]; suf=[" kkkkk"," rsrs"," 😂"," 🤡","..."," !!",""]
    text=random.choice(pre)+text+random.choice(suf)
    if random.random()<0.2: text=text.upper()
    return text

def gen_rewrite_from(df_train, seed):
    random.seed(seed); np.random.seed(seed)
    rows=[]
    for cls,n in [('1 - Contra a Anistia',800),('3 - Indeterminado',400),('4 - Neutro',400)]:
        pool=df_train[df_train['teor_classificado']==cls]['comentario'].astype(str).tolist()
        if not pool: continue
        for _ in range(n): rows.append({'comentario':apply_rewrite(random.choice(pool)),'teor_classificado':cls})
    return pd.DataFrame(rows)

def fuse(s): return s.replace({'3 - Indeterminado':'4 - Neutro'})

def fit_eval(Xtr, ytr, Xte, yte):
    vec=TfidfVectorizer(stop_words=SW, max_features=5000, ngram_range=(1,2))
    m=SVC(kernel='linear', class_weight='balanced', random_state=SEED)
    m.fit(vec.fit_transform(Xtr), ytr)
    pred=m.predict(vec.transform(Xte))
    macro=f1_score(yte,pred,average='macro')
    neutro=f1_score(yte,pred,average=None,labels=['4 - Neutro'])[0] if '4 - Neutro' in set(yte) else np.nan
    return macro, neutro

X = df['comentario'].astype(str); y = df['teor_classificado']
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

res = {sc:{c:{'macro':[], 'neutro':[]} for c in ['Baseline','Sinônimos','SLAF','Rewriting']} for sc in ['4classes','3classes']}

print(f"Rodando {N_SPLITS}x{N_REPEATS} = {N_SPLITS*N_REPEATS} folds...")
for k,(tr,te) in enumerate(rskf.split(X,y)):
    dtr = df.iloc[tr]; dte = df.iloc[te]
    rew = gen_rewrite_from(dtr, seed=SEED+k)   # regenerado por fold, só do treino
    for sc, fus in [('4classes',False),('3classes',True)]:
        ytr0 = fuse(dtr['teor_classificado']) if fus else dtr['teor_classificado']
        yte0 = fuse(dte['teor_classificado']) if fus else dte['teor_classificado']
        Xtr0 = dtr['comentario'].astype(str); Xte0 = dte['comentario'].astype(str)
        def with_syn(syn):
            ys = fuse(syn['teor_classificado']) if fus else syn['teor_classificado']
            return pd.concat([Xtr0, syn['comentario'].astype(str)]), pd.concat([ytr0, ys])
        # baseline
        for cond, builder in [('Baseline', lambda:(Xtr0,ytr0)),
                              ('Sinônimos', lambda:with_syn(sin)),
                              ('SLAF', lambda:with_syn(slaf)),
                              ('Rewriting', lambda:with_syn(rew))]:
            Xt,yt = builder()
            ma,ne = fit_eval(Xt,yt,Xte0,yte0)
            res[sc][cond]['macro'].append(ma); res[sc][cond]['neutro'].append(ne)
    if (k+1)%5==0: print(f"  {k+1} folds concluídos")

def stars(p):
    return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

for sc in ['4classes','3classes']:
    print("\n"+"="*78)
    print(f"CENÁRIO {sc}  —  Macro F1 (média ± dp de {N_SPLITS*N_REPEATS} folds)  |  vs Baseline")
    print("="*78)
    base = np.array(res[sc]['Baseline']['macro'])
    print(f"{'Baseline':12s}  Macro={base.mean():.3f} ± {base.std():.3f}    (Neutro F1={np.nanmean(res[sc]['Baseline']['neutro']):.3f})")
    for cond in ['Sinônimos','SLAF','Rewriting']:
        arr=np.array(res[sc][cond]['macro'])
        diff=arr.mean()-base.mean()
        try: w_p=wilcoxon(arr, base).pvalue
        except Exception: w_p=float('nan')
        t_p=ttest_rel(arr, base).pvalue
        ne=np.nanmean(res[sc][cond]['neutro'])
        print(f"{cond:12s}  Macro={arr.mean():.3f} ± {arr.std():.3f}    Δ={diff:+.3f}  "
              f"Wilcoxon p={w_p:.4f} {stars(w_p)}  t p={t_p:.4f}  (Neutro F1={ne:.3f})")

print("\nLegenda: *** p<0,001  ** p<0,01  * p<0,05  ns = não significativo (vs baseline).")
print("Δ = diferença de Macro F1 média em relação ao baseline.")
