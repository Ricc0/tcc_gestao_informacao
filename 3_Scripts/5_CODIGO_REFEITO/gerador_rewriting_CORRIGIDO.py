# -*- coding: utf-8 -*-
# =====================================================================================
#  GERADOR REWRITING  ***VERSÃO CORRIGIDA (SEM VAZAMENTO)***
# -------------------------------------------------------------------------------------
#  Correção do bug descrito em ../4_CODIGO_COM_ERRO/gerador_rewriting_COM_ERRO.py.
#
#  >>> A CORREÇÃO <<<
#  Fazemos o train_test_split ANTES de gerar qualquer dado sintético e usamos como
#  semente APENAS a partição de TREINO (80%). Assim, nenhum comentário do conjunto de
#  teste pode ser reescrito para dentro do treino -> não há mais data leakage.
#
#  Saída: 2_Dados/Sinteticos/prompt_7_rewriting_CORRIGIDO.csv
# =====================================================================================
import pandas as pd
import random
import re
import os
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

def _raiz_projeto():
    p = os.path.dirname(os.path.abspath(__file__))
    while p != os.path.dirname(p) and not os.path.isdir(os.path.join(p, '2_Dados')):
        p = os.path.dirname(p)
    return p
RAIZ = _raiz_projeto()
DIR_ORIGINAIS  = os.path.join(RAIZ, '2_Dados', 'Originais')
DIR_SINTETICOS = os.path.join(RAIZ, '2_Dados', 'Sinteticos')


def apply_slang_and_rewrite(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    replacements = {
        r'\bcomunistas?\b': random.choice(['esquerdopatas', 'petistas', 'pão com mortadela', 'turma do L']),
        r'\besquerda\b': random.choice(['esquerdalha', 'canhotos', 'turma da mortadela']),
        r'\bmanifestação\b': random.choice(['bagunça', 'baderneiros', 'festa', 'passeata']),
        r'\bato\b': random.choice(['bagunça', 'baderneiros']),
        r'\babsurdo\b': random.choice(['fala sério', 'mt absurdo', 'inaceitável véi', 'slá mano mt absurdo']),
        r'\bbandidos?\b': random.choice(['vagabundos', 'corruptos', 'meliantes', 'ladrões']),
        r'\bpolíticos?\b': random.choice(['bandidos de terno', 'engravatados', 'corruptos']),
        r'\bdinheiro\b': random.choice(['grana', 'grana nossa', 'impostos', 'nosso suor']),
        r'\bconcordo\b': random.choice(['tbm acho', 'isso aí', 'disse tudo', 'falou pouco mas falou bonito']),
        r'\bnão sei\b': random.choice(['slá', 'tô perdido', 'sei lá mano', 'difícil dizer']),
        r'\blegal\b': random.choice(['daora', 'top', 'mto bom']),
        r'\bverdade\b': random.choice(['vdd', 'papo reto', 'isso aí', 'fato'])
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    prefixes = ["mano, ", "véi, ", "mds... ", "cara, na boa, ", "olha, ", ""]
    suffixes = [" kkkkk", " rsrs", " 😂", " 🤡", "...", " !!", ""]
    text = random.choice(prefixes) + text + random.choice(suffixes)
    if random.random() < 0.2:
        text = text.upper()
    return text


base_file = os.path.join(DIR_ORIGINAIS, "Atos_esquerda_pec_classificado.csv")
df_real = pd.read_csv(base_file).dropna(subset=['teor_classificado']).reset_index(drop=True)

# =====================================================================================
#  >>> CORREÇÃO: separar TREINO/TESTE ANTES de gerar. Só o treino vira semente. <<<
# =====================================================================================
X = df_real['comentario'].astype(str)
y = df_real['teor_classificado']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
df_train = pd.DataFrame({'comentario': X_train.values, 'teor_classificado': y_train.values})

# Sementes extraídas APENAS da partição de treino (80%)
df_contra = df_train[df_train['teor_classificado'] == '1 - Contra a Anistia']['comentario'].tolist()
df_indet  = df_train[df_train['teor_classificado'] == '3 - Indeterminado']['comentario'].tolist()
df_neutro = df_train[df_train['teor_classificado'] == '4 - Neutro']['comentario'].tolist()

data_sintetica = []
for cls, pool, n in [('1 - Contra a Anistia', df_contra, 800),
                     ('3 - Indeterminado', df_indet, 400),
                     ('4 - Neutro', df_neutro, 400)]:
    print(f"Gerando {n} comentários de {cls} (sementes só do treino)...")
    for _ in range(n):
        semente = random.choice(pool)  # garantidamente do TREINO -> sem vazamento
        data_sintetica.append({'comentario': apply_slang_and_rewrite(semente),
                               'teor_classificado': cls})

df_sintetico = pd.DataFrame(data_sintetica).sample(frac=1, random_state=SEED).reset_index(drop=True)
output_path = os.path.join(DIR_SINTETICOS, "prompt_7_rewriting_CORRIGIDO.csv")
df_sintetico.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n[CORRIGIDO] Gerado {len(df_sintetico)} comentários (SEM vazamento) em: {output_path}")
print(df_sintetico['teor_classificado'].value_counts())
