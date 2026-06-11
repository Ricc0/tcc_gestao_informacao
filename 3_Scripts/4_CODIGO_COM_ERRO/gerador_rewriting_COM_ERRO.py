# -*- coding: utf-8 -*-
# =====================================================================================
#  GERADOR REWRITING  ***VERSÃO COM ERRO (VAZAMENTO DE TESTE / DATA LEAKAGE)***
# -------------------------------------------------------------------------------------
#  Esta é a versão ORIGINAL do gerador da técnica de Rewriting (prompt_7).
#  Ela CONTÉM O BUG que inflou artificialmente o resultado do TCC (Macro F1 0,82).
#
#  >>> RESUMO DO ERRO <<<
#  A técnica de Rewriting parte de um comentário humano REAL ("semente") e o reescreve
#  trocando ~12 termos por gírias. Como ~90% do texto original é preservado, a saída é
#  uma QUASE-DUPLICATA da semente. O problema: aqui as sementes são sorteadas do CORPUS
#  COMPLETO (3.422 comentários), incluindo os 20% que serão usados como TESTE.
#  Resultado: versões quase idênticas dos comentários de teste entram no TREINO, com o
#  rótulo correto  ->  o modelo "decora" o teste  ->  VAZAMENTO DE TESTE (data leakage).
#  Medição: ~21% das sementes vinham do conjunto de teste.
#
#  >>> VEJA A CORREÇÃO EM:  ../5_CODIGO_REFEITO/gerador_rewriting_CORRIGIDO.py
# =====================================================================================
import pandas as pd
import random
import re
import os

# Raiz do projeto (resolve a pasta TCC a partir de qualquer subpasta)
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
    # Dicionário de 12 substituições por Expressões Regulares (termo formal -> gíria)
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
    # Injeção de marcadores de informalidade
    prefixes = ["mano, ", "véi, ", "mds... ", "cara, na boa, ", "olha, ", ""]
    suffixes = [" kkkkk", " rsrs", " 😂", " 🤡", "...", " !!", ""]
    text = random.choice(prefixes) + text + random.choice(suffixes)
    if random.random() < 0.2:
        text = text.upper()
    return text


base_file = os.path.join(DIR_ORIGINAIS, "Atos_esquerda_pec_classificado.csv")
df_real = pd.read_csv(base_file)
df_real = df_real.dropna(subset=['teor_classificado'])

# =====================================================================================
#  >>> ERRO Nº 1 — AQUI ESTÁ A RAIZ DO VAZAMENTO <<<
#  As listas de sementes abaixo são extraídas do df_real INTEIRO (corpus completo),
#  SEM separar antes os 20% de teste. Logo, comentários que depois serão usados como
#  teste estão disponíveis como semente e podem ser reescritos para dentro do treino.
#  CORREÇÃO: fazer o train_test_split ANTES e usar apenas a partição de TREINO aqui.
# =====================================================================================
df_contra = df_real[df_real['teor_classificado'] == '1 - Contra a Anistia']['comentario'].tolist()
df_indet  = df_real[df_real['teor_classificado'] == '3 - Indeterminado']['comentario'].tolist()
df_neutro = df_real[df_real['teor_classificado'] == '4 - Neutro']['comentario'].tolist()

data_sintetica = []

# >>> ERRO Nº 1 (continuação): random.choice(df_contra) pode sortear um comentário de TESTE
print("Gerando comentários Contra a Anistia...")
for _ in range(800):
    semente = random.choice(df_contra)          # <-- pode ser um comentário do conjunto de TESTE!
    novo_comentario = apply_slang_and_rewrite(semente)
    data_sintetica.append({'comentario': novo_comentario, 'teor_classificado': '1 - Contra a Anistia'})

print("Gerando comentários Indeterminado...")
for _ in range(400):
    semente = random.choice(df_indet)           # <-- idem
    novo_comentario = apply_slang_and_rewrite(semente)
    data_sintetica.append({'comentario': novo_comentario, 'teor_classificado': '3 - Indeterminado'})

print("Gerando comentários Neutro...")
for _ in range(400):
    semente = random.choice(df_neutro)          # <-- idem
    novo_comentario = apply_slang_and_rewrite(semente)
    data_sintetica.append({'comentario': novo_comentario, 'teor_classificado': '4 - Neutro'})

df_sintetico = pd.DataFrame(data_sintetica).sample(frac=1).reset_index(drop=True)

# Observação: este arquivo (prompt_7_rewriting.csv) é o CONTAMINADO usado no TCC original.
output_path = os.path.join(DIR_SINTETICOS, "prompt_7_rewriting.csv")
df_sintetico.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n[COM ERRO] Gerado {len(df_sintetico)} comentários (com vazamento) em: {output_path}")
print(df_sintetico['teor_classificado'].value_counts())
