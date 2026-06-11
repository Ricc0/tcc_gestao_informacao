import pandas as pd
import random
import os
# === Raiz do projeto (resolve a pasta TCC a partir de qualquer subpasta) ===
def _raiz_projeto():
    p = os.path.dirname(os.path.abspath(__file__))
    while p != os.path.dirname(p) and not os.path.isdir(os.path.join(p, '2_Dados')):
        p = os.path.dirname(p)
    return p
RAIZ = _raiz_projeto()
DIR_ORIGINAIS  = os.path.join(RAIZ, '2_Dados', 'Originais')
DIR_SINTETICOS = os.path.join(RAIZ, '2_Dados', 'Sinteticos')
DIR_GRAFICOS   = os.path.join(RAIZ, '4_Resultados_e_Graficos')


# Classes
# 1 - Contra a Anistia (600)
# 3 - Indeterminado (300)
# 4 - Neutro (300)

def generate_contra():
    sujeitos = ["golpistas", "bandidos", "terroristas", "criminosos", "esses caras", "vândalos"]
    verbos = ["tem que ir pra", "merecem", "lugar deles é na", "devem mofar na", "vão para a"]
    locais = ["cadeia", "prisão", "papuda", "xadrez", "trás das grades"]
    adjetivos = ["sem anistia", "crime é crime", "bando de baderneiro", "inaceitável", "tem que pagar", "golpe não"]
    
    comentario = f"{random.choice(sujeitos)} {random.choice(verbos)} {random.choice(locais)}. {random.choice(adjetivos)}!"
    
    # Adicionar ruído de internet (letras repetidas, sem pontuação, etc)
    if random.random() > 0.5:
        comentario = comentario.replace("!", "!!!" * random.randint(1, 3))
    if random.random() > 0.8:
        comentario = comentario.upper()
    return comentario

def generate_indeterminado():
    inicio = ["não sei não,", "sinceramente,", "olha,", "complicado,", "difícil dizer,"]
    meio = ["tem erro dos dois lados", "a situação tá feia", "o brasil ta polarizado demais", "isso não vai dar em nada", "muita confusão", "é tudo a mesma coisa"]
    fim = ["mas vamos ver", "fazer o que né", "complicado", "só jesus na causa", "...", "enfim"]
    
    comentario = f"{random.choice(inicio)} {random.choice(meio)}... {random.choice(fim)}"
    return comentario

def generate_neutro():
    inicio = ["assistindo aqui de", "bom video de", "vídeo de", "acompanhando do", "salve de", "bom dia"]
    locais = ["são paulo", "sp", "rj", "rio", "minas", "bh", "curitiba", "portugal", "nordeste"]
    extras = ["audio ta bom", "legal o canal", "inscrito novo", "bom trabalho da reportagem", "primeiro a comentar"]
    
    if random.random() > 0.5:
        comentario = f"{random.choice(inicio)} {random.choice(locais)}"
    else:
        comentario = f"{random.choice(extras)} {random.choice(inicio)} {random.choice(locais)}"
    return comentario

data = []

# Generate Contra
for _ in range(800):
    data.append({'comentario': generate_contra(), 'teor_classificado': '1 - Contra a Anistia'})

# Generate Indeterminado
for _ in range(400):
    data.append({'comentario': generate_indeterminado(), 'teor_classificado': '3 - Indeterminado'})

# Generate Neutro
for _ in range(400):
    data.append({'comentario': generate_neutro(), 'teor_classificado': '4 - Neutro'})

df = pd.DataFrame(data)

# Embaralhar as linhas
df = df.sample(frac=1).reset_index(drop=True)

# Salvar
output_path = os.path.join(DIR_SINTETICOS, 'prompt_6_sinonimos.csv')
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Gerado {len(df)} comentários sintéticos em {output_path}")
print(df['teor_classificado'].value_counts())
