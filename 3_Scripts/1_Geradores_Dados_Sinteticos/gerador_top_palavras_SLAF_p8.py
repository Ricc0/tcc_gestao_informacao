import pandas as pd
import random
import os

def generate_contra():
    # Usando EXATAMENTE as top palavras da base original:
    # esquerda, anistia, povo, brasil, bandeira, direita, pec, contra, bandidagem, blindagem
    
    estruturas = [
        "o {povo} do {brasil} é {contra} essa {pec} de {anistia} para a {direita}.",
        "a {esquerda} com a {bandeira} na mão é {contra} a {anistia} da {direita}.",
        "{manifestacao} do {povo} de {esquerda} contra essa {anistia}.",
        "os {politicos} de {direita} querem {anistia} e {blindagem}, mas o {brasil} é {contra}.",
        "essa {pec} é para ajudar a {bandidagem} da {direita}.",
        "a {esquerda} com seu {vermelho} na {bandeira} é {contra} a {anistia}.",
        "{artistas} são a {favor} da {anistia}, mas o {povo} de {esquerda} é {contra}."
    ]
    
    escolha = random.choice(estruturas)
    
    comentario = escolha.format(
        povo=random.choice(["povo", "brasileiro", "gente"]),
        brasil=random.choice(["brasil", "país"]),
        contra="contra",
        pec=random.choice(["pec", "lei"]),
        anistia="anistia",
        esquerda=random.choice(["esquerda", "vermelho", "comunistas"]),
        direita=random.choice(["direita", "conservadores", "patriotas"]),
        bandeira="bandeira",
        blindagem="blindagem",
        bandidagem=random.choice(["bandidagem", "políticos", "criminosos"]),
        manifestacao=random.choice(["manifestação", "protesto"]),
        politicos=random.choice(["políticos", "políticos corruptos"]),
        artistas=random.choice(["artistas", "celebridades"]),
        vermelho="vermelho",
        favor="favor"
    )
    
    if random.random() > 0.7:
        comentario = comentario.upper()
    
    return comentario

def generate_indeterminado():
    # Palavras genéricas mas que orbitam o tema sem tomar lado claro
    estruturas = [
        "não sei se essa {pec} da {anistia} é boa para o {brasil}.",
        "o {povo} fica no meio da briga entre {direita} e {esquerda}.",
        "vendo essa {manifestacao} sobre a {pec}, muito complicado.",
        "os {politicos} discutem a {anistia}, vamos {ver} o que o {brasil} ganha com isso.",
        "difícil dizer quem tá certo sobre a {anistia}."
    ]
    
    comentario = random.choice(estruturas).format(
        pec="pec", anistia="anistia", brasil="brasil", povo="povo", 
        direita="direita", esquerda="esquerda", manifestacao="manifestação",
        politicos="políticos", ver="ver"
    )
    return comentario

def generate_neutro():
    # Totalmente neutro, usando palavras comuns de engajamento que apareceram (ver, show, dinheiro, etc)
    estruturas = [
        "bom {video} sobre o {brasil}.",
        "gostei de {ver} a {manifestacao}, {show} de imagens.",
        "só queria {ver} como vai ficar o {dinheiro} do {pais}.",
        "acompanhando a cobertura da {pec} aqui no canal.",
        "salve para todos do {brasil}."
    ]
    
    comentario = random.choice(estruturas).format(
        video="vídeo", brasil="brasil", ver="ver", manifestacao="manifestação", 
        show="show", dinheiro="dinheiro", pais="país", pec="pec"
    )
    return comentario

data = []

# Gerar 800 Contra a Anistia
for _ in range(800):
    data.append({'comentario': generate_contra(), 'teor_classificado': '1 - Contra a Anistia'})

# Gerar 400 Indeterminado
for _ in range(400):
    data.append({'comentario': generate_indeterminado(), 'teor_classificado': '3 - Indeterminado'})

# Gerar 400 Neutro
for _ in range(400):
    data.append({'comentario': generate_neutro(), 'teor_classificado': '4 - Neutro'})

df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)

# Salvar na pasta correta
output_path = r'C:\Users\Henricco Santos\OneDrive - exedconsulting\Documentos\TCC\2_Dados\Sinteticos\prompt_9_top_palavras_2.csv'
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Gerado {len(df)} comentários sintéticos focados nas TOP palavras originais em: {output_path}")
print(df['teor_classificado'].value_counts())
