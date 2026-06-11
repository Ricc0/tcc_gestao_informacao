# -*- coding: utf-8 -*-
# =====================================================================================
#  COLETA DE COMENTÁRIOS DO YOUTUBE  (1ª etapa do pipeline)
# -------------------------------------------------------------------------------------
#  Baixa os comentários de um vídeo do YouTube e salva em planilha (.xlsx) para a
#  rotulação manual posterior (coluna "teor": a favor / contra / neutro / indeterminado).
#
#  Vídeo utilizado neste TCC: https://www.youtube.com/watch?v=Na0GGRfTVWI
#
#  Dependências:
#    pip install youtube-comment-downloader pandas openpyxl
# =====================================================================================
from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd

# URL do vídeo
url = "https://www.youtube.com/watch?v=Na0GGRfTVWI"

# Criar objeto
downloader = YoutubeCommentDownloader()
comments = downloader.get_comments_from_url(url, sort_by=1)  # 1 = mais recentes, 0 = mais relevantes

# Guardar os comentários
data = []
for comment in comments:
    texto = comment.get("text", "")

    # Agora considera resposta se a flag da API for True OU se tiver @ no texto
    if comment.get("is_reply", False) or "@" in texto:
        is_reply = "Sim"
    else:
        is_reply = "Não"

    data.append({
        "comentario": texto,
        "respondendo outro comentario": is_reply,
        "teor": ""  # a favor / contra / neutro / indeterminado
    })

# Criar DataFrame
df = pd.DataFrame(data)

# Salvar em Excel
df.to_excel("comentarios_youtube.xlsx", index=False)

print(f"Total de comentários coletados: {len(df)}")
print("Arquivo salvo: comentarios_youtube.xlsx")
