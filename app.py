import streamlit as st
import pymongo
from google import genai
from google.genai import types
import numpy as np

# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan las variables de entorno GOOGLE_API_KEY o MONGODB_URI")
    st.stop()

# =======================
# CLIENTES (cacheados)
# =======================

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["pdf_embeddings_db"]
    return db["pdf_vectors"]

client_genai = get_genai_client()
collection = get_mongo_collection()

# =======================
# FUNCIONES
# =======================

def crear_embedding(texto: str):
    """
    Genera embedding de la query con el mismo modelo y dimensión que se usó
    al indexar (gemini-embedding-001, 768 dims, normalizado L2).

    IMPORTANTE: para queries de búsqueda usar task_type='RETRIEVAL_QUERY'
    (al indexar se usó 'RETRIEVAL_DOCUMENT').
    """
    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return response.embeddings[0].values

def buscar_similares(embedding, k=5):
    """
    Busca los documentos más similares en MongoDB Atlas Vector Search.
    Requiere el índice 'vector_index' creado sobre el campo 'embedding'.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))

def generar_respuesta(pregunta: str, contextos: list[dict]) -> str:
    """Usa Gemini para responder con contexto recuperado (RAG)."""
    contexto = "\n\n".join([c["texto"] for c in contextos])
    prompt = f"""Eres un asistente experto. Usa EXCLUSIVAMENTE el siguiente contexto para responder la pregunta del usuario. Si la respuesta no está en el contexto, indícalo claramente.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde de forma concisa y clara en español."""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

# =======================
# INTERFAZ STREAMLIT
# =======================

st.set_page_config(page_title="Chat PDF con MongoDB + Gemini", page_icon="💬")
st.title("💬 Chatbot de tu PDF (MongoDB + Gemini)")

if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostrar historial PRIMERO (antes de procesar la nueva pregunta)
for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])

pregunta = st.chat_input("Escribe tu pregunta sobre el PDF...")

if pregunta:
    # Mostrar inmediatamente la pregunta del usuario
    st.chat_message("user").write(pregunta)
    st.session_state.historial.append({"rol": "usuario", "texto": pregunta})

    with st.chat_message("assistant"):
        with st.spinner("Buscando respuesta..."):
            try:
                emb = crear_embedding(pregunta)
                similares = buscar_similares(emb, k=5)

                if not similares:
                    respuesta = "No encontré información relevante en el documento."
                else:
                    respuesta = generar_respuesta(pregunta, similares)
            except Exception as e:
                respuesta = f"⚠️ Ocurrió un error: {e}"

        st.write(respuesta)

        # Opcional: mostrar fuentes recuperadas
        if 'similares' in locals() and similares:
            with st.expander("🔍 Fragmentos recuperados"):
                for i, c in enumerate(similares, 1):
                    st.markdown(f"**Fragmento {i}** — score: `{c['score']:.4f}`")
                    st.write(c["texto"][:500] + ("…" if len(c["texto"]) > 500 else ""))
                    st.divider()

    st.session_state.historial.append({"rol": "bot", "texto": respuesta})
