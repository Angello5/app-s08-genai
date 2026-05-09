import streamlit as st
import pymongo
from google import genai
from google.genai import types
from supabase import create_client, Client


# =======================
# CONFIGURACIÓN
# =======================

st.set_page_config(
    page_title="Chat PDF con MongoDB + Gemini + Supabase",
    page_icon="🤖"
)

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]
SUPABASE_URL = st.secrets["app"]["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["app"]["SUPABASE_KEY"]

if not GOOGLE_API_KEY or not MONGODB_URI or not SUPABASE_URL or not SUPABASE_KEY:
    st.error(
        "❌ Faltan variables en Secrets: "
        "GOOGLE_API_KEY, MONGODB_URI, SUPABASE_URL o SUPABASE_KEY"
    )
    st.stop()


# =======================
# CLIENTES CACHEADOS
# =======================

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)


@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["pdf_embeddings_db"]
    return db["pdf_vectors"]


@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


client_genai = get_genai_client()
collection = get_mongo_collection()
supabase = get_supabase_client()


# =======================
# FUNCIONES RAG
# =======================

def crear_embedding(texto: str):
    """
    Genera el embedding de la pregunta usando Gemini.
    Debe coincidir con el modelo usado al indexar los documentos.
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
    Requiere un índice llamado 'vector_index' sobre el campo 'embedding'.
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
    """
    Genera una respuesta usando Gemini con los fragmentos recuperados.
    """
    contexto = "\n\n".join([c["texto"] for c in contextos])

    prompt = f"""
Eres un asistente experto. Usa EXCLUSIVAMENTE el siguiente contexto para responder
la pregunta del usuario. Si la respuesta no está en el contexto, indícalo claramente.

Contexto:
{contexto}

Pregunta:
{pregunta}

Responde de forma concisa y clara en español.
"""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


# =======================
# FUNCIONES SUPABASE
# =======================

def guardar_feedback_supabase(
    pregunta: str,
    respuesta: str,
    feedback: str | None = None
):
    """
    Guarda la pregunta, respuesta y feedback del usuario en Supabase.
    """
    try:
        data = {
            "pregunta": pregunta,
            "respuesta": respuesta,
            "feedback": feedback,
        }

        supabase.table("rag_feedback").insert(data).execute()

    except Exception as e:
        st.warning(f"No se pudo guardar el feedback en Supabase: {e}")


# =======================
# INTERFAZ STREAMLIT
# =======================

st.title("🤖 Chatbot RAG sobre el Covid-19")

st.caption(
    "Aplicación desplegada en Streamlit Cloud usando MongoDB Atlas, "
    "Google Gemini API y Supabase."
)

if "historial" not in st.session_state:
    st.session_state.historial = []

if "ultima_interaccion" not in st.session_state:
    st.session_state.ultima_interaccion = None


# Mostrar historial
for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])


pregunta = st.chat_input("Escribe tu pregunta sobre el PDF...")


if pregunta:
    st.chat_message("user").write(pregunta)
    st.session_state.historial.append(
        {
            "rol": "usuario",
            "texto": pregunta,
        }
    )

    similares = []

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

        if similares:
            with st.expander("Fragmentos recuperados"):
                for i, c in enumerate(similares, 1):
                    st.markdown(
                        f"**Fragmento {i}** — score: `{c['score']:.4f}`"
                    )
                    st.write(
                        c["texto"][:500]
                        + ("…" if len(c["texto"]) > 500 else "")
                    )
                    st.divider()

    st.session_state.historial.append(
        {
            "rol": "bot",
            "texto": respuesta,
        }
    )

    st.session_state.ultima_interaccion = {
        "pregunta": pregunta,
        "respuesta": respuesta,
    }


# =======================
# FEEDBACK DEL USUARIO
# =======================

if st.session_state.ultima_interaccion:
    st.markdown("### ¿La última respuesta fue útil?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Sí"):
            guardar_feedback_supabase(
                pregunta=st.session_state.ultima_interaccion["pregunta"],
                respuesta=st.session_state.ultima_interaccion["respuesta"],
                feedback="positivo",
            )
            st.success("Feedback positivo guardado en Supabase.")

    with col2:
        if st.button("👎 No"):
            guardar_feedback_supabase(
                pregunta=st.session_state.ultima_interaccion["pregunta"],
                respuesta=st.session_state.ultima_interaccion["respuesta"],
                feedback="negativo",
            )
            st.info("Feedback negativo guardado en Supabase.")


# =======================
# SIDEBAR
# =======================

with st.sidebar:
    st.header("Stack tecnológico")

    st.markdown(
        """
        **Frontend / Hosting**
        - Streamlit
        - Streamlit Cloud

        **Base de datos vectorial**
        - MongoDB Atlas
        - Atlas Vector Search

        **IA Generativa**
        - Google Gemini API
        - Gemini Embeddings
        - Gemini 2.5 Flash

        **Base de datos cloud adicional**
        - Supabase
        - PostgreSQL
        - Feedback del usuario
        """
    )

    if st.button("Limpiar historial"):
        st.session_state.historial = []
        st.session_state.ultima_interaccion = None
        st.rerun()
