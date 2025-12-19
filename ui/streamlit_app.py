import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Legal Assistant POC", page_icon="⚖️", layout="wide")

st.title("⚖️ Legal Assistant Chatbot — POC (RAG)")
st.caption("POC technique : réponses ancrées dans un corpus + citations. Pas un avis juridique.")

with st.sidebar:
    st.header("Paramètres")
    api = st.text_input("Backend API", API_URL)
    show_chunks = st.checkbox("Afficher les sources retrouvées", value=True)
    st.markdown("---")
    st.markdown("**Astuce** : pose une question et vérifie que la réponse cite les sources [C1], [C2]…")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("chunks") and show_chunks:
            with st.expander("Sources retrouvées"):
                for c in m["chunks"]:
                    st.markdown(f"- **{c['title']}** (doc `{c['doc_id']}`, score={c['score']:.3f})")

prompt = st.chat_input("Ta question (ex: Quelles sont les bases légales du RGPD ?)")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche + génération…"):
            try:
                r = requests.post(f"{api.rstrip('/')}/chat", json={"question": prompt}, timeout=120)
                r.raise_for_status()
                data = r.json()
                st.markdown(data["answer"])
                if show_chunks:
                    with st.expander("Sources retrouvées"):
                        for c in data["chunks"]:
                            st.markdown(f"- **{c['title']}** (doc `{c['doc_id']}`, score={c['score']:.3f})")
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"], "chunks": data["chunks"]}
                )
            except Exception as e:
                st.error(f"Erreur: {e}")
