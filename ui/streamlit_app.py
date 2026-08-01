"""Streamlit chat UI for the Enterprise RAG Knowledge Assistant.

Talks to the FastAPI backend over HTTP (so the UI and API can scale/deploy
independently, or run in the same container in the default Docker setup).
Run with: `streamlit run ui/streamlit_app.py`
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN", "")

st.set_page_config(page_title="Enterprise RAG Knowledge Assistant", page_icon="📚", layout="wide")


def _headers() -> dict:
    if API_AUTH_TOKEN:
        return {"Authorization": f"Bearer {API_AUTH_TOKEN}"}
    return {}


def _api_get(path: str, **kwargs):
    return requests.get(f"{API_BASE_URL}{path}", headers=_headers(), timeout=30, **kwargs)


def _api_post(path: str, **kwargs):
    return requests.post(f"{API_BASE_URL}{path}", headers=_headers(), timeout=120, **kwargs)


st.title("📚 Enterprise RAG Knowledge Assistant")

with st.sidebar:
    st.header("Status")
    try:
        health = _api_get("/health").json()
        status_icon = "🟢" if health["status"] == "ok" else "🔴"
        st.write(f"{status_icon} API status: **{health['status']}**")
        st.write(f"Index loaded: **{health['index_loaded']}**")
        st.write(f"LLM provider: **{health['llm_provider']}**")
    except requests.RequestException:
        st.error(f"Could not reach API at {API_BASE_URL}")

    st.divider()
    st.header("Knowledge base")
    try:
        docs = _api_get("/documents").json().get("documents", [])
        st.write(f"{len(docs)} document(s) indexed")
        for d in docs:
            st.caption(f"• {d}")
    except requests.RequestException:
        pass

    if st.button("🔄 Reindex knowledge base", use_container_width=True):
        with st.spinner("Rebuilding index..."):
            try:
                resp = _api_post("/ingest", params={"force": "true"})
                if resp.ok:
                    st.success(f"Indexed {resp.json()['chunks_indexed']} chunks.")
                else:
                    st.error(f"Ingest failed: {resp.text}")
            except requests.RequestException as exc:
                st.error(f"Ingest failed: {exc}")

    st.divider()
    st.header("Upload a document")
    uploaded = st.file_uploader("Add a .txt or .md file to the knowledge base", type=["txt", "md"])
    if uploaded is not None:
        st.info(
            "File upload writes are not wired to a dedicated endpoint in this "
            "build — copy the file into the `knowledge_base/` directory and "
            "click Reindex above, or mount it into the container's "
            "knowledge_base volume."
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

question = st.chat_input("Ask a question about your internal documentation...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, and generating..."):
            try:
                resp = _api_post("/query", json={"question": question})
                if resp.ok:
                    data = resp.json()
                    st.markdown(data["answer"])
                    if data["grounded"]:
                        st.caption("✅ Grounded — Sources: " + ", ".join(data["sources"]))
                    else:
                        st.caption("⚠️ Not grounded in the knowledge base")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data.get("sources", []),
                        }
                    )
                else:
                    st.error(f"Query failed ({resp.status_code}): {resp.text}")
            except requests.RequestException as exc:
                st.error(f"Could not reach API: {exc}")
