"""
Enterprise RAG Knowledge Assistant
--------------------------------------------------
Answers questions from internal documentation. Beyond a basic RAG pipeline,
this adds a reranking step (bi-encoder retrieval is fast but imprecise; a
cross-encoder reranker fixes ordering before generation) and an explicit
response-grounding check so answers stay tied to retrieved text instead of
drifting into the model's parametric knowledge.

Stack: LangChain, Hugging Face (embeddings + cross-encoder reranker),
       ChromaDB / FAISS, OpenAI API
"""

from dataclasses import dataclass
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder


@dataclass
class RAGConfig:
    docs_dir: str = "./knowledge_base"
    persist_dir: str = "./chroma_store"
    chunk_size: int = 700
    chunk_overlap: int = 100
    retrieve_k: int = 10       # cast a wide net...
    rerank_top_n: int = 4      # ...then narrow with a cross-encoder
    model_name: str = "gpt-4o-mini"


class EnterpriseKnowledgeAssistant:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.llm = ChatOpenAI(model=config.model_name, temperature=0)
        self.vector_store = None

    # ---------- Ingestion ----------
    def build_index(self) -> None:
        loader = DirectoryLoader(self.config.docs_dir, loader_cls=TextLoader)
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)

        self.vector_store = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=self.config.persist_dir
        )
        print(f"Indexed {len(chunks)} chunks from {len(raw_docs)} documents.")

    def load_index(self) -> None:
        self.vector_store = Chroma(
            persist_directory=self.config.persist_dir, embedding_function=self.embeddings
        )

    # ---------- Retrieve -> Rerank -> Generate -> Ground-check ----------
    def _retrieve(self, question: str) -> List:
        return self.vector_store.similarity_search(question, k=self.config.retrieve_k)

    def _rerank(self, question: str, docs: List) -> List:
        pairs = [(question, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[: self.config.rerank_top_n]]

    def _generate(self, question: str, docs: List) -> str:
        context = "\n\n".join(f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)
        prompt = f"""Answer the question using ONLY the context below. Cite the
        source file for each claim. If the context is insufficient, say so.

        Context:
        {context}

        Question: {question}"""
        return self.llm.invoke(prompt).content

    def _is_grounded(self, answer: str, docs: List) -> bool:
        """Cheap grounding check: does the answer reference an actual source tag?"""
        return any(d.metadata.get("source", "") in answer for d in docs)

    def ask(self, question: str) -> str:
        if self.vector_store is None:
            raise RuntimeError("Call build_index() or load_index() first.")

        candidates = self._retrieve(question)
        top_docs = self._rerank(question, candidates)
        answer = self._generate(question, top_docs)

        if not self._is_grounded(answer, top_docs):
            return "I couldn't find a well-grounded answer in the knowledge base for that question."
        return answer


if __name__ == "__main__":
    assistant = EnterpriseKnowledgeAssistant(RAGConfig())
    assistant.build_index()
    print(assistant.ask("What is our on-call escalation policy?"))
