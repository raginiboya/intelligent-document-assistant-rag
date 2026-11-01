import os
from pathlib import Path
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph


# Project configuration
VECTOR_STORE_PATH = Path("vector_store/faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RESULTS = 5


class RAGState(TypedDict, total=False):
    """Information passed between the LangGraph steps."""

    question: str
    documents: List[Document]
    context: str
    answer: str
    sources: List[str]


class RAGPipeline:
    """Retrieve relevant PDF sections and generate a grounded answer."""

    def __init__(self, top_k: int = TOP_K_RESULTS):
        load_dotenv(override=True)

        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("GROQ_MODEL")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to the root .env file."
            )

        if not model_name:
            raise ValueError(
                "GROQ_MODEL is missing. Add it to the root .env file."
            )

        if not VECTOR_STORE_PATH.exists():
            raise FileNotFoundError(
                "FAISS index was not found. Run: python -m src.vector_store"
            )

        print("Loading embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            show_progress=False,
        )

        print("Loading FAISS vector store...")

        self.vector_store = FAISS.load_local(
            str(VECTOR_STORE_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        self.llm = ChatGroq(
            model=model_name,
            temperature=0,
            max_retries=2,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an intelligent document assistant.

Answer the user's question using ONLY the retrieved document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts, figures, or page numbers.
3. Cite supporting information using [Page X].
4. If the answer is not present in the context, say:
   "I could not find this information in the uploaded document."
5. Keep the answer clear, accurate, and concise.
""",
                ),
                (
                    "human",
                    """
Question:
{question}

Retrieved document context:
{context}

Provide a grounded answer with page citations.
""",
                ),
            ]
        )

        self.graph = self._build_graph()

    def _retrieve(self, state: RAGState) -> RAGState:
        """Retrieve the most relevant document chunks."""

        question = state["question"].strip()

        if not question:
            return {
                "documents": [],
                "context": "",
                "sources": [],
            }

        documents = self.retriever.invoke(question)

        context_sections = []
        sources = []

        for document in documents:
            page = document.metadata.get("page", "Unknown")
            source = document.metadata.get("source", "Document")
            source_name = Path(str(source)).name

            context_sections.append(
                f"[Source: {source_name} | Page: {page}]\n"
                f"{document.page_content}"
            )

            source_label = f"{source_name} - Page {page}"

            if source_label not in sources:
                sources.append(source_label)

        context = "\n\n---\n\n".join(context_sections)

        return {
            "documents": documents,
            "context": context,
            "sources": sources,
        }

    def _generate(self, state: RAGState) -> RAGState:
        """Generate an answer using the retrieved context."""

        if not state.get("documents"):
            return {
                "answer": (
                    "I could not find this information in the uploaded document."
                )
            }

        prompt_value = self.prompt.invoke(
            {
                "question": state["question"],
                "context": state["context"],
            }
        )

        response = self.llm.invoke(prompt_value)

        return {"answer": response.content.strip()}

    def _build_graph(self):
        """Create the retrieve-to-generate LangGraph workflow."""

        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def ask(self, question: str) -> dict:
        """Run a question through the complete RAG workflow."""

        result = self.graph.invoke({"question": question})

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "documents": result.get("documents", []),
        }


def main():
    """Test the RAG pipeline from the terminal."""

    print("\nInitializing Intelligent Document Assistant...\n")
    pipeline = RAGPipeline()

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a valid question.")
        return

    result = pipeline.ask(question)

    print("\nAnswer")
    print("-" * 70)
    print(result["answer"])

    print("\nRetrieved sources")
    print("-" * 70)

    for source in result["sources"]:
        print(f"- {source}")


if __name__ == "__main__":
    main()