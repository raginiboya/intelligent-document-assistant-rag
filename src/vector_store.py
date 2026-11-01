import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(chunk_file: str | Path) -> list[Document]:
    """Load processed document chunks from JSON."""
    chunk_file = Path(chunk_file)

    if not chunk_file.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunk_file}"
        )

    with chunk_file.open("r", encoding="utf-8") as file:
        chunk_data = json.load(file)

    documents = [
        Document(
            page_content=item["text"],
            metadata=item["metadata"],
        )
        for item in chunk_data
        if item.get("text", "").strip()
    ]

    if not documents:
        raise ValueError(
            "No valid document chunks were found."
        )

    return documents


def create_embedding_model() -> HuggingFaceEmbeddings:
    """Load the open-source embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
        show_progress=True,
    )


def build_vector_store(
    documents: list[Document],
    output_directory: str | Path,
) -> FAISS:
    """Generate embeddings, build FAISS, and save it."""
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_model = create_embedding_model()

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model,
    )

    vector_store.save_local(
        str(output_directory)
    )

    return vector_store


def save_vector_summary(
    vector_store: FAISS,
    number_of_chunks: int,
    output_path: str | Path,
) -> None:
    """Save index information for evaluation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": vector_store.index.d,
        "number_of_vectors": vector_store.index.ntotal,
        "number_of_chunks": number_of_chunks,
        "vector_database": "FAISS",
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )


if __name__ == "__main__":
    chunks_path = Path(
        "data/processed/chunks_1000_200.json"
    )
    index_path = Path(
        "vector_store/faiss_index"
    )
    summary_path = Path(
        "reports/vector_store_summary.json"
    )

    chunks = load_chunks(chunks_path)

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Creating embeddings and FAISS index...")

    faiss_store = build_vector_store(
        documents=chunks,
        output_directory=index_path,
    )

    save_vector_summary(
        vector_store=faiss_store,
        number_of_chunks=len(chunks),
        output_path=summary_path,
    )

    print("\nVector store created successfully!")
    print(
        f"Vectors stored: "
        f"{faiss_store.index.ntotal}"
    )
    print(
        f"Embedding dimension: "
        f"{faiss_store.index.d}"
    )
    print(
        f"FAISS index saved to: "
        f"{index_path}"
    )
    print(
        f"Summary saved to: "
        f"{summary_path}"
    )