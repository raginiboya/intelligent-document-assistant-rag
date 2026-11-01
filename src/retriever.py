from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.vector_store import create_embedding_model


def load_vector_store(
    index_directory: str | Path,
) -> FAISS:
    """Load the locally created FAISS vector store."""
    index_directory = Path(index_directory)

    index_file = index_directory / "index.faiss"
    metadata_file = index_directory / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"FAISS index files not found in: "
            f"{index_directory}"
        )

    embedding_model = create_embedding_model()

    vector_store = FAISS.load_local(
        folder_path=str(index_directory),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    return vector_store


def retrieve_relevant_chunks(
    vector_store: FAISS,
    question: str,
    number_of_results: int = 5,
) -> list[Document]:
    """Retrieve and rank the most relevant document chunks."""
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    search_results = vector_store.similarity_search_with_score(
        query=question,
        k=number_of_results,
    )

    ranked_documents = []

    for rank, (document, distance) in enumerate(
        search_results,
        start=1,
    ):
        document.metadata["rank"] = rank
        document.metadata["distance"] = round(
            float(distance),
            4,
        )
        ranked_documents.append(document)

    return ranked_documents


def display_results(
    question: str,
    documents: list[Document],
) -> None:
    """Display retrieved chunks for testing."""
    print(f"\nQuestion: {question}")
    print("=" * 80)

    for document in documents:
        metadata = document.metadata
        preview = document.page_content[:500].replace(
            "\n",
            " ",
        )

        print(
            f"\nRank: {metadata['rank']} | "
            f"Page: {metadata.get('page', 'Unknown')} | "
            f"Distance: {metadata['distance']}"
        )
        print("-" * 80)
        print(preview)
        print("...")


if __name__ == "__main__":
    faiss_path = Path("vector_store/faiss_index")

    test_question = (
        "What were Apple's total net sales in 2024?"
    )

    store = load_vector_store(faiss_path)

    retrieved_documents = retrieve_relevant_chunks(
        vector_store=store,
        question=test_question,
        number_of_results=5,
    )

    display_results(
        question=test_question,
        documents=retrieved_documents,
    )