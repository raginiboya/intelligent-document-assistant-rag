import csv
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_processor import extract_pdf


CHUNK_SETTINGS = [
    (500, 100),
    (1000, 200),
    (1500, 300),
]


def create_chunks(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping text chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = chunk_number
        chunk.metadata["chunk_size"] = chunk_size
        chunk.metadata["chunk_overlap"] = chunk_overlap

    return chunks


def compare_chunk_sizes(
    documents: list[Document],
) -> list[dict]:
    """Compare the number and length of chunks from multiple settings."""
    comparison_results = []

    for chunk_size, chunk_overlap in CHUNK_SETTINGS:
        chunks = create_chunks(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        lengths = [len(chunk.page_content) for chunk in chunks]

        comparison_results.append(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "number_of_chunks": len(chunks),
                "average_chunk_length": round(
                    sum(lengths) / len(lengths),
                    2,
                ),
                "minimum_chunk_length": min(lengths),
                "maximum_chunk_length": max(lengths),
            }
        )

    return comparison_results


def save_comparison(
    results: list[dict],
    output_path: str | Path,
) -> None:
    """Save the chunk-size comparison as a CSV report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)


def save_chunks(
    chunks: list[Document],
    output_path: str | Path,
) -> None:
    """Save selected chunks and their metadata as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_data = [
        {
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            chunk_data,
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    pdf_path = Path("data/raw/apple_2024_annual_report.pdf")

    page_documents = extract_pdf(pdf_path)

    comparison = compare_chunk_sizes(page_documents)
    save_comparison(
        comparison,
        "reports/chunk_size_comparison.csv",
    )

    print("\nChunk-size comparison")
    print("-" * 70)

    for result in comparison:
        print(
            f"Size: {result['chunk_size']}, "
            f"Overlap: {result['chunk_overlap']}, "
            f"Chunks: {result['number_of_chunks']}, "
            f"Average length: "
            f"{result['average_chunk_length']}"
        )

    selected_chunks = create_chunks(
        documents=page_documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    save_chunks(
        selected_chunks,
        "data/processed/chunks_1000_200.json",
    )

    print("-" * 70)
    print(f"Selected chunks saved: {len(selected_chunks)}")
    print("Selected configuration: size=1000, overlap=200")
    print(
        "Comparison saved to: "
        "reports/chunk_size_comparison.csv"
    )