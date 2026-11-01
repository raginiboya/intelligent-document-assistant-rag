import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Clean PDF text while preserving useful paragraphs."""
    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")

    # Join words separated by a line-break hyphen.
    text = re.sub(r"-\s*\n\s*(?=\w)", "", text)

    # Remove unnecessary spaces and blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(file_path: str | Path) -> list[Document]:
    """Extract and clean text from every readable PDF page."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(file_path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_text(raw_text)

        if cleaned_text:
            documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": file_path.name,
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise ValueError("No searchable text was found in the PDF.")

    return documents


def save_clean_text(
    documents: list[Document],
    output_path: str | Path,
) -> None:
    """Save cleaned text for inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for document in documents:
            page_number = document.metadata["page"]
            file.write(f"\n--- PAGE {page_number} ---\n")
            file.write(document.page_content)
            file.write("\n")


if __name__ == "__main__":
    input_pdf = Path("data/raw/apple_2024_annual_report.pdf")
    output_text = Path("data/processed/apple_2024_cleaned.txt")

    extracted_documents = extract_pdf(input_pdf)
    save_clean_text(extracted_documents, output_text)

    total_characters = sum(
        len(document.page_content)
        for document in extracted_documents
    )

    print("Document processing completed successfully!")
    print(f"Readable pages: {len(extracted_documents)}")
    print(f"Total cleaned characters: {total_characters}")
    print(f"Cleaned text saved to: {output_text}")