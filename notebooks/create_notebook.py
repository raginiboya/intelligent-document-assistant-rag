import json
from pathlib import Path


OUTPUT_PATH = Path(__file__).with_name("rag_experiments.ipynb")


def markdown_cell(content: str) -> dict:
    """Create a Markdown notebook cell."""

    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.splitlines(keepends=True),
    }


def code_cell(content: str) -> dict:
    """Create an empty executable notebook cell."""

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        """# Intelligent Document Assistant using RAG

## Experiment and Evaluation Notebook

This notebook demonstrates the complete Retrieval-Augmented Generation workflow:

1. Inspect the PDF dataset.
2. Review processed document statistics.
3. Compare text-chunk configurations.
4. Inspect the vector-store configuration.
5. Test semantic retrieval.
6. Test the complete LangGraph RAG pipeline.
7. Review automated evaluation results.

**Development document:** Apple 2024 Annual Report
"""
    ),

    code_cell(
        """from pathlib import Path
import json
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

# Make the notebook work whether it starts from the project root
# or from inside the notebooks directory.
PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Python executable:", sys.executable)
"""
    ),

    markdown_cell(
        """## 1. Dataset Inspection

The dataset for this project is contained in the PDF itself. PyPDF extracts text page by page while preserving page metadata for citations.
"""
    ),

    code_cell(
        """pdf_path = Path("data/raw/apple_2024_annual_report.pdf")
cleaned_text_path = Path("data/processed/apple_2024_cleaned.txt")

print("PDF exists:", pdf_path.exists())

if pdf_path.exists():
    print("PDF size:", f"{pdf_path.stat().st_size / 1_000_000:.2f} MB")

print("Cleaned text exists:", cleaned_text_path.exists())

if cleaned_text_path.exists():
    cleaned_text = cleaned_text_path.read_text(encoding="utf-8")
    print("Cleaned characters:", f"{len(cleaned_text):,}")
    print("\\nText preview:\\n")
    print(cleaned_text[:1000])
"""
    ),

    markdown_cell(
        """## 2. Chunk-Size Comparison

Multiple chunk sizes were evaluated because very small chunks may lose context, while very large chunks may retrieve unnecessary information.

The selected configuration is **1,000 characters with 200 characters of overlap**.
"""
    ),

    code_cell(
        """chunk_report_path = Path("reports/chunk_size_comparison.csv")
chunk_results = pd.read_csv(chunk_report_path)

display(chunk_results)
"""
    ),

    code_cell(
        """# Detect relevant columns even if their exact labels differ.
normalized_columns = {
    column: column.lower().replace(" ", "_").replace("-", "_")
    for column in chunk_results.columns
}

size_column = next(
    (
        column
        for column, normalized in normalized_columns.items()
        if "size" in normalized
    ),
    None,
)

count_column = next(
    (
        column
        for column, normalized in normalized_columns.items()
        if normalized in {
            "chunks",
            "num_chunks",
            "number_of_chunks",
            "chunk_count",
        }
        or (
            "chunk" in normalized
            and (
                "count" in normalized
                or "number" in normalized
            )
        )
    ),
    None,
)

average_column = next(
    (
        column
        for column, normalized in normalized_columns.items()
        if "average" in normalized or normalized.startswith("avg")
    ),
    None,
)

figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

if size_column and count_column:
    axes[0].bar(
        chunk_results[size_column].astype(str),
        chunk_results[count_column],
        color="#2563EB",
    )
    axes[0].set_title("Number of Chunks")
    axes[0].set_xlabel("Chunk Size")
    axes[0].set_ylabel("Chunk Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
else:
    axes[0].text(
        0.5,
        0.5,
        "Chunk count columns not detected",
        ha="center",
        va="center",
    )
    axes[0].set_axis_off()

if size_column and average_column:
    axes[1].bar(
        chunk_results[size_column].astype(str),
        chunk_results[average_column],
        color="#059669",
    )
    axes[1].set_title("Average Chunk Length")
    axes[1].set_xlabel("Chunk Size")
    axes[1].set_ylabel("Characters")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
else:
    axes[1].text(
        0.5,
        0.5,
        "Average-length columns not detected",
        ha="center",
        va="center",
    )
    axes[1].set_axis_off()

figure.suptitle(
    "Text Chunking Experiment",
    fontsize=14,
    fontweight="bold",
)

figure.tight_layout()
plt.show()
"""
    ),

    markdown_cell(
        """## 3. Vector-Store Summary

The selected chunks are converted into 384-dimensional embeddings using `all-MiniLM-L6-v2`. FAISS stores these vectors for semantic similarity search.
"""
    ),

    code_cell(
        """vector_summary_path = Path("reports/vector_store_summary.json")

with vector_summary_path.open("r", encoding="utf-8") as file:
    vector_summary = json.load(file)

display(pd.DataFrame([vector_summary]))
"""
    ),

    markdown_cell(
        """## 4. Semantic Retrieval Test

The question and document chunks are embedded in the same vector space. FAISS returns the five chunks with the smallest semantic distance.
"""
    ),

    code_cell(
        """from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    show_progress=False,
)

vector_store = FAISS.load_local(
    "vector_store/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True,
)

test_question = "What were Apple's total net sales in 2024?"

retrieved_results = vector_store.similarity_search_with_score(
    test_question,
    k=5,
)

retrieval_rows = []

for rank, (document, distance) in enumerate(
    retrieved_results,
    start=1,
):
    retrieval_rows.append(
        {
            "rank": rank,
            "page": document.metadata.get("page"),
            "distance": round(float(distance), 4),
            "text_preview": document.page_content[:250],
        }
    )

retrieval_dataframe = pd.DataFrame(retrieval_rows)
display(retrieval_dataframe)
"""
    ),

    markdown_cell(
        """## 5. Complete LangGraph RAG Test

The LangGraph workflow contains two main nodes:

- **Retrieve:** Finds relevant evidence from FAISS.
- **Generate:** Uses the Groq language model to produce a grounded answer.

The `.env` file must contain a valid `GROQ_API_KEY` and `GROQ_MODEL`.
"""
    ),

    code_cell(
        """from src.rag_pipeline import RAGPipeline

rag_pipeline = RAGPipeline()
rag_result = rag_pipeline.ask(test_question)

print("Question:")
print(rag_result["question"])

print("\\nAnswer:")
print(rag_result["answer"])

print("\\nRetrieved sources:")
for source in rag_result["sources"]:
    print("-", source)
"""
    ),

    markdown_cell(
        """## 6. Automated Evaluation Results

The evaluation checks:

- Factual answer accuracy
- Retrieval hit rate
- Citation grounding
- Unsupported-question refusal
- Overall pass rate
- Response time
"""
    ),

    code_cell(
        """evaluation_results_path = Path(
    "reports/rag_evaluation_results.csv"
)
evaluation_summary_path = Path(
    "reports/rag_evaluation_summary.json"
)

evaluation_results = pd.read_csv(evaluation_results_path)

with evaluation_summary_path.open(
    "r",
    encoding="utf-8",
) as file:
    evaluation_summary = json.load(file)

selected_columns = [
    column
    for column in [
        "question_type",
        "question",
        "answer_keyword_score",
        "retrieval_hit",
        "citation_grounded",
        "refusal_detected",
        "response_time_seconds",
        "passed",
    ]
    if column in evaluation_results.columns
]

display(evaluation_results[selected_columns])

summary_table = pd.DataFrame(
    {
        "Metric": [
            "Answer accuracy",
            "Retrieval hit rate",
            "Citation grounding rate",
            "Refusal accuracy",
            "Overall pass rate",
            "Average response time",
        ],
        "Result": [
            f"{evaluation_summary['answer_accuracy'] * 100:.1f}%",
            f"{evaluation_summary['retrieval_hit_rate'] * 100:.1f}%",
            (
                f"{evaluation_summary['citation_grounding_rate'] * 100:.1f}%"
            ),
            f"{evaluation_summary['refusal_accuracy'] * 100:.1f}%",
            f"{evaluation_summary['overall_pass_rate'] * 100:.1f}%",
            (
                f"{evaluation_summary['average_response_time_seconds']:.2f} "
                "seconds"
            ),
        ],
    }
)

display(summary_table)
"""
    ),

    code_cell(
        """evaluation_chart_path = Path(
    "reports/rag_evaluation_chart.png"
)

if evaluation_chart_path.exists():
    display(Image(filename=str(evaluation_chart_path)))
else:
    print("Evaluation chart was not found.")
"""
    ),

    markdown_cell(
        """## 7. Conclusion

The experiment demonstrates a complete PDF question-answering system using Retrieval-Augmented Generation.

### Final results

- Answer accuracy: **100%**
- Retrieval hit rate: **100%**
- Citation grounding rate: **100%**
- Unsupported-question refusal accuracy: **100%**
- Overall pass rate: **100%**
- Average response time: **0.66 seconds**

The system successfully retrieves relevant PDF evidence, generates grounded answers, provides page citations, and refuses unsupported questions in the evaluation set.

### Current limitations

- Scanned PDFs require OCR.
- Complex table layouts may lose some structure during extraction.
- The evaluation dataset should be expanded for production use.
- Groq inference requires an internet connection.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUTPUT_PATH.write_text(
    json.dumps(
        notebook,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("Notebook created successfully!")
print(f"Saved to: {OUTPUT_PATH}")