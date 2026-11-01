import json
from pathlib import Path

from src.evaluate_rag import (
    calculate_keyword_score,
    detect_refusal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_source_files_exist():
    """Verify that all core project modules exist."""

    required_files = [
        "app.py",
        "README.md",
        "requirements.txt",
        "src/__init__.py",
        "src/document_processor.py",
        "src/text_chunker.py",
        "src/vector_store.py",
        "src/retriever.py",
        "src/rag_pipeline.py",
        "src/evaluate_rag.py",
        "docs/architecture.md",
        "notebooks/rag_experiments.ipynb",
    ]

    missing_files = [
        relative_path
        for relative_path in required_files
        if not (PROJECT_ROOT / relative_path).exists()
    ]

    assert not missing_files, (
        f"Required project files are missing: {missing_files}"
    )


def test_processed_document_artifacts_exist():
    """Verify that document-processing outputs were created."""

    required_artifacts = [
        "data/raw/apple_2024_annual_report.pdf",
        "data/processed/apple_2024_cleaned.txt",
        "data/processed/chunks_1000_200.json",
        "reports/chunk_size_comparison.csv",
        "reports/vector_store_summary.json",
    ]

    for relative_path in required_artifacts:
        artifact_path = PROJECT_ROOT / relative_path

        assert artifact_path.exists(), (
            f"Missing artifact: {relative_path}"
        )

        assert artifact_path.stat().st_size > 0, (
            f"Artifact is empty: {relative_path}"
        )


def test_selected_chunks_are_not_empty():
    """Verify that the selected chunk file contains document chunks."""

    chunks_path = (
        PROJECT_ROOT
        / "data/processed/chunks_1000_200.json"
    )

    with chunks_path.open("r", encoding="utf-8") as file:
        chunk_data = json.load(file)

    if isinstance(chunk_data, dict):
        chunks = chunk_data.get(
            "chunks",
            chunk_data.get("documents", []),
        )
    else:
        chunks = chunk_data

    assert isinstance(chunks, list)
    assert len(chunks) >= 500


def test_faiss_index_exists():
    """Verify that both FAISS index files were saved."""

    faiss_directory = (
        PROJECT_ROOT
        / "vector_store/faiss_index"
    )

    index_file = faiss_directory / "index.faiss"
    metadata_file = faiss_directory / "index.pkl"

    assert index_file.exists()
    assert index_file.stat().st_size > 0

    assert metadata_file.exists()
    assert metadata_file.stat().st_size > 0


def test_evaluation_outputs_exist():
    """Verify that automated evaluation reports were generated."""

    required_reports = [
        "reports/rag_evaluation_results.csv",
        "reports/rag_evaluation_summary.json",
        "reports/rag_evaluation_chart.png",
    ]

    for relative_path in required_reports:
        report_path = PROJECT_ROOT / relative_path

        assert report_path.exists(), (
            f"Missing evaluation output: {relative_path}"
        )

        assert report_path.stat().st_size > 0


def test_evaluation_scores_meet_threshold():
    """Verify that all principal RAG scores are at least 80%."""

    summary_path = (
        PROJECT_ROOT
        / "reports/rag_evaluation_summary.json"
    )

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    score_names = [
        "answer_accuracy",
        "retrieval_hit_rate",
        "citation_grounding_rate",
        "refusal_accuracy",
        "overall_pass_rate",
    ]

    for score_name in score_names:
        assert summary[score_name] >= 0.80, (
            f"{score_name} is below the required threshold."
        )


def test_keyword_score_function():
    """Verify deterministic factual-answer scoring."""

    answer = (
        "Apple's total net sales were "
        "$391,035 million in 2024."
    )

    expected_groups = [
        ["391,035", "391.035"],
        ["net sales", "sales"],
    ]

    score = calculate_keyword_score(
        answer,
        expected_groups,
    )

    assert score == 1.0


def test_refusal_detection_function():
    """Verify unsupported-question refusal detection."""

    refusal_answer = (
        "I could not find this information "
        "in the uploaded document."
    )

    assert detect_refusal(refusal_answer)


def test_environment_file_is_ignored():
    """Verify that private environment files cannot be committed."""

    gitignore_path = PROJECT_ROOT / ".gitignore"

    assert gitignore_path.exists()

    ignored_entries = {
        line.strip()
        for line in gitignore_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    assert ".env" in ignored_entries
    assert "venv/" in ignored_entries


def test_no_groq_key_is_hardcoded():
    """Check public project files for accidentally hard-coded Groq keys."""

    files_to_check = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs/architecture.md",
    ]

    files_to_check.extend(
        (PROJECT_ROOT / "src").glob("*.py")
    )

    for file_path in files_to_check:
        file_content = file_path.read_text(
            encoding="utf-8"
        )

        assert "gsk_" not in file_content, (
            f"Possible Groq API key found in {file_path.name}"
        )


def test_python_files_compile():
    """Verify that all main Python files have valid syntax."""

    python_files = [
        PROJECT_ROOT / "app.py",
        *list((PROJECT_ROOT / "src").glob("*.py")),
    ]

    for python_file in python_files:
        source_code = python_file.read_text(
            encoding="utf-8"
        )

        compile(
            source_code,
            str(python_file),
            "exec",
        )