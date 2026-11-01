import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.rag_pipeline import RAGPipeline


REPORTS_DIRECTORY = Path("reports")
RESULTS_CSV_PATH = REPORTS_DIRECTORY / "rag_evaluation_results.csv"
SUMMARY_JSON_PATH = REPORTS_DIRECTORY / "rag_evaluation_summary.json"
CHART_PATH = REPORTS_DIRECTORY / "rag_evaluation_chart.png"


# Each inner list contains acceptable alternatives for one expected fact.
EVALUATION_CASES = [
    {
        "type": "factual",
        "question": "What were Apple's total net sales in 2024?",
        "expected_groups": [
            ["391,035", "391.035"],
            ["net sales", "sales"],
        ],
    },
    {
        "type": "factual",
        "question": "What were Apple's Services net sales in 2024?",
        "expected_groups": [
            ["96,169", "96.169"],
            ["services"],
        ],
    },
    {
        "type": "factual",
        "question": "What was Apple's net income in 2024?",
        "expected_groups": [
            ["93,736", "93.736"],
            ["net income"],
        ],
    },
    {
        "type": "factual",
        "question": (
            "How much were Apple's research and development expenses in 2024?"
        ),
        "expected_groups": [
            ["31,370", "31.370"],
            ["research and development", "r&d"],
        ],
    },
    {
        "type": "out_of_document",
        "question": "What was Apple's revenue from operations on Mars in 2035?",
        "expected_groups": [],
    },
]


REFUSAL_PATTERNS = [
    "could not find",
    "not found",
    "not available",
    "does not contain",
    "doesn't contain",
    "not present",
    "no information",
    "cannot determine",
]


PAGE_CITATION_PATTERN = re.compile(
    r"Page\s*:?\s*(\d+)",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize text for deterministic keyword checks."""

    return (
        text.lower()
        .replace("**", "")
        .replace("$", "")
        .replace("\n", " ")
        .strip()
    )


def calculate_keyword_score(answer: str, expected_groups: list) -> float:
    """
    Calculate how many expected fact groups appear in the answer.

    Example:
    [["391,035", "391.035"], ["net sales", "sales"]]

    One value from each group must appear for a perfect score.
    """

    if not expected_groups:
        return 0.0

    normalized_answer = normalize_text(answer)
    matched_groups = 0

    for alternatives in expected_groups:
        if any(
            normalize_text(alternative) in normalized_answer
            for alternative in alternatives
        ):
            matched_groups += 1

    return matched_groups / len(expected_groups)


def retrieval_contains_expected_fact(
    retrieved_text: str,
    expected_groups: list,
) -> bool:
    """Check whether retrieved chunks contain the expected facts."""

    if not expected_groups:
        return False

    normalized_context = normalize_text(retrieved_text)

    for alternatives in expected_groups:
        group_found = any(
            normalize_text(alternative) in normalized_context
            for alternative in alternatives
        )

        if not group_found:
            return False

    return True


def detect_refusal(answer: str) -> bool:
    """Check whether the model correctly refuses an unsupported question."""

    normalized_answer = normalize_text(answer)

    return any(
        pattern in normalized_answer
        for pattern in REFUSAL_PATTERNS
    )


def extract_cited_pages(answer: str) -> list:
    """Extract page numbers cited by the generated answer."""

    return list(dict.fromkeys(PAGE_CITATION_PATTERN.findall(answer)))


def extract_retrieved_pages(documents: list) -> list:
    """Extract unique page numbers from retrieved documents."""

    pages = []

    for document in documents:
        page = str(document.metadata.get("page", "Unknown"))

        if page not in pages:
            pages.append(page)

    return pages


def evaluate_case(pipeline: RAGPipeline, test_case: dict) -> dict:
    """Run and score one evaluation question."""

    question = test_case["question"]
    question_type = test_case["type"]
    expected_groups = test_case["expected_groups"]

    start_time = time.perf_counter()
    result = pipeline.ask(question)
    response_time = time.perf_counter() - start_time

    answer = result["answer"]
    documents = result["documents"]

    retrieved_text = "\n".join(
        document.page_content
        for document in documents
    )

    retrieved_pages = extract_retrieved_pages(documents)
    cited_pages = extract_cited_pages(answer)

    citation_present = len(cited_pages) > 0

    citation_grounded = (
        citation_present
        and all(page in retrieved_pages for page in cited_pages)
    )

    if question_type == "factual":
        keyword_score = calculate_keyword_score(
            answer,
            expected_groups,
        )

        retrieval_hit = retrieval_contains_expected_fact(
            retrieved_text,
            expected_groups,
        )

        refusal_detected = False

        passed = (
            keyword_score == 1.0
            and retrieval_hit
            and citation_grounded
        )

    else:
        keyword_score = 0.0
        retrieval_hit = False
        refusal_detected = detect_refusal(answer)

        # Unsupported questions should be refused instead of answered.
        passed = refusal_detected

    expected_facts = " | ".join(
        "/".join(group)
        for group in expected_groups
    )

    return {
        "question_type": question_type,
        "question": question,
        "expected_facts": expected_facts,
        "answer": answer,
        "answer_keyword_score": round(keyword_score, 3),
        "retrieval_hit": retrieval_hit,
        "citation_present": citation_present,
        "citation_grounded": citation_grounded,
        "refusal_detected": refusal_detected,
        "retrieved_pages": ", ".join(retrieved_pages),
        "cited_pages": ", ".join(cited_pages),
        "response_time_seconds": round(response_time, 3),
        "passed": passed,
        "error": "",
    }


def calculate_summary(results_dataframe: pd.DataFrame) -> dict:
    """Calculate overall RAG evaluation metrics."""

    factual_results = results_dataframe[
        results_dataframe["question_type"] == "factual"
    ]

    refusal_results = results_dataframe[
        results_dataframe["question_type"] == "out_of_document"
    ]

    factual_count = len(factual_results)
    refusal_count = len(refusal_results)

    answer_accuracy = (
        float(
            (
                factual_results["answer_keyword_score"] == 1.0
            ).mean()
        )
        if factual_count
        else 0.0
    )

    retrieval_hit_rate = (
        float(factual_results["retrieval_hit"].mean())
        if factual_count
        else 0.0
    )

    citation_grounding_rate = (
        float(factual_results["citation_grounded"].mean())
        if factual_count
        else 0.0
    )

    refusal_accuracy = (
        float(refusal_results["refusal_detected"].mean())
        if refusal_count
        else 0.0
    )

    pass_rate = float(results_dataframe["passed"].mean())

    average_response_time = float(
        results_dataframe["response_time_seconds"].mean()
    )

    return {
        "evaluation_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_questions": int(len(results_dataframe)),
        "factual_questions": int(factual_count),
        "out_of_document_questions": int(refusal_count),
        "answer_accuracy": round(answer_accuracy, 3),
        "retrieval_hit_rate": round(retrieval_hit_rate, 3),
        "citation_grounding_rate": round(
            citation_grounding_rate,
            3,
        ),
        "refusal_accuracy": round(refusal_accuracy, 3),
        "overall_pass_rate": round(pass_rate, 3),
        "average_response_time_seconds": round(
            average_response_time,
            3,
        ),
    }


def create_evaluation_chart(summary: dict):
    """Create a chart showing the principal evaluation scores."""

    metric_names = [
        "Answer\naccuracy",
        "Retrieval\nhit rate",
        "Citation\ngrounding",
        "Refusal\naccuracy",
        "Overall\npass rate",
    ]

    metric_values = [
        summary["answer_accuracy"] * 100,
        summary["retrieval_hit_rate"] * 100,
        summary["citation_grounding_rate"] * 100,
        summary["refusal_accuracy"] * 100,
        summary["overall_pass_rate"] * 100,
    ]

    colors = [
        "#2563EB",
        "#059669",
        "#7C3AED",
        "#EA580C",
        "#0891B2",
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    bars = axis.bar(
        metric_names,
        metric_values,
        color=colors,
        width=0.65,
    )

    axis.set_title(
        "Intelligent Document Assistant — RAG Evaluation",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )

    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 110)
    axis.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, metric_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            fontweight="bold",
        )

    figure.tight_layout()
    figure.savefig(
        CHART_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    """Run the complete automated evaluation."""

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nInitializing RAG evaluation pipeline...\n")
    pipeline = RAGPipeline()

    evaluation_results = []

    for index, test_case in enumerate(
        EVALUATION_CASES,
        start=1,
    ):
        print(
            f"[{index}/{len(EVALUATION_CASES)}] "
            f"{test_case['question']}"
        )

        try:
            case_result = evaluate_case(
                pipeline,
                test_case,
            )

            status = (
                "PASS"
                if case_result["passed"]
                else "FAIL"
            )

            print(
                f"Result: {status} | "
                f"Time: "
                f"{case_result['response_time_seconds']} seconds\n"
            )

        except Exception as error:
            case_result = {
                "question_type": test_case["type"],
                "question": test_case["question"],
                "expected_facts": "",
                "answer": "",
                "answer_keyword_score": 0.0,
                "retrieval_hit": False,
                "citation_present": False,
                "citation_grounded": False,
                "refusal_detected": False,
                "retrieved_pages": "",
                "cited_pages": "",
                "response_time_seconds": 0.0,
                "passed": False,
                "error": str(error),
            }

            print(f"Result: ERROR | {error}\n")

        evaluation_results.append(case_result)

    results_dataframe = pd.DataFrame(evaluation_results)

    results_dataframe.to_csv(
        RESULTS_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    summary = calculate_summary(results_dataframe)

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=4,
        )

    create_evaluation_chart(summary)

    print("\nRAG evaluation completed!")
    print("-" * 60)
    print(
        f"Answer accuracy: "
        f"{summary['answer_accuracy'] * 100:.1f}%"
    )
    print(
        f"Retrieval hit rate: "
        f"{summary['retrieval_hit_rate'] * 100:.1f}%"
    )
    print(
        f"Citation grounding rate: "
        f"{summary['citation_grounding_rate'] * 100:.1f}%"
    )
    print(
        f"Refusal accuracy: "
        f"{summary['refusal_accuracy'] * 100:.1f}%"
    )
    print(
        f"Overall pass rate: "
        f"{summary['overall_pass_rate'] * 100:.1f}%"
    )
    print(
        f"Average response time: "
        f"{summary['average_response_time_seconds']:.2f} seconds"
    )

    print("\nFiles created:")
    print(f"- {RESULTS_CSV_PATH}")
    print(f"- {SUMMARY_JSON_PATH}")
    print(f"- {CHART_PATH}")


if __name__ == "__main__":
    main()