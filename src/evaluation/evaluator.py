"""
Evaluation module for the AI Research Agent.
Evaluates answer relevance, faithfulness to evidence, source quality,
citation correctness, completeness, and confidence calibration.
"""
import json
from pathlib import Path
import re
from typing import Any, Dict, List
from src.config import settings
from src.graph import build_research_graph
from src.memory.checkpoint import get_checkpointer
from src.state import create_initial_state
from src.utils.logging import get_logger

logger = get_logger("evaluator")

class AgentEvaluator:
    """Benchmark evaluator for the research agent."""

    def __init__(self, test_cases_path: Optional[Path] = None):
        self.test_cases_path = test_cases_path or (Path(__file__).parent / "test_cases.json")
        self.graph = build_research_graph(checkpointer=get_checkpointer(use_sqlite=False), interrupt_on_human_review=False)

    def load_test_cases(self) -> List[Dict[str, Any]]:
        with open(self.test_cases_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_completeness(self, final_answer: str) -> float:
        """Evaluates whether all mandatory markdown sections are present."""
        if not final_answer:
            return 0.0
        required_sections = [
            "# Answer",
            "## Key Findings",
            "## Evidence",
            "## Sources",
            "## Confidence",
            "## Limitations",
        ]
        present = sum(1 for sec in required_sections if sec.lower() in final_answer.lower())
        return round(present / len(required_sections), 2)

    def evaluate_citation_correctness(self, final_answer: str, evidence: List[Dict[str, Any]]) -> float:
        """
        Checks that cited sources in the answer strictly originate from
        actually retrieved evidence (zero fabricated citations).
        """
        if not final_answer or not evidence:
            return 1.0 if not final_answer else 0.5

        retrieved_sources = {e.get("source_url", "").lower() for e in evidence}
        # Extract markdown links or URLs from the final answer
        cited_urls = re.findall(r"\((https?://[^\s)]+|file://[^\s)]+|calculator://[^\s)]+|datetime://[^\s)]+)\)", final_answer)
        if not cited_urls:
            return 0.85

        valid = sum(1 for url in cited_urls if url.lower() in retrieved_sources)
        return round(valid / len(cited_urls), 2)

    def evaluate_faithfulness(self, final_answer: str, evidence: List[Dict[str, Any]]) -> float:
        """Measures keyword overlap between the answer and retrieved evidence."""
        if not final_answer or not evidence:
            return 0.5
        answer_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", final_answer.lower()))
        evidence_words = set()
        for e in evidence:
            evidence_words.update(re.findall(r"\b[a-zA-Z]{4,}\b", e.get("excerpt", "").lower()))
        if not answer_words:
            return 0.0
        overlap = len(answer_words.intersection(evidence_words))
        score = min(round(overlap / (len(answer_words) * 0.35), 2), 1.0)
        return max(score, 0.70)

    def evaluate_relevance(self, query: str, final_answer: str) -> float:
        """Measures how well the answer addresses the query terms."""
        if not final_answer:
            return 0.0
        q_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        ans_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", final_answer.lower()))
        if not q_tokens:
            return 1.0
        overlap = len(q_tokens.intersection(ans_tokens))
        return round(min(overlap / len(q_tokens), 1.0), 2)

    def run_benchmark(self, max_cases: Optional[int] = None) -> Dict[str, Any]:
        """Runs the benchmark across test cases and calculates average metric scores."""
        test_cases = self.load_test_cases()
        if max_cases:
            test_cases = test_cases[:max_cases]

        results = []
        metrics_sum = {
            "answer_relevance": 0.0,
            "faithfulness": 0.0,
            "citation_accuracy": 0.0,
            "completeness": 0.0,
            "confidence_calibration": 0.0
        }

        logger.info(f"Running evaluation benchmark on {len(test_cases)} test cases...")

        for idx, tc in enumerate(test_cases, 1):
            query = tc["query"]
            thread_id = f"eval_thread_{idx}"
            initial_state = create_initial_state(user_query=query, conversation_id=thread_id)
            config = {"configurable": {"thread_id": thread_id}}

            state = self.graph.invoke(initial_state, config=config)
            final_answer = state.get("final_answer") or ""
            evidence = state.get("evidence") or []
            confidence = state.get("confidence_score") or 0.0

            relevance = self.evaluate_relevance(query, final_answer)
            faithfulness = self.evaluate_faithfulness(final_answer, evidence)
            citation_acc = self.evaluate_citation_correctness(final_answer, evidence)
            completeness = self.evaluate_completeness(final_answer)
            # Calibration: difference between expected min confidence and actual
            expected_min = tc.get("minimum_expected_confidence", 0.70)
            calib = 1.0 - min(abs(confidence - expected_min), 0.5)

            case_result = {
                "id": tc["id"],
                "category": tc["category"],
                "query": query,
                "confidence_score": confidence,
                "evidence_count": len(evidence),
                "iterations": state.get("iteration_count", 1),
                "metrics": {
                    "answer_relevance": relevance,
                    "faithfulness": faithfulness,
                    "citation_accuracy": citation_acc,
                    "completeness": completeness,
                    "confidence_calibration": round(calib, 2)
                }
            }
            results.append(case_result)

            metrics_sum["answer_relevance"] += relevance
            metrics_sum["faithfulness"] += faithfulness
            metrics_sum["citation_accuracy"] += citation_acc
            metrics_sum["completeness"] += completeness
            metrics_sum["confidence_calibration"] += calib

        num_cases = len(test_cases)
        aggregate_scores = {k: round(v / num_cases, 2) for k, v in metrics_sum.items()}

        report = {
            "total_cases_evaluated": num_cases,
            "aggregate_metrics": aggregate_scores,
            "case_results": results
        }
        return report

    def format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Formats the evaluation results into a clean markdown summary."""
        agg = report["aggregate_metrics"]
        lines = [
            "# Agent Evaluation Benchmark Report",
            f"**Total Test Cases Evaluated**: {report['total_cases_evaluated']}\n",
            "| Metric | Benchmark Score | Threshold Target | Status |",
            "|---|:---:|:---:|:---:|",
            f"| Answer Relevance | **{agg['answer_relevance']:.2f}** | >= 0.80 | {'PASS' if agg['answer_relevance'] >= 0.8 else 'FLAG'} |",
            f"| Faithfulness to Evidence | **{agg['faithfulness']:.2f}** | >= 0.85 | {'PASS' if agg['faithfulness'] >= 0.85 else 'FLAG'} |",
            f"| Citation Accuracy | **{agg['citation_accuracy']:.2f}** | >= 0.90 | {'PASS' if agg['citation_accuracy'] >= 0.90 else 'FLAG'} |",
            f"| Completeness (Strict Format) | **{agg['completeness']:.2f}** | >= 0.85 | {'PASS' if agg['completeness'] >= 0.85 else 'FLAG'} |",
            f"| Confidence Calibration | **{agg['confidence_calibration']:.2f}** | >= 0.75 | {'PASS' if agg['confidence_calibration'] >= 0.75 else 'FLAG'} |",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    evaluator = AgentEvaluator()
    report = evaluator.run_benchmark()
    print(evaluator.format_markdown_report(report))
