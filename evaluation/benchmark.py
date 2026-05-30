"""
Benchmarks the search agent across multiple models and all 10 use cases.
Measures: response time, memory usage, answer length, and a simple quality score.
Results are saved to evaluation/results.csv
"""

import sys
import os
import time
import tracemalloc
import json
import csv
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.search_agent import ask
from evaluation.use_cases import USE_CASES
from configs.settings import MODELS

# Which models to benchmark — must match keys in settings.MODELS
BENCHMARK_MODELS = ["llama4", "qwen", "mistral"]

# Delay between calls per model (seconds) — prevents rate limit errors on free tiers
RATE_LIMIT_DELAY = {
    "llama4": 2,
    "qwen":   2,
    "mistral": 2,
    "groq": 2,
}

MAX_RETRIES = 2  # How many times to retry a failed call before giving up

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")


def score_answer(answer: str, expected_topics: list[str]) -> float:
    """
    Simple keyword-based quality score (0.0 to 1.0).
    Checks how many expected topics appear in the answer.
    """
    if not answer or not expected_topics:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
    return round(hits / len(expected_topics), 2)


def run_single(model_key: str, use_case: dict) -> dict:
    """Run one use case on one model and return metrics. Retries on rate limit errors."""
    question = use_case["question"]

    tracemalloc.start()
    start_time = time.time()

    answer = ""
    success = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            answer = ask(question, model_key=model_key)
            success = True
            break
        except Exception as e:
            error_msg = str(e)
            # Rate limit hit — wait and retry
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait = RATE_LIMIT_DELAY.get(model_key, 10) * attempt
                print(f"  ⚠ Rate limit hit. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                answer = f"ERROR: {error_msg}"
                break

    if not success and not answer:
        answer = "ERROR: Max retries exceeded due to rate limiting."

    elapsed = round(time.time() - start_time, 2)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    quality = score_answer(answer, use_case["expected_topics"]) if success else 0.0

    return {
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":         model_key,
        "use_case_id":   use_case["id"],
        "category":      use_case["category"],
        "question":      question,
        "success":       success,
        "time_sec":      elapsed,
        "memory_kb":     round(peak_memory / 1024, 2),
        "answer_length": len(answer),
        "quality_score": quality,
        "answer":        answer,   # Full answer — no truncation
    }


def run_benchmark(models: list[str] = BENCHMARK_MODELS, use_cases: list[dict] = USE_CASES):
    """Run all use cases on all models and save results to CSV."""
    results = []
    total = len(models) * len(use_cases)
    count = 0

    print(f"Starting benchmark: {len(models)} models × {len(use_cases)} use cases = {total} runs\n")

    for model_key in models:
        if model_key not in MODELS:
            print(f"Skipping unknown model: {model_key}")
            continue

        print(f"\n=== Model: {model_key} ===")
        for uc in use_cases:
            count += 1
            print(f"  [{count}/{total}] Use case {uc['id']}: {uc['question'][:60]}...")

            result = run_single(model_key, uc)
            results.append(result)

            # Wait between calls to respect rate limits
            delay = RATE_LIMIT_DELAY.get(model_key, 2)
            time.sleep(delay)

            status = "✓" if result["success"] else "✗"
            print(f"  {status} Time: {result['time_sec']}s | Memory: {result['memory_kb']}KB | Quality: {result['quality_score']}")

    save_results(results)
    print_summary(results)
    return results


def save_results(results: list[dict]):
    """Save results to a CSV file."""
    if not results:
        return

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {RESULTS_FILE}")


def print_summary(results: list[dict]):
    """Print a summary table of average metrics per model."""
    print("\n=== BENCHMARK SUMMARY ===\n")
    print(f"{'Model':<12} {'Avg Time':>10} {'Avg Mem(KB)':>12} {'Avg Quality':>12} {'Success Rate':>13}")
    print("-" * 62)

    models_seen = list(dict.fromkeys(r["model"] for r in results))
    for model in models_seen:
        model_results = [r for r in results if r["model"] == model]
        avg_time    = round(sum(r["time_sec"]     for r in model_results) / len(model_results), 2)
        avg_mem     = round(sum(r["memory_kb"]    for r in model_results) / len(model_results), 2)
        avg_quality = round(sum(r["quality_score"]for r in model_results) / len(model_results), 2)
        success_rate = round(sum(1 for r in model_results if r["success"]) / len(model_results) * 100)

        print(f"{model:<12} {avg_time:>10}s {avg_mem:>11}KB {avg_quality:>12} {success_rate:>12}%")


if __name__ == "__main__":
    run_benchmark()
