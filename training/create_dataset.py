"""
Generates a fine-tuning dataset by running the search + scrape pipeline
on all 10 use cases and saving the results as instruction-following examples.

Output: training/dataset.json
Each example: { "instruction": question, "context": scraped_content, "response": summarized_answer }
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.search_tool import search_web
from tools.scraper_tool import scrape_multiple_pages
from tools.summarizer_tool import summarize_content
from evaluation.use_cases import USE_CASES

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "dataset.json")


def generate_example(use_case: dict) -> dict | None:
    """Run the full pipeline on one use case and return a training example."""
    question = use_case["question"]
    print(f"\n[{use_case['id']}] Generating: {question[:60]}...")

    try:
        # Step 1: Search
        search_result = search_web.invoke({"query": question})
        urls = []
        for line in search_result.split("\n"):
            if line.strip().startswith("URL:"):
                url = line.replace("URL:", "").strip()
                if url:
                    urls.append(url)
            if len(urls) == 3:
                break

        if not urls:
            print("  ✗ No URLs found")
            return None

        # Step 2: Scrape
        urls_str = ", ".join(urls)
        scraped = scrape_multiple_pages.invoke({"urls": urls_str})
        if not scraped or len(scraped) < 100:
            print("  ✗ Could not scrape content")
            return None

        # Step 3: Summarize — this becomes the expected output
        summarize_input = f"QUESTION: {question} ||| CONTENT: {scraped}"
        answer = summarize_content.invoke({"input": summarize_input})

        if "failed" in answer.lower() or "error" in answer.lower():
            print(f"  ✗ Summarization failed: {answer[:80]}")
            return None

        print(f"  ✓ Generated ({len(answer)} chars)")
        return {
            "instruction": question,
            "context":     scraped[:2000],
            "response":    answer,
            "category":    use_case["category"],
        }

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def create_dataset(use_cases: list = USE_CASES) -> list[dict]:
    """Generate training examples for all use cases."""
    dataset = []
    print(f"Generating dataset from {len(use_cases)} use cases...\n")

    for uc in use_cases:
        example = generate_example(uc)
        if example:
            dataset.append(example)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved: {len(dataset)}/{len(use_cases)} examples → {OUTPUT_FILE}")
    return dataset


if __name__ == "__main__":
    create_dataset()
