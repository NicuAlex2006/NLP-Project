"""
Fine-tuned-model agent — same 3-step pipeline as local_agent.py (search → scrape →
summarize) but the summarize step uses a PRETRAINED seq2seq model that was
fine-tuned by training/finetune.py.

One module serves BOTH fine-tuned brains, selected by model_key:
    "finetuned-bart" -> distilbart   (training/finetuned_distilbart/)
    "finetuned-t5"   -> flan-t5       (training/finetuned_flan-t5/)

If a fine-tuned folder does not exist yet, it transparently falls back to the
base pretrained weights from the HuggingFace Hub, so the agent works end-to-end
*before* you have run any fine-tuning. Output quality just improves once you do.
"""

import sys
import os
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tools.search_tool import search_web
from tools.scraper_tool import scrape_multiple_pages
from training.model import get_device

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# model_key -> where to load from + task prefix (T5 needs one, BART does not)
MODEL_REGISTRY = {
    "finetuned-bart": {
        "local_dir": os.path.join(PROJECT_ROOT, "training", "finetuned_distilbart"),
        "base_hf":   "sshleifer/distilbart-cnn-12-6",
        "prefix":    "",
    },
    "finetuned-t5": {
        "local_dir": os.path.join(PROJECT_ROOT, "training", "finetuned_flan-t5"),
        "base_hf":   "google/flan-t5-small",
        "prefix":    "summarize: ",
    },
}

TOOL_DESCRIPTIONS = {
    "search_web":            "🔎 Searching the web",
    "scrape_multiple_pages": "📄 Scraping multiple pages",
    "local_summarize":       "🧠 Summarizing (fine-tuned model)",
}

# Cache loaded models so each is loaded once and reused.
_loaded = {}   # model_key -> (model, tokenizer, prefix)
_device = None


def _get_device():
    global _device
    if _device is not None:
        return _device
    _device = get_device()
    print(f"[FinetunedAgent] Using {_device}")
    return _device


def _get_model(model_key: str):
    """Load (and cache) the seq2seq model for the given key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown fine-tuned model '{model_key}'. "
                         f"Options: {list(MODEL_REGISTRY)}")

    if model_key in _loaded:
        return _loaded[model_key]

    spec = MODEL_REGISTRY[model_key]
    device = _get_device()

    # Prefer the locally fine-tuned folder; fall back to the base Hub weights.
    if os.path.isdir(spec["local_dir"]) and os.listdir(spec["local_dir"]):
        source = spec["local_dir"]
        print(f"[FinetunedAgent] Loading fine-tuned '{model_key}' from {source}")
    else:
        source = spec["base_hf"]
        print(f"[FinetunedAgent] No fine-tuned folder for '{model_key}' yet — "
              f"falling back to base weights '{source}'")

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSeq2SeqLM.from_pretrained(source).to(device).eval()

    _loaded[model_key] = (model, tokenizer, spec["prefix"])
    print(f"[FinetunedAgent] '{model_key}' ready.")
    return _loaded[model_key]


def _summarize(question: str, content: str, model_key: str) -> str:
    model, tokenizer, prefix = _get_model(model_key)
    device = _get_device()

    # Give the model the question as context plus the scraped content.
    text = f"{prefix}Question: {question}\n\n{content[:3000]}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=160,
            min_length=30,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
        )
    summary = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return summary or "(the model produced an empty summary)"


def ask_with_steps(question: str, model_key: str = "finetuned-bart",
                   chat_history: list = []) -> dict:
    """Full 3-step pipeline: search → scrape → summarize (fine-tuned model)."""
    steps = []

    # --- Step 1: Search ---
    search_args = {"query": question}
    steps.append({"type": "tool_call", "tool": "search_web",
                  "label": TOOL_DESCRIPTIONS["search_web"], "args": search_args})
    search_result = search_web.invoke(search_args)
    steps[-1]["result"] = search_result[:500]

    urls = []
    for line in search_result.split("\n"):
        if line.strip().startswith("URL:"):
            url = line.replace("URL:", "").strip()
            if url:
                urls.append(url)
        if len(urls) == 3:
            break

    if not urls:
        return {"answer": "No search results found. Please try a different question.",
                "steps": steps}

    # --- Step 2: Scrape ---
    urls_str = ", ".join(urls)
    scrape_args = {"urls": urls_str}
    steps.append({"type": "tool_call", "tool": "scrape_multiple_pages",
                  "label": TOOL_DESCRIPTIONS["scrape_multiple_pages"], "args": scrape_args})
    scraped_content = scrape_multiple_pages.invoke(scrape_args)
    steps[-1]["result"] = scraped_content[:500]

    # --- Step 3: Summarize ---
    steps.append({"type": "tool_call", "tool": "local_summarize",
                  "label": TOOL_DESCRIPTIONS["local_summarize"],
                  "args": {"question": question, "model": model_key}})
    answer = _summarize(question, scraped_content, model_key)
    steps[-1]["result"] = answer[:500]

    return {"answer": answer, "steps": steps}


def ask(question: str, model_key: str = "finetuned-bart", chat_history: list = []) -> str:
    return ask_with_steps(question, model_key)["answer"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="finetuned-bart",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--question",
                        default="What are the latest developments in open source LLMs in 2025?")
    args = parser.parse_args()

    print(f"=== Fine-tuned Agent Test ({args.model}) ===\n")
    result = ask_with_steps(args.question, model_key=args.model)
    print("\n--- Steps ---")
    for i, step in enumerate(result["steps"], 1):
        print(f"[{i}] {step['label']}")
        print(f"     {str(step.get('result', ''))[:150]}\n")
    print(f"\nFinal Answer:\n{result['answer']}")
