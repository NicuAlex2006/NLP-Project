"""
Local agent — 3-step pipeline (search -> scrape -> summarize) using
the custom-trained tool-augmented transformer.

Key changes:
  - Tool context validation: validates that tool outputs are captured,
    formatted with special tokens, and fit within the model's context window
  - Cross-platform device routing (CUDA / MPS / CPU)
  - Special token formatting matches training data format exactly
"""

import sys
import os
import json
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import torch
from transformers import GPT2TokenizerFast

from training.model import (
    ScratchTransformer, TransformerConfig,
    get_device, get_special_token_ids, add_special_tokens,
)
from tools.search_tool import search_web
from tools.scraper_tool import scrape_multiple_pages

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "training", "finetuned_model")

TOOL_DESCRIPTIONS = {
    "search_web":            "Searching the web",
    "scrape_multiple_pages": "Scraping multiple pages",
    "local_summarize":       "Summarizing locally (custom model)",
}

_model = None
_tokenizer = None
_device = None


def _get_model():
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    _device = get_device()
    print(f"[LocalAgent] Using {_device}")

    model_path = os.path.join(MODEL_DIR, "model.pt")
    config_path = os.path.join(MODEL_DIR, "config.json")

    _tokenizer = _load_tokenizer()

    if not os.path.exists(model_path):
        print(f"[LocalAgent] WARNING: No trained model found at {model_path}")
        print("[LocalAgent] Run 'python training/train_scratch.py' to train first.")
        print("[LocalAgent] Falling back to extractive summarization.")
        _model = None
        return _model, _tokenizer, _device

    print(f"[LocalAgent] Loading custom model from {MODEL_DIR}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = TransformerConfig.from_dict(config_dict)

    _model = ScratchTransformer(config)
    _model.resize_token_embeddings(len(_tokenizer), _tokenizer)

    state_dict = torch.load(model_path, map_location=_device, weights_only=True)
    _model.load_state_dict(state_dict)
    _model.to(_device)
    _model.eval()

    print("[LocalAgent] Custom model loaded and ready.")
    return _model, _tokenizer, _device


def _load_tokenizer():
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
        tokenizer.pad_token = tokenizer.eos_token
        add_special_tokens(tokenizer)
    except Exception:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        add_special_tokens(tokenizer)
    return tokenizer


# =============================================================================
# Tool Context Validation & Formatting
# =============================================================================

class ToolContextError(Exception):
    """Raised when tool output fails validation."""
    pass


def validate_tool_output(output: str, tool_name: str) -> str:
    """Validate that a tool produced usable output.

    Checks:
      - Output is not empty or error-only
      - Output contains actual content (not just error messages)
      - Output is within reasonable length bounds
    """
    if not output or not output.strip():
        raise ToolContextError(f"{tool_name} returned empty output")

    error_markers = ["Search failed:", "Could not extract content", "No valid URLs"]
    for marker in error_markers:
        if output.strip().startswith(marker):
            raise ToolContextError(f"{tool_name} returned an error: {output[:200]}")

    if len(output.strip()) < 20:
        raise ToolContextError(f"{tool_name} output too short ({len(output.strip())} chars)")

    return output


def format_tool_context(
    question: str,
    search_result: str,
    scraped_content: str,
    tokenizer,
    max_context_tokens: int = 1600,
) -> str:
    """Format tool outputs into the special-token structure matching training data.

    Returns the formatted context string, truncated per-source to fit within
    the model's context budget while preserving breadth across sources.
    """
    search_block = (
        f"<|tool_start|><|search_result|>Query: {question}\n{search_result}<|tool_end|>"
    )

    source_blocks = _parse_source_blocks(scraped_content)

    if source_blocks:
        formatted_sources = []
        for i, (url, content) in enumerate(source_blocks, 1):
            block = f"<|source|> {i}"
            if url:
                block += f" <|url|>{url}"
            block += f"\n{content}"
            formatted_sources.append(block)
        scrape_block = (
            "<|tool_start|><|scrape_result|>"
            + "\n".join(formatted_sources)
            + "<|tool_end|>"
        )
    else:
        scrape_block = (
            f"<|tool_start|><|scrape_result|>{scraped_content}<|tool_end|>"
        )

    full_context = search_block + "\n" + scrape_block + "\n<|report_start|>"

    context_ids = tokenizer.encode(full_context, add_special_tokens=False)
    if len(context_ids) > max_context_tokens:
        per_source_budget = max_context_tokens // max(len(source_blocks), 1) if source_blocks else max_context_tokens
        truncated_sources = []
        for i, (url, content) in enumerate(source_blocks or [(None, scraped_content)], 1):
            content_ids = tokenizer.encode(content, add_special_tokens=False)
            if len(content_ids) > per_source_budget:
                content = tokenizer.decode(content_ids[:per_source_budget])
            block = f"<|source|> {i}"
            if url:
                block += f" <|url|>{url}"
            block += f"\n{content}"
            truncated_sources.append(block)

        scrape_block = (
            "<|tool_start|><|scrape_result|>"
            + "\n".join(truncated_sources)
            + "<|tool_end|>"
        )
        full_context = search_block + "\n" + scrape_block + "\n<|report_start|>"

    return full_context


def _parse_source_blocks(scraped_content: str) -> list[tuple[str, str]]:
    """Parse scraper output into (url, content) pairs."""
    import re
    blocks = re.split(r'\[Source \d+\]', scraped_content)
    blocks = [b.strip() for b in blocks if b.strip()]

    result = []
    for block in blocks:
        lines = block.split('\n', 1)
        url = None
        content = block
        if lines[0].strip().startswith('http'):
            url = lines[0].strip().rstrip(':')
            content = lines[1].strip() if len(lines) > 1 else ""
        result.append((url, content))

    return result


# =============================================================================
# Summarization
# =============================================================================

def _extractive_fallback(content: str, num_sentences: int = 5) -> str:
    sentences = content.replace('\n', ' ').split('. ')
    selected = sentences[:num_sentences]
    return '. '.join(selected).strip() + '.'


def _local_summarize(question: str, search_result: str, scraped_content: str) -> str:
    """Summarize using the custom model with validated tool context."""
    model, tokenizer, device = _get_model()

    if model is None:
        return _extractive_fallback(scraped_content)

    formatted_context = format_tool_context(
        question, search_result, scraped_content, tokenizer,
        max_context_tokens=model.config.max_seq_len - 400,
    )

    input_ids = tokenizer.encode(
        formatted_context, return_tensors="pt", truncation=True,
        max_length=model.config.max_seq_len - 300,
    )
    input_ids = input_ids.to(device)

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=300,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = generated[0, input_ids.shape[1]:]
    summary = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    report_end = "<|report_end|>"
    if report_end in summary:
        summary = summary[:summary.index(report_end)]

    for tok in ["<|tool_start|>", "<|tool_end|>", "<|report_start|>", "<|report_end|>"]:
        summary = summary.replace(tok, "")

    summary = summary.strip()
    if not summary:
        return _extractive_fallback(scraped_content)

    return summary


# =============================================================================
# Agent Pipeline
# =============================================================================

def ask_with_steps(question: str, chat_history: list = []) -> dict:
    steps = []

    # --- Step 1: Search ---
    search_args = {"query": question}
    steps.append({
        "type": "tool_call",
        "tool": "search_web",
        "label": TOOL_DESCRIPTIONS["search_web"],
        "args": search_args,
    })
    search_result = search_web.invoke(search_args)
    steps[-1]["result"] = search_result[:500]

    try:
        search_result = validate_tool_output(search_result, "search_web")
    except ToolContextError as e:
        return {
            "answer": f"Search failed: {e}",
            "steps": steps,
        }

    # Extract top 3 URLs
    urls = []
    for line in search_result.split("\n"):
        if line.strip().startswith("URL:"):
            url = line.replace("URL:", "").strip()
            if url:
                urls.append(url)
        if len(urls) == 3:
            break

    if not urls:
        return {
            "answer": "No search results found. Please try a different question.",
            "steps": steps,
        }

    # --- Step 2: Scrape ---
    urls_str = ", ".join(urls)
    scrape_args = {"urls": urls_str}
    steps.append({
        "type": "tool_call",
        "tool": "scrape_multiple_pages",
        "label": TOOL_DESCRIPTIONS["scrape_multiple_pages"],
        "args": scrape_args,
    })
    scraped_content = scrape_multiple_pages.invoke(scrape_args)
    steps[-1]["result"] = scraped_content[:500]

    try:
        scraped_content = validate_tool_output(scraped_content, "scrape_multiple_pages")
    except ToolContextError as e:
        return {
            "answer": f"Scraping failed: {e}. Using search snippets instead.",
            "steps": steps,
        }

    # --- Step 3: Summarize with validated tool context ---
    steps.append({
        "type": "tool_call",
        "tool": "local_summarize",
        "label": TOOL_DESCRIPTIONS["local_summarize"],
        "args": {"question": question},
    })
    answer = _local_summarize(question, search_result, scraped_content)
    steps[-1]["result"] = answer[:500]

    return {"answer": answer, "steps": steps}


def ask(question: str, chat_history: list = []) -> str:
    return ask_with_steps(question, chat_history)["answer"]


if __name__ == "__main__":
    print("=== Local Agent Test (Tool-Augmented Transformer) ===\n")
    question = "What are the latest developments in open source LLMs in 2025?"
    print(f"Question: {question}\n")

    result = ask_with_steps(question)

    print("\n--- Steps ---")
    for i, step in enumerate(result["steps"], 1):
        print(f"[{i}] {step['label']}")
        print(f"     {str(step.get('result', ''))[:150]}\n")

    print(f"\nFinal Answer:\n{result['answer']}")
