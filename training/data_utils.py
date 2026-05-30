"""
Data Utilities for Training the Tool-Augmented Transformer.

Changes from original:
  - Tool-aware special token formatting for all training examples
  - Expanded context window (2048 tokens)
  - Phase 1: existing data reformatted into tool template
  - Phase 2: synthetic multi-source tool-formatted examples
  - Per-source truncation to preserve breadth over depth
"""

import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

from training.model import SPECIAL_TOKENS, add_special_tokens


def _load_hf_dataset(path, *args, **kwargs):
    """Lazy import of datasets.load_dataset — avoids ImportError when the
    `datasets` package isn't installed (e.g. quick-test or inference-only)."""
    from datasets import load_dataset
    return load_dataset(path, *args, **kwargs)


# =============================================================================
# Tokenizer Setup
# =============================================================================

def get_tokenizer() -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    add_special_tokens(tokenizer)
    return tokenizer


# =============================================================================
# Tool-Aware Formatting
# =============================================================================

def format_tool_example(
    article: str,
    summary: str,
    question: str = None,
    multi_source: bool = False,
) -> tuple[str, str]:
    """Format a training example using tool-aware special tokens.

    Returns (context_str, report_str) where:
      - context_str contains the tool output wrapped in special tokens
      - report_str contains the target report wrapped in report tokens
    """
    if multi_source and len(article) > 200:
        chunks = _split_into_sources(article)
        source_blocks = []
        for i, chunk in enumerate(chunks, 1):
            source_blocks.append(
                f"<|source|> {i}\n{chunk}"
            )
        tool_content = "\n".join(source_blocks)
    else:
        tool_content = article

    if question:
        context = (
            f"<|tool_start|><|search_result|>Query: {question}<|tool_end|>\n"
            f"<|tool_start|><|scrape_result|>{tool_content}<|tool_end|>"
        )
    else:
        context = f"<|tool_start|><|scrape_result|>{tool_content}<|tool_end|>"

    report = f"<|report_start|>{summary}<|report_end|>"
    return context, report


def _split_into_sources(text: str, n_sources: int = 3) -> list[str]:
    """Split a single article into n pseudo-source blocks for training diversity."""
    sentences = text.replace('\n', ' ').split('. ')
    if len(sentences) < n_sources * 2:
        return [text]

    chunk_size = len(sentences) // n_sources
    chunks = []
    for i in range(n_sources):
        start = i * chunk_size
        end = start + chunk_size if i < n_sources - 1 else len(sentences)
        chunk = '. '.join(sentences[start:end]).strip()
        if chunk and not chunk.endswith('.'):
            chunk += '.'
        if chunk:
            chunks.append(chunk)
    return chunks


# =============================================================================
# Dataset Classes
# =============================================================================

class SummarizationDataset(Dataset):
    """Tool-aware summarization dataset.

    Each example is formatted as:
        <|tool_start|><|scrape_result|>{content}<|tool_end|><|report_start|>{summary}<|report_end|><eos>

    Labels: -100 for tool context tokens, actual IDs for report tokens.
    """

    def __init__(
        self,
        articles: list[str],
        summaries: list[str],
        tokenizer: GPT2TokenizerFast,
        max_len: int = 2048,
        max_context_len: int = 1600,
        max_report_len: int = 400,
        multi_source_ratio: float = 0.3,
        questions: list[str] = None,
        chunk_size: int = 5000,
    ):
        self.max_len = max_len

        report_start_tokens = tokenizer.encode("<|report_start|>", add_special_tokens=False)
        report_end_tokens = tokenizer.encode("<|report_end|>", add_special_tokens=False)
        eos_id = tokenizer.eos_token_id

        self.input_ids: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        skipped = 0

        articles = list(articles)
        summaries = list(summaries)
        total = len(articles)
        print(f"Tokenizing {total} examples (tool-aware format, chunk={chunk_size})...")

        for start in range(0, total, chunk_size):
            a_chunk = articles[start:start + chunk_size]
            s_chunk = summaries[start:start + chunk_size]
            q_chunk = questions[start:start + chunk_size] if questions else [None] * len(a_chunk)

            for article, summary, question in zip(a_chunk, s_chunk, q_chunk):
                if not article or not summary or len(article.strip()) < 20 or len(summary.strip()) < 10:
                    skipped += 1
                    continue

                use_multi = random.random() < multi_source_ratio
                context_str, report_str = format_tool_example(
                    article, summary, question=question, multi_source=use_multi,
                )

                ctx_ids = tokenizer.encode(
                    context_str, add_special_tokens=False,
                    truncation=True, max_length=max_context_len,
                )
                rpt_ids = tokenizer.encode(
                    report_str, add_special_tokens=False,
                    truncation=True, max_length=max_report_len,
                )

                if len(ctx_ids) < 10 or len(rpt_ids) < 5:
                    skipped += 1
                    continue

                ids = ctx_ids + rpt_ids + [eos_id]
                lbl = [-100] * len(ctx_ids) + rpt_ids + [eos_id]

                if len(ids) > max_len:
                    ids = ids[:max_len]
                    lbl = lbl[:max_len]

                if all(l == -100 for l in lbl):
                    skipped += 1
                    continue

                self.input_ids.append(torch.tensor(ids, dtype=torch.int32))
                self.labels.append(torch.tensor(lbl, dtype=torch.int32))

            done = min(start + chunk_size, total)
            print(f"  tokenized {done}/{total}")

        print(f"Dataset: {len(self.input_ids)} examples loaded, {skipped} skipped")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'labels': self.labels[idx]}


class PadCollate:
    """Right-pads each batch to its longest sequence (dynamic padding)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(item['input_ids'].size(0) for item in batch)

        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            ids, lbl = item['input_ids'], item['labels']
            n = ids.size(0)
            pad = max_len - n
            input_ids.append(torch.cat([ids, torch.full((pad,), self.pad_id, dtype=ids.dtype)]))
            labels.append(torch.cat([lbl, torch.full((pad,), -100, dtype=lbl.dtype)]))
            attention_mask.append(
                torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
            )

        return {
            'input_ids': torch.stack(input_ids).long(),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels).long(),
        }


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_cnn_dailymail(
    split: str = "train",
    max_examples: int = None,
) -> tuple[list[str], list[str]]:
    print(f"Loading CNN/DailyMail ({split})...")
    dataset = _load_hf_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    articles = dataset["article"]
    summaries = dataset["highlights"]
    print(f"  Loaded {len(articles)} examples")
    return articles, summaries


def load_xsum(
    split: str = "train",
    max_examples: int = None,
) -> tuple[list[str], list[str]]:
    print(f"Loading XSum ({split})...")
    dataset = _load_hf_dataset("EdinburghNLP/xsum", split=split)
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    articles = dataset["document"]
    summaries = dataset["summary"]
    print(f"  Loaded {len(articles)} examples")
    return articles, summaries


def load_billsum(
    split: str = "train",
    max_examples: int = None,
) -> tuple[list[str], list[str]]:
    print(f"Loading BillSum ({split})...")
    dataset = _load_hf_dataset("FiscalNote/billsum", split=split)
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    articles = dataset["text"]
    summaries = dataset["summary"]
    print(f"  Loaded {len(articles)} examples")
    return articles, summaries


def load_samsum(
    split: str = "train",
    max_examples: int = None,
) -> tuple[list[str], list[str]]:
    print(f"Loading SAMSum ({split})...")
    dataset = _load_hf_dataset("Samsung/samsum", split=split)
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    articles = dataset["dialogue"]
    summaries = dataset["summary"]
    print(f"  Loaded {len(articles)} examples")
    return articles, summaries


def load_custom_dataset(
    path: str = None,
) -> tuple[list[str], list[str], list[str]]:
    """Load custom Tavily-generated dataset. Returns (articles, summaries, questions)."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "dataset.json")

    if not os.path.exists(path):
        print(f"  Custom dataset not found at {path}")
        return [], [], []

    print(f"Loading custom dataset from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)

    articles, summaries, questions = [], [], []
    for item in data:
        context = item.get("context", "")
        response = item.get("response", "")
        question = item.get("instruction", "")
        if context and response:
            articles.append(context)
            summaries.append(response)
            questions.append(question)

    print(f"  Loaded {len(articles)} examples")
    return articles, summaries, questions


# =============================================================================
# DataLoader Builder
# =============================================================================

def create_dataloaders(
    dataset_name: str = "combined_all",
    max_train_examples: int = 1_000_000,
    max_val_examples: int = 2000,
    max_seq_len: int = 2048,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, GPT2TokenizerFast]:
    tokenizer = get_tokenizer()

    if dataset_name == "cnn_dailymail":
        train_articles, train_summaries = load_cnn_dailymail("train", max_train_examples)
        val_articles, val_summaries = load_cnn_dailymail("validation", max_val_examples)
        train_questions, val_questions = None, None

    elif dataset_name == "xsum":
        train_articles, train_summaries = load_xsum("train", max_train_examples)
        val_articles, val_summaries = load_xsum("validation", max_val_examples)
        train_questions, val_questions = None, None

    elif dataset_name == "custom":
        articles, summaries, questions = load_custom_dataset()
        split_idx = max(1, int(len(articles) * 0.9))
        train_articles, train_summaries = articles[:split_idx], summaries[:split_idx]
        val_articles, val_summaries = articles[split_idx:], summaries[split_idx:]
        train_questions = questions[:split_idx]
        val_questions = questions[split_idx:]

    elif dataset_name == "combined":
        train_a1, train_s1 = load_cnn_dailymail("train", max_train_examples)
        val_a1, val_s1 = load_cnn_dailymail("validation", max_val_examples)
        train_a2, train_s2, train_q2 = load_custom_dataset()
        train_articles = train_a1 + train_a2
        train_summaries = train_s1 + train_s2
        train_questions = [None] * len(train_a1) + train_q2
        val_articles, val_summaries = val_a1, val_s1
        val_questions = None

    elif dataset_name == "combined_all":
        train_articles, train_summaries = [], []
        val_articles, val_summaries = [], []
        train_questions = []

        ta, ts = load_cnn_dailymail("train", max_train_examples)
        va, vs = load_cnn_dailymail("validation", max_val_examples)
        train_articles += ta; train_summaries += ts
        train_questions += [None] * len(ta)
        val_articles += va; val_summaries += vs

        ta, ts = load_xsum("train", max_train_examples)
        va, vs = load_xsum("validation", max_val_examples)
        train_articles += ta; train_summaries += ts
        train_questions += [None] * len(ta)
        val_articles += va; val_summaries += vs

        ta, ts = load_billsum("train", 18000)
        va, vs = load_billsum("ca_test", 500)
        train_articles += ta; train_summaries += ts
        train_questions += [None] * len(ta)
        val_articles += va; val_summaries += vs

        try:
            ta, ts = load_samsum("train", 14000)
            va, vs = load_samsum("validation", 500)
            train_articles += ta; train_summaries += ts
            train_questions += [None] * len(ta)
            val_articles += va; val_summaries += vs
        except Exception as e:
            print(f"  (skipping SAMSum: {e})")

        ta, ts, tq = load_custom_dataset()
        train_articles += ta; train_summaries += ts
        train_questions += tq

        val_questions = None
        print(f"\n  Combined total: {len(train_articles)} train, {len(val_articles)} val")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nCreating datasets (max_seq_len={max_seq_len}, tool-aware format)...")

    train_dataset = SummarizationDataset(
        train_articles, train_summaries, tokenizer, max_len=max_seq_len,
        questions=train_questions if dataset_name in ("custom", "combined", "combined_all") else None,
    )
    val_dataset = SummarizationDataset(
        val_articles, val_summaries, tokenizer, max_len=max_seq_len,
        questions=val_questions,
        multi_source_ratio=0.0,
    )

    collate = PadCollate(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate,
    )

    print(f"Train: {len(train_dataset)} examples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} examples, {len(val_loader)} batches")

    return train_loader, val_loader, tokenizer


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Utilities (Tool-Aware)")
    print("=" * 60)

    tokenizer = get_tokenizer()
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"PAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"Special tokens: {SPECIAL_TOKENS}")

    test_articles = [
        "The quick brown fox jumped over the lazy dog. " * 20,
        "Scientists discovered a new frog in the Amazon. The frog was found in Brazil. " * 10,
    ]
    test_summaries = [
        "A fox jumped over a dog in a test scenario.",
        "A new frog species was discovered in the Amazon.",
    ]

    dataset = SummarizationDataset(test_articles, test_summaries, tokenizer, max_len=512)
    print(f"\nTest dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Input IDs shape (unpadded): {sample['input_ids'].shape}")
        print(f"Labels shape   (unpadded): {sample['labels'].shape}")

        collate = PadCollate(tokenizer.pad_token_id)
        batch = collate([dataset[i] for i in range(len(dataset))])
        print(f"Batched input_ids: {batch['input_ids'].shape}")
        print(f"Batched attention_mask: {batch['attention_mask'].shape}")

        tokens = sample['input_ids'].tolist()
        text = tokenizer.decode(tokens)
        print(f"\nDecoded text (first 300 chars):\n{text[:300]}...")

    print("\nData utilities test passed!")
