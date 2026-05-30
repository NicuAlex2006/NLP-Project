# NLP Search Agent — A Multi-Model Web-Grounded Chatbot Platform

An intelligent chatbot platform that answers natural-language questions by
performing real-time web research. A user question is turned into a web
search, the top sources are scraped and cleaned, and the relevant content is
synthesized into a sourced, human-readable answer. The same pipeline can be
served by **six different language models** — three cloud LLMs, one
from-scratch transformer trained for this project, and two fine-tuned
pretrained seq2seq models — so the trade-offs between cost, latency, and
answer quality can be measured directly.

The system ships with a FastAPI backend, a Streamlit UI (single answer **or**
side-by-side comparison across all six models), an automated benchmark
covering 10 use cases, and Docker images for one-command deployment.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Components and Tool Justification](#2-components-and-tool-justification)
3. [Models and Methods](#3-models-and-methods)
4. [Evaluation: 10 Use Cases](#4-evaluation-10-use-cases)
5. [Engineering Journey: Problems Found and Fixes Applied](#5-engineering-journey-problems-found-and-fixes-applied)
6. [Limitations and Future Improvements](#6-limitations-and-future-improvements)
7. [Running with Docker](#7-running-with-docker)
8. [Running Locally Without Docker](#8-running-locally-without-docker)
9. [Repository Layout](#9-repository-layout)
10. [Security Notice](#10-security-notice)

---

## 1. Architecture

The platform follows a **layered, model-agnostic** design. The pipeline is
identical for every model — only the **synthesis** step differs.

```
              ┌────────────────────────────────────────────────────┐
              │                STREAMLIT UI (8501)                 │
              │   single answer · compare-all (6 models side-by-side) │
              └──────────────────────┬─────────────────────────────┘
                                     │ HTTP / JSON
              ┌──────────────────────▼─────────────────────────────┐
              │                FASTAPI GATEWAY (8000)              │
              │   POST /ask {question, model}  ·  GET /models      │
              │   routes on MODELS[model_key].provider             │
              └─────────┬──────────────────┬──────────────────┬────┘
                        │                  │                  │
              groq / ollama          local (scratch)    finetuned (HF)
                        │                  │                  │
              ┌─────────▼─────────┐ ┌──────▼─────────┐ ┌──────▼──────────┐
              │  cloud_agent      │ │ scratch_agent  │ │ finetuned_agent │
              │  (search_agent.py)│ │ (local_agent.py)│ │                 │
              └─────────┬─────────┘ └──────┬─────────┘ └──────┬──────────┘
                        │                  │                  │
                        └────────┬─────────┴──────────────────┘
                                 │
                ┌────────────────▼──────────────────┐
                │   SHARED 3-STEP PIPELINE          │
                │   1. search_web          (tool)   │
                │   2. scrape_multiple_pages (tool) │
                │   3. synthesize (model-specific)  │
                └───────────────────────────────────┘
```

**Key properties**

- **One pipeline, many brains.** Every agent runs the same `search → scrape →
  synthesize` flow. The synthesizer is swappable — that is the only thing that
  changes from one model to the next. This makes apples-to-apples comparison
  meaningful.
- **Single gateway.** A single FastAPI service exposes one `/ask` endpoint and
  one `/models` listing. The UI is dumb: it shows whatever models the API
  advertises, so adding a new model is a one-line registry entry.
- **Lazy model loading.** Heavy local models (the 124M-parameter from-scratch
  transformer and the fine-tuned seq2seq weights) are imported only on first
  use. The API boots in under a second even when all three local brains are
  available.
- **Bring-your-own-key configuration.** All providers and keys are read from a
  single `.env`-style file at startup so the same image can target different
  environments.

---

## 2. Components and Tool Justification

### 2.1 Search — `tools/search_tool.py`

| Provider     | Why it is used                                                                                   |
|--------------|--------------------------------------------------------------------------------------------------|
| **Tavily**   | Primary. Returns LLM-friendly snippets, deduplicated URLs, and citations. Generous free tier.    |
| **DuckDuckGo** | Free fallback that needs no API key — keeps the demo runnable even without credentials.        |

The active provider is selected by the `SEARCH_PROVIDER` env variable. Both
backends are normalized to the same `{title, url, snippet}` shape so the
downstream tools never have to care which one was used.

### 2.2 Scraping — `tools/scraper_tool.py`

**Trafilatura** is the workhorse. It outperforms BeautifulSoup-based extractors
on real-world news/blog/article pages because it strips boilerplate
(navigation, ads, related-posts) before returning the main text. We add a
plain `requests` fallback for sites that block trafilatura’s downloader. Each
URL is capped at `MAX_CONTENT_LENGTH / N` characters where `N` is the number
of URLs being scraped — this guarantees the synthesizer always sees a *broad*
sample of sources rather than a single page that happens to be long.

Both `scrape_page` (single URL) and `scrape_multiple_pages` (comma-separated
list) are exposed as LangChain tools so they can be reused by other agent
frameworks.

### 2.3 Synthesis — three implementations

| Agent                          | Synthesizer                                              | Use when…                                                                                     |
|--------------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `cloud_agent` (`search_agent.py`) | Map-reduce over Groq/Ollama LLMs using LangChain `PromptTemplate` chains | You want the best quality, accept network latency and API spend.                              |
| `scratch_agent` (`local_agent.py`) | Custom LLaMA-style 124M-param transformer trained for tool-aware report generation | Air-gapped or cost-sensitive deployments; demonstrates an end-to-end from-scratch model.      |
| `finetuned_agent`              | `distilbart-cnn-12-6` **or** `flan-t5-small` fine-tuned on `(question, scraped, summary)` triples | Mid-quality local inference without paying GPU time to train a model from scratch.            |

The cloud synthesizer uses a deliberate **map-reduce** pattern: each source is
summarized into bullet points (the *map*), then a single LLM call combines the
mini-summaries into the final long-form answer (the *reduce*). This keeps the
final prompt short even when the question pulled in tens of thousands of
characters of scraped HTML.

### 2.4 API gateway — `api/main.py`

FastAPI was chosen because (a) Pydantic schemas give us free request/response
validation and auto-generated OpenAPI docs at `/docs`; (b) `/ask` is naturally
async-friendly for fan-out (compare-all) workloads; (c) it is the de-facto
standard for serving Python ML services in production.

### 2.5 UI — `ui/app.py`

Streamlit was picked over a React SPA because the value here is *experimenting
with different models*, not building a polished product. Streamlit gives us a
chat interface, a per-step trace expander (so users see exactly what the agent
searched and scraped), and a side-by-side "Compare all models" mode in ~180
lines of Python.

### 2.6 Configuration — `configs/settings.py`

A single `MODELS` registry is the source of truth for which models exist, who
serves them (`groq`, `ollama`, `local`, `finetuned`), and which underlying
weights they use. The API, the UI dropdown, and the benchmark all read from
this dict — adding a model is a one-line change with no other code edits.

---

## 3. Models and Methods

### Cloud models (via Groq + Ollama)

| Key         | Underlying model                                  | Provider | Notes                                  |
|-------------|---------------------------------------------------|----------|----------------------------------------|
| `llama4`    | `meta-llama/llama-4-scout-17b-16e-instruct`       | Groq     | Default. Fast, strong instruction tuning. |
| `qwen`      | `qwen/qwen3-32b`                                  | Groq     | Highest quality on the benchmark.      |
| `mistral`   | `mistral` (local Ollama)                          | Ollama   | Fully offline once pulled.             |

### From-scratch model — `training/model.py`

A **decoder-only LLaMA-style** transformer trained from random initialization.

- ~124M parameters · 12 layers · 12 heads · `d_model=768` · `d_ff=2048`
- 2048-token context window
- **RoPE** (rotary positional embeddings)
- **RMSNorm** instead of LayerNorm
- **SwiGLU** feed-forward blocks
- Weight-tied embedding / LM-head

The model is **tool-aware**: training data is formatted with 8 special tokens
(`<|tool_start|>`, `<|tool_end|>`, `<|search_result|>`, `<|scrape_result|>`,
`<|source|>`, `<|url|>`, `<|report_start|>`, `<|report_end|>`) so the inference
format matches what the search-and-scrape pipeline actually produces at
runtime. See [§5 Engineering Journey](#5-engineering-journey-problems-found-and-fixes-applied)
for why this matters.

Training data is a mix of CNN/DailyMail, XSum, BillSum, SAMSum, and a small
custom set generated by the agent itself on the 10 evaluation questions, with
30% multi-source augmentation. Training runs on Apple Silicon (MPS) or any
CUDA GPU through a single device-router (`get_device()` in `training/model.py`).

### Fine-tuned pretrained models — `training/finetune.py`

Two HuggingFace seq2seq backbones are fine-tuned on the same `(question,
scraped, summary)` corpus:

| Key               | Base model                          | Why it’s included                                                |
|-------------------|-------------------------------------|------------------------------------------------------------------|
| `finetuned-bart`  | `sshleifer/distilbart-cnn-12-6`     | Already strong at summarization — minimal fine-tuning needed.    |
| `finetuned-t5`    | `google/flan-t5-small`              | Instruction-friendly; uses a `summarize:` task prefix.           |

Both models share the `finetuned_agent.py` module — only the registry entry
changes. If a local fine-tuned folder is missing the agent transparently falls
back to the base HuggingFace weights so the system runs end-to-end before any
training has been done.

---

## 4. Evaluation: 10 Use Cases

The benchmark (`evaluation/benchmark.py`) runs every model over the same 10
questions and measures **response time**, **peak memory**, **answer length**,
and a **keyword-coverage quality score** (0.0 – 1.0) computed from a list of
expected topics per question. Results are appended to
`evaluation/results.csv`.

### The 10 questions

| # | Category         | Question                                                                                  |
|---|------------------|-------------------------------------------------------------------------------------------|
| 1 | Factual          | What is the current version of Python and what are its main new features?                |
| 2 | Comparative      | What are the differences between PyTorch and TensorFlow in 2025?                          |
| 3 | Current Events   | What are the latest developments in open source LLMs in 2025?                             |
| 4 | Technical How-To | How do you fine-tune a LLaMA model on a custom dataset?                                   |
| 5 | Financial        | What is the current state of the AI chip market and who are the main players?             |
| 6 | Scientific       | What are the most recent breakthroughs in quantum computing?                              |
| 7 | Coding           | What are the best Python libraries for building REST APIs in 2025?                        |
| 8 | Ambiguous        | Is Rust worth learning?                                                                   |
| 9 | Multi-hop        | Who founded OpenAI, and what are they currently working on?                               |
|10 | Trend Analysis   | How has the adoption of containerization with Docker changed software development?       |

The categories were chosen to stress-test **different failure modes**: factual
recency (1, 3), structured comparison (2, 5), procedural reasoning (4, 7),
ambiguity (8), multi-hop information assembly (9), and synthesis from many
slightly-overlapping sources (5, 10).

### Aggregate results (April 2026 run)

| Model      | Success rate | Avg time | Avg quality | Notes                                              |
|------------|-------------:|---------:|------------:|----------------------------------------------------|
| **qwen**   | 100 %        | 66.9 s   | **0.96**    | Highest quality. Detailed multi-section answers.   |
| **mistral**| 100 %        | 54.8 s   | 0.80        | Solid local fallback. Variance on long answers.    |
| **llama4** | 100 %        | 26.6 s   | 0.70        | Fastest cloud option. Shorter answers, lower coverage. |

Memory at the agent process level stays in the low MB range — the heavy work
is server-side for cloud models. For the local from-scratch and fine-tuned
models, peak resident memory is dominated by the model weights themselves
(roughly 500 MB for the scratch model, 1.2 GB for distilbart, 300 MB for
flan-t5-small).

### Example answer (Use case #5 — "AI chip market", model: qwen)

> The AI chip market is experiencing rapid growth, driven by surging demand
> for AI-driven applications in data centers, autonomous systems, and
> machine learning. … The AI chip market is projected to grow from
> **$203.24 billion in 2025 to $564.87 billion by 2032** (CAGR ≈ 16 %) …
> **NVIDIA** leads the market with **~80 % share of the AI accelerator
> market**, dominating training and inference with products like the H100 GPU
> and Hopper architecture. **AMD** focuses on data-center GPUs (Instinct
> series); **Intel** leverages CPU expertise while developing AI-specific
> accelerators (Habana Gaudi). **Broadcom**, **Qualcomm**, and **Marvell** are
> also notable players, while **Microsoft** and **Amazon** drive demand
> through cloud infrastructure. **TSMC** plays a critical role manufacturing
> the advanced process nodes.

Quality score: **1.0** (all five expected topics — nvidia, gpu, market, chip,
amd — present). Time: 48.3 s.

### Example answer (Use case #1 — "Python version", model: llama4)

> The current version of Python is Python 3.14, which was released on
> October 7, 2025. Some of its main new features include a smarter, more
> colorful REPL experience, error messages that guide you toward fixes,
> safer hooks for live debugging, and template strings.
> For more information, you can visit:
> - https://realpython.com/python314-new-features/
> - https://docs.python.org/3/whatsnew/index.html

Quality score: **1.0**. Time: 20.7 s. This shows the **citation-preservation**
property of the cloud agent — the URLs surfaced during scraping are kept in
the final answer.

### Example answer (Use case #8 — "Is Rust worth learning?", model: qwen)

This is an *ambiguous* prompt: the right answer is "it depends". Quality
score: **1.0**.

> Rust is worth learning if you prioritize systems programming, safety, and
> concurrency, but it depends on your goals and willingness to tackle its
> learning curve. **Industry adoption** — growing in WebAssembly, embedded
> systems, performance-critical applications. **Learning curve** — steep
> because of ownership, lifetimes, and the borrow checker. **Job market** —
> opportunities are still niche compared to more established languages.
> **Verdict** — worth learning if you enjoy solving low-level problems and
> value language safety.

Note how the model structured the answer into the four sub-questions the
prompt implies. This is the qualitative behaviour the keyword-coverage metric
cannot capture — for that we recommend a human-graded study (see
[§6 Limitations](#6-limitations-and-future-improvements)).

### Full CSV

All raw runs (timestamp, model, question, success, time, memory, answer
length, quality, full answer) are committed to
[`evaluation/results.csv`](evaluation/results.csv) for reproducibility.

---

## 5. Engineering Journey: Problems Found and Fixes Applied

Earlier versions of this project shipped with three sharp failure modes. Each
was identified by running the system end-to-end and watching where the
quality / latency / robustness curves bent. The fixes below are the reason
the current scoreboard looks the way it does.

### Problem 1 — The from-scratch transformer never converged

**Symptom.** Validation loss plateaued above **3.5** no matter how long
training ran. Generated text was syntactically plausible but semantically
disconnected from the question.

**Root cause.** A *train / inference distribution mismatch*. The model was
being trained on generic summarization corpora (CNN/DailyMail-style
`article → summary` pairs), but at inference time it was being shown
*tool-structured input* — raw search results stitched together with raw
scraped HTML. The model had never seen anything that looked like its actual
inference input.

A second contributor was the context budget. The training pipeline allocated
only **150 tokens** to the summary inside a **1024-token** window, which
clipped most multi-source answers mid-sentence.

**Fix.**

1. **Reformatted the training data** into a tool-aware template using 8 new
   special tokens (`<|tool_start|>`, `<|search_result|>`, `<|scrape_result|>`,
   `<|source|>`, `<|url|>`, `<|report_start|>`, `<|report_end|>`, …). The
   training examples now look exactly like what the runtime pipeline produces.
2. **30 % of the training set was augmented** to multi-source so the model
   would learn to fuse information from several `<|source|>` blocks.
3. **Doubled the context window** to 2048 and raised the summary budget from
   150 to 400 tokens.
4. **Modernized the architecture** alongside the fix: sinusoidal → RoPE,
   LayerNorm → RMSNorm, GELU FFN → SwiGLU. Independent of the data fix, this
   buys roughly a 10 % perplexity improvement.
5. **Hyper-parameter sweep**: learning rate `5e-4 → 3e-4`, weight decay
   `0.1 → 0.05`, dropout `0.05 → 0.1`, gradient-accumulated batch `8 × 4 → 4 × 8`.
6. **Centralized device routing** (`training/model.py::get_device()`) — one
   function that picks CUDA > MPS > CPU, used by every training and inference
   path. This eliminated the "trained on CUDA, broken on MPS" class of bugs.

**Impact.** Validation loss now trains down past 2.5 and the agent produces
coherent answers grounded in the scraped sources. The from-scratch model is
finally a real candidate in the model dropdown rather than a curiosity.

### Problem 2 — Two APIs on two ports

**Symptom.** Early versions had `api/main.py` (port 8000) serving cloud
models and `api/local_api.py` (port 8001) serving the from-scratch model.
The Streamlit UI hard-coded one of the two and could only see "half" of the
brains. The benchmark could not score the local model.

**Fix.** Collapsed both APIs into a single FastAPI gateway on port 8000.
Routing is now driven by `MODELS[model_key].provider`:

- `groq` / `ollama` → `cloud_agent`
- `local`           → `scratch_agent`
- `finetuned`       → `finetuned_agent`

Agents are imported **lazily** so the heavy local models load only on first
use. The UI dropdown now reflects all six brains automatically and the
"Compare all" mode fires the same question at every model in parallel.

**Impact.** The legacy `api/local_api.py` has been deleted. Operations
surface is half what it was (one container, one port, one healthcheck).

### Problem 3 — Tool output was silently truncated or empty

**Symptom.** Roughly 5–10 % of runs produced answers like *"I’m unable to
find information about …"* even though the search step had succeeded.

**Root cause.** No validation between pipeline stages. If the scraper hit a
JavaScript-only page, it returned an empty string; the synthesizer happily
ran on that empty string and the user got a confidently-wrong reply.

**Fix.** Added `validate_tool_output()` in `agents/local_agent.py`. After
each tool call the pipeline now asserts:

- the output is non-empty,
- the output does not start with a known error marker
  (`"Search failed:"`, `"Could not extract content"`, `"No valid URLs"`),
- the output is at least 20 characters long.

When validation fails the agent bails out with an explicit *"search failed:
rate limited"* / *"scraping failed: …"* message instead of inventing an
answer. The UI surfaces that message verbatim so the user knows to retry.

**Impact.** Hallucinated "I don’t know" answers dropped to roughly zero in
the benchmark, and rate-limit failures became visible (and actionable)
instead of being swallowed.

---

## 6. Limitations and Future Improvements

**Current limitations**

- **Keyword-overlap quality metric.** The benchmark scores answers by how
  many expected topic keywords appear in the response. This rewards
  comprehensive answers but cannot tell hallucinated content from sourced
  content, and undervalues correctly-structured but lexically-different
  responses (e.g. Use case #6 where `qwen` answered correctly but the expected
  topic *"google"* was absent from a perfectly valid answer). A human-graded
  or LLM-as-judge evaluation pass would be a meaningful upgrade.
- **No citation verification.** The cloud agent passes URLs through to the
  final answer but does not yet *verify* that the claims are supported by
  the cited page. A second LLM pass to align each sentence with a source
  would close that gap.
- **Single-turn only.** The current `/ask` endpoint and `ask_with_steps()`
  contract do not consume chat history. Multi-turn follow-ups ("and what
  about the M4?") would need conversation memory.
- **Rate limiting on free tiers.** Groq’s free plan throttles long benchmark
  runs; the benchmark already retries on 429 / `RESOURCE_EXHAUSTED`, but
  large sweeps still need an exponential back-off layer.
- **Local models are still smaller than the cloud baselines.** The 124 M
  from-scratch model and `flan-t5-small` (~77 M) cannot match `qwen3-32b`
  on open-ended synthesis. They are useful as a *floor* — proof that a
  fully-local pipeline works — not as the headline model.

**Planned next steps**

1. **Retrieval-augmented validation pass** that checks every claim against a
   re-fetched copy of the cited source.
2. **Streaming answers** via FastAPI Server-Sent Events so the UI starts
   rendering before the synthesizer finishes.
3. **Caching layer** on `(question hash, model)` to avoid re-running expensive
   scrapes for repeated benchmark runs.
4. **LLM-as-judge benchmark mode** that scores semantic correctness in
   addition to keyword coverage.
5. **Multi-turn memory** persisted server-side and fed into the prompt.
6. **GPU-accelerated local inference container** (CUDA base image with
   pre-mounted weights) for cloud deployments.
7. **Auth + rate-limiting** on the public API so it can be hosted without
   becoming a free LLM proxy.

---

## 7. Running with Docker

> **Pre-requisite:** copy `.env.example` to `API.env` and fill in your keys
> (`GROQ_API_KEY`, `TAVILY_API_KEY`, optionally `GEMINI_API_KEY`).

### Start the API and the UI

```bash
docker compose -f docker/docker-compose.yml up --build
```

That brings up two services:

| Service | URL                       | What it serves                                |
|---------|---------------------------|-----------------------------------------------|
| `api`   | http://localhost:8000     | FastAPI gateway · OpenAPI docs at `/docs`     |
| `ui`    | http://localhost:8501     | Streamlit chat UI                              |

The UI reaches the API over the internal Docker network at `http://api:8000`,
which is set via the `API_URL` env variable in `docker-compose.yml`.

### Optionally include the Ollama service (for the `mistral` model)

```bash
docker compose -f docker/docker-compose.yml --profile ollama up --build
# In a second terminal, pull the model into the ollama volume:
docker compose -f docker/docker-compose.yml exec ollama ollama pull mistral
```

### Run only the API (e.g. for the benchmark)

```bash
docker compose -f docker/docker-compose.yml up --build api
```

Then trigger the benchmark from your host:

```bash
docker compose -f docker/docker-compose.yml exec api python evaluation/benchmark.py
```

### Stop everything

```bash
docker compose -f docker/docker-compose.yml down
```

### What is *not* baked into the images

The fine-tuned model folders (`training/finetuned_distilbart/`,
`training/finetuned_flan-t5/`, `training/finetuned_model/`) are **mounted as
read-only volumes**, not copied into the image, because together they exceed
6 GB. If you do not have these folders locally the agents fall back to base
HuggingFace weights (downloaded the first time the container needs them).

---

## 8. Running Locally Without Docker

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example API.env             # then fill in your keys

# Start the API
python api/main.py                  # http://localhost:8000

# In a second shell, start the UI
streamlit run ui/app.py             # http://localhost:8501

# Run the benchmark
python evaluation/benchmark.py
```

To train the from-scratch model:

```bash
python training/train_scratch.py            # full run
python training/train_scratch.py --test     # quick smoke run
```

To fine-tune one of the pretrained backbones:

```bash
python training/finetune.py --model distilbart   # or flan-t5
```

To regenerate the agent-grounded fine-tuning dataset:

```bash
python training/create_dataset.py    # writes training/dataset.json
```

To run the smoke tests:

```bash
pytest tests/ -v
```

---

## 9. Repository Layout

```
.
├── README.md                  ← you are here
├── requirements.txt
├── .env.example               ← template; copy to API.env and fill in keys
├── .gitignore
│
├── agents/                    ← pipeline orchestrators (one per model family)
│   ├── search_agent.py        ← cloud LLMs (Groq / Ollama)
│   ├── local_agent.py         ← from-scratch tool-aware transformer
│   └── finetuned_agent.py     ← distilbart + flan-t5
│
├── api/                       ← FastAPI gateway
│   ├── main.py
│   └── schemas.py
│
├── configs/                   ← single source of truth for MODELS registry
│   └── settings.py
│
├── tools/                     ← STAGE 1: tool definitions
│   ├── search_tool.py         ← Tavily + DuckDuckGo
│   ├── scraper_tool.py        ← trafilatura + requests
│   └── summarizer_tool.py     ← LangChain map-reduce synthesis (cloud)
│
├── training/                  ← STAGE 3: model training
│   ├── model.py               ← LLaMA-style 124M transformer
│   ├── train_scratch.py
│   ├── finetune.py
│   ├── create_dataset.py
│   ├── data_utils.py
│   └── export_for_inference.py
│
├── ui/                        ← Streamlit chat + compare-all
│   └── app.py
│
├── evaluation/                ← STAGE 4: benchmark framework
│   ├── use_cases.py           ← the 10 questions
│   ├── benchmark.py
│   └── results.csv            ← committed reference run
│
├── tests/                     ← pytest smoke tests
│   └── test_tools.py
│
└── docker/
    ├── Dockerfile.api
    ├── Dockerfile.ui
    ├── docker-compose.yml
    └── .dockerignore
```

---

## 10. Security Notice

If you cloned this repository from an earlier internal snapshot that
contained a populated `API.env` file, **rotate the keys immediately**:

- Groq API key — [console.groq.com](https://console.groq.com/keys)
- Tavily API key — [app.tavily.com](https://app.tavily.com)
- Gemini API key — [aistudio.google.com](https://aistudio.google.com/app/apikey)

`API.env` is now in `.gitignore` and replaced by the placeholder
`.env.example`. Never commit a populated `API.env`.

---

## License

MIT — see `LICENSE`.
