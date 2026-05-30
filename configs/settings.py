from dotenv import load_dotenv
import os

load_dotenv("API.env")

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Ollama (local run) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Models ---
# provider tells the API which agent to route a request to:
#   groq / ollama -> agents.search_agent      (cloud LLMs)
#   local         -> agents.local_agent       (from-scratch transformer)
#   finetuned     -> agents.finetuned_agent   (fine-tuned pretrained seq2seq)
MODELS = {
    "mistral":        {"provider": "ollama",    "model_name": "mistral"},
    "llama4":         {"provider": "groq",      "model_name": "meta-llama/llama-4-scout-17b-16e-instruct"},
    "qwen":           {"provider": "groq",      "model_name": "qwen/qwen3-32b"},
    "scratch":        {"provider": "local",     "model_name": "scratch-transformer"},
    "finetuned-bart": {"provider": "finetuned", "model_name": "distilbart"},
    "finetuned-t5":   {"provider": "finetuned", "model_name": "flan-t5"},
}

DEFAULT_MODEL = "llama4"

# --- Search ---
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily")
MAX_SEARCH_RESULTS = 10

# --- Scraper ---
MAX_CONTENT_LENGTH = 1500

# --- Validation ---
def validate():
    warnings = []
    if SEARCH_PROVIDER == "tavily" and not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY is missing — search will fall back to DuckDuckGo")
    if not GROQ_API_KEY:
        warnings.append("GROQ_API_KEY is missing — Groq models unavailable")
    for w in warnings:
        print(f"[settings] WARNING: {w}")
    return len(warnings) == 0


if __name__ == "__main__":
    validate()
    print("\nLoaded settings:")
    print(f"  Default model   : {DEFAULT_MODEL}")
    print(f"  Search provider : {SEARCH_PROVIDER}")
    print(f"  Ollama URL      : {OLLAMA_BASE_URL}")
    print(f"  Groq key        : {'set' if GROQ_API_KEY else 'MISSING'}")
    print(f"  Tavily key      : {'set' if TAVILY_API_KEY else 'MISSING'}")
