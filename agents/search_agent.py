import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", message=".*create_react_agent.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.messages import HumanMessage, SystemMessage

from tools.search_tool import search_web
from tools.scraper_tool import scrape_multiple_pages
from tools.summarizer_tool import summarize_content, get_llm
from configs.settings import DEFAULT_MODEL, validate

TOOL_DESCRIPTIONS = {
    "search_web":            "🔎 Searching the web",
    "scrape_multiple_pages": "📄 Scraping multiple pages",
    "summarize_content":     "🧠 Summarizing content",
}


def ask_with_steps(question: str, model_key: str = DEFAULT_MODEL, chat_history: list = []) -> dict:
    """
    Always runs the full 3-step pipeline regardless of model:
      1. search_web       — find relevant URLs
      2. scrape_multiple_pages — get content from top 3 URLs
      3. summarize_content — synthesize into a final answer

    Returns:
        { "answer": str, "steps": [ { type, tool, label, args, result } ] }
    """
    steps = []

    # --- Step 1: Search ---
    search_args = {"query": question}
    steps.append({
        "type":  "tool_call",
        "tool":  "search_web",
        "label": TOOL_DESCRIPTIONS["search_web"],
        "args":  search_args,
    })
    search_result = search_web.invoke(search_args)
    steps[-1]["result"] = search_result[:500]

    # Extract top 3 URLs from the search results
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
        "type":  "tool_call",
        "tool":  "scrape_multiple_pages",
        "label": TOOL_DESCRIPTIONS["scrape_multiple_pages"],
        "args":  scrape_args,
    })
    scraped_content = scrape_multiple_pages.invoke(scrape_args)
    steps[-1]["result"] = scraped_content[:500]

    # --- Step 3: Summarize ---
    summarize_input = f"QUESTION: {question} ||| CONTENT: {scraped_content}"
    summarize_args = {"input": summarize_input}
    steps.append({
        "type":  "tool_call",
        "tool":  "summarize_content",
        "label": TOOL_DESCRIPTIONS["summarize_content"],
        "args":  {"input": f"QUESTION: {question} ||| CONTENT: [scraped content]"},
    })
    answer = summarize_content.invoke(summarize_args)
    steps[-1]["result"] = answer[:500]

    return {"answer": answer, "steps": steps}


def ask(question: str, model_key: str = DEFAULT_MODEL, chat_history: list = []) -> str:
    """Ask and return just the final answer string."""
    return ask_with_steps(question, model_key, chat_history)["answer"]


def ask_debug(question: str, model_key: str = DEFAULT_MODEL):
    """Same as ask_with_steps() but prints every step to the terminal."""
    result = ask_with_steps(question, model_key)

    print("\n--- AGENT STEPS ---")
    for i, step in enumerate(result["steps"], 1):
        print(f"\n[{i}] {step['label']}")
        print(f"     Args   : {step['args']}")
        print(f"     Result : {step.get('result', 'N/A')[:200]}")
    print("\n-------------------\n")
    print(f"Final Answer:\n{result['answer']}")
    return result["answer"]


if __name__ == "__main__":
    validate()
    print("=== Search Agent Test ===\n")
    question = "Give me a paper from Nvidia on SVO"
    print(f"Question: {question}\n")
    ask_debug(question, model_key="llama4")
