import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.tools import tool
from tavily import TavilyClient
from duckduckgo_search import DDGS
from configs.settings import (
    TAVILY_API_KEY,
    SEARCH_PROVIDER,
    MAX_SEARCH_RESULTS,
)


def _search_tavily(query: str) -> list[dict]:
    """Search using Tavily API."""
    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query=query, max_results=MAX_SEARCH_RESULTS)
    results = []
    for r in response.get("results", []):
        results.append({
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "snippet": r.get("content", ""),
        })
    return results


def _search_duckduckgo(query: str) -> list[dict]:
    """Search using DuckDuckGo (no API key needed)."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=MAX_SEARCH_RESULTS):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return results


@tool
def search_web(query: str) -> str:
    """
    Search the web for a given query.
    Returns a list of results with title, URL, and a short snippet.
    Use this tool when you need up-to-date information from the internet.
    """
    try:
        if SEARCH_PROVIDER == "tavily" and TAVILY_API_KEY:
            results = _search_tavily(query)
        else:
            results = _search_duckduckgo(query)

        if not results:
            return "No results found for this query."

        # Format results as a readable string for the LLM
        output = f"Search results for: '{query}'\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r['title']}\n"
            output += f"   URL: {r['url']}\n"
            output += f"   {r['snippet']}\n\n"

        return output.strip()

    except Exception as e:
        return f"Search failed: {str(e)}"


if __name__ == "__main__":
    # Quick test — run: python tools/search_tool.py
    result = search_web.invoke("last research paper Nvidia")
    print(result)
