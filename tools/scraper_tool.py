import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trafilatura
import requests
from langchain.tools import tool
from configs.settings import MAX_CONTENT_LENGTH


def _fetch_content(url: str) -> str:
    """Download and extract the main text content from a URL."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text

        # Fallback: plain requests + trafilatura extract
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        text = trafilatura.extract(response.text)
        return text or ""

    except Exception as e:
        return f"Failed to fetch {url}: {str(e)}"


@tool
def scrape_page(url: str) -> str:
    """
    Extract the main text content from a single web page URL.
    Use this tool after a web search to get the full content of a specific page.
    Returns the cleaned text of the page, trimmed to avoid overloading the model.
    """
    content = _fetch_content(url)

    if not content:
        return f"Could not extract content from: {url}"

    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH] + "... [content trimmed]"

    return f"Content from {url}:\n\n{content}"


@tool
def scrape_multiple_pages(urls: str) -> str:
    """
    Extract and combine text content from multiple web page URLs.
    Input must be a comma-separated list of URLs, e.g.: 'https://url1.com, https://url2.com, https://url3.com'
    Use this tool when you want to gather information from several sources and build a richer answer.
    Returns combined content from all pages, each trimmed to avoid overloading the model.
    """
    url_list = [u.strip() for u in urls.split(",") if u.strip()]

    if not url_list:
        return "No valid URLs provided."

    # Limit to 4 URLs max to avoid overwhelming the model
    url_list = url_list[:10]

    # Each URL gets an equal share of the total allowed content
    per_url_limit = MAX_CONTENT_LENGTH // len(url_list)

    combined = ""
    for i, url in enumerate(url_list, 1):
        content = _fetch_content(url)
        if not content:
            combined += f"\n[Source {i}] {url}: Could not extract content.\n"
            continue

        if len(content) > per_url_limit:
            content = content[:per_url_limit] + "... [trimmed]"

        combined += f"\n[Source {i}] {url}:\n{content}\n"

    return combined.strip()


if __name__ == "__main__":
    # Test single page
    print("=== Single page ===")
    result = scrape_page.invoke("https://en.wikipedia.org/wiki/Large_language_model")
    print(result[:500])

    # Test multiple pages
    print("\n=== Multiple pages ===")
    result = scrape_multiple_pages.invoke(
        "https://en.wikipedia.org/wiki/Large_language_model, https://en.wikipedia.org/wiki/Mistral_AI"
    )
    print(result[:500])
