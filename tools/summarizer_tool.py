import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from configs.settings import MODELS, DEFAULT_MODEL, GROQ_API_KEY, OLLAMA_BASE_URL


# Max chars sent to LLM per individual source summary
MAX_CHUNK_SIZE = 800
# Max chars sent to LLM for the final combine step
MAX_COMBINE_SIZE = 1800


def get_llm(model_key: str = DEFAULT_MODEL):
    """Return the correct LangChain LLM based on the model key in settings."""
    config = MODELS[model_key]
    provider = config["provider"]
    model_name = config["model_name"]

    if provider == "ollama":
        return ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL)
    elif provider == "groq":
        return ChatGroq(model=model_name, api_key=GROQ_API_KEY)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Step 1 — extract detailed facts from a single source
MAP_PROMPT = PromptTemplate.from_template("""
You are a research assistant. Extract all relevant facts, details, numbers, and insights
from the content below that relate to the question. Be thorough — include specific data
points, examples, names, and explanations. Use 5 to 8 bullet points.

Question: {question}
Content: {content}

Detailed facts:
""")

# Step 2 — combine all facts into a long, well-structured final answer
REDUCE_PROMPT = PromptTemplate.from_template("""
You are an expert research assistant. Using the detailed facts gathered from multiple
sources below, write a comprehensive, well-structured answer to the question.

Requirements:
- Write at least 3 to 5 full paragraphs
- Include specific facts, numbers, and examples from the sources
- Organize with clear sections if the topic has multiple aspects
- Cite sources inline (Source 1, Source 2, etc.) where relevant
- End with a short conclusion or summary

Question: {question}

Facts from sources:
{combined_summaries}

Comprehensive answer:
""")


def _summarize_chunk(question: str, content: str, source_label: str) -> str:
    """Summarize a single source into bullet points."""
    if len(content) > MAX_CHUNK_SIZE:
        content = content[:MAX_CHUNK_SIZE] + "... [trimmed]"
    llm = get_llm()
    chain = MAP_PROMPT | llm
    result = chain.invoke({"question": question, "content": content})
    return f"{source_label}:\n{result.content.strip()}"


@tool
def summarize_content(input: str) -> str:
    """
    Summarize content from one or multiple sources to answer a question.
    Input must be formatted as: 'QUESTION: <question> ||| CONTENT: <content>'
    The content can include multiple [Source N] sections from scrape_multiple_pages.
    Uses map-reduce: summarizes each source individually, then combines into a final answer.
    """
    try:
        if "|||" not in input:
            return "Invalid input format. Use: 'QUESTION: <question> ||| CONTENT: <content>'"

        parts = input.split("|||")
        question = parts[0].replace("QUESTION:", "").strip()
        full_content = parts[1].replace("CONTENT:", "").strip()

        # --- MAP step: split by [Source N] if multiple sources present ---
        if "[Source 1]" in full_content:
            # Split content into individual sources
            import re
            source_blocks = re.split(r'\[Source \d+\]', full_content)
            source_blocks = [s.strip() for s in source_blocks if s.strip()]

            mini_summaries = []
            for i, block in enumerate(source_blocks, 1):
                mini = _summarize_chunk(question, block, f"Source {i}")
                mini_summaries.append(mini)

            combined = "\n\n".join(mini_summaries)

        else:
            # Single source — summarize directly
            combined = _summarize_chunk(question, full_content, "Source 1")

        # --- REDUCE step: combine all mini-summaries into final answer ---
        if len(combined) > MAX_COMBINE_SIZE:
            combined = combined[:MAX_COMBINE_SIZE] + "... [trimmed]"

        llm = get_llm()
        chain = REDUCE_PROMPT | llm
        result = chain.invoke({"question": question, "combined_summaries": combined})
        return result.content.strip()

    except Exception as e:
        return f"Summarization failed: {str(e)}"


if __name__ == "__main__":
    test_input = (
        "QUESTION: What are the latest open source LLMs? ||| "
        "CONTENT: [Source 1] https://example1.com:\n"
        "Mistral AI released Mistral 7B, a highly efficient open-source model. "
        "It outperforms Llama 2 on many benchmarks while being smaller in size. "
        "The model uses grouped-query attention for faster inference.\n\n"
        "[Source 2] https://example2.com:\n"
        "Meta released Llama 3.2 with improved reasoning and coding capabilities. "
        "The model is available in 1B, 3B, 11B and 90B sizes. "
        "It supports multilingual text and has a 128K context window."
    )
    result = summarize_content.invoke(test_input)
    print(result)
