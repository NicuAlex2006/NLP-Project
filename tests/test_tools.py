"""Smoke tests for the tool layer and the public API surface.

These tests do not hit the network, OpenAI, Groq, or load the heavy local
models — they verify that modules import cleanly, public functions exist
with the expected signature, and the LangChain tool wrappers behave.

Run with:
    pytest tests/ -v
"""

import os
import sys

# Make the project root importable when pytest is run from anywhere.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------------------------- imports

def test_tool_modules_import():
    from tools import search_tool, scraper_tool, summarizer_tool  # noqa: F401


def test_config_loads():
    from configs.settings import MODELS, DEFAULT_MODEL
    assert DEFAULT_MODEL in MODELS
    for key, cfg in MODELS.items():
        assert "provider" in cfg, f"{key} missing provider"


def test_use_cases_shape():
    from evaluation.use_cases import USE_CASES
    assert len(USE_CASES) == 10
    expected = {"id", "category", "question", "expected_topics"}
    for uc in USE_CASES:
        assert expected.issubset(uc.keys()), f"Use case {uc} is missing fields"
        assert isinstance(uc["expected_topics"], list) and uc["expected_topics"]


def test_api_schemas():
    from api.schemas import QuestionRequest, AnswerResponse, ModelsResponse
    req = QuestionRequest(question="hello?")
    assert req.model == "llama4"   # default
    resp = AnswerResponse(question="q", answer="a", model="m", success=True)
    assert resp.success is True
    listing = ModelsResponse(models=["a", "b"], default="a")
    assert listing.default == "a"


# --------------------------------------------------------------------- search

def test_search_tool_is_langchain_tool():
    from tools.search_tool import search_web
    # LangChain @tool decorator exposes .invoke() and .name on the function.
    assert hasattr(search_web, "invoke")
    assert hasattr(search_web, "name")


def test_search_tool_handles_failure_gracefully(monkeypatch):
    """If both backends raise, the tool should swallow the exception and
    return a human-readable error string rather than crashing the agent."""
    from tools import search_tool

    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(search_tool, "_search_tavily", boom)
    monkeypatch.setattr(search_tool, "_search_duckduckgo", boom)

    out = search_tool.search_web.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert "Search failed" in out or "No results" in out


# --------------------------------------------------------------------- scraper

def test_scraper_invalid_urls_returns_message():
    from tools.scraper_tool import scrape_multiple_pages
    out = scrape_multiple_pages.invoke({"urls": "   ,   ,"})
    assert "No valid URLs" in out


# --------------------------------------------------------------------- agent contract

def test_finetuned_agent_registry():
    """Both fine-tuned model keys must be registered with required fields."""
    from agents.finetuned_agent import MODEL_REGISTRY
    for key in ("finetuned-bart", "finetuned-t5"):
        assert key in MODEL_REGISTRY
        spec = MODEL_REGISTRY[key]
        assert {"local_dir", "base_hf", "prefix"}.issubset(spec.keys())


def test_scratch_agent_tool_context_validator():
    from agents.local_agent import validate_tool_output, ToolContextError
    import pytest

    # Happy path
    assert validate_tool_output("real content " * 10, "search_web")

    # Empty
    with pytest.raises(ToolContextError):
        validate_tool_output("", "search_web")

    # Error string from a tool
    with pytest.raises(ToolContextError):
        validate_tool_output("Search failed: rate limited", "search_web")

    # Too short
    with pytest.raises(ToolContextError):
        validate_tool_output("hi", "search_web")
