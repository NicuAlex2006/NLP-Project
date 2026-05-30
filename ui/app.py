import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import json
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="NLP Search Agent",
    page_icon="🔍",
    layout="wide",
)


def get_available_models() -> list[str]:
    try:
        res = requests.get(f"{API_URL}/models", timeout=5)
        return res.json().get("models", ["llama4"])
    except Exception:
        return ["llama4", "qwen", "mistral", "gemini"]


def ask_agent(question: str, model: str) -> dict:
    try:
        t0 = time.perf_counter()
        res = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "model": model},
            timeout=300,
        )
        data = res.json()
        data["elapsed"] = time.perf_counter() - t0
        return data
    except requests.exceptions.ConnectionError:
        return {
            "success": False, "answer": "", "steps": [],
            "error": "Could not connect to the API. Make sure it is running with: python api/main.py",
        }
    except Exception as e:
        return {"success": False, "answer": "", "steps": [], "error": str(e)}


def render_steps(steps: list[dict]):
    """Render the agent's tool calls in a nice expandable section."""
    if not steps:
        return

    with st.expander(f"🛠️ Agent used {len(steps)} tool(s) — click to see details", expanded=False):
        for i, step in enumerate(steps, 1):
            tool = step.get("tool", "unknown")
            label = step.get("label", tool)
            args = step.get("args", {})
            result = step.get("result", "")

            # Tool header
            st.markdown(f"**Step {i} — {label}**")

            # Args
            if tool == "search_web":
                query = args.get("query", "")
                st.markdown(f"🔎 **Query:** `{query}`")

            elif tool in ("scrape_page", "scrape_multiple_pages"):
                urls = args.get("url") or args.get("urls", "")
                if isinstance(urls, str):
                    for url in urls.split(","):
                        url = url.strip()
                        if url:
                            st.markdown(f"🌐 **URL:** {url}")

            elif tool == "summarize_content":
                raw = args.get("input", "")
                if "|||" in raw:
                    question_part = raw.split("|||")[0].replace("QUESTION:", "").strip()
                    st.markdown(f"❓ **Summarizing for:** {question_part}")
                else:
                    st.markdown(f"📝 **Input:** {raw[:100]}...")

            # Result preview
            if result:
                with st.container():
                    st.markdown("**Result preview:**")
                    st.code(result[:400] + ("..." if len(result) > 400 else ""), language=None)

            if i < len(steps):
                st.divider()


# --- UI ---
st.title("🔍 NLP Search Agent")
st.markdown("Ask any question — the agent searches the web and gives you a sourced, summarized answer.")

with st.sidebar:
    st.header("⚙️ Settings")
    models = get_available_models()
    selected_model = st.selectbox("Model", models, index=0)

    compare_all = st.checkbox("⚖️ Compare all models", value=False,
                              help="Ask every brain the same question and show answers side by side.")

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. 🔎 Searches the web")
    st.markdown("2. 📄 Scrapes top pages")
    st.markdown("3. 🧠 Summarizes with LLM")
    st.markdown("---")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            render_steps(msg["steps"])
        if msg.get("model"):
            st.caption(f"Model: {msg['model']}")

# New message input
if question := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if compare_all:
            # Compare one cloud model against the three local brains, side by side.
            wanted = [selected_model, "scratch", "finetuned-bart", "finetuned-t5"]
            compare_set = [m for i, m in enumerate(wanted)
                           if m in models and m not in wanted[:i]]

            cols = st.columns(len(compare_set))
            summary_lines = []
            for col, model in zip(cols, compare_set):
                with col:
                    st.markdown(f"**{model}**")
                    with st.spinner(f"{model}..."):
                        result = ask_agent(question, model)
                    if result.get("success"):
                        st.markdown(result["answer"])
                        st.caption(f"⏱️ {result.get('elapsed', 0):.1f}s")
                        summary_lines.append(f"**{model}** ({result.get('elapsed', 0):.1f}s): {result['answer']}")
                    else:
                        st.error(result.get("error", "Unknown error"))
                        summary_lines.append(f"**{model}**: ❌ {result.get('error')}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": "\n\n---\n\n".join(summary_lines),
                "steps": [],
                "model": "compare-all",
            })
        else:
            with st.spinner(f"Searching with **{selected_model}**..."):
                result = ask_agent(question, selected_model)

            if result["success"]:
                st.markdown(result["answer"])
                render_steps(result.get("steps", []))
                st.caption(f"Model: {result['model']} · ⏱️ {result.get('elapsed', 0):.1f}s")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "steps": result.get("steps", []),
                    "model": result["model"],
                })
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ {result.get('error')}",
                    "steps": [],
                    "model": selected_model,
                })
