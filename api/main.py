import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import QuestionRequest, AnswerResponse, ModelsResponse
from configs.settings import MODELS, DEFAULT_MODEL, validate


def run_pipeline(question: str, model_key: str) -> dict:
    """Route a question to the correct agent based on the model's provider.

    cloud LLMs (groq/ollama) -> search_agent
    from-scratch transformer -> local_agent
    fine-tuned seq2seq       -> finetuned_agent
    Agents are imported lazily so the heavy local models only load when used.
    """
    provider = MODELS[model_key]["provider"]
    if provider == "local":
        from agents.local_agent import ask_with_steps as run
        return run(question)
    if provider == "finetuned":
        from agents.finetuned_agent import ask_with_steps as run
        return run(question, model_key=model_key)
    # groq / ollama
    from agents.search_agent import ask_with_steps as run
    return run(question, model_key=model_key)

# --- App setup ---
app = FastAPI(
    title="NLP Search Agent API",
    description="An intelligent chatbot that searches the web and returns summarized answers.",
    version="1.0.0",
)

# Allow the Streamlit UI to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """Validate settings when the server starts."""
    validate()
    print("[API] Server ready.")


# --- Routes ---

@app.get("/", summary="Health check")
def root():
    """Check if the API is running."""
    return {"status": "ok", "message": "NLP Search Agent API is running."}


@app.get("/models", response_model=ModelsResponse, summary="List available models")
def get_models():
    """Return all available models and the current default."""
    return ModelsResponse(
        models=list(MODELS.keys()),
        default=DEFAULT_MODEL,
    )


@app.post("/ask", response_model=AnswerResponse, summary="Ask the agent a question")
def ask_question(request: QuestionRequest):
    """
    Send a natural language question to the search agent.
    The agent will search the web, scrape relevant pages, and return a summarized answer.
    """
    if request.model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Available: {list(MODELS.keys())}"
        )

    try:
        result = run_pipeline(request.question, request.model)
        return AnswerResponse(
            question=request.question,
            answer=result["answer"],
            model=request.model,
            success=True,
            steps=result["steps"],
        )
    except Exception as e:
        return AnswerResponse(
            question=request.question,
            answer="",
            model=request.model,
            success=False,
            steps=[],
            error=str(e),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
