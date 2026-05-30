from pydantic import BaseModel
from typing import Optional, Any


class QuestionRequest(BaseModel):
    """Request body for asking the agent a question."""
    question: str
    model: Optional[str] = "llama4"   # Default model to use

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the latest developments in AI in 2025?",
                "model": "llama4"
            }
        }


class AnswerResponse(BaseModel):
    """Response returned by the agent."""
    question: str
    answer: str
    model: str
    success: bool
    steps: Optional[list[dict[str, Any]]] = []
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the latest developments in AI in 2025?",
                "answer": "In 2025, the AI landscape has seen...",
                "model": "llama4",
                "success": True,
                "error": None
            }
        }


class ModelsResponse(BaseModel):
    """List of available models."""
    models: list[str]
    default: str
