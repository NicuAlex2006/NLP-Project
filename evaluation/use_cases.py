"""
10 use cases for benchmarking the search agent.
Each use case covers a different type of query to test robustness.
"""

USE_CASES = [
    {
        "id": 1,
        "category": "Factual",
        "question": "What is the current version of Python and what are its main new features?",
        "expected_topics": ["python", "version", "release", "feature"],
    },
    {
        "id": 2,
        "category": "Comparative",
        "question": "What are the differences between PyTorch and TensorFlow in 2025?",
        "expected_topics": ["pytorch", "tensorflow", "performance", "framework", "training"],
    },
    {
        "id": 3,
        "category": "Current Events",
        "question": "What are the latest developments in open source LLMs in 2025?",
        "expected_topics": ["model", "open", "language", "training", "2025"],
    },
    {
        "id": 4,
        "category": "Technical How-To",
        "question": "How do you fine-tune a LLaMA model on a custom dataset?",
        "expected_topics": ["fine-tune", "llama", "dataset", "training", "model"],
    },
    {
        "id": 5,
        "category": "Financial",
        "question": "What is the current state of the AI chip market and who are the main players?",
        "expected_topics": ["nvidia", "gpu", "market", "chip", "amd"],
    },
    {
        "id": 6,
        "category": "Scientific",
        "question": "What are the most recent breakthroughs in quantum computing?",
        "expected_topics": ["quantum", "qubit", "error", "computing", "google"],
    },
    {
        "id": 7,
        "category": "Coding",
        "question": "What are the best Python libraries for building REST APIs in 2025?",
        "expected_topics": ["fastapi", "flask", "python", "api", "rest"],
    },
    {
        "id": 8,
        "category": "Ambiguous",
        "question": "Is Rust worth learning?",
        "expected_topics": ["rust", "performance", "memory", "systems", "language"],
    },
    {
        "id": 9,
        "category": "Multi-hop",
        "question": "Who founded OpenAI, and what are they currently working on?",
        "expected_topics": ["openai", "sam altman", "founded", "artificial intelligence", "research"],
    },
    {
        "id": 10,
        "category": "Trend Analysis",
        "question": "How has the adoption of containerization with Docker changed software development?",
        "expected_topics": ["docker", "container", "deployment", "kubernetes", "microservice"],
    },
]


def get_questions() -> list[str]:
    """Return just the questions as a list."""
    return [uc["question"] for uc in USE_CASES]


def get_by_category(category: str) -> list[dict]:
    """Return all use cases of a specific category."""
    return [uc for uc in USE_CASES if uc["category"] == category]


if __name__ == "__main__":
    print(f"Total use cases: {len(USE_CASES)}\n")
    for uc in USE_CASES:
        print(f"[{uc['id']}] ({uc['category']}) {uc['question']}")
