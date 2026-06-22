from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from datasets import Dataset
import json


def get_ragas_llm(model: str = "qwen2.5:3b"):
    """Configure RAGAS to use local Ollama LLM."""
    return LangchainLLMWrapper(
        ChatOllama(model=model, temperature=0)
    )

def get_ragas_embeddings(model: str = "nomic-embed-text"):
    """Configure RAGAS to use local embeddings."""
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=model)
    )

def run_evaluation(test_cases: list[dict], model: str = "qwen2.5:3b") -> dict:
    """
    Run RAGAS evaluation on a list of test cases.
    Uses the same model being evaluated as the RAGAS judge.

    Each test case should have:
    - question: str
    - answer: str
    - contexts: list[str]
    - ground_truth: str (optional, needed for context_recall)
    """

    # Build dataset
    data = {
        "question":     [tc["question"]             for tc in test_cases],
        "answer":       [tc["answer"]               for tc in test_cases],
        "contexts":     [tc["contexts"]             for tc in test_cases],
        "ground_truth": [tc.get("ground_truth", "") for tc in test_cases]
    }
    dataset = Dataset.from_dict(data)

    # Use the same model being evaluated as the judge 
    ragas_llm        = get_ragas_llm(model)        # ← uses passed model
    ragas_embeddings = get_ragas_embeddings()       # ← always nomic-embed-text

    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    return result

def save_results(results: dict, path: str = "eval/results.json"):
    """Save evaluation results to disk."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {path}")