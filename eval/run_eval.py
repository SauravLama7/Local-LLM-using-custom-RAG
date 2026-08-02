import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_chain import get_prompt
from rag.llm import generate
from eval.ragas_eval import run_evaluation, save_results


def collect_answers(test_cases: list[dict], model: str) -> list[dict]:
    """Run RAG pipeline on each test case and collect answers + contexts."""
    results = []

    for i, tc in enumerate(test_cases):
        print(f"🔍 [{model}] Test {i+1}/{len(test_cases)}: {tc['question'][:50]}")

        # Get prompt and context from your RAG pipeline
        prompt, sources, context_chunks, indexed_chunks = get_prompt(
            query=tc["question"],
            model=model
        )

        # Generate answer
        full_response = ""
        for token in generate(prompt, model=model):
            full_response += token

        results.append({
            "question":     tc["question"],
            "answer":       full_response,
            "contexts":     context_chunks,
            "indexed_chunks": indexed_chunks,
            "ground_truth": tc.get("ground_truth", "")
        })

        print(f"   ✅ {full_response[:80]}...")

    return results


if __name__ == "__main__":
    MODELS = ["qwen2.5:3b", "ministral-3:3b"]  # ← both models

    # Load test cases
    with open("eval/test_cases.json", "r") as f:
        test_cases = json.load(f)

    all_results = {}

    for model in MODELS:
        print(f"\n{'='*50}")
        print(f"📊 Evaluating model: {model}")
        print(f"{'='*50}")

        # Collect answers
        answered = collect_answers(test_cases, model=model)

        # Run RAGAS evaluation
        print(f"🧪 Running RAGAS metrics for {model}...")
        results = run_evaluation(answered, model=model)

        all_results[model] = dict(results)

        print(f"\n📊 Results for {model}:")
        for metric, score in dict(results).items():
            print(f"   {metric}: {score:.3f}")

        # Save per-model results
        safe_name = model.replace(":", "_").replace(".", "_")
        save_results(dict(results), f"eval/results_{safe_name}.json")

    # Compare both models 
    print(f"\n{'='*50}")
    print("📊 Model Comparison:")
    print(f"{'='*50}")

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(f"{'Metric':<25} {'qwen2.5:3b':>12} {'ministral-3:3b':>15} {'Winner':>10}")
    print("-" * 65)

    for metric in metrics:
        scores = {m: all_results[m].get(metric, 0) for m in MODELS}
        winner = max(scores, key=scores.get)
        print(
            f"{metric:<25} "
            f"{scores['qwen2.5:3b']:>12.3f} "
            f"{scores['ministral-3:3b']:>15.3f} "
            f"{'← ' + winner.split(':')[0]:>10}"
        )

    # Save combined results
    save_results(all_results, "eval/results_comparison.json")
    print(f"\n✅ Comparison saved to eval/results_comparison.json")