"""
PatchContext - Stage 4: RAGAs evaluation.

Runs the pipeline over the 50-question benchmark and scores it with RAGAs
metrics: faithfulness, answer_relevancy, context_utilization (the reference-free
variant of context_precision — see note below).

RAGAs needs an LLM "judge" internally. We use Google Gemini's free tier for this,
deliberately separate from Groq (which the pipeline itself uses to generate
answers) — see the comment above judge_llm below for why. Everything is still
free; no paid API keys anywhere in this project.

Generation is resumable and incremental: progress is checkpointed to
data/pipeline_outputs_checkpoint.json after EVERY question, so this script can
be safely stopped and re-run across multiple days if you're limited by a daily
token quota (e.g. `python evaluate_ragas.py --limit 5` to only generate 5 new
questions this run). RAGAs scoring is skipped automatically until all questions
are generated, unless you pass --score-partial.
"""

import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, ContextUtilization
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from rag_pipeline import PatchContextPipeline

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_PATH = os.path.join(DATA_DIR, "ragas_results.csv")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "pipeline_outputs_checkpoint.json")


def load_questions():
    with open(os.path.join(DATA_DIR, "questions.json")) as f:
        return json.load(f)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return []


def save_checkpoint(rows):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(rows, f, indent=2)


def run_pipeline_incremental(pipeline, questions, limit=None, existing_rows=None):
    """
    Resumable, incremental generation: skips questions already in existing_rows
    (matched by question text), saves after EVERY question (not just at the end)
    so a rate-limit failure mid-run never loses progress, and stops early once
    `limit` NEW questions have been processed in this run (to respect a daily
    token budget across multiple days).
    """
    rows = list(existing_rows) if existing_rows is not None else load_checkpoint()
    done_questions = {r["question"] for r in rows}
    remaining = [q for q in questions if q not in done_questions]

    if not remaining:
        print(f"All {len(questions)} questions already in checkpoint. Nothing to generate.")
        return rows

    todo = remaining if limit is None else remaining[:limit]
    print(f"{len(done_questions)}/{len(questions)} already done. "
          f"Generating {len(todo)} more this run ({len(remaining) - len(todo)} left after this).")

    for i, q in enumerate(todo, 1):
        try:
            result = pipeline.answer(q)
        except Exception as e:
            print(f"\nStopped at question {i}/{len(todo)} due to an error (likely rate limit): {e}")
            print(f"Progress so far is saved in {CHECKPOINT_PATH} — just re-run this script to resume.")
            break
        rows.append({
            "question": q,
            "answer": result["answer"],
            "contexts": [d.page_content for d in result["retrieved_docs"]],
        })
        save_checkpoint(rows)  # save after EVERY question, not just at the end
        print(f"  [{i}/{len(todo)}] done: {q[:70]}")

    remaining_after = len(questions) - len(rows)
    if remaining_after > 0:
        print(f"\n{len(rows)}/{len(questions)} total done. {remaining_after} remain — "
              f"re-run this script (same command) to continue, e.g. tomorrow once your "
              f"daily quota resets.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Max NEW questions to generate this run (use this to stay "
                              "under a daily token quota, e.g. --limit 5). Default: no limit "
                              "(process everything remaining).")
    parser.add_argument("--score-partial", action="store_true",
                         help="Run RAGAs scoring even if not all 50 questions are generated yet. "
                              "By default, scoring is skipped until generation is complete.")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found in .env. Set it before running (needed for generation).")
        return
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not found in .env. Generation will still work, but "
              "RAGAs scoring (which needs it) will fail once you reach that step. Set it in "
              ".env now so you don't discover this after today's/tomorrow's generation run.")

    questions = load_questions()
    question_set = set(questions)

    # Match the checkpoint against the CURRENT questions.json by content, not just
    # count — if questions.json changes between runs, stale entries for questions
    # that no longer exist must not silently count toward "complete".
    rows = load_checkpoint()
    stale = [r for r in rows if r["question"] not in question_set]
    if stale:
        print(f"Dropping {len(stale)} stale checkpoint entries that don't match the "
              f"current questions.json (likely from a previous version of the question set).")
        rows = [r for r in rows if r["question"] in question_set]
        save_checkpoint(rows)

    done_questions = {r["question"] for r in rows}
    remaining = [q for q in questions if q not in done_questions]

    if remaining:
        print("Loading pipeline...")
        pipeline = PatchContextPipeline()
        rows = run_pipeline_incremental(pipeline, questions, limit=args.limit, existing_rows=rows)
    else:
        print(f"All {len(questions)} questions already in checkpoint (and match the current "
              f"question set). Nothing to generate.")

    is_complete = {r["question"] for r in rows} == question_set
    if not is_complete and not args.score_partial:
        print("\nGeneration is incomplete — skipping RAGAs scoring for now. "
              "Re-run this script (with --limit if needed) to keep going, or pass "
              "--score-partial to score what's done so far.")
        return

    if not is_complete:
        print(f"\nScoring partial results ({len(rows)}/{len(questions)} questions) "
              f"as requested via --score-partial.")

    dataset = Dataset.from_list(rows)

    # RAGAs judge: deliberately a DIFFERENT model (Gemini, free tier) than the one
    # being evaluated (Groq/Llama, used by the pipeline itself). Two reasons:
    #   1. Methodological: using the same model to both generate and judge its own
    #      answers is a known bias in RAG evaluation (self-preference bias).
    #   2. Practical: RAGAs scoring makes many LLM calls per question (faithfulness
    #      alone decomposes each answer into claims and verifies each one), which
    #      would otherwise burn through Groq's free daily token cap on top of what
    #      the pipeline itself already used to generate the answers.
    # Embeddings stay local/free (no API quota involved either way).
    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key=os.getenv("GOOGLE_API_KEY"))
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    # context_precision (RAGAs' default) requires a manually-authored ground-truth
    # 'reference' answer per question, which we don't have for this project. We use
    # context_utilization instead — the reference-free variant of the same underlying
    # metric — which judges precision using only question + retrieved context + answer.
    context_utilization = ContextUtilization()

    print("Scoring with RAGAs (faithfulness, answer_relevancy, context_utilization)...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_utilization],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = scores.to_pandas()
    df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved per-question scores to {RESULTS_PATH}")

    print("\n=== Aggregate scores ===")
    for metric in ["faithfulness", "answer_relevancy", "context_utilization"]:
        if metric in df.columns:
            print(f"{metric}: {df[metric].mean():.3f}")


if __name__ == "__main__":
    main()