"""
PatchContext - Stage 4: RAGAs evaluation.
"""

import os
import json
import argparse
import time
import asyncio
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, ContextUtilization
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

class PatchedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        temp = kwargs.pop("temperature", None)
        if temp is not None:
            gen_config = kwargs.get("generation_config", None) or {}
            gen_config = {**gen_config, "temperature": temp}
            kwargs["generation_config"] = gen_config
        return super()._generate(messages, stop, run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        temp = kwargs.pop("temperature", None)
        if temp is not None:
            gen_config = kwargs.get("generation_config", None) or {}
            gen_config = {**gen_config, "temperature": temp}
            kwargs["generation_config"] = gen_config
        return await super()._agenerate(messages, stop, run_manager, **kwargs)

class PacedChatGoogleGenerativeAI(PatchedChatGoogleGenerativeAI):
    """Subclass of PatchedChatGoogleGenerativeAI to handle request pacing (RPM limiting)."""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(5.0)  # Pace sync calls to stay under 15 RPM
        return super()._generate(messages, stop, run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.sleep(5.0)  # Pace async calls to stay under 15 RPM
        return await super()._agenerate(messages, stop, run_manager, **kwargs)

from rag_pipeline import PatchContextPipeline

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_PATH = os.path.join(DATA_DIR, "ragas_results.csv")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "pipeline_outputs.json")


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


def run_pipeline_incremental(pipeline, questions, limit=None, existing_rows=None, delay_seconds=15):
    """
    Resumable, incremental generation: skips questions already in existing_rows
    (matched by question text), saves after EVERY question (not just at the end)
    so a rate-limit failure mid-run never loses progress, and stops early once
    `limit` NEW questions have been processed in this run (to respect a daily
    token budget across multiple days).

    Also sleeps `delay_seconds` between questions to respect Groq's PER-MINUTE
    token limit (12K TPM), separate from the daily cap. Observed cost was
    ~2K tokens/question on average (more if a repair attempt fires), so
    running back-to-back without pacing risks a 429 mid-run well before the
    daily quota is the binding constraint.
    """
    rows = list(existing_rows) if existing_rows is not None else load_checkpoint()
    done_questions = {r["question"] for r in rows}
    remaining = [q for q in questions if q not in done_questions]

    if not remaining:
        print(f"All {len(questions)} questions already in checkpoint. Nothing to generate.")
        return rows

    todo = remaining if limit is None else remaining[:limit]
    print(f"{len(done_questions)}/{len(questions)} already done. "
          f"Generating {len(todo)} more this run ({len(remaining) - len(todo)} left after this). "
          f"Pacing with a {delay_seconds}s delay between questions to stay under Groq's per-minute token limit.")

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
        if i < len(todo):  # no need to sleep after the last question in this run
            time.sleep(delay_seconds)

    remaining_after = len(questions) - len(rows)
    if remaining_after > 0:
        print(f"\n{len(rows)}/{len(questions)} total done. {remaining_after} remain — "
              f"re-run this script (same command) to continue, e.g. tomorrow once your "
              f"daily quota resets.")
    return rows


def gemini_is_finished(response) -> bool:
    is_finished_list = []
    for g in response.flatten():
        resp = g.generations[0][0]
        if resp.generation_info is not None:
            finish_reason = resp.generation_info.get("finish_reason")
            if finish_reason is not None:
                is_finished_list.append(str(finish_reason).lower() == "stop")
            else:
                is_finished_list.append(True)
        elif hasattr(resp, "message") and resp.message is not None:
            meta = resp.message.response_metadata
            finish_reason = meta.get("finish_reason") or meta.get("stop_reason")
            if finish_reason is not None:
                is_finished_list.append(str(finish_reason).lower() in ["stop", "end_turn"])
            else:
                is_finished_list.append(True)
        else:
            is_finished_list.append(True)
    return all(is_finished_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Max NEW questions to generate this run (use this to stay "
                              "under a daily token quota, e.g. --limit 5). Default: no limit "
                              "(process everything remaining).")
    parser.add_argument("--score-partial", action="store_true",
                         help="Run RAGAs scoring even if not all 50 questions are generated yet. "
                              "By default, scoring is skipped until generation is complete.")
    parser.add_argument("--delay", type=int, default=15,
                         help="Seconds to wait between questions during generation, to stay "
                              "under Groq's per-minute token limit (default: 15s).")
    parser.add_argument("--scoring-workers", type=int, default=1,
                         help="Number of concurrent workers to use for RAGAs scoring. "
                              "Keep at 1 for Gemini free tier to stay under the 15 RPM rate limit.")
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
        rows = run_pipeline_incremental(pipeline, questions, limit=args.limit, existing_rows=rows, delay_seconds=args.delay)
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

    # Load existing results if they exist to avoid duplicate evaluations
    existing_df = None
    evaluated_questions = set()
    if os.path.exists(RESULTS_PATH):
        try:
            existing_df = pd.read_csv(RESULTS_PATH)
            if "user_input" in existing_df.columns:
                # Filter out rows that are no longer in the active questions set
                existing_df = existing_df[existing_df["user_input"].isin(question_set)]
                evaluated_questions = set(existing_df["user_input"].dropna().tolist())
        except Exception as e:
            print(f"Warning: Could not read existing results from {RESULTS_PATH}: {e}")

    rows_to_evaluate = [r for r in rows if r["question"] not in evaluated_questions]

    if not rows_to_evaluate:
        print("All currently generated questions have already been scored by RAGAs.")
        df = existing_df
    else:
        print(f"Scoring {len(rows_to_evaluate)} new/unscored questions with RAGAs...")
        dataset = Dataset.from_list(rows_to_evaluate)

        # Initialize RAGAs judge LLM (using Gemini to prevent generator bias)
        judge_llm = LangchainLLMWrapper(
            PacedChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY")),
            is_finished_parser=gemini_is_finished
        )
        judge_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        )

        # Use reference-free context_utilization metric
        context_utilization = ContextUtilization()

        run_config = RunConfig(
            max_workers=args.scoring_workers,
            max_retries=10,
            max_wait=60
        )

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_utilization],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
        )

        new_df = scores.to_pandas()
        if existing_df is not None:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df

        df.to_csv(RESULTS_PATH, index=False)
        print(f"Saved per-question scores to {RESULTS_PATH}")

    print("\n=== Aggregate scores ===")
    for metric in ["faithfulness", "answer_relevancy", "context_utilization"]:
        if df is not None and metric in df.columns:
            print(f"{metric}: {df[metric].mean():.3f}")


if __name__ == "__main__":
    main()