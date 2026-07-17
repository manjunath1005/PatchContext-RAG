# PatchContext

A RAG pipeline over the FastAPI repository's commit history, pull requests, and
issue threads. Ask "why was this designed this way?" and get answers grounded
in actual developer discussions, with clickable citations to commit SHAs, PR
numbers, and issue IDs.

## Architecture

```
GitHub REST API (commits, PRs, issues)
        │
        ▼
  ingest_github.py  ──►  data/*.json
        │
        ▼
  build_index.py         chunk (LangChain RecursiveCharacterTextSplitter)
        │                embed (BAAI/bge-small-en-v1.5, local & free)
        ▼
  index/  (FAISS vector store, saved to disk)
        │
        ▼
  rag_pipeline.py
     ├─ MMR retrieval (diverse top-k chunks from FAISS)
     ├─ Generation (Groq: llama-3.3-70b-versatile, free API)
     ├─ hallucination_guard.py — THREE independent checks:
     │    ├─ citation grounding check (every cited SHA/PR#/Issue# must be
     │    │    among the retrieved chunks' metadata)
     │    ├─ NLI entailment check (facebook/bart-large-mnli, local & free —
     │    │    true premise/hypothesis NLI: context = premise, answer = hypothesis)
     │    └─ speculation-language check (flags hedging phrases like "it can
     │         be inferred", "suggests that" — catches claims stated with
     │         unwarranted confidence even when citations are technically real)
     └─ bounded repair loop: if the guard flags a problem, the model gets
          ONE retry with the specific problem named, before falling back to
          an honest "not enough evidence" refusal
        │
        ▼
  app.py  (Streamlit UI — question box, answer, guard status, citations)

  evaluate_ragas.py  →  runs pipeline over data/questions.json (50 Qs) using
                         Groq (the actual system), checkpointed incrementally
                         to survive rate limits/multi-day runs, then scores
                         with RAGAs (faithfulness, answer_relevancy,
                         context_utilization) using Gemini as an INDEPENDENT
                         judge model (see "Why Gemini as judge" below)
```

## Why these substitutions (vs. the original paid spec)

| Original | Used here | Reasoning |
|---|---|---|
| OpenAI `text-embedding-ada-002` | `BAAI/bge-small-en-v1.5` (sentence-transformers) | Free, runs locally, outperforms ada-002 on most public retrieval benchmarks |
| `gpt-4o-mini` | Groq `llama-3.3-70b-versatile` | Free API tier, fast inference, comparable reasoning quality |

Everything else — LangChain orchestration, FAISS, MMR retrieval, the NLI
hallucination guard, Streamlit UI, and RAGAs evaluation — matches the
original spec.

### Why Gemini as the RAGAs judge (not Groq)

The pipeline itself (the thing being evaluated) runs entirely on Groq/Llama.
RAGAs scoring uses a **separate** free model, Google Gemini, as the judge.
This is a deliberate choice, not a limitation:

1. **Methodological** — using the same model to both generate and judge its
   own answers is a known bias in RAG evaluation (self-preference bias).
   Using a different judge model avoids that.
2. **Practical** — RAGAs' `faithfulness` metric decomposes each answer into
   individual claims and verifies each one with a separate LLM call, so
   scoring 50 questions makes many more calls than generating them. Keeping
   this on a separate quota (Gemini) protects Groq's daily token cap for
   what actually matters: the system's own answer generation.

### Why `context_utilization` instead of `context_precision`

RAGAs' default `context_precision` metric requires a manually-authored
ground-truth "reference" answer per question, which this project doesn't
have (no one hand-wrote 50 reference answers). `context_utilization` is the
reference-free variant of the same underlying metric — it judges retrieval
precision using only the question, retrieved context, and generated answer.

## Setup

1. **Clone FastAPI is not needed** — we pull history via the GitHub API, not a git clone.

2. **Get a GitHub token** (raises your rate limit from 60/hr to 5000/hr; not
   strictly required for small test runs):
   - Go to github.com → Settings → Developer settings → Personal access tokens
     → Fine-grained tokens → Generate new token
   - No special scopes needed (you're only reading a public repo)
   - Copy the token

3. **Get a free Groq API key** (powers the actual PatchContext system):
   - Go to console.groq.com → API Keys → Create API Key
   - Copy the key

4. **Get a free Google Gemini API key** (used ONLY as the RAGAs evaluation
   judge — see "Why Gemini as judge" above):
   - Go to aistudio.google.com → Get API key → Create API key
   - Copy the key

5. **Configure environment**:
   ```bash
   cp .env.example .env
   # edit .env and paste in GITHUB_TOKEN, GROQ_API_KEY, and GOOGLE_API_KEY
   ```

6. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Run order

```bash
cd src

# Stage 1: pull commits/PRs/issues from GitHub
python ingest_github.py

# Stage 2: chunk, embed, build FAISS index
python build_index.py

# Stage 3: try it from the command line
python rag_pipeline.py

# Stage 4: launch the UI
streamlit run app.py

# Stage 5: run the RAGAs benchmark (50 questions)
python evaluate_ragas.py
```

### Running the benchmark under a tight daily API quota

`evaluate_ragas.py` is resumable and checkpoints progress after **every**
question to `data/pipeline_outputs_checkpoint.json` — safe to stop and
restart across multiple days if you're limited by a daily token cap:

```bash
# Generate only 5 new questions this run (stay under a small remaining quota)
python evaluate_ragas.py --limit 5

# Re-run the same command later/tomorrow — it picks up where it left off
python evaluate_ragas.py --limit 5

# RAGAs scoring runs automatically once all 50 are generated. To score
# early with an incomplete set instead, pass:
python evaluate_ragas.py --score-partial
```

If `data/questions.json` is edited between runs, stale checkpoint entries
for questions no longer in the current set are detected and dropped
automatically.

## Tuning knobs

- `src/ingest_github.py`: `MAX_COMMITS` (default 300), `MAX_PRS` (200),
  `MAX_ISSUES` (200), `FETCH_COMMENTS_FOR_TOP_N` (200, i.e. all of them —
  comments carry most of the "why" discussion and the rate-limit cost of
  fetching all of them is trivial). Raise these for a richer corpus if you
  have time/rate-limit budget; these defaults were chosen to prioritize PR/
  issue discussion depth over raw commit count, since design rationale lives
  mostly in reviews and comments, not commit messages.
- `src/rag_pipeline.py`: `k`, `fetch_k`, `lambda_mult` in `get_retriever()`
  control MMR's relevance/diversity tradeoff. `max_repair_attempts` in
  `PatchContextPipeline.answer()` controls the guard's self-correction loop
  (default 1).
- `src/hallucination_guard.py`: `contradiction_threshold` controls how
  aggressively the NLI check flags answers; `SPECULATION_PATTERNS` controls
  which hedging phrases trigger the speculation check.
- `data/questions.json`: the 50-question benchmark — broad coverage across
  FastAPI subsystems, with a handful of questions deliberately about recent
  maintenance/process topics rather than founding-era design decisions (see
  "Known limitations" below for why).

## Known limitations (worth stating explicitly in the report)

- **Recency-sampled corpus can't answer founding-era design questions.**
  Ingestion samples the most *recently updated* commits/PRs/issues. FastAPI's
  foundational decisions (why Pydantic, why Starlette, why dependency
  injection) were made ~2018-2019 and are structurally outside a recency
  window at any reasonable sample size. The system correctly says "not
  enough evidence" for these rather than fabricating an answer — this is
  intentional guard behavior, not a retrieval failure.
- **The speculation check is phrase-pattern based, not true reasoning
  verification.** It catches the *language* of hedging (e.g. "it can be
  inferred"), not all speculation — a model could in principle state an
  unsupported claim confidently, without hedge words, and this check would
  miss it.
- **Citation misattribution isn't caught by any current check.** A citation
  can be real (i.e. it was actually retrieved) but still be used to support
  a claim it doesn't actually substantiate — e.g. citing a "proposal to
  improve X" as evidence for "why X was originally built." The grounding
  check only verifies a citation *exists* in the retrieved set, not that
  it's used correctly. This was observed directly during testing (see
  report/dev notes) and is a good discussion point on the guard's real scope
  vs. its limits.

## Notes for the report/viva

- The hallucination guard runs THREE independent checks, not two: a
  citation grounding check (symbolic — verifies referenced IDs were
  actually retrieved), an NLI entailment check (semantic — genuine
  premise/hypothesis comparison between the retrieved context and the
  answer), and a speculation-language check (catches hedged/inferential
  claims). They catch different failure modes and are worth discussing
  separately rather than as one "the guard" black box.
- The bounded repair loop is a real, observed-working feature: when the
  guard flags a problem, the model gets one retry with the exact problem
  named (e.g. "you cited PR#X which isn't in the context" or "you used
  inferential language"), and in testing this successfully turned a
  fabricated-citation answer into a correctly-hedged one. Good before/after
  material for the report.
