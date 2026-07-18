# PatchContext

PatchContext is a Retrieval-Augmented Generation (RAG) pipeline built to analyze the FastAPI repository's development history. It indexes commit history, merged pull requests, and issue discussions to answer design-rationale questions (e.g., "Why was this designed this way?") with grounding in actual developer conversations and clickable citations.

**Live Deployment:** [patchcontext.streamlit.app](https://patchcontext-rag-pbhrwzsrfengw8gbmcbaz7.streamlit.app/)

## Features

- Retrieves evidence from FastAPI commits, pull requests, and issue discussions.
- Supports both current and historical repository ingestion for design-rationale analysis.
- Uses FAISS with MMR retrieval for diverse and relevant context selection.
- Generates grounded answers using Groq's Llama 3.3 70B model.
- Applies a three-stage hallucination guard before returning responses.
- Provides clickable citations linking back to GitHub resources.
- Includes a Streamlit interface for interactive querying.
- Supports resumable RAGAs benchmarking with incremental checkpoints.

## Architecture

```
GitHub REST API (commits, PRs, issues)
   ├── ingest_github.py      ──►  data/*.json
   └── ingest_historical.py  ──►  data/historical/*.json
         │
         ▼
   build_index.py         chunk (LangChain RecursiveCharacterTextSplitter)
                          embed (BAAI/bge-small-en-v1.5, local & free)
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
                         Groq (the actual system), saved incrementally
                         to survive rate limits/multi-day runs, then scores
                         with RAGAs (faithfulness, answer_relevancy,
                         context_utilization) using Gemini as an independent
                         judge model (see "Gemini as Judge" below)
```

## Model Configurations & Substitutions

To keep the pipeline local, highly performant, and free of paid API key requirements, the following model substitutions are used:

| Original Spec | Substitution | Rationale |
|---|---|---|
| OpenAI `text-embedding-ada-002` | `BAAI/bge-small-en-v1.5` | Run locally via `sentence-transformers`. Free, fast, and outperforms `ada-002` on public retrieval benchmarks. |
| OpenAI `gpt-4o-mini` | Groq `llama-3.3-70b-versatile` | High-speed inference via free-tier Groq API, comparable reasoning and structuring quality. |

All orchestrations (LangChain), vector storage (FAISS), MMR retrieval, the NLI-based hallucination guard, and RAGAs evaluation parameters remain identical to production specifications.

### Gemini as the RAGAs Judge
While the primary pipeline runs on Groq, RAGAs evaluation uses a separate free model, **Google Gemini 3.1 Flash Lite**, as the evaluation judge. This is a deliberate design choice:
1. **Methodological Bias Prevention**: Using the same model to generate answers and then judge those same answers introduces self-preference bias. Using an independent model (Gemini) ensures objective scoring.
2. **Rate Limit Preservation**: RAGAs decomposites answers into multiple atomic claims and evaluates them individually. This triggers many API requests. Running them on a separate, dedicated Gemini quota protects the Groq API limits.

### Reference-Free Metric: `context_utilization`
RAGAs' default `context_precision` requires hand-written human reference answers for all benchmark questions. In the absence of manual references, PatchContext uses `context_utilization` (the reference-free equivalent), which evaluates how effectively the model uses the retrieved context chunks directly from the question and context themselves.

## Setup

1. **Clone FastAPI is not needed**: Data is fetched directly via the GitHub API.

2. **GitHub API Token**:
   * Generate a fine-grained token at `github.com → Settings → Developer settings → Personal access tokens`. No special scopes are required (only public repo access).
   * This increases the GitHub API rate limit from 60/hr to 5000/hr.

3. **Groq API Key**:
   * Create a free API key at [console.groq.com](https://console.groq.com).

4. **Google Gemini API Key**:
   * Create a free API key at [aistudio.google.com](https://aistudio.google.com).

5. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Open .env and populate GITHUB_TOKEN, GROQ_API_KEY, and GOOGLE_API_KEY
   ```

6. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Run Order

Run the stages sequentially from the `src/` directory:

```bash
cd src

# Stage 1 (Optional): Build historical repository corpus
python ingest_historical.py

# Stage 2: Fetch current repository data
python ingest_github.py

# Stage 3: Chunk, embed, and build the FAISS index
python build_index.py

# Stage 4: Test retrieval from the command line
python rag_pipeline.py

# Stage 5: Launch the Streamlit web interface
streamlit run app.py

# Stage 6: Run the 50-question RAGAs benchmark
python evaluate_ragas.py
```

### Resumable Benchmarking
`evaluate_ragas.py` is built to be fully resumable and saves progress incrementally after **every** question to `data/pipeline_outputs.json`. If your runs are interrupted by daily token caps:
```bash
# Process only 5 new questions this run
python evaluate_ragas.py --limit 5

# Run RAGAs scoring early on an incomplete set of answers
python evaluate_ragas.py --score-partial
```

## Tuning Configurations

* `src/ingest_github.py`: Modify `MAX_COMMITS` (default 300), `MAX_PRS` (default 200), or `MAX_ISSUES` (default 200) to adjust corpus sizes.
* `src/rag_pipeline.py`: Modify `k`, `fetch_k`, and `lambda_mult` in `get_retriever()` to fine-tune MMR retrieval relevance vs. diversity. Modify `max_repair_attempts` to adjust self-correction loops.
* `src/hallucination_guard.py`: Modify `contradiction_threshold` to tune the sensitivity of the local BART NLI model.

## Limitations & Scope

* **Foundational Recency Bias**: The repository ingestion samples recently updated commits, PRs, and issues. Foundational architectural design choices made during FastAPI's creation (circa 2018-2019) fall outside this recency window. The pipeline correctly triggers an honest "not enough evidence" refusal rather than fabricating answers for these questions.
* **Pattern-Based Speculation Checks**: The speculation check searches for specific hedging phrase patterns rather than performing semantic reasoning analysis. Speculative claims stated with absolute confidence without hedge words will bypass this check.
* **Semantic Citation Mapping**: The symbolic citation guard verifies that a cited resource (e.g., `PR#1234`) exists in the retrieved context pool. It does not check if the content of that specific citation actually supports the claim it is mapped to.

## Benchmark & Evaluation Results

The pipeline was benchmarked over the 50-question set (`data/questions.json`) and scored using the RAGAs framework (with Google Gemini 3.1 Flash Lite as the independent judge).

The final aggregate scores across all 50 benchmark questions are:
* **Faithfulness**: `0.628`
* **Answer Relevancy**: `0.455`
* **Context Utilization**: `0.345`
