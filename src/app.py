"""
PatchContext - Streamlit UI
Run with: streamlit run src/app.py
"""

import json
import os
import streamlit as st
from rag_pipeline import PatchContextPipeline

st.set_page_config(page_title="PatchContext", page_icon="🔍", layout="wide")

# Global CSS styles for badges, cards, and text layout
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    .pc-subtitle { color: #6b7280; font-size: 1.02rem; margin-top: -0.6rem; }
    .pc-badge {
        display: inline-block; padding: 3px 11px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-right: 6px; white-space: nowrap;
    }
    .pc-badge-pass   { background: #dcfce7; color: #15803d; }
    .pc-badge-repair { background: #dbeafe; color: #1d4ed8; }
    .pc-badge-fail   { background: #fee2e2; color: #b91c1c; }
    .pc-badge-type   { background: #f3f4f6; color: #374151; }
    .pc-source-title { font-weight: 600; font-size: 0.95rem; }
    .pc-footer { color: #9ca3af; font-size: 0.8rem; margin-top: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

SOURCE_ICON = {"commit": "🔨", "pull_request": "🔀", "issue": "💬"}

EXAMPLE_QUESTIONS = [
    "Why does FastAPI support WebSockets as a first-class feature?",
    "Why does FastAPI include security utilities like OAuth2PasswordBearer out of the box?",
    "Why did FastAPI add support for lifespan events instead of only startup/shutdown events?",
    "Why does FastAPI support dependency overrides for testing?",
]


# Main header
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.title("🔍 PatchContext")
    st.markdown(
        '<div class="pc-subtitle">Ask <i>why</i> the FastAPI codebase was designed a '
        "certain way — answers grounded in real commits, PRs, and issue threads, "
        "with citations.</div>",
        unsafe_allow_html=True,
    )
with col_meta:
    st.markdown(
        '<div style="text-align:right; padding-top:0.6rem;">'
        '<span class="pc-badge pc-badge-type">Llama 3.3 70B · Groq</span><br><br>'
        '<span class="pc-badge pc-badge-type">Gemini-judged eval</span>'
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()


# Sidebar statistics and pipeline summary
@st.cache_data
def get_corpus_stats():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    commits, prs, issues = [], [], []
    for name, lst in [("commits.json", commits), ("prs.json", prs), ("issues.json", issues)]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lst.extend(json.load(f))
            except Exception:
                pass

    primary_commit_shas = {c["sha"] for c in commits if "sha" in c}
    primary_pr_numbers = {p["number"] for p in prs if "number" in p}
    primary_issue_numbers = {i["number"] for i in issues if "number" in i}

    historical_dir = os.path.join(data_dir, "historical")
    if os.path.exists(historical_dir):
        for fname in sorted(os.listdir(historical_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(historical_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for c in data.get("commits", []):
                        if c.get("sha") not in primary_commit_shas:
                            commits.append(c)
                            primary_commit_shas.add(c["sha"])
                    for p in data.get("prs", []):
                        if p.get("number") not in primary_pr_numbers:
                            prs.append(p)
                            primary_pr_numbers.add(p["number"])
                    for i in data.get("issues", []):
                        if i.get("number") not in primary_issue_numbers:
                            issues.append(i)
                            primary_issue_numbers.add(i["number"])
                except Exception:
                    pass

    return {
        "commits": len(commits) if commits else None,
        "PRs": len(prs) if prs else None,
        "issues": len(issues) if issues else None,
    }


with st.sidebar:
    st.markdown("### About PatchContext")
    st.markdown(
        "A RAG system over FastAPI's development history — commits, merged "
        "PRs, and issues — built to answer *design rationale* questions with "
        "citations, not just documentation lookups."
    )

    st.markdown("### Corpus")
    stats = get_corpus_stats()
    for label, key in [("Commits", "commits"), ("Merged PRs", "PRs"), ("Issues", "issues")]:
        if stats.get(key) is not None:
            st.markdown(f"**{stats[key]}** {label.lower()}")

    st.markdown("### Pipeline")
    st.markdown(
        "- **Retrieval:** FAISS + MMR (BAAI/bge-small-en-v1.5)\n"
        "- **Generation:** Groq — Llama 3.3 70B\n"
        "- **Guard:** citation grounding + NLI entailment "
        "(bart-large-mnli) + speculation detection\n"
        "- **Self-correction:** 1 bounded repair attempt on flagged answers\n"
        "- **Eval judge:** Gemini 3.1 Flash Lite (kept separate from the "
        "generator to avoid self-preference bias)"
    )


# Load vector index and models
@st.cache_resource(show_spinner="Loading index and models (first run only)...")
def get_pipeline():
    try:
        return PatchContextPipeline()
    except FileNotFoundError:
        st.error(
            "Index not found. Run `python build_index.py` (after `python ingest_github.py`) "
            "from the src/ folder before starting the app."
        )
        st.stop()
    except Exception as e:
        st.error(
            f"Failed to initialize the pipeline: {e}\n\n"
            "Check that GROQ_API_KEY is set in your .env file (or Streamlit secrets, if deployed)."
        )
        st.stop()


pipeline = get_pipeline()


# Example question chips and text input
if "pc_question" not in st.session_state:
    st.session_state.pc_question = ""

st.markdown("**Try an example, or ask your own:**")
chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, eq in zip(chip_cols, EXAMPLE_QUESTIONS):
    short_label = eq if len(eq) <= 42 else eq[:39] + "..."
    if col.button(short_label, key=f"chip_{eq}", use_container_width=True):
        st.session_state.pc_question = eq

question = st.text_input(
    "Your question",
    key="pc_question",
    placeholder="e.g. Why does FastAPI use dependency injection instead of middleware for auth?",
    label_visibility="collapsed",
)

ask_clicked = st.button("🔍 Ask", type="primary")


# Generate and display answers
if ask_clicked and question:
    try:
        with st.spinner("Retrieving relevant history and generating a grounded answer..."):
            result = pipeline.answer(question)
    except Exception as e:
        st.error(
            f"Something went wrong generating an answer: {e}\n\n"
            "This is usually a temporary API issue (e.g. rate limit) — try again in a moment."
        )
        st.stop()

    guard = result["guard"]
    first_guard = result["attempt_history"][0]["guard"]

    # Display execution stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sources retrieved", len(result["retrieved_docs"]))
    m2.metric("Citations used", len(result["citations"]))
    m3.metric("Repair attempts", result["repair_attempts"])
    m4.metric("Guard status", "Safe" if guard["is_safe"] or guard["grounding_passed"] else "Flagged")

    # Show warning or success banner based on guard result
    if not result["guard_intervened"] and guard["is_safe"]:
        st.success(
            "✅ **Hallucination guard: passed on first try** — citations grounded, "
            "no contradiction detected, no speculative language."
        )
    elif result["guard_intervened"]:
        problems = []
        if not first_guard["grounding_passed"]:
            problems.append("a fabricated citation")
        if first_guard["speculation"]["flagged"]:
            problems.append("speculative/hedging language")
        problem_text = " and ".join(problems) if problems else "an issue"

        if guard["grounding_passed"]:
            st.info(
                f"🔧 **Hallucination guard caught {problem_text}** in the original draft — "
                f"the model self-corrected after {result['repair_attempts']} repair "
                f"attempt(s). See *Guard details* below for the original draft."
            )
        else:
            st.warning(
                f"⚠️ **Hallucination guard caught {problem_text}** and the model could not "
                "fully self-correct — showing a safe fallback instead."
            )
    else:
        st.warning(
            "⚠️ **Hallucination guard flagged this answer** and the model could not "
            "self-correct — showing a safe fallback instead."
        )

    # Display the generated answer
    st.markdown("### Answer")
    with st.container(border=True):
        st.markdown(result["answer"])

    # Collapsible attempt-by-attempt guard logs
    with st.expander("🛡️ Guard details (per-attempt breakdown)"):
        for i, attempt in enumerate(result["attempt_history"]):
            label = "Original draft" if i == 0 else f"Repair attempt {i}"
            g = attempt["guard"]
            badge = (
                '<span class="pc-badge pc-badge-pass">grounded</span>'
                if g["grounding_passed"]
                else '<span class="pc-badge pc-badge-fail">fabricated citation</span>'
            )
            badge += (
                '<span class="pc-badge pc-badge-fail">speculative language</span>'
                if g["speculation"]["flagged"]
                else '<span class="pc-badge pc-badge-pass">no speculation</span>'
            )
            nli_class = "pc-badge-fail" if g["nli"]["flagged"] else "pc-badge-pass"
            badge += (
                f'<span class="pc-badge {nli_class}">NLI: {g["nli"]["top_label"]} '
                f'({g["nli"]["top_score"]:.2f})</span>'
            )
            st.markdown(f"**{label}**  {badge}", unsafe_allow_html=True)
            st.text(attempt["answer"])
            if g["unverified_citations"]:
                st.caption(f"Unverified citations: {', '.join(sorted(g['unverified_citations']))}")
            if i < len(result["attempt_history"]) - 1:
                st.divider()

    # Clickable source cards
    st.markdown("### Sources")
    if result["citations"]:
        src_cols = st.columns(min(len(result["citations"]), 3))
        for i, c in enumerate(result["citations"]):
            icon = SOURCE_ICON.get(c["type"], "📄")
            with src_cols[i % len(src_cols)]:
                with st.container(border=True):
                    st.markdown(f'<span class="pc-source-title">{icon} {c["ref_id"]}</span>', unsafe_allow_html=True)
                    st.caption(c["type"].replace("_", " ").title())
                    if c["url"]:
                        st.markdown(f"[Open on GitHub ↗]({c['url']})")
    else:
        st.caption("No citations were retrieved for this question.")

    # Collapsible raw document contexts
    with st.expander("📄 Raw retrieved context"):
        for d in result["retrieved_docs"]:
            icon = SOURCE_ICON.get(d.metadata.get("source_type"), "📄")
            st.markdown(f"**{icon} {d.metadata.get('ref_id')}**")
            st.text(d.page_content[:500])
            st.divider()

elif ask_clicked and not question:
    st.warning("Type a question first (or click one of the examples above).")

st.markdown(
    '<div class="pc-footer">PatchContext — a Retrieval-Augmented Generation (RAG) pipeline over the FastAPI repository.</div>',
    unsafe_allow_html=True,
)