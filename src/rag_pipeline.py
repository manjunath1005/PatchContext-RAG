"""
PatchContext - Stage 3: RAG pipeline.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

from hallucination_guard import run_guard

load_dotenv()

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"  # free via Groq; comparable class to gpt-4o-mini

SYSTEM_PROMPT = """You are PatchContext, an assistant that explains WHY the FastAPI \
codebase was designed the way it was, using only the provided excerpts from real \
commits, pull requests, and issue threads.

Rules:
- Base your answer ONLY on the provided context. Do not use outside knowledge of FastAPI.
- Every claim must be backed by a specific citation using the exact ref_id shown for \
each excerpt (e.g. "PR#1234", "Issue#567", or a commit sha like "a1b2c3d").
- If the context does not contain enough information to answer, say so explicitly \
instead of guessing.
- Never invent a PR number, issue number, or commit sha that isn't shown in the context.
- If the retrieved context does not explicitly answer the question, say plainly that the \
indexed repository does not contain enough evidence to answer it. Do not infer, speculate, \
or fill gaps using outside knowledge of FastAPI, even if a related excerpt is present. Only \
state a claim as fact if the context explicitly supports it.
"""

USER_PROMPT = """Question: {question}

Context excerpts:
{context}

Answer the question, citing ref_ids inline (e.g. "...as discussed in PR#1234")."""

REPAIR_PROMPT = """Your previous answer has a problem that must be fixed: {problem_description}

Question: {question}

Context excerpts:
{context}

Your previous (flawed) answer was:
{previous_answer}

Rewrite the answer following these rules:
- Only cite ref_ids that literally appear in the context above (e.g. "PR#1234", "Issue#567", \
or a commit sha shown in brackets like [abc1234]).
- Do not use inferential or hedging language ("it can be inferred", "suggests that", "likely", \
"probably", "may have", "implies"). Either state a claim as fact because the context explicitly \
supports it, or say plainly that the context does not contain enough evidence — do not speculate \
in between."""


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def get_retriever(vectorstore, k=5, fetch_k=20, lambda_mult=0.5):
    """MMR retrieval for result diversity, as specified in the project brief."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def format_context(docs):
    blocks = []
    for d in docs:
        ref = d.metadata.get("ref_id", "unknown")
        blocks.append(f"[{ref}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def format_citations(docs):
    citations = []
    seen = set()
    for d in docs:
        ref = d.metadata.get("ref_id")
        url = d.metadata.get("url")
        if ref and ref not in seen:
            seen.add(ref)
            citations.append({"ref_id": ref, "url": url, "type": d.metadata.get("source_type")})
    return citations


class PatchContextPipeline:
    def __init__(self, k=5, fetch_k=20, lambda_mult=0.5):
        self.vectorstore = load_vectorstore()
        self.retriever = get_retriever(self.vectorstore, k, fetch_k, lambda_mult)
        self.llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=os.getenv("GROQ_API_KEY"))
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ])
        self.repair_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", REPAIR_PROMPT),
        ])

    def answer(self, question: str, max_repair_attempts: int = 1):
        docs = self.retriever.invoke(question)
        context = format_context(docs)
        citations = format_citations(docs)

        chain = self.prompt | self.llm
        response = chain.invoke({"question": question, "context": context})
        answer_text = response.content
        guard_result = run_guard(answer_text, docs)

        attempts = [{"answer": answer_text, "guard": guard_result}]

        # Bounded self-correction loop for repair attempts
        repair_chain = self.repair_prompt | self.llm
        attempt_count = 0

        def needs_repair(g):
            return (not g["grounding_passed"]) or g["speculation"]["flagged"]

        while needs_repair(guard_result) and attempt_count < max_repair_attempts:
            attempt_count += 1
            problems = []
            if not guard_result["grounding_passed"]:
                bad = ", ".join(sorted(guard_result["unverified_citations"]))
                problems.append(f"it cited reference(s) that do NOT appear anywhere in the context: {bad}")
            if guard_result["speculation"]["flagged"]:
                problems.append(
                    "it used inferential/hedging language (e.g. 'it can be inferred', 'suggests that') "
                    "to state a claim the context doesn't explicitly support"
                )
            problem_description = "; and ".join(problems)

            response = repair_chain.invoke({
                "question": question,
                "context": context,
                "previous_answer": answer_text,
                "problem_description": problem_description,
            })
            answer_text = response.content
            guard_result = run_guard(answer_text, docs)
            attempts.append({"answer": answer_text, "guard": guard_result})

        # Fallback response if grounding fails after self-correction
        final_safe = guard_result["grounding_passed"]
        if not final_safe:
            answer_text = (
                "I couldn't produce a fully grounded answer to this question — "
                "the model kept referencing sources that weren't actually retrieved "
                "from the corpus. Try rephrasing the question, or it may be that this "
                "topic isn't well covered by the current data sample."
            )

        return {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "guard": guard_result,
            "guard_intervened": attempt_count > 0,
            "repair_attempts": attempt_count,
            "attempt_history": attempts,
            "retrieved_docs": docs,
        }


if __name__ == "__main__":
    pipeline = PatchContextPipeline()
    result = pipeline.answer("Why does FastAPI use Pydantic for request validation?")
    print(result["answer"])
    print("\nCitations:", result["citations"])
    print("\nGuard intervened:", result["guard_intervened"], "| repair attempts:", result["repair_attempts"])
    print("Final grounding passed:", result["guard"]["grounding_passed"])