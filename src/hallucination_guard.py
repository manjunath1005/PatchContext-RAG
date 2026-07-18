"""
PatchContext - NLI-based Hallucination Guard.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_NLI_MODEL = "facebook/bart-large-mnli"
_nli_tokenizer = None
_nli_model = None  # lazy-loaded, it's a ~1.6GB model


def _get_nli_model():
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model


SPECULATION_PATTERNS = [
    re.compile(r"\bit can be inferred\b", re.IGNORECASE),
    re.compile(r"\bcan be inferred\b", re.IGNORECASE),
    re.compile(r"\bsuggests? that\b", re.IGNORECASE),
    re.compile(r"\blikely\b", re.IGNORECASE),
    re.compile(r"\bmay have\b", re.IGNORECASE),
    re.compile(r"\bprobably\b", re.IGNORECASE),
    re.compile(r"\bpresum(ably|e)\b", re.IGNORECASE),
    re.compile(r"\bassum(ing|e)\b", re.IGNORECASE),
    re.compile(r"\bimplie(s|d)\b", re.IGNORECASE),
    re.compile(r"\bit(?:'s| is) reasonable to (?:assume|think)\b", re.IGNORECASE),
]


def speculation_check(answer_text):
    """
    Flags hedging or inferential language based on regex phrase patterns.
    """
    text_lower = answer_text.lower()
    matched = [p.pattern for p in SPECULATION_PATTERNS if p.search(text_lower)]
    return {"flagged": len(matched) > 0, "matched_patterns": matched}


CITATION_PATTERNS = [
    re.compile(r"\bPR\s*#\s*(\d+)", re.IGNORECASE),
    re.compile(r"\bIssue\s*#\s*(\d+)", re.IGNORECASE),
    re.compile(r"\b([0-9a-f]{7,40})\b"),  # commit sha-like tokens
]


def extract_citations(answer_text):
    found = set()
    for pattern in CITATION_PATTERNS:
        for m in pattern.findall(answer_text):
            found.add(m.lower())
    return found


def citation_grounding_check(answer_text, retrieved_docs):
    """Returns (passed: bool, unverified: set of citation strings)."""
    cited = extract_citations(answer_text)
    if not cited:
        return True, set()  # nothing to verify

    grounded_ids = set()
    for doc in retrieved_docs:
        meta = doc.metadata
        if "number" in meta:
            grounded_ids.add(str(meta["number"]))
        if "sha" in meta:
            grounded_ids.add(meta["sha"].lower())
            grounded_ids.add(meta["sha"][:7].lower())

    unverified = {c for c in cited if c not in grounded_ids}
    passed = len(unverified) == 0
    return passed, unverified


def nli_entailment_check(answer_text, context_text, contradiction_threshold=0.5):
    """
    Entailment check: premise = retrieved context, hypothesis = generated answer.
    """
    tokenizer, model = _get_nli_model()
    inputs = tokenizer(context_text, answer_text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    # Read label order from the model's own config rather than assuming a fixed
    # order, since that's the only fully robust way to interpret the logits.
    id2label = model.config.id2label
    scores = {id2label[i].lower(): probs[i].item() for i in range(probs.shape[0])}
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]

    flagged = (top_label == "contradiction" and top_score >= contradiction_threshold)
    return {
        "top_label": top_label,
        "top_score": top_score,
        "flagged": flagged,
        "full_result": scores,
    }


def run_guard(answer_text, retrieved_docs):
    """Combined guard. Returns a dict summarizing all checks."""
    grounding_passed, unverified = citation_grounding_check(answer_text, retrieved_docs)
    context_text = "\n\n".join(d.page_content for d in retrieved_docs)[:4000]  # truncate for NLI model limits
    nli_result = nli_entailment_check(answer_text, context_text)
    speculation_result = speculation_check(answer_text)

    is_safe = grounding_passed and not nli_result["flagged"] and not speculation_result["flagged"]
    return {
        "is_safe": is_safe,
        "grounding_passed": grounding_passed,
        "unverified_citations": unverified,
        "nli": nli_result,
        "speculation": speculation_result,
    }