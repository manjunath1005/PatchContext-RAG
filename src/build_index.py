"""
PatchContext - Stage 2: Chunk, embed, and index.

Turns commits.json / prs.json / issues.json into LangChain Documents with
citation metadata (sha / pr number / issue number + url), embeds them with
a free local sentence-transformers model (BAAI/bge-small-en-v1.5), and
builds a FAISS index saved to disk.
"""

import os
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_json(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run ingest_github.py first to generate the data files."
        )
    with open(path) as f:
        return json.load(f)


def commit_to_doc(c):
    text = f"Commit {c['short_sha']} by {c.get('author')}\n\n{c['message']}"
    meta = {
        "source_type": "commit",
        "ref_id": c["short_sha"],
        "sha": c["sha"],
        "url": c["url"],
        "date": c.get("date"),
    }
    return Document(page_content=text, metadata=meta)


def pr_to_doc(p):
    comment_block = "\n\n".join(p.get("comments", [])[:10])  # cap to keep chunks reasonable
    text = (
        f"Pull Request #{p['number']}: {p['title']}\n\n"
        f"{p.get('body','')}\n\n"
        f"--- Discussion ---\n{comment_block}"
    )
    meta = {
        "source_type": "pull_request",
        "ref_id": f"PR#{p['number']}",
        "number": p["number"],
        "url": p["url"],
        "date": p.get("merged_at"),
    }
    return Document(page_content=text, metadata=meta)


def issue_to_doc(i):
    comment_block = "\n\n".join(i.get("comments", [])[:10])
    text = (
        f"Issue #{i['number']}: {i['title']}\n\n"
        f"{i.get('body','')}\n\n"
        f"--- Discussion ---\n{comment_block}"
    )
    meta = {
        "source_type": "issue",
        "ref_id": f"Issue#{i['number']}",
        "number": i["number"],
        "url": i["url"],
    }
    return Document(page_content=text, metadata=meta)


def build_documents():
    commits = load_json("commits.json")
    prs = load_json("prs.json")
    issues = load_json("issues.json")

    docs = []
    docs += [commit_to_doc(c) for c in commits]
    docs += [pr_to_doc(p) for p in prs]
    docs += [issue_to_doc(i) for i in issues]
    print(f"Built {len(docs)} raw documents "
          f"({len(commits)} commits, {len(prs)} PRs, {len(issues)} issues).")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def main():
    docs = build_documents()
    chunks = chunk_documents(docs)

    print(f"Loading embedding model: {EMBED_MODEL} (runs locally, free, no API key)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Building FAISS index (this may take a while for large corpora)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}")


if __name__ == "__main__":
    main()