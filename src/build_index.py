"""
PatchContext - Stage 2: Chunk, embed, and index.
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
    comment_bodies = []
    for c in p.get("comments", []):
        if isinstance(c, dict) and "body" in c:
            comment_bodies.append(c["body"])
        elif isinstance(c, str):
            comment_bodies.append(c)
    comment_block = "\n\n".join(comment_bodies[:10])  # cap to keep chunks reasonable
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
    comment_bodies = []
    for c in i.get("comments", []):
        if isinstance(c, dict) and "body" in c:
            comment_bodies.append(c["body"])
        elif isinstance(c, str):
            comment_bodies.append(c)
    comment_block = "\n\n".join(comment_bodies[:10])
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

    # Deduplication sets
    primary_commit_shas = {c["sha"] for c in commits if "sha" in c}
    primary_pr_numbers = {p["number"] for p in prs if "number" in p}
    primary_issue_numbers = {i["number"] for i in issues if "number" in i}

    historical_commits_added = 0
    historical_prs_added = 0
    historical_issues_added = 0

    historical_dir = os.path.join(DATA_DIR, "historical")
    if os.path.exists(historical_dir):
        for fname in sorted(os.listdir(historical_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(historical_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Merge historical commits
                    for c in data.get("commits", []):
                        if c.get("sha") not in primary_commit_shas:
                            commits.append(c)
                            primary_commit_shas.add(c["sha"])
                            historical_commits_added += 1
                            
                    # Merge historical PRs
                    for p in data.get("prs", []):
                        if p.get("number") not in primary_pr_numbers:
                            prs.append(p)
                            primary_pr_numbers.add(p["number"])
                            historical_prs_added += 1
                            
                    # Merge historical issues
                    for i in data.get("issues", []):
                        if i.get("number") not in primary_issue_numbers:
                            issues.append(i)
                            primary_issue_numbers.add(i["number"])
                            historical_issues_added += 1
                except Exception as e:
                    print(f"Error loading historical file {fname}: {e}")

    docs = []
    docs += [commit_to_doc(c) for c in commits]
    docs += [pr_to_doc(p) for p in prs]
    docs += [issue_to_doc(i) for i in issues]
    print(f"Built {len(docs)} raw documents "
          f"({len(commits)} commits, {len(prs)} PRs, {len(issues)} issues).")
    print(f"Added {historical_commits_added} historical commits, "
          f"{historical_prs_added} historical PRs, and "
          f"{historical_issues_added} historical issues.")
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

    # Calculate index size on disk
    index_size = 0
    if os.path.exists(INDEX_DIR):
        for fname in os.listdir(INDEX_DIR):
            fpath = os.path.join(INDEX_DIR, fname)
            if os.path.isfile(fpath):
                index_size += os.path.getsize(fpath)
    print(f"Actual FAISS index size on disk: {index_size / (1024 * 1024):.2f} MB ({index_size} bytes)")


if __name__ == "__main__":
    main()