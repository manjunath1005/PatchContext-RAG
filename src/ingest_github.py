"""
PatchContext - Stage 1: Data Ingestion
"""

import os
import json
import time
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "fastapi/fastapi"
BASE_URL = f"https://api.github.com/repos/{REPO}"
HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

# ---- Tunable limits ----
MAX_COMMITS = 300
MAX_PRS = 200
MAX_ISSUES = 200
FETCH_COMMENTS_FOR_TOP_N = 200  # Fetch comments for all items

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def _get(url, params=None):
    """GET with rate-limit handling and retries."""
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - time.time(), 5)
                print(f"Rate limited. Sleeping {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"Request failed ({e}); retrying ({attempt + 1}/3)...")
            time.sleep(5)
    raise RuntimeError(f"Failed to GET {url} after retries: {last_error}")


def paginate(url, max_items, params=None):
    params = dict(params or {})
    params["per_page"] = 100
    page = 1
    items = []
    with tqdm(total=max_items, desc=url.split("/")[-1]) as pbar:
        while len(items) < max_items:
            params["page"] = page
            resp = _get(url, params=params)
            batch = resp.json()
            if not batch:
                break
            items.extend(batch)
            pbar.update(len(batch))
            page += 1
    return items[:max_items]


def fetch_commits():
    raw = paginate(f"{BASE_URL}/commits", MAX_COMMITS, params={"sha": "master"})
    commits = []
    for c in raw:
        commits.append({
            "sha": c["sha"],
            "short_sha": c["sha"][:7],
            "message": c["commit"]["message"],
            "author": (c["commit"]["author"] or {}).get("name"),
            "date": (c["commit"]["author"] or {}).get("date"),
            "url": c["html_url"],
        })
    return commits


def fetch_issue_comments(issue_number):
    try:
        resp = _get(f"{BASE_URL}/issues/{issue_number}/comments")
        return [c["body"] for c in resp.json() if c.get("body")]
    except Exception:
        return []


def fetch_pull_requests():
    raw = paginate(f"{BASE_URL}/pulls", MAX_PRS, params={"state": "closed", "sort": "updated", "direction": "desc"})
    prs = []
    for i, p in enumerate(raw):
        if not p.get("merged_at"):
            continue  # skip closed-but-unmerged PRs; we want accepted design decisions
        comments = fetch_issue_comments(p["number"]) if i < FETCH_COMMENTS_FOR_TOP_N else []
        prs.append({
            "number": p["number"],
            "title": p["title"],
            "body": p.get("body") or "",
            "comments": comments,
            "merged_at": p["merged_at"],
            "url": p["html_url"],
        })
    return prs


def fetch_issues():
    """Fetch issues using the Search API."""
    items = []
    page = 1
    with tqdm(total=MAX_ISSUES, desc="issues (search)") as pbar:
        while len(items) < MAX_ISSUES:
            resp = _get(
                "https://api.github.com/search/issues",
                params={
                    "q": f"repo:{REPO} type:issue",
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            batch = resp.json().get("items", [])
            if not batch:
                break
            items.extend(batch)
            pbar.update(len(batch))
            page += 1

    raw = items[:MAX_ISSUES]
    issues = []
    for i, item in enumerate(raw):
        comments = fetch_issue_comments(item["number"]) if i < FETCH_COMMENTS_FOR_TOP_N else []
        issues.append({
            "number": item["number"],
            "title": item["title"],
            "body": item.get("body") or "",
            "comments": comments,
            "state": item["state"],
            "url": item["html_url"],
        })
    return issues


def main():
    print("Fetching commits...")
    commits = fetch_commits()
    with open(os.path.join(OUT_DIR, "commits.json"), "w") as f:
        json.dump(commits, f, indent=2)
    print(f"Saved {len(commits)} commits.")

    print("Fetching pull requests (with comments for most recent ones)...")
    prs = fetch_pull_requests()
    with open(os.path.join(OUT_DIR, "prs.json"), "w") as f:
        json.dump(prs, f, indent=2)
    print(f"Saved {len(prs)} merged PRs.")

    print("Fetching issues (with comments for most recent ones)...")
    issues = fetch_issues()
    with open(os.path.join(OUT_DIR, "issues.json"), "w") as f:
        json.dump(issues, f, indent=2)
    print(f"Saved {len(issues)} issues.")


if __name__ == "__main__":
    main()