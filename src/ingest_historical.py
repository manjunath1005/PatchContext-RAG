"""
PatchContext - Stage 1b: Historical Data Ingestion
"""

import os
import json
import time
import requests
import re
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "fastapi/fastapi"
BASE_URL = f"https://api.github.com/repos/{REPO}"
HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

# Eras definition
ERAS = [
    {
        "id": "foundation",
        "name": "Foundation (Repository creation -> 2018)",
        "commits_since": None,
        "commits_until": "2018-12-31T23:59:59Z",
        "search_suffix": "<=2018-12-31",
    },
    {
        "id": "early_adoption",
        "name": "Early Adoption (2019)",
        "commits_since": "2019-01-01T00:00:00Z",
        "commits_until": "2019-12-31T23:59:59Z",
        "search_suffix": "2019-01-01..2019-12-31",
    },
    {
        "id": "stabilization",
        "name": "Stabilization (2020)",
        "commits_since": "2020-01-01T00:00:00Z",
        "commits_until": "2020-12-31T23:59:59Z",
        "search_suffix": "2020-01-01..2020-12-31",
    },
    {
        "id": "expansion",
        "name": "Expansion (2021-2022)",
        "commits_since": "2021-01-01T00:00:00Z",
        "commits_until": "2022-12-31T23:59:59Z",
        "search_suffix": "2021-01-01..2022-12-31",
    },
    {
        "id": "modernization",
        "name": "Modernization (2023-2024)",
        "commits_since": "2023-01-01T00:00:00Z",
        "commits_until": "2024-12-31T23:59:59Z",
        "search_suffix": "2023-01-01..2024-12-31",
    },
    {
        "id": "current",
        "name": "Current (2025 -> Today)",
        "commits_since": "2025-01-01T00:00:00Z",
        "commits_until": None,
        "search_suffix": ">=2025-01-01",
    },
]

TARGET_PER_ERA = 100

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical")
PRIMARY_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Keep track of globally unique URLs to avoid duplicate indexing
seen_urls = set()


def _get(url, params=None):
    """GET request with robust rate-limit handling and retries."""
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            
            # Rate limit check (GitHub REST & Search APIs)
            if resp.status_code in (403, 429):
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - time.time(), 5)
                print(f"\nRate limit reached (status {resp.status_code}). Sleeping {wait:.0f}s...")
                time.sleep(wait)
                continue
                
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"\nRequest failed ({e}); retrying ({attempt + 1}/3)...")
            time.sleep(5)
    raise RuntimeError(f"Failed to GET {url} after retries: {last_error}")


def load_existing_corpus_identifiers():
    """Loads identifiers from commits.json, prs.json, issues.json to avoid duplicates."""
    existing_commits = set()
    existing_prs = set()
    existing_issues = set()
    
    def load_json(name):
        path = os.path.join(PRIMARY_DATA_DIR, name)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: could not load primary file {name}: {e}")
        return []

    for c in load_json("commits.json"):
        if "sha" in c:
            existing_commits.add(c["sha"])
            seen_urls.add(c["url"])
            
    for p in load_json("prs.json"):
        if "number" in p:
            existing_prs.add(p["number"])
            seen_urls.add(p["url"])
            
    for i in load_json("issues.json"):
        if "number" in i:
            existing_issues.add(i["number"])
            seen_urls.add(i["url"])

    return existing_commits, existing_prs, existing_issues


def evaluate_quality_score(title_or_msg, body_or_desc="", author=""):
    """Calculates a quality score based on keyword rules and bot checks."""
    title_lower = title_or_msg.lower()
    body_lower = (body_or_desc or "").lower()
    author_lower = (author or "").lower()
    
    # 1. Bot check - automatically disqualified
    if any(bot in author_lower for bot in ["bot", "dependabot", "github-actions"]):
        return -100, False
        
    score = 0
    
    # Rewards: strongly reward architectural discussions
    rewards = [
        "architecture", "design", "rationale", "proposal", "feature",
        "dependency injection", "depends", "apirouter", "routing", "validation",
        "response model", "request model", "openapi", "middleware", "lifespan",
        "startup", "shutdown", "oauth2", "security", "websockets", "serialization",
        "jsonable_encoder", "backward compatibility", "breaking change", "performance"
    ]
    
    # Penalties: strongly penalize noise
    penalties = [
        "typo", "docs", "readme", "changelog", "release notes", "formatting",
        "whitespace", "ci", "github actions", "github-actions", "dependabot",
        "dependency bumps", "bump ", "translation", "translate", "trivial maintenance",
        "bot commit"
    ]
    
    # Reward evaluation
    for reward in rewards:
        if reward in title_lower:
            score += 15
        if reward in body_lower:
            score += 5
            
    # Penalty evaluation
    for penalty in penalties:
        if penalty in title_lower:
            score -= 20
        if penalty in body_lower:
            score -= 10
            
    # Prefix blacklists are automatic high penalty
    prefix_blacklist = ["docs:", "chore:", "style:", "ci:", "build(deps):", "build(dependencies):"]
    for prefix in prefix_blacklist:
        if title_lower.startswith(prefix):
            score -= 50
            
    # Check body for obvious documentation templates
    if "this is a documentation-only change" in body_lower or "documentation issue" in body_lower:
        score -= 50

    return score, score > 0


def is_high_quality(title_or_msg, body_or_desc="", author=""):
    """Determines if a commit, PR, or issue is of high quality using scoring."""
    score, passed = evaluate_quality_score(title_or_msg, body_or_desc, author)
    return passed


def is_descriptive_merge_commit(msg):
    """
    Checks if a merge commit message contains a descriptive summary
    instead of just standard merge headers.
    """
    msg_clean = msg.strip()
    lines = [line.strip() for line in msg_clean.splitlines() if line.strip()]
    if not lines:
        return False
    first_line = lines[0]
    is_merge = first_line.startswith("Merge pull request #") or first_line.startswith("Merge branch")
    if not is_merge:
        return True  # Not a merge commit, so it doesn't fail this check
    # Descriptive merge commits will have a custom summary body
    return len(lines) > 1


def fetch_issue_comments(issue_number):
    """Fetches comments for issues/PRs with rich metadata."""
    try:
        # Pause slightly to respect REST rate limit
        time.sleep(0.02)
        resp = _get(f"{BASE_URL}/issues/{issue_number}/comments")
        comments = []
        for c in resp.json():
            if c.get("body"):
                comments.append({
                    "author": (c.get("user") or {}).get("login") or "",
                    "created_at": c.get("created_at"),
                    "body": c["body"]
                })
        return comments
    except Exception as e:
        print(f"\nWarning: could not fetch comments for #{issue_number}: {e}")
        return []


def ingest_era(era, existing_commits, existing_prs, existing_issues):
    """Ingests commits, PRs, and issues for a specific era."""
    print(f"\nProcessing era: {era['name']}")
    
    era_commits = []
    era_prs = []
    era_issues = []
    
    stats = {
        "commits_scanned": 0, "commits_kept": 0, "commits_discarded": 0,
        "prs_scanned": 0, "prs_kept": 0, "prs_discarded": 0,
        "issues_scanned": 0, "issues_kept": 0, "issues_discarded": 0
    }
    
    # 1. Commits Ingestion
    print("Fetching historical commits...")
    params = {"per_page": 100}  # Fetch default branch
    if era["commits_since"]:
        params["since"] = era["commits_since"]
    if era["commits_until"]:
        params["until"] = era["commits_until"]
        
    page = 1
    consecutive_empty_pages = 0
    while len(era_commits) < TARGET_PER_ERA:
        params["page"] = page
        try:
            resp = _get(f"{BASE_URL}/commits", params=params)
            batch = resp.json()
        except Exception as e:
            print(f"Error fetching commits page {page}: {e}")
            break
            
        if not batch:
            break
            
        kept_in_batch = 0
        for c in batch:
            stats["commits_scanned"] += 1
            sha = c["sha"]
            msg = c["commit"]["message"]
            author_name = (c["commit"]["author"] or {}).get("name") or ""
            url = c["html_url"]
            
            # Check existing, unique URL
            if sha in existing_commits or url in seen_urls:
                stats["commits_discarded"] += 1
                continue
                
            # Filter noise
            if not is_high_quality(msg, author=author_name):
                stats["commits_discarded"] += 1
                continue
                
            # Check merge commit summary quality
            if not is_descriptive_merge_commit(msg):
                stats["commits_discarded"] += 1
                continue
                
            # Keep
            era_commits.append({
                "sha": sha,
                "short_sha": sha[:7],
                "message": msg,
                "author": author_name,
                "date": (c["commit"]["author"] or {}).get("date"),
                "url": url,
                "created_at": (c["commit"]["author"] or {}).get("date"),
            })
            seen_urls.add(url)
            stats["commits_kept"] += 1
            kept_in_batch += 1
            
            if len(era_commits) >= TARGET_PER_ERA:
                break
                
        if kept_in_batch == 0:
            consecutive_empty_pages += 1
        else:
            consecutive_empty_pages = 0
            
        if consecutive_empty_pages >= 5:
            print(f"Early stopping commits: 5 consecutive pages produced 0 high-quality commits.")
            break
            
        page += 1
        # Prevent hammering API too fast
        time.sleep(0.5)

    # 2. Pull Requests Ingestion
    print("Fetching historical pull requests...")
    page = 1
    consecutive_empty_pages = 0
    while len(era_prs) < TARGET_PER_ERA:
        q = f"repo:{REPO} type:pr is:merged merged:{era['search_suffix']}"
        try:
            # Respect search API rate limits
            time.sleep(2.0)
            resp = _get("https://api.github.com/search/issues", params={
                "q": q, "per_page": 100, "page": page
            })
            batch = resp.json().get("items", [])
        except Exception as e:
            print(f"Error fetching PRs page {page}: {e}")
            break
            
        if not batch:
            break
            
        kept_in_batch = 0
        for item in batch:
            stats["prs_scanned"] += 1
            num = item["number"]
            url = item["html_url"]
            
            if num in existing_prs or url in seen_urls:
                stats["prs_discarded"] += 1
                continue
                
            title = item["title"]
            body = item.get("body") or ""
            author_login = (item.get("user") or {}).get("login") or ""
            
            if not is_high_quality(title, body, author_login):
                stats["prs_discarded"] += 1
                continue
                
            # Keep PR: fetch comments first
            comments = fetch_issue_comments(num)
            
            era_prs.append({
                "number": num,
                "title": title,
                "body": body,
                "comments": comments,
                "merged_at": item.get("closed_at"),  # merged is closed for merged:PRs
                "url": url,
                "user": author_login,
                "labels": [lbl["name"] for lbl in item.get("labels", [])],
                "comments_count": item.get("comments", 0),
                "created_at": item.get("created_at"),
                "closed_at": item.get("closed_at"),
            })
            seen_urls.add(url)
            stats["prs_kept"] += 1
            kept_in_batch += 1
            
            if len(era_prs) >= TARGET_PER_ERA:
                break
                
        if kept_in_batch == 0:
            consecutive_empty_pages += 1
        else:
            consecutive_empty_pages = 0
            
        if consecutive_empty_pages >= 5:
            print(f"Early stopping PRs: 5 consecutive pages produced 0 high-quality PRs.")
            break
            
        page += 1

    # 3. Issues Ingestion
    print("Fetching historical issues...")
    page = 1
    consecutive_empty_pages = 0
    while len(era_issues) < TARGET_PER_ERA:
        q = f"repo:{REPO} type:issue created:{era['search_suffix']}"
        try:
            time.sleep(2.0)
            resp = _get("https://api.github.com/search/issues", params={
                "q": q, "per_page": 100, "page": page
            })
            batch = resp.json().get("items", [])
        except Exception as e:
            print(f"Error fetching issues page {page}: {e}")
            break
            
        if not batch:
            break
            
        kept_in_batch = 0
        for item in batch:
            stats["issues_scanned"] += 1
            num = item["number"]
            url = item["html_url"]
            
            if num in existing_issues or url in seen_urls:
                stats["issues_discarded"] += 1
                continue
                
            title = item["title"]
            body = item.get("body") or ""
            author_login = (item.get("user") or {}).get("login") or ""
            
            if not is_high_quality(title, body, author_login):
                stats["issues_discarded"] += 1
                continue
                
            # Keep Issue: fetch comments first
            comments = fetch_issue_comments(num)
            
            era_issues.append({
                "number": num,
                "title": title,
                "body": body,
                "comments": comments,
                "state": item.get("state"),
                "url": url,
                "user": author_login,
                "labels": [lbl["name"] for lbl in item.get("labels", [])],
                "comments_count": item.get("comments", 0),
                "created_at": item.get("created_at"),
                "closed_at": item.get("closed_at"),
            })
            seen_urls.add(url)
            stats["issues_kept"] += 1
            kept_in_batch += 1
            
            if len(era_issues) >= TARGET_PER_ERA:
                break
                
        if kept_in_batch == 0:
            consecutive_empty_pages += 1
        else:
            consecutive_empty_pages = 0
            
        if consecutive_empty_pages >= 5:
            print(f"Early stopping issues: 5 consecutive pages produced 0 high-quality issues.")
            break
            
        page += 1

    # Print reporting for this timeline
    print("\n================================================")
    print(f"{era['name']}")
    print("================================================")
    print(f"Commits scanned: {stats['commits_scanned']}")
    print(f"Commits kept: {stats['commits_kept']}")
    print(f"Commits discarded: {stats['commits_discarded']}")
    print()
    print(f"PRs scanned: {stats['prs_scanned']}")
    print(f"PRs kept: {stats['prs_kept']}")
    print(f"PRs discarded: {stats['prs_discarded']}")
    print()
    print(f"Issues scanned: {stats['issues_scanned']}")
    print(f"Issues kept: {stats['issues_kept']}")
    print(f"Issues discarded: {stats['issues_discarded']}")
    print("================================================\n")

    return {
        "commits": era_commits,
        "prs": era_prs,
        "issues": era_issues,
        "stats": stats
    }


def main():
    print("Loading primary corpus identifiers for deduplication...")
    existing_commits, existing_prs, existing_issues = load_existing_corpus_identifiers()
    print(f"Loaded: {len(existing_commits)} existing commits, {len(existing_prs)} PRs, {len(existing_issues)} issues.")
    
    total_stats = {
        "commits_scanned": 0, "commits_indexed": 0,
        "prs_scanned": 0, "prs_indexed": 0,
        "issues_scanned": 0, "issues_indexed": 0,
        "historical_docs_added": 0
    }
    
    for era in ERAS:
        era_file = os.path.join(OUT_DIR, f"{era['id']}.json")
        
        # Check if already processed to support resuming
        if os.path.exists(era_file):
            print(f"\nEra '{era['name']}' already processed. Loading from disk...")
            try:
                with open(era_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Update global seen_urls and deduplication sets so later eras don't duplicate
                for c in data.get("commits", []):
                    seen_urls.add(c["url"])
                    if "sha" in c:
                        existing_commits.add(c["sha"])
                for p in data.get("prs", []):
                    seen_urls.add(p["url"])
                    if "number" in p:
                        existing_prs.add(p["number"])
                for i in data.get("issues", []):
                    seen_urls.add(i["url"])
                    if "number" in i:
                        existing_issues.add(i["number"])
                    
                # Update total counts
                total_stats["commits_indexed"] += len(data.get("commits", []))
                total_stats["prs_indexed"] += len(data.get("prs", []))
                total_stats["issues_indexed"] += len(data.get("issues", []))
                continue
            except Exception as e:
                print(f"Warning: failed to load existing era file {era_file}: {e}. Reprocessing...")

        # Process new era
        result = ingest_era(era, existing_commits, existing_prs, existing_issues)
        
        # Save immediately to disk
        try:
            with open(era_file, "w", encoding="utf-8") as f:
                json.dump({
                    "commits": result["commits"],
                    "prs": result["prs"],
                    "issues": result["issues"]
                }, f, indent=2)
            print(f"Saved {era['name']} data to {era_file}")
        except Exception as e:
            print(f"Error saving {era_file}: {e}")
            
        # Update total stats
        estats = result["stats"]
        total_stats["commits_scanned"] += estats["commits_scanned"]
        total_stats["commits_indexed"] += len(result["commits"])
        
        total_stats["prs_scanned"] += estats["prs_scanned"]
        total_stats["prs_indexed"] += len(result["prs"])
        
        total_stats["issues_scanned"] += estats["issues_scanned"]
        total_stats["issues_indexed"] += len(result["issues"])
        
    # Calculate corpus increase in bytes
    total_bytes = 0
    for era in ERAS:
        era_file = os.path.join(OUT_DIR, f"{era['id']}.json")
        if os.path.exists(era_file):
            total_bytes += os.path.getsize(era_file)
            
    total_docs = total_stats["commits_indexed"] + total_stats["prs_indexed"] + total_stats["issues_indexed"]
    
    print("\n================================================")
    print("FINAL SUMMARY REPORT")
    print("================================================")
    print(f"Total commits scanned: {total_stats['commits_scanned']}")
    print(f"Total commits indexed: {total_stats['commits_indexed']}")
    print(f"Total PRs scanned: {total_stats['prs_scanned']}")
    print(f"Total PRs indexed: {total_stats['prs_indexed']}")
    print(f"Total issues scanned: {total_stats['issues_scanned']}")
    print(f"Total issues indexed: {total_stats['issues_indexed']}")
    print(f"Total historical documents added: {total_docs}")
    print(f"Estimated increase in corpus size: {total_bytes / 1024:.2f} KB ({total_bytes} bytes)")
    print("Estimated increase in FAISS index size: Rebuild index via build_index.py to check actual size.")
    print("================================================\n")


if __name__ == "__main__":
    main()
