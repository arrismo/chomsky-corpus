from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datasets import Dataset
from huggingface_hub import HfApi


SOURCE_URLS = {
    "articles": "https://chomsky.info/articles/",
    "interviews": "https://chomsky.info/interviews/",
    "letters": "https://chomsky.info/letters/",
    "talks": "https://chomsky.info/talks/",
    "debates": "https://chomsky.info/debates/",
}

DATE_REGEX = re.compile(
    r"\b("
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:\s+\d{1,2},?)?\s*,?\s+\d{4}"
    r"|\d{4}"
    r")\b"
)
WHITESPACE_REGEX = re.compile(r"\s+")
REMOVABLE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "img",
    "header",
    "footer",
    "nav",
    "form",
    "button",
)

PUBLISH_DROP_FIELDS = {
    "url",
    "scraped_at",
    "pipeline_version",
    "source_site",
    "raw_date",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleaned Chomsky corpus dataset suitable for Hugging Face upload."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where raw and processed artifacts are written.",
    )
    parser.add_argument(
        "--pipeline-version",
        default="v1.0.0",
        help="Version string recorded in the dataset artifacts.",
    )
    parser.add_argument(
        "--min-content-length",
        type=int,
        default=200,
        help="Reject records with content shorter than this many characters.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="Optional dataset repository id to push to, for example username/chomsky-corpus.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the Hugging Face dataset repo as private.",
    )
    return parser.parse_args()


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; ChomskyCorpusDatasetBuilder/1.0; +https://huggingface.co/datasets)"
        }
    )
    return session


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_REGEX.sub(" ", text or "").strip()


def normalize_date(raw_date: str | None) -> str | None:
    if not raw_date:
        return None

    cleaned = raw_date.replace(",", "").strip()
    for date_format in ("%B %d %Y", "%B %Y", "%Y"):
        try:
            parsed = datetime.strptime(cleaned, date_format)
        except ValueError:
            continue

        if date_format == "%Y":
            return f"{parsed.year:04d}"
        if date_format == "%B %Y":
            return parsed.strftime("%Y-%m")
        return parsed.date().isoformat()

    return raw_date


def normalized_content_hash(text: str) -> str:
    normalized = normalize_whitespace(text).lower().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def stable_record_id(url: str, text: str) -> str:
    payload = f"{url}::{normalize_whitespace(text)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def is_valid_source_url(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return netloc == "chomsky.info" or netloc.endswith(".chomsky.info")


def extract_listing_entries(
    session: requests.Session, section_name: str, index_url: str, timeout: int
) -> list[dict]:
    response = session.get(index_url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    container = soup.select_one("#main_container") or soup

    entries: list[dict] = []
    seen_urls: set[str] = set()

    for item in container.select("li"):
        anchor = item.find("a", href=True)
        if anchor is None:
            continue

        href = (anchor.get("href") or "").strip()
        if not href:
            continue

        full_url = urljoin(index_url, href)
        if full_url in seen_urls or not is_valid_source_url(full_url):
            continue

        title = normalize_whitespace(anchor.get_text(" ", strip=True))
        item_text = normalize_whitespace(item.get_text(" ", strip=True))
        match = DATE_REGEX.search(item_text)
        raw_date = match.group(0) if match else None

        entries.append(
            {
                "section": section_name,
                "index_url": index_url,
                "url": full_url,
                "title": title,
                "raw_date": raw_date,
                "date": normalize_date(raw_date),
            }
        )
        seen_urls.add(full_url)

    return entries


def extract_page_payload(session: requests.Session, url: str, timeout: int) -> dict:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for selector in REMOVABLE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()

    root = (
        soup.select_one("#detail_main_container")
        or soup.select_one("article .entry-content")
        or soup.select_one(".post-content")
        or soup.find("article")
        or soup.select_one("#content")
        or soup.find("main")
        or soup.select_one("#main_container")
        or soup.body
        or soup
    )
    page_title = ""
    heading = root.find("h1") if hasattr(root, "find") else None
    if heading is not None:
        page_title = normalize_whitespace(heading.get_text(" ", strip=True))

    content = normalize_whitespace(root.get_text("\n", strip=True))
    if not content:
        content = normalize_whitespace(soup.get_text("\n", strip=True))

    return {
        "page_title": page_title,
        "content": content,
    }


def validate_record(record: dict, min_content_length: int) -> list[str]:
    issues: list[str] = []

    if not is_valid_source_url(record["url"]):
        issues.append("invalid_domain")
    if not record["article_title"]:
        issues.append("missing_title")
    if not record["content"]:
        issues.append("missing_content")
    if record["content_length"] < min_content_length:
        issues.append("content_too_short")

    return issues


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset_card(report: dict, source_urls: dict[str, str]) -> str:
    source_lines = "\n".join(f"- {url}" for url in source_urls.values())
    return f"""---
pretty_name: Chomsky Corpus
license: other
language:
- en
task_categories:
- text-generation
---

# Chomsky Corpus

## Summary

A cleaned text dataset scraped from selected public sections of `chomsky.info`.

## Source URLs

{source_lines}

## Schema

- `record_id`: stable record identifier
- `section`: source section on the website
- `article_title`: title extracted from the listing page
- `page_title`: title extracted from the detail page when available
- `article_date`: normalized date when available
- `content`: cleaned page text
- `content_length`: number of characters in content
- `content_hash`: normalized content hash for deduplication
- `index_url`: listing page where the document was discovered

## Cleaning

- whitespace normalization
- exact deduplication using normalized content hash
- invalid domain rejection
- missing title rejection
- missing content rejection
- short-content rejection

## Quality Snapshot

- catalog_count: {report['catalog_count']}
- raw_record_count: {report['raw_record_count']}
- validated_record_count: {report['validated_record_count']}
- rejected_record_count: {report['rejected_record_count']}
- deduped_record_count: {report['deduped_record_count']}
- duplicate_count: {report['duplicate_count']}
- min_content_length: {report['min_content_length']}
- median_content_length: {report['median_content_length']}
- max_content_length: {report['max_content_length']}

## Usage Notes

Review the source website's terms, robots.txt, and redistribution rights before publishing publicly. If rights are unclear, keep the dataset private or publish metadata and scripts only.

## Version

{report['pipeline_version']}
"""


def push_to_hub(df: pd.DataFrame, dataset_card_path: Path, repo_id: str, private: bool) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    dataset = Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)
    dataset.push_to_hub(repo_id, split="train", private=private)
    api.upload_file(
        path_or_fileobj=str(dataset_card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


def build_publish_records(records: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in record.items() if key not in PUBLISH_DROP_FIELDS}
        for record in records
    ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    scraped_at = datetime.now(timezone.utc).isoformat()
    session = build_session()

    catalog: list[dict] = []
    for section_name, index_url in SOURCE_URLS.items():
        section_entries = extract_listing_entries(session, section_name, index_url, args.timeout)
        print(f"{section_name}: discovered {len(section_entries)} listing entries")
        catalog.extend(section_entries)

    deduped_catalog_by_url: dict[str, dict] = {}
    for entry in catalog:
        deduped_catalog_by_url.setdefault(entry["url"], entry)
    catalog = list(deduped_catalog_by_url.values())
    print(f"unique URLs across all sections: {len(catalog)}")

    catalog_path = raw_dir / "catalog.jsonl"
    write_jsonl(catalog_path, catalog)

    raw_records: list[dict] = []
    rejected_records: list[dict] = []
    validated_records: list[dict] = []

    for entry in catalog:
        page_payload = extract_page_payload(session, entry["url"], args.timeout)
        article_title = entry["title"] or page_payload["page_title"]
        record = {
            "record_id": stable_record_id(entry["url"], page_payload["content"]),
            "url": entry["url"],
            "section": entry["section"],
            "article_title": article_title,
            "page_title": page_payload["page_title"],
            "article_date": entry["date"],
            "raw_date": entry["raw_date"],
            "content": page_payload["content"],
            "content_length": len(page_payload["content"]),
            "content_hash": normalized_content_hash(page_payload["content"]),
            "source_site": "chomsky.info",
            "index_url": entry["index_url"],
            "scraped_at": scraped_at,
            "pipeline_version": args.pipeline_version,
        }
        raw_records.append(record)

        issues = validate_record(record, args.min_content_length)
        if issues:
            rejected_records.append({"record": record, "issues": issues})
        else:
            validated_records.append(record)

    write_jsonl(raw_dir / "raw_records.jsonl", raw_records)

    deduped_records: list[dict] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0
    for record in validated_records:
        if record["content_hash"] in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(record["content_hash"])
        deduped_records.append(record)

    publish_records = build_publish_records(deduped_records)
    write_jsonl(processed_dir / "chomsky_corpus.jsonl", publish_records)
    write_jsonl(processed_dir / "rejected_records.jsonl", rejected_records)

    section_counts = Counter(record["section"] for record in deduped_records)
    rejection_counts = Counter(issue for item in rejected_records for issue in item["issues"])
    content_lengths = [record["content_length"] for record in deduped_records]
    quality_report = {
        "pipeline_version": args.pipeline_version,
        "scraped_at": scraped_at,
        "source_sections": list(SOURCE_URLS.keys()),
        "catalog_count": len(catalog),
        "raw_record_count": len(raw_records),
        "validated_record_count": len(validated_records),
        "rejected_record_count": len(rejected_records),
        "deduped_record_count": len(deduped_records),
        "duplicate_count": duplicate_count,
        "min_content_length": min(content_lengths) if content_lengths else 0,
        "median_content_length": int(median(content_lengths)) if content_lengths else 0,
        "max_content_length": max(content_lengths) if content_lengths else 0,
        "sections_breakdown": dict(section_counts),
        "rejection_breakdown": dict(rejection_counts),
    }

    quality_report_path = processed_dir / "quality_report.json"
    quality_report_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")

    df = pd.DataFrame(publish_records)
    if not df.empty:
        df = df.sort_values(["section", "article_date", "article_title"], na_position="last")
    parquet_path = processed_dir / "train.parquet"
    df.to_parquet(parquet_path, index=False)

    dataset_card_path = processed_dir / "README.md"
    dataset_card_path.write_text(
        build_dataset_card(quality_report, SOURCE_URLS),
        encoding="utf-8",
    )

    print(f"wrote {catalog_path}")
    print(f"wrote {raw_dir / 'raw_records.jsonl'}")
    print(f"wrote {processed_dir / 'chomsky_corpus.jsonl'}")
    print(f"wrote {processed_dir / 'rejected_records.jsonl'}")
    print(f"wrote {quality_report_path}")
    print(f"wrote {parquet_path}")
    print(f"wrote {dataset_card_path}")

    if args.hf_repo:
        push_to_hub(df, dataset_card_path, args.hf_repo, args.private)
        print(f"pushed dataset to https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()