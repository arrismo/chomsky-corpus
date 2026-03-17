# Dataset Schema

The processed dataset written to `data/processed/chomsky_corpus.jsonl` and `data/processed/train.parquet` uses the following schema.

| Field | Type | Description |
| --- | --- | --- |
| `record_id` | string | Stable identifier derived from URL and normalized content |
| `section` | string | One of `articles`, `interviews`, `letters`, `talks`, `debates` |
| `article_title` | string | Title extracted from the listing page or page content |
| `page_title` | string | Title extracted from the detail page when available |
| `article_date` | string or null | Normalized date in `YYYY`, `YYYY-MM`, or `YYYY-MM-DD` format |
| `content` | string | Cleaned page text |
| `content_length` | integer | Character length of `content` |
| `content_hash` | string | SHA-256 hash of normalized content used for deduplication |
| `index_url` | string | Listing page where the URL was discovered |

## Validation Rules

- `url` must be under `https://chomsky.info/`
- `article_title` must not be empty
- `content` must not be empty
- `content_length` must be at least 200 characters

## Deduplication Rule

Exact duplicates are removed using `content_hash` computed from lowercased, whitespace-normalized text.