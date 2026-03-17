
# Chomsky Corpus Dataset Pipeline

Scrape selected public sections from `chomsky.info`, normalize and validate the t
text, deduplicate records, and export a publishable dataset.
## Install


```bash
uv sync
```

## Run The Pipeline

```bash
python scripts/build_dataset.py
```

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── docs/
│   └── schema.md
├── scripts/
│   └── build_dataset.py
├── pyproject.toml
└── README.md
```

## Data Sources

The pipeline scrapes these listing pages:

- `https://chomsky.info/articles/`
- `https://chomsky.info/interviews/`
- `https://chomsky.info/letters/`
- `https://chomsky.info/talks/`
- `https://chomsky.info/debates/`

## What The Script Produces

Running the script creates these generated artifacts:

- `data/raw/catalog.jsonl`: discovered URLs and listing metadata
- `data/raw/raw_records.jsonl`: scraped page records before filtering
- `data/processed/chomsky_corpus.jsonl`: cleaned dataset rows
- `data/processed/train.parquet`: Hugging Face friendly publish artifact
- `data/processed/rejected_records.jsonl`: rejected rows with reasons
- `data/processed/quality_report.json`: dataset quality summary
- `data/processed/README.md`: generated dataset card draft

