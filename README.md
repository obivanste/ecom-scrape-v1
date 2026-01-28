# Serbian E-commerce Lead Scraper v2.0

A Python tool for finding and qualifying Serbian e-commerce business leads through automated website analysis.

## Features

- **Async Scraping** - 5-10x faster than synchronous with controlled concurrency
- **Resume Capability** - Continue interrupted scrapes without reprocessing
- **Multi-Format Export** - CSV, JSON, or Excel output
- **Smart Lead Scoring** - E-commerce and Serbia-specific keyword matching
- **Contact Extraction** - Emails, phone numbers, and social media links
- **Input Validation** - Filters invalid domains automatically
- **Proxy Support** - Avoid IP blocks on large runs
- **Configurable** - YAML/JSON config files supported

## Installation

```bash
cd ecom-scrape-v1
pip install -r requirements.txt
```

## Quick Start

```bash
# Basic usage
python scraper.py --source-file domains.txt --out leads.csv

# Export to Excel
python scraper.py -s domains.txt -o leads.xlsx -f xlsx

# Resume interrupted scrape
python scraper.py -s domains.txt -o leads.csv --resume

# With custom settings
python scraper.py -s domains.txt -o leads.csv --timeout 20 --max-concurrent 5
```

## Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--source-file` | `-s` | Input file with domains (required) |
| `--out` | `-o` | Output file path (default: leads.csv) |
| `--format` | `-f` | Output format: csv, json, xlsx |
| `--config` | `-c` | Config file path (YAML/JSON) |
| `--resume` | `-r` | Skip already-processed domains |
| `--proxy` | | HTTP proxy URL |
| `--timeout` | | Request timeout in seconds |
| `--max-concurrent` | | Max parallel requests |
| `--log-file` | | Save logs to file |
| `--verbose` | `-v` | Verbose output |

## Lead Classification

| Status | Criteria |
|--------|----------|
| `VALID` | Has email + ecom_score ≥ 2 + serbia_score ≥ 2 |
| `REVIEW_NO_EMAIL` | Good scores but no email found |
| `REVIEW_WEAK_SIGNALS` | Has email but low confidence |
| `SKIP` | Site unreachable or insufficient signals |

## Output Fields

| Field | Description |
|-------|-------------|
| `domain` | Website domain |
| `url` | Full URL (with protocol) |
| `email` | Best business email found |
| `phone` | Serbian phone number (+381...) |
| `ecom_score` | E-commerce keyword matches |
| `serbia_score` | Serbian business keyword matches |
| `status` | Lead classification |
| `instagram` | Instagram profile URL |
| `facebook` | Facebook page URL |
| `linkedin` | LinkedIn page URL |
| `notes` | Additional notes |

## Configuration File

Create `config.yaml` based on `config.example.yaml`:

```yaml
timeout: 15
sleep_min: 0.5
sleep_max: 1.5
max_concurrent: 10
user_agent: "YourBot/1.0 (+contact: you@email.com)"
```

## Examples

### Process a list of domains
```bash
python scraper.py -s my_domains.txt -o results.csv
```

### Resume after interruption
```bash
python scraper.py -s my_domains.txt -o results.csv --resume
```

### Export to Excel with logging
```bash
python scraper.py -s domains.txt -o leads.xlsx -f xlsx --log-file scrape.log
```

### Use with proxy
```bash
python scraper.py -s domains.txt -o leads.csv --proxy http://127.0.0.1:8080
```

## Legacy Script

The original `test-scrape1.py` is kept for reference. Use `scraper.py` for new projects.

## License

MIT
