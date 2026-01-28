#!/usr/bin/env python3
"""
Serbian E-commerce Lead Scraper v2.0
Enhanced version with async scraping, resume capability, and multi-format export.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
import ssl
import sys
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Set
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

# -------------------- Configuration --------------------

DEFAULT_CONFIG = {
    "timeout": 15,
    "sleep_min": 0.5,
    "sleep_max": 1.5,
    "max_concurrent": 10,
    "user_agent": "QNCT-LeadScraper/2.0 (+contact: stevanm.biz@gmail.com)",
}

DEFAULT_ECOM_KEYWORDS = [
    "korpa", "kasa", "checkout", "cart", "add to cart", "dodaj u korpu",
    "poruči", "narudž", "narudz", "order", "kupovina", "plaćanje", "payment",
    "dostava", "isporuka", "povraćaj", "reklamacij", "uslovi kupovine",
    "webshop", "online prodavnica", "cena", "popust", "akcija",
]

DEFAULT_SERBIA_KEYWORDS = [
    "pib", "matični broj", "maticni broj", "+381", "rsd", "din", "srbija",
    "beograd", "novi sad", "niš", "nis", "kragujevac", "subotica",
    "agencija za privredne registre", "apr", "11000", "21000", "18000",
    "ul.", "ulica", "bb", "d.o.o", "doo", "serbia",
]

CONTACT_PATHS = [
    "/", "/kontakt", "/contact", "/o-nama", "/about", "/about-us",
    "/uslovi-koriscenja", "/uslovi-kupovine", "/terms", "/privacy",
    "/politika-privatnosti", "/isporuka", "/dostava",
    "/reklamacije", "/povracaj", "/returns", "/impressum",
]

PREFERRED_LOCALPARTS = [
    "sales", "prodaja", "office", "info", "support",
    "kontakt", "contact", "marketing", "ecommerce",
    "webshop", "narudzbine", "order", "shop",
]

JUNK_EMAIL_PREFIXES = [
    "noreply", "no-reply", "mailer-daemon", "postmaster",
    "admin", "webmaster", "hostmaster", "abuse",
]

INVALID_DOMAIN_WORDS = [
    "import", "from", "def", "class", "python", "print", "return",
    "async", "await", "self", "none", "true", "false", "lambda",
]

# -------------------- Regex Patterns --------------------

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+381[\s\-./]?\d{1,2}[\s\-./]?\d{2,3}[\s\-./]?\d{3,4})")
DOMAIN_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})\b", re.IGNORECASE)

# -------------------- Logging Setup --------------------

def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers,
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# -------------------- Data Classes --------------------

@dataclass
class LeadResult:
    domain: str
    url: str
    email: Optional[str] = None
    phone: Optional[str] = None
    ecom_score: int = 0
    serbia_score: int = 0
    status: str = "SKIP"
    instagram: Optional[str] = None
    facebook: Optional[str] = None
    linkedin: Optional[str] = None
    notes: str = ""

# -------------------- Validation --------------------

def is_valid_domain(d: str) -> bool:
    """Filter out obvious non-domains."""
    if not d or len(d) < 4:
        return False
    if '.' not in d:
        return False
    if d.lower() in INVALID_DOMAIN_WORDS:
        return False
    if any(d.lower() == word for word in INVALID_DOMAIN_WORDS):
        return False
    # Check for file extensions that aren't TLDs
    if d.endswith(('.py', '.js', '.css', '.html', '.txt', '.json', '.xml')):
        return False
    return True

def normalize_domain(u: str) -> Optional[str]:
    """Normalize URL to bare domain."""
    u = (u or "").strip()
    if not u:
        return None
    if "://" not in u:
        u = "https://" + u
    try:
        p = urlparse(u)
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None

# -------------------- Extraction Functions --------------------

def extract_domains_from_text(text: str) -> List[str]:
    """Extract domain names from raw text."""
    found = []
    for m in DOMAIN_RE.finditer(text or ""):
        d = m.group(1).lower().strip(".")
        if is_valid_domain(d):
            found.append(d)
    # Dedupe while preserving order
    seen = set()
    return [d for d in found if not (d in seen or seen.add(d))]

def extract_emails(html: str, soup: BeautifulSoup = None) -> List[str]:
    """Extract email addresses from HTML."""
    if not html:
        return []

    emails = set(EMAIL_RE.findall(html))

    if soup is None:
        soup = BeautifulSoup(html, "lxml")

    for a in soup.select("a[href^='mailto:']"):
        href = a.get("href", "").replace("mailto:", "").split("?")[0]
        if EMAIL_RE.fullmatch(href):
            emails.add(href)

    # Filter junk emails
    filtered = []
    for e in emails:
        local = e.split("@")[0].lower()
        if not any(local.startswith(j) for j in JUNK_EMAIL_PREFIXES):
            filtered.append(e)

    return sorted(filtered)

def extract_phones(html: str) -> List[str]:
    """Extract Serbian phone numbers."""
    if not html:
        return []
    phones = PHONE_RE.findall(html)
    # Normalize format
    normalized = []
    for p in phones:
        clean = re.sub(r'[\s\-./]', '', p)
        if len(clean) >= 10:
            normalized.append(clean)
    return list(set(normalized))

def extract_socials(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract social media profile links."""
    socials = {}
    if not soup:
        return socials

    for a in soup.select("a[href]"):
        href = a.get("href", "").lower()
        if "instagram.com/" in href and "instagram" not in socials:
            socials["instagram"] = a.get("href")
        elif "facebook.com/" in href and "facebook" not in socials:
            socials["facebook"] = a.get("href")
        elif "linkedin.com/" in href and "linkedin" not in socials:
            socials["linkedin"] = a.get("href")

    return socials

def pick_best_email(emails: List[str]) -> Optional[str]:
    """Select the most relevant business email."""
    if not emails:
        return None

    def score(e):
        local = e.split("@")[0].lower()
        for i, pref in enumerate(PREFERRED_LOCALPARTS):
            if local == pref:
                return 100 - i
            if local.startswith(pref):
                return 80 - i
        return 10

    return sorted(emails, key=score, reverse=True)[0]

def pick_best_phone(phones: List[str]) -> Optional[str]:
    """Select the first valid phone number."""
    return phones[0] if phones else None

def score_text(text: str, keywords: List[str]) -> int:
    """Count keyword matches in text."""
    t = (text or "").lower()
    return sum(1 for kw in keywords if kw in t)

# -------------------- File I/O --------------------

def load_seeds_from_file(path: str) -> List[str]:
    """Load and extract domains from a seed file."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")
    domains = extract_domains_from_text(text)

    normalized = []
    for d in domains:
        nd = normalize_domain(d)
        if nd and is_valid_domain(nd):
            normalized.append(nd)

    return normalized

def load_existing_results(path: str) -> Set[str]:
    """Load already-processed domains from output file."""
    if not os.path.exists(path):
        return set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row["domain"] for row in reader if row.get("domain")}
    except Exception as e:
        logger.warning(f"Could not load existing results: {e}")
        return set()

def load_config(path: str) -> dict:
    """Load configuration from YAML or JSON file."""
    if not path or not os.path.exists(path):
        return DEFAULT_CONFIG.copy()

    with open(path, "r") as f:
        if path.endswith(".json"):
            return {**DEFAULT_CONFIG, **json.load(f)}
        elif path.endswith((".yaml", ".yml")):
            try:
                import yaml
                return {**DEFAULT_CONFIG, **yaml.safe_load(f)}
            except ImportError:
                logger.warning("PyYAML not installed, using default config")
                return DEFAULT_CONFIG.copy()

    return DEFAULT_CONFIG.copy()

# -------------------- Async HTTP --------------------

async def fetch_async(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int
) -> Optional[str]:
    """Fetch URL content asynchronously."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status >= 400:
                return None
            ct = response.headers.get("Content-Type", "").lower()
            if "text/html" not in ct:
                return None
            return await response.text()
    except Exception:
        return None

async def crawl_site_async(
    session: aiohttp.ClientSession,
    domain: str,
    timeout: int,
    sleep_range: tuple
) -> LeadResult:
    """Crawl a single site asynchronously."""
    result = LeadResult(domain=domain, url="")

    # Try HTTPS first, then HTTP
    base_url = None
    html = None

    for proto in ("https://", "http://"):
        html = await fetch_async(session, proto + domain, timeout)
        if html:
            base_url = proto + domain
            break

    if not base_url or not html:
        result.status = "SKIP"
        result.notes = "home page fetch failed"
        return result

    result.url = base_url
    soup = BeautifulSoup(html, "lxml")

    # Initial scores from homepage
    best_ecom = score_text(html, DEFAULT_ECOM_KEYWORDS)
    best_srb = score_text(html, DEFAULT_SERBIA_KEYWORDS)
    all_emails = extract_emails(html, soup)
    all_phones = extract_phones(html)
    socials = extract_socials(soup)

    # Crawl additional pages
    for path in CONTACT_PATHS[1:]:  # Skip "/" (already fetched)
        await asyncio.sleep(random.uniform(*sleep_range))

        page_url = urljoin(base_url + "/", path.lstrip("/"))
        page_html = await fetch_async(session, page_url, timeout)

        if not page_html:
            continue

        page_soup = BeautifulSoup(page_html, "lxml")

        best_ecom = max(best_ecom, score_text(page_html, DEFAULT_ECOM_KEYWORDS))
        best_srb = max(best_srb, score_text(page_html, DEFAULT_SERBIA_KEYWORDS))
        all_emails.extend(extract_emails(page_html, page_soup))
        all_phones.extend(extract_phones(page_html))

        page_socials = extract_socials(page_soup)
        for k, v in page_socials.items():
            if k not in socials:
                socials[k] = v

    # Dedupe
    all_emails = list(set(all_emails))
    all_phones = list(set(all_phones))

    # Populate result
    result.ecom_score = best_ecom
    result.serbia_score = best_srb
    result.email = pick_best_email(all_emails)
    result.phone = pick_best_phone(all_phones)
    result.instagram = socials.get("instagram")
    result.facebook = socials.get("facebook")
    result.linkedin = socials.get("linkedin")

    # Classify lead
    if result.email and best_ecom >= 2 and best_srb >= 2:
        result.status = "VALID"
    elif best_ecom >= 2 and best_srb >= 2:
        result.status = "REVIEW_NO_EMAIL"
    elif result.email and (best_ecom >= 1 or best_srb >= 1):
        result.status = "REVIEW_WEAK_SIGNALS"
    else:
        result.status = "SKIP"

    return result

async def crawl_batch(
    domains: List[str],
    config: dict,
    proxy: Optional[str] = None
) -> List[LeadResult]:
    """Crawl multiple domains with controlled concurrency."""
    # Create SSL context that doesn't verify certificates (needed for some sites)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(
        limit=config.get("max_concurrent", 10),
        ssl=ssl_context
    )

    headers = {"User-Agent": config.get("user_agent", DEFAULT_CONFIG["user_agent"])}

    timeout = config.get("timeout", 15)
    sleep_range = (config.get("sleep_min", 0.5), config.get("sleep_max", 1.5))

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = []
        for domain in domains:
            task = crawl_site_async(session, domain, timeout, sleep_range)
            tasks.append(task)

        results = []
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            status_symbol = "✓" if result.status == "VALID" else "○"
            logger.info(f"[{i}/{len(domains)}] {status_symbol} {result.domain} → {result.status}")
            results.append(result)

        return results

# -------------------- Export Functions --------------------

def export_csv(results: List[LeadResult], path: str, append: bool = False):
    """Export results to CSV."""
    mode = "a" if append else "w"
    write_header = not append or not os.path.exists(path)

    fieldnames = [
        "domain", "url", "email", "phone", "ecom_score", "serbia_score",
        "status", "instagram", "facebook", "linkedin", "notes"
    ]

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

def export_json(results: List[LeadResult], path: str):
    """Export results to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

def export_xlsx(results: List[LeadResult], path: str):
    """Export results to Excel."""
    try:
        import pandas as pd
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_excel(path, index=False, engine="openpyxl")
    except ImportError:
        logger.error("pandas/openpyxl not installed. Install with: pip install pandas openpyxl")
        # Fallback to CSV
        csv_path = path.replace(".xlsx", ".csv")
        export_csv(results, csv_path)
        logger.info(f"Exported to CSV instead: {csv_path}")

# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Serbian E-commerce Lead Scraper v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --source-file domains.txt --out leads.csv
  %(prog)s --source-file domains.txt --out leads.json --format json
  %(prog)s --source-file domains.txt --out leads.xlsx --format xlsx --resume
        """
    )

    parser.add_argument("--source-file", "-s", required=True, help="File containing domains (TXT/RTF)")
    parser.add_argument("--out", "-o", default="leads.csv", help="Output file path")
    parser.add_argument("--format", "-f", choices=["csv", "json", "xlsx"], default="csv", help="Output format")
    parser.add_argument("--config", "-c", help="Config file (JSON/YAML)")
    parser.add_argument("--resume", "-r", action="store_true", help="Skip already-processed domains")
    parser.add_argument("--proxy", help="HTTP proxy URL")
    parser.add_argument("--timeout", type=int, help="Request timeout in seconds")
    parser.add_argument("--max-concurrent", type=int, help="Max concurrent requests")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(args.log_file, args.verbose)

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.timeout:
        config["timeout"] = args.timeout
    if args.max_concurrent:
        config["max_concurrent"] = args.max_concurrent

    # Load seed domains
    logger.info(f"Loading domains from: {args.source_file}")
    seeds = load_seeds_from_file(args.source_file)

    if not seeds:
        logger.error("No valid domains found in seed file")
        sys.exit(1)

    logger.info(f"Found {len(seeds)} valid domains")

    # Resume: skip already processed
    if args.resume:
        existing = load_existing_results(args.out)
        original_count = len(seeds)
        seeds = [d for d in seeds if d not in existing]
        skipped = original_count - len(seeds)
        if skipped > 0:
            logger.info(f"Resuming: skipping {skipped} already-processed domains")

    if not seeds:
        logger.info("All domains already processed. Nothing to do.")
        sys.exit(0)

    # Run async crawler
    logger.info(f"Starting crawl of {len(seeds)} domains...")
    results = asyncio.run(crawl_batch(seeds, config, args.proxy))

    # Sort results by status priority
    status_order = {"VALID": 0, "REVIEW_NO_EMAIL": 1, "REVIEW_WEAK_SIGNALS": 2, "SKIP": 3}
    results.sort(key=lambda r: (status_order.get(r.status, 99), -r.ecom_score))

    # Export
    append_mode = args.resume and os.path.exists(args.out)

    if args.format == "csv":
        export_csv(results, args.out, append=append_mode)
    elif args.format == "json":
        export_json(results, args.out)
    elif args.format == "xlsx":
        export_xlsx(results, args.out)

    # Summary
    valid = sum(1 for r in results if r.status == "VALID")
    review = sum(1 for r in results if r.status.startswith("REVIEW"))

    logger.info("=" * 50)
    logger.info(f"DONE! Processed {len(results)} domains")
    logger.info(f"  VALID leads:  {valid}")
    logger.info(f"  REVIEW leads: {review}")
    logger.info(f"  SKIP:         {len(results) - valid - review}")
    logger.info(f"Output saved to: {args.out}")

if __name__ == "__main__":
    main()
