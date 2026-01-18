#!/usr/bin/env python3
import argparse
import csv
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

DEFAULT_ECOM_KEYWORDS = [
    "korpa", "kasa", "checkout", "cart", "add to cart", "dodaj u korpu",
    "poruči", "narudž", "narudz", "order", "kupovina", "plaćanje", "payment",
    "dostava", "isporuka", "povraćaj", "reklamacij", "uslovi kupovine",
]

DEFAULT_SERBIA_KEYWORDS = [
    "pib", "matični broj", "maticni broj", "+381", "rsd", "din", "srbija",
    "beograd", "novi sad", "niš", "nis", "kragujevac", "subotica",
    "agencija za privredne registre", "apr", "11000", "21000", "18000",
    "ul.", "ulica", "bb",
]

CONTACT_PATHS = [
    "/", "/kontakt", "/contact", "/o-nama", "/about",
    "/uslovi-koriscenja", "/uslovi-kupovine", "/terms", "/privacy",
    "/politika-privatnosti", "/isporuka", "/dostava",
    "/reklamacije", "/povracaj", "/returns",
]

PREFERRED_LOCALPARTS = [
    "sales", "prodaja", "office", "info", "support",
    "kontakt", "contact", "marketing", "ecommerce",
    "webshop", "narudzbine", "order",
]

@dataclass
class LeadResult:
    domain: str
    url: str
    email: Optional[str]
    ecom_score: int
    serbia_score: int
    status: str
    notes: str
    source: str


# -------------------- helpers --------------------

def normalize_domain(u: str) -> Optional[str]:
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


def extract_domains_from_text(text: str) -> List[str]:
    domain_re = re.compile(
        r"\b(?:https?://)?(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})\b",
        re.IGNORECASE,
    )
    found = []
    for m in domain_re.finditer(text or ""):
        found.append(m.group(1).lower().strip("."))
    # dedupe, keep order
    seen = set()
    out = []
    for d in found:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def load_seeds_from_file(path: str) -> List[str]:
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")
    domains = extract_domains_from_text(text)
    normalized = []
    for d in domains:
        nd = normalize_domain(d)
        if nd:
            normalized.append(nd)
    return normalized


def fetch(session: requests.Session, url: str, timeout: int) -> Optional[str]:
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        ct = (r.headers.get("Content-Type") or "").lower()
        if r.status_code >= 400:
            return None
        if "text/html" not in ct:
            return None
        return r.text or ""
    except requests.RequestException:
        return None


def score_text(text: str, keywords: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for kw in keywords if kw in t)


def extract_emails(html: str) -> List[str]:
    if not html:
        return []
    emails = set(EMAIL_RE.findall(html))
    soup = BeautifulSoup(html, "lxml")
    for a in soup.select("a[href^='mailto:']"):
        e = a.get("href", "").replace("mailto:", "").split("?")[0]
        if EMAIL_RE.fullmatch(e):
            emails.add(e)
    return sorted(emails)


def pick_best_email(emails: List[str]) -> Optional[str]:
    if not emails:
        return None

    def score(e):
        local = e.split("@")[0].lower()
        for i, p in enumerate(PREFERRED_LOCALPARTS):
            if local == p:
                return 100 - i
            if local.startswith(p):
                return 80 - i
        return 10

    return sorted(emails, key=score, reverse=True)[0]


# -------------------- main crawl --------------------

def crawl_site(session, domain, timeout, sleep_min, sleep_max):
    best_email = None
    best_ecom = 0
    best_srb = 0
    base_url = ""

    for proto in ("https://", "http://"):
        html = fetch(session, proto + domain, timeout)
        if html:
            base_url = proto + domain
            best_ecom = score_text(html, DEFAULT_ECOM_KEYWORDS)
            best_srb = score_text(html, DEFAULT_SERBIA_KEYWORDS)
            best_email = pick_best_email(extract_emails(html))
            break

    if not base_url:
        return LeadResult(domain, "", None, 0, 0, "SKIP", "home fetch failed", "")

    for p in CONTACT_PATHS:
        time.sleep(random.uniform(sleep_min, sleep_max))
        html = fetch(session, urljoin(base_url + "/", p.lstrip("/")), timeout)
        if not html:
            continue

        best_ecom = max(best_ecom, score_text(html, DEFAULT_ECOM_KEYWORDS))
        best_srb = max(best_srb, score_text(html, DEFAULT_SERBIA_KEYWORDS))

        e = pick_best_email(extract_emails(html))
        if e:
            best_email = e

    if best_email and best_ecom >= 2 and best_srb >= 2:
        status = "VALID"
    elif best_ecom >= 2 and best_srb >= 2:
        status = "REVIEW_NO_EMAIL"
    elif best_email:
        status = "REVIEW_WEAK_SIGNALS"
    else:
        status = "SKIP"

    return LeadResult(
        domain, base_url, best_email,
        best_ecom, best_srb, status, "", ""
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-file", required=True, help="RTF/TXT file with domains")
    ap.add_argument("--out", default="leads.csv")
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--sleep-min", type=float, default=1.0)
    ap.add_argument("--sleep-max", type=float, default=2.0)
    args = ap.parse_args()

    seeds = load_seeds_from_file(args.source_file)
    if not seeds:
        print("No domains found in seed file", file=sys.stderr)
        sys.exit(1)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "QNCT-FileSeedScraper/1.0 (+contact: stevanm.biz@gmail.com)"
    })

    results = []
    valid_count = 0

    for i, domain in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] {domain}")
        r = crawl_site(session, domain, args.timeout, args.sleep_min, args.sleep_max)
        results.append(r)
        if r.status == "VALID":
            valid_count += 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "url", "email", "ecom_score", "serbia_score", "status"])
        for r in results:
            w.writerow([r.domain, r.url, r.email or "", r.ecom_score, r.serbia_score, r.status])

    print(f"\nDone. VALID leads: {valid_count} / {len(results)}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()