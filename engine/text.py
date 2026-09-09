import re


CASHTAG_RE = re.compile(r"(?<![A-Z0-9])\$([A-Z]{1,5})(?![A-Z0-9])")
BARE_TOKEN_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{1,5})(?![A-Z0-9])")
STOP_TICKERS = {
    "A", "I", "DD", "CEO", "CFO", "FDA", "SEC", "IPO", "ETF", "USA",
    "USD", "EPS", "AI", "EV", "YOLO", "IMO", "ATH", "EOD", "FOMO",
    "OTC", "RS", "PR", "PM", "AH",
}


def normalize_ticker(value: str | None) -> str | None:
    if not value:
        return None
    ticker = value.upper().strip().replace("$", "")
    if not re.fullmatch(r"[A-Z]{1,5}", ticker) or ticker in STOP_TICKERS:
        return None
    return ticker


def extract_tickers(
    title: str | None,
    body: str | None = None,
    allowed_tickers: set[str] | None = None,
) -> list[str]:
    content = f"{title or ''}\n{body or ''}".upper()
    candidates = set(CASHTAG_RE.findall(content))
    if allowed_tickers:
        allowed = {
            ticker
            for value in allowed_tickers
            if (ticker := normalize_ticker(value))
        }
        candidates.update(set(BARE_TOKEN_RE.findall(content)).intersection(allowed))
    return sorted(
        ticker
        for candidate in candidates
        if (ticker := normalize_ticker(candidate))
    )
