"""
Invoice field extraction logic.
Parses raw, free-form invoice text into 6 fixed structured fields:
    invoice_no, date, vendor, amount, tax, currency

Rules implemented:
  1. Always return all 6 keys; use null (None) if a field cannot be found.
  2. date is normalized to ISO format YYYY-MM-DD.
  3. amount = subtotal (before tax); tax = tax amount only.
  4. currency is inferred from symbols/codes in the text (defaults to None if unclear).
"""

import re
from dateutil import parser as dateparser

# Label patterns (case-insensitive), anchored to the START of a (stripped) line.
# The label may be followed by ANY separator style: ":", "-", ".", dot-leaders, or
# just whitespace (e.g. "Subtotal .......... Rs. 2,199.00" or "Issued: 2026-01-22").
LABELS = {
    "invoice_no": [
        r"^invoice\s*(?:no\.?|number|#|id)\b",
        r"^inv\.?\s*(?:no\.?|#|id)\b",
        r"^bill\s*(?:no\.?|number|id)\b",
        r"^receipt\s*(?:no\.?|number|id)\b",
        r"^ref(?:erence)?\.?\s*(?:no\.?|number|id)?\b",
        r"^voucher\s*(?:no\.?|number|id)\b",
        r"^doc(?:ument)?\.?\s*(?:no\.?|number|id)\b",
        r"^order\s*(?:no\.?|number|id)\b",
        r"^transaction\s*(?:no\.?|number|id)\b",
        r"^txn\.?\s*(?:no\.?|id)\b",
        r"^control\s*(?:no\.?|number)\b",
        r"^serial\s*(?:no\.?|number)\b",
        r"^sl\.?\s*no\.?\b",
    ],
    "date": [
        r"^invoice\s*date\b",
        r"^invoice\s*dt\.?\b",
        r"^billed\s*on\b",
        r"^bill\s*date\b",
        r"^issued\s*(?:on)?\b",
        r"^issue\s*date\b",
        r"^dated\b",
        r"^transaction\s*date\b",
        r"^date\b",
    ],
    "date_fallback": [
        r"^value\s*date\b",
    ],
    "vendor": [
        r"^vendor\s*(?:name)?\b",
        r"^seller\s*(?:name)?\b",
        r"^supplier\s*(?:name)?\b",
        r"^bill(?:ed)?\s*from\b",
        r"^company\s*(?:name)?\b",
        r"^issued\s*by\b",
        r"^billed\s*by\b",
        r"^provider\b",
        r"^merchant\s*(?:name)?\b",
        r"^from\b",
    ],
    "amount": [
        r"^sub\s*-?\s*total\b",
        r"^net\s*amount\b",
        r"^amount\s*\(before\s*tax\)\b",
        r"^amount\s*\(excl(?:uding|\.)?\s*tax\)\b",
        r"^amount\s*\(without\s*tax\)\b",
        r"^taxable\s*(?:value|amount)\b",
        r"^basic\s*(?:value|amount)\b",
        r"^base\s*(?:value|amount)\b",
        r"^gross\s*amount\b",
        r"^price\b",
        r"^cost\b",
        r"^value\b(?!\s*date)",
        r"^charges?\b",
        r"^amount\b",
    ],
    "tax": [
        r"^\w{0,3}gst\s*\([\d.]+%\)",
        r"^\w{0,3}gst\s*@\s*[\d.]+%",
        r"^\w{0,3}gst\b",
        r"^vat\s*\([\d.]+%\)",
        r"^vat\s*@\s*[\d.]+%",
        r"^vat\b",
        r"^service\s*tax\b",
        r"^sales\s*tax\b",
        r"^output\s*tax\b",
        r"^tax\s*amount\b",
        r"^tax\s*\([\d.]+%\)",
        r"^tax\s*@\s*[\d.]+%",
        r"^duty\b",
        r"^cess\b",
        r"^tax\b",
    ],
    "total": [
        r"^grand\s*total\b",
        r"^total\s*due\b",
        r"^total\s*payable\b",
        r"^amount\s*due\b",
        r"^amount\s*payable\b",
        r"^net\s*payable\b",
        r"^balance\s*due\b",
        r"^total\b",
    ],
}

# Non-vendor header lines to skip when falling back to an unlabeled vendor/company name.
NON_VENDOR_LINE_PATTERNS = [p for plist in LABELS.values() for p in plist] + [
    r"^client\b",
    r"^bill\s*to\b",
    r"^currency\b",
    r"^items?\b",
    r"^service\b",
    r"^description\b",
    r"^qty\b",
    r"^price\b",
]
GENERIC_HEADER_WORDS = {"invoice", "tax invoice", "receipt", "bill", "statement", "quotation"}

# Separator characters (colon, dash, dot-leaders, whitespace) that can follow a label
# before the actual value starts.
_SEPARATOR_RE = re.compile(r"^[\s.:\-]+")

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")

CURRENCY_MARKERS = [
    (r"₹|\bRs\.?(?=\s|\d|$)|\bINR\b", "INR"),
    (r"\$|\bUSD\b", "USD"),
    (r"€|\bEUR\b", "EUR"),
    (r"£|\bGBP\b", "GBP"),
    (r"¥|\bJPY\b", "JPY"),
]

# Fallback pattern for invoice-number-like codes when no explicit label matches:
# e.g. "ZB-4490", "INV-2026-0041", "AB/1234", "NS/2026/778"
FALLBACK_CODE_RE = re.compile(
    r"\b([A-Z]{1,5}[-/][A-Z0-9]{2,}(?:[-/][A-Z0-9]+)*|[A-Z]{2,5}\d{3,})\b"
)


def _find_value_for_line(text: str, patterns: list) -> str | None:
    """Find the first line whose start matches any of the label patterns, then
    return whatever follows the label (after stripping separator characters)."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in patterns:
            m = re.match(pattern, stripped, flags=re.IGNORECASE)
            if m:
                remainder = stripped[m.end():]
                remainder = _SEPARATOR_RE.sub("", remainder).strip()
                if remainder:
                    return remainder
    return None


def _find_numeric_value_for_line(text: str, patterns: list) -> float | None:
    """Like _find_value_for_line, but for numeric fields (amount/tax/total).
    Skips a matched label line if no actual number follows it (e.g. a false
    positive like 'Cost Center: Engineering' or 'Value Date: ...'), and keeps
    searching later lines/patterns instead of giving up."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in patterns:
            m = re.match(pattern, stripped, flags=re.IGNORECASE)
            if m:
                remainder = stripped[m.end():]
                remainder = _SEPARATOR_RE.sub("", remainder).strip()
                if remainder:
                    number = _extract_number(remainder)
                    if number is not None:
                        return number
                    # No usable number on this line — keep searching other lines/patterns.
    return None


def _extract_number(value_str: str) -> float | None:
    if value_str is None:
        return None
    for m in NUMBER_RE.finditer(value_str):
        # Skip figures that are actually percentages (e.g. the "18" in "@ 18%: Rs. 31,500.00")
        following = value_str[m.end():m.end() + 2].strip()
        if following.startswith("%"):
            continue
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            continue
    return None


_ISO_DATE_RE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")


def _normalize_date(value_str: str) -> str | None:
    if value_str is None:
        return None
    candidate = value_str.strip()

    # If the date is already unambiguous ISO-style (YYYY-MM-DD or YYYY/MM/DD),
    # parse it directly — dateutil's dayfirst=True incorrectly swaps
    # month/day even for this unambiguous year-first format.
    m = _ISO_DATE_RE.match(candidate)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            from datetime import date
            return date(year, month, day).isoformat()
        except ValueError:
            pass

    try:
        dt = dateparser.parse(candidate, dayfirst=True, fuzzy=True)
        if dt:
            return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        pass
    return None


def _detect_currency(text: str) -> str | None:
    for pattern, code in CURRENCY_MARKERS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return code
    return None


def _extract_invoice_no(invoice_text: str) -> str | None:
    value = _find_value_for_line(invoice_text, LABELS["invoice_no"])
    if value:
        m = re.match(r"[A-Za-z0-9\-\/]+", value.strip())
        return m.group(0) if m else value.strip()

    # No labeled line found — fall back to scanning for a code-like token anywhere in the text.
    m = FALLBACK_CODE_RE.search(invoice_text)
    if m:
        return m.group(0)
    return None


def _extract_vendor(invoice_text: str) -> str | None:
    value = _find_value_for_line(invoice_text, LABELS["vendor"])
    if value:
        return value.strip().strip(",")

    # Fallback: the vendor/company name is often the very first substantive
    # line of the document, sometimes followed by a dash and a document type
    # (e.g. "NovaSoft Solutions — Tax Invoice").
    for line in invoice_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().strip(" .") in GENERIC_HEADER_WORDS:
            continue
        if any(re.match(p, stripped, flags=re.IGNORECASE) for p in NON_VENDOR_LINE_PATTERNS):
            continue
        candidate = re.split(r"\s+[\u2014\u2013\-]\s+", stripped, maxsplit=1)[0].strip()
        if candidate:
            return candidate
    return None


def extract_invoice_fields(invoice_text: str) -> dict:
    invoice_no = _extract_invoice_no(invoice_text)
    date_raw = _find_value_for_line(invoice_text, LABELS["date"]) or \
        _find_value_for_line(invoice_text, LABELS["date_fallback"])
    vendor = _extract_vendor(invoice_text)
    amount = _find_numeric_value_for_line(invoice_text, LABELS["amount"])
    tax = _find_numeric_value_for_line(invoice_text, LABELS["tax"])
    total = _find_numeric_value_for_line(invoice_text, LABELS["total"])

    date = _normalize_date(date_raw)

    # Fallback: derive amount (subtotal before tax) from total - tax if amount missing.
    if amount is None and total is not None and tax is not None:
        amount = round(total - tax, 2)

    # Fallback: derive tax from total - amount if tax missing.
    if tax is None and total is not None and amount is not None:
        tax = round(total - amount, 2)

    currency = _detect_currency(invoice_text)

    return {
        "invoice_no": invoice_no,
        "date": date,
        "vendor": vendor,
        "amount": amount,
        "tax": tax,
        "currency": currency,
    }
