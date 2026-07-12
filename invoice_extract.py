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

# Label patterns (case-insensitive) that can precede each field's value on a line,
# ordered by how specific/likely they are (first match wins per field).
LABELS = {
    "invoice_no": [
        r"invoice\s*(?:no\.?|number|#)\s*[:\-]",
        r"inv\.?\s*(?:no\.?|#)\s*[:\-]",
        r"bill\s*(?:no\.?|number)\s*[:\-]",
        r"receipt\s*(?:no\.?|number)\s*[:\-]",
    ],
    "date": [
        r"invoice\s*date\s*[:\-]",
        r"billed\s*on\s*[:\-]",
        r"issued\s*on\s*[:\-]",
        r"^date\s*[:\-]",
        r"\bdate\s*[:\-]",
    ],
    "vendor": [
        r"vendor\s*(?:name)?\s*[:\-]",
        r"seller\s*(?:name)?\s*[:\-]",
        r"supplier\s*(?:name)?\s*[:\-]",
        r"bill(?:ed)?\s*from\s*[:\-]",
        r"company\s*(?:name)?\s*[:\-]",
        r"from\s*[:\-]",
    ],
    "amount": [
        r"sub\s*-?\s*total\s*[:\-]",
        r"net\s*amount\s*[:\-]",
        r"amount\s*\(before\s*tax\)\s*[:\-]",
        r"amount\s*[:\-]",
    ],
    "tax": [
        r"gst\s*\([\d.]+%\)\s*[:\-]",
        r"gst\s*[:\-]",
        r"vat\s*\([\d.]+%\)\s*[:\-]",
        r"vat\s*[:\-]",
        r"service\s*tax\s*[:\-]",
        r"tax\s*\([\d.]+%\)\s*[:\-]",
        r"tax\s*[:\-]",
    ],
    "total": [
        r"grand\s*total\s*[:\-]",
        r"total\s*[:\-]",
    ],
}

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")

CURRENCY_MARKERS = [
    (r"₹|Rs\.?|INR", "INR"),
    (r"\$|USD", "USD"),
    (r"€|EUR", "EUR"),
    (r"£|GBP", "GBP"),
    (r"¥|JPY", "JPY"),
]


def _find_value_for_line(text: str, patterns: list) -> str | None:
    """Find the first line matching any of the label patterns and return the text after the label."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in patterns:
            m = re.search(pattern, stripped, flags=re.IGNORECASE)
            if m:
                value = stripped[m.end():].strip()
                if value:
                    return value
    return None


def _extract_number(value_str: str) -> float | None:
    if value_str is None:
        return None
    m = NUMBER_RE.search(value_str)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _normalize_date(value_str: str) -> str | None:
    if value_str is None:
        return None
    # Take just the date-looking portion (in case trailing text exists on the line).
    candidate = value_str.strip()
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


def extract_invoice_fields(invoice_text: str) -> dict:
    invoice_no_raw = _find_value_for_line(invoice_text, LABELS["invoice_no"])
    date_raw = _find_value_for_line(invoice_text, LABELS["date"])
    vendor_raw = _find_value_for_line(invoice_text, LABELS["vendor"])
    amount_raw = _find_value_for_line(invoice_text, LABELS["amount"])
    tax_raw = _find_value_for_line(invoice_text, LABELS["tax"])
    total_raw = _find_value_for_line(invoice_text, LABELS["total"])

    invoice_no = invoice_no_raw.strip().strip(",") if invoice_no_raw else None
    date = _normalize_date(date_raw)
    vendor = vendor_raw.strip().strip(",") if vendor_raw else None
    amount = _extract_number(amount_raw)
    tax = _extract_number(tax_raw)
    total = _extract_number(total_raw)

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
