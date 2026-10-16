import re
from typing import Dict, Any, Optional

def parse_bbraun_discrepancy_form(text: str) -> Dict[str, Any]:
    """Dedicated parser for B. Braun Discrepancy Notification (DN) forms."""
    
    patterns = {
        "document_id": r"Document\s*ID[:#]?\s*(SOP-[A-Z0-9-]+|[A-Z0-9-]+)",
        "version": r"Version[:#]?\s*([0-9.]+)",
        "effective_date": r"Effective\s*Date[:#]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
        "proc_ref": r"Proc\.?\s*Ref#?[:#]?\s*(SOP-[A-Z0-9-]+|[A-Z0-9-]+)",
        "item_number": r"Item\s*#?[:#]?\s*([A-Z0-9]+)",
        "batch_number": r"Batch\s*#?[:#]?\s*([A-Z0-9]+)",
        "machine_line": r"Machine\s*/?\s*Line\s*#?[:#]?\s*([0-9/]+)",
        "who": r"Who\s*is\s*involved\??[:#]?\s*(.+?)(?=\r?\n|Where|When|Other|Z7|A7|$)",
        "where": r"Where\s*is\s*the\s*error\??[:#]?\s*(.+?)(?=\r?\n|When|Who|Other|Z7|A7|$)",
        "when": r"When\s*did\s*it\s*happen\??[:#]?\s*([0-9/-]+)",
        "other_details": r"Other\s*Details[:#]?\s*(.+?)(?=\r?\n|Z7|A7|$)",
        "z7_dn": r"Z7\s*DN\s*#?[:#]?\s*([0-9]+)",
        "a7_dn": r"A7\s*DN\s*#?[:#]?\s*([0-9]+)"
    }

    extracted_fields = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            # Clean leading colons or punctuation
            val = re.sub(r"^[:#\s]+", "", val).strip()
            extracted_fields[key] = val
        else:
            extracted_fields[key] = None

    dates = [val for val in [extracted_fields.get("effective_date"), extracted_fields.get("when")] if val]
    ids = [val for val in [
        extracted_fields.get("document_id"),
        extracted_fields.get("proc_ref"),
        extracted_fields.get("item_number"),
        extracted_fields.get("batch_number"),
        extracted_fields.get("z7_dn"),
        extracted_fields.get("a7_dn")
    ] if val]

    return {
        "document_type": "Discrepancy Notification (DN)",
        "title": "Discrepancy Notification",
        "organization": "B. Braun Medical Inc.",
        "fields": extracted_fields,
        "dates": dates,
        "ids": ids,
        "checked_options": [],
        "handwritten_entries": {
            "who": extracted_fields.get("who"),
            "where": extracted_fields.get("where"),
            "when": extracted_fields.get("when"),
            "other_details": extracted_fields.get("other_details")
        },
        "summary": (
            f"B. Braun Discrepancy Notification (DN) {extracted_fields.get('document_id', '')} "
            f"for Item #{extracted_fields.get('item_number', '')}, Batch #{extracted_fields.get('batch_number', '')}."
        ),
        "raw_text": text
    }


def _extract_fallback_structured(text: str) -> Dict[str, Any]:
    """Generic fallback regex document parser for structured fields."""
    generic_patterns = {
        "document_id": r"(?:Document\s*ID|Doc\s*ID|SOP\s*ID)[:#]?\s*([A-Z0-9-]+)",
        "proc_ref": r"Proc\.?\s*Ref#?[:#]?\s*([A-Z0-9-]+)",
        "effective_date": r"Effective\s*Date[:#]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
        "item_number": r"Item\s*#?[:#]?\s*([A-Z0-9]+)",
        "batch_number": r"(?:Batch\s*#?|Lot\s*#?)[:#]?\s*([A-Z0-9]+)",
        "machine_line": r"Machine\s*/?\s*Line\s*#?[:#]?\s*([0-9/]+)",
        "who": r"Who\s*is\s*involved\??[:#]?\s*(.+?)(?=\r?\n|Where|When|Other|$)",
        "where": r"Where\s*is\s*the\s*error\??[:#]?\s*(.+?)(?=\r?\n|When|Who|Other|$)",
        "when": r"When\s*did\s*it\s*happen\??[:#]?\s*([0-9/-]+)",
        "other_details": r"Other\s*Details[:#]?\s*(.+?)(?=\r?\n|$)",
        "invoice_number": r"(?:Invoice\s*#?|INV-?)\s*([A-Z0-9-]+)"
    }

    fields = {}
    for key, pattern in generic_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            val = re.sub(r"^[:#\s]+", "", val).strip()
            fields[key] = val

    # Collect dates and numbers
    all_dates = list(set(re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", text)))
    all_ids = list(set(re.findall(r"\b[A-Z0-9]{5,}(?:-[A-Z0-9]+)*\b", text)))

    # Organization detector heuristics
    org = None
    if "B. Braun" in text or "Braun" in text:
        org = "B. Braun Medical Inc."
    
    # Document type heuristic
    doc_type = "Generic Document"
    if "Discrepancy Notification" in text:
        doc_type = "Discrepancy Notification"
    elif "Invoice" in text:
        doc_type = "Invoice"
    elif "Receipt" in text:
        doc_type = "Receipt"

    return {
        "document_type": doc_type,
        "title": doc_type,
        "organization": org,
        "fields": fields,
        "dates": all_dates,
        "ids": all_ids,
        "checked_options": [],
        "handwritten_entries": {},
        "summary": f"Structured document ({doc_type}) with {len(fields)} key fields extracted.",
        "raw_text": text
    }


def parse_document(text: str) -> Dict[str, Any]:
    """Parse text into structured data, preferring template parsers before generic fallback."""
    if not text or not text.strip():
        return {
            "document_type": "Unknown",
            "title": None,
            "organization": None,
            "fields": {},
            "dates": [],
            "ids": [],
            "checked_options": [],
            "handwritten_entries": {},
            "summary": "No text detected.",
            "raw_text": ""
        }

    # Direct match for B. Braun Discrepancy Notification form
    lower_text = text.lower()
    if "discrepancy notification" in lower_text or "b. braun" in lower_text or "sop-md-" in lower_text:
        return parse_bbraun_discrepancy_form(text)

    # Generic parser
    return _extract_fallback_structured(text)
