from __future__ import annotations

import base64
import io
import re
from typing import Any

_STOPWORDS = {
    "the", "of", "and", "to", "in", "is", "that", "for", "be", "by", "as", "on",
    "with", "or", "which", "are", "under", "this", "it", "an", "from", "goods",
    "heading", "customs", "invoice", "item", "value", "duty", "hs", "code",
}
_WORDLIKE = re.compile(
    r"^(?:[A-Za-z][a-z]+|[A-Z]+|\d+(?:[.,/-]\d+)*|[(\[]?[a-z0-9]{1,3}[)\].]|&)[.,;:)]*$"
)


def filter_english_lines(text: str) -> str:
    """Drop garbled OCR lines (e.g., bilingual Hindi halves parsed by English OCR)."""
    keep: list[str] = []
    for line in text.splitlines():
        toks = line.split()
        if not toks:
            keep.append("")
            continue
        wordlike = sum(bool(_WORDLIKE.match(t)) for t in toks) / len(toks)
        stop = sum(t.lower().strip(".,;:()") in _STOPWORDS for t in toks) / len(toks)
        if wordlike >= 0.65 and (len(toks) < 6 or stop >= 0.06):
            keep.append(line)
    return "\n".join(keep).strip()


def sanitize_untrusted_document(filename: str, text: str, max_chars: int = 12000) -> str:
    """Wrap extracted document text to isolate untrusted invoice/notice instructions."""
    cleaned = re.sub(r"[ \t]+", " ", text or "").strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n...[truncated]"
    return f'<untrusted_document_content filename="{filename}">\n{cleaned}\n</untrusted_document_content>'


def parse_document_bytes(
    filename: str,
    raw_bytes: bytes,
    mimetype: str = "application/octet-stream",
) -> dict[str, Any]:
    """Extract and sanitize text from PDF, plain text, CSV, JSON, or image attachments."""
    warnings: list[str] = []
    pages_extracted = 1
    lower_name = filename.lower()
    text = ""

    if mimetype == "application/pdf" or lower_name.endswith(".pdf"):
        text, pages_extracted, pdf_warnings = _extract_pdf_bytes(raw_bytes)
        warnings.extend(pdf_warnings)
    elif mimetype.startswith("image/") or lower_name.endswith((".png", ".jpg", ".jpeg")):
        text, ocr_warn = _extract_image_ocr(raw_bytes)
        if ocr_warn:
            warnings.append(ocr_warn)
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    cleaned = filter_english_lines(text) if len(text.splitlines()) > 3 else text.strip()
    if not cleaned and text.strip():
        cleaned = text.strip()

    sanitized = sanitize_untrusted_document(filename, cleaned) if cleaned else ""
    return {
        "filename": filename,
        "mimetype": mimetype,
        "pages_extracted": pages_extracted,
        "raw_text": cleaned,
        "sanitized_text": sanitized,
        "warnings": warnings,
    }


def _extract_pdf_bytes(raw_bytes: bytes) -> tuple[str, int, list[str]]:
    warnings: list[str] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            page_txt = (page.extract_text() or "").strip()
            if page_txt:
                pages.append(f"--- Page {idx} ---\n{page_txt}")
        if pages:
            return "\n\n".join(pages), len(reader.pages), warnings
        warnings.append("PDF appears to be image-only (no embedded text layer).")
        return "", len(reader.pages), warnings
    except Exception as exc:
        warnings.append(f"PDF extraction failed: {exc}")
        return "", 0, warnings


def _extract_image_ocr(raw_bytes: bytes) -> tuple[str, str | None]:
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(raw_bytes))
        return pytesseract.image_to_string(img, lang="eng"), None
    except Exception:
        return "", "Image OCR is unavailable in this runtime; please include text description alongside image."


def extract_attachments_context(attachments: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Convert request attachment payloads into sanitized context blocks and warnings."""
    blocks: list[str] = []
    warnings: list[str] = []
    for att in attachments:
        name = att.get("name", "attachment")
        mimetype = att.get("mimetype", "application/octet-stream")
        extracted = (att.get("extracted_text") or "").strip()
        if extracted:
            blocks.append(sanitize_untrusted_document(name, extracted))
            continue
        b64 = att.get("content_base64")
        if b64:
            try:
                raw_bytes = base64.b64decode(b64)
                parsed = parse_document_bytes(name, raw_bytes, mimetype)
                if parsed["sanitized_text"]:
                    blocks.append(parsed["sanitized_text"])
                warnings.extend(parsed["warnings"])
            except Exception as exc:
                warnings.append(f"Could not decode attachment {name}: {exc}")
    return "\n\n".join(blocks), warnings
