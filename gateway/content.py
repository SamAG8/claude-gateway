"""Helpers for decoding/validating inbound media into canonical content blocks."""
import base64
import binascii
import io

from . import config
from .errors import GatewayError


def _decode_b64(b64: str, limit: int | None = None) -> bytes:
    """Decode inbound base64 (standard or URL-safe), enforce a size limit, return raw bytes.

    ``limit`` defaults to MAX_FILE_SIZE; PDFs pass MAX_PDF_SIZE.

    The official google-genai SDK encodes inline media with URL-safe base64
    (``-``/``_`` alphabet); the OpenAI/Anthropic SDKs send standard base64. We accept
    either on the way in. Non-string, empty, malformed, or oversize input is rejected
    as a 4xx GatewayError (never an uncaught 500).
    """
    if not isinstance(b64, str):
        raise GatewayError(400, "invalid base64 data")
    s = b64.strip().replace("-", "+").replace("_", "/")
    s += "=" * ((-len(s)) % 4)  # tolerate stripped padding
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise GatewayError(400, "invalid base64 data")
    if not raw:
        raise GatewayError(400, "empty base64 data")
    if limit is None:
        if len(raw) > config.MAX_FILE_SIZE:
            raise GatewayError(413, "file exceeds MAX_FILE_SIZE")
    elif len(raw) > limit:
        raise GatewayError(413, "PDF exceeds MAX_PDF_SIZE")
    return raw


def _encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def normalize_b64(b64: str) -> str:
    """Validate inbound base64 and return canonical standard base64 (what the CLI requires).

    Accepts standard or URL-safe base64 and always hands the CLI the standard form.
    The Claude CLI (Anthropic API) only accepts standard base64.
    """
    return _encode(_decode_b64(b64))


def image_block(media_type: str, data: str) -> dict:
    """An inline image block (base64 normalized + size-checked) for the CLI's `image` source."""
    if not media_type:
        raise GatewayError(400, "image missing media_type")
    return {"type": "image", "media_type": media_type, "data": normalize_b64(data)}


def document_block(media_type: str, data: str) -> dict:
    """A native document (e.g. PDF) block, sent to the CLI as a `document` source.

    Unlike `pdf_to_text_block` (which flattens a PDF to extracted text), this
    preserves the original bytes so Claude reads the document with vision —
    keeping layout, handwriting, and highlights intact.
    """
    if not media_type:
        raise GatewayError(400, "document missing media_type")
    return {"type": "document", "media_type": media_type,
            "data": _encode(_decode_b64(data, config.MAX_PDF_SIZE))}


def pdf_page_count(raw: bytes) -> int | None:
    """Pages in a PDF, or None when pdfplumber cannot open it.

    None is not a refusal: Claude's own PDF reader is more forgiving than
    pdfminer's, so an unparseable file still goes to it natively.
    """
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return len(pdf.pages)
    except Exception:  # noqa: BLE001
        return None


def pdf_block(data: str, mode: str | None = None) -> dict:
    """The block an Anthropic-surface PDF becomes.

    Native by default, so Claude sees the pages — a drawing's lines, a scan, a
    table's layout — and not only whatever text pdfminer could pull out of them
    (a drawing loses nearly everything; a scan loses everything and used to be
    refused). ``mode == "text"`` (the ``x-pdf-mode: text`` header) asks for the
    old flattening. A PDF longer than MAX_PDF_PAGES is flattened too, because the
    Messages API refuses it natively and text is better than an error.

    Synchronous and possibly slow (pdfplumber): callers run it off the loop.
    """
    if mode == "text":
        return pdf_to_text_block(data)
    raw = _decode_b64(data, config.MAX_PDF_SIZE)
    pages = pdf_page_count(raw)
    if pages is not None and pages > config.MAX_PDF_PAGES:
        return pdf_to_text_block(data)
    return {"type": "document", "media_type": "application/pdf", "data": _encode(raw)}


def pdf_to_text_block(data: str) -> dict:
    """Extract PDF text via pdfplumber and inline it as a text block."""
    raw = _decode_b64(data, config.MAX_PDF_SIZE)
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n\n".join(pages).strip()
    except GatewayError:
        raise
    except Exception as e:  # noqa: BLE001
        raise GatewayError(400, f"PDF extraction failed: {e}")
    if not text:
        raise GatewayError(400, "no text could be extracted from the PDF")
    return {"type": "text", "text": "Document content:\n" + text}


def parse_data_uri(url: str) -> tuple[str, str]:
    """Parse an OpenAI image_url data: URI into (media_type, base64_data)."""
    if not url.startswith("data:"):
        raise GatewayError(400, "only data: image URLs are supported")
    try:
        header, b64 = url.split(",", 1)
    except ValueError:
        raise GatewayError(400, "malformed data URI")
    media_type = header[len("data:"):].split(";")[0] or "image/png"
    return media_type, b64
