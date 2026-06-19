"""Helpers for decoding/validating inbound media into canonical content blocks."""
import base64
import binascii
import io

from . import config
from .errors import GatewayError


def normalize_b64(b64: str) -> str:
    """Validate inbound base64, enforce MAX_FILE_SIZE, and return canonical standard base64.

    The official google-genai SDK encodes inline media with URL-safe base64
    (``-``/``_`` alphabet); the OpenAI/Anthropic SDKs send standard base64. The
    Claude CLI (Anthropic API) only accepts standard base64, so we accept either
    on the way in and always hand the CLI the standard form.
    """
    s = (b64 or "").strip().replace("-", "+").replace("_", "/")
    s += "=" * ((-len(s)) % 4)  # tolerate stripped padding
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise GatewayError(400, "invalid base64 data")
    if len(raw) > config.MAX_FILE_SIZE:
        raise GatewayError(413, "file exceeds MAX_FILE_SIZE")
    return base64.b64encode(raw).decode("ascii")


def image_block(media_type: str, data: str) -> dict:
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
    return {"type": "document", "media_type": media_type, "data": normalize_b64(data)}


def pdf_to_text_block(data: str) -> dict:
    """Extract PDF text via pdfplumber and inline it as a text block."""
    data = normalize_b64(data)
    try:
        import pdfplumber
        raw = base64.b64decode(data)
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
