"""Shared request-translation helpers used by all three adapters.

Only the genuinely-common shape lives here: joining system text and normalising
roles. Each protocol's content-block structure (image/document/inline_data) is
dictated by its own spec and stays in the adapter — forcing it through one
interface would be a false seam.
"""
from typing import Optional


def join_texts(parts: list[dict], key: str = "text", sep: str = "\n") -> Optional[str]:
    """Join the `key` field of every part that has it; None if nothing remains."""
    joined = sep.join(p[key] for p in parts if p.get(key))
    return joined or None


def to_role(role: Optional[str], assistant_aliases: tuple[str, ...] = ("assistant",)) -> str:
    """Normalise a protocol message role to canonical 'user' | 'assistant'."""
    return "assistant" if role in assistant_aliases else "user"
