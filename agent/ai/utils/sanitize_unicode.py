"""Unicode cleanup applied before values cross an SDK/JSON boundary."""

from __future__ import annotations


def sanitize_surrogates(value: str) -> str:
    """Replace unpaired UTF-16 surrogate code points with U+FFFD."""
    return "".join(
        "\ufffd" if 0xD800 <= ord(character) <= 0xDFFF else character for character in value
    )


__all__ = ["sanitize_surrogates"]
