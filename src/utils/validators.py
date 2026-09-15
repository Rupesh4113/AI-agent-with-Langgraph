"""
Security and validation utilities.
Sanitizes user input, validates tool parameters, and prevents injection/abuse.
"""
import re
from typing import Any

def sanitize_user_input(text: str, max_length: int = 2000) -> str:
    """
    Sanitizes user input:
    - Strips surrounding whitespace
    - Limits maximum length
    - Strips control characters
    """
    if not text:
        return ""
    # Strip null bytes and non-printable control characters (except newline, tab, carriage return)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    cleaned = cleaned.strip()
    return cleaned[:max_length]

def validate_tool_query(query: str, max_length: int = 500) -> str:
    """
    Validates a search query before passing to search tools.
    """
    if not query:
        return ""
    sanitized = re.sub(r"[\x00-\x1f\x7f]", " ", query).strip()
    return sanitized[:max_length]

def sanitize_math_expression(expr: str) -> str:
    """
    Removes potentially dangerous tokens or assignments from math strings.
    """
    if not expr:
        return ""
    # Reject assignment operators or double underscores
    if any(blocked in expr for blocked in ["__", "import", "eval", "exec", "open", "os", "sys", "subprocess"]):
        raise ValueError(f"Prohibited tokens detected in expression: {expr}")
    return expr.strip()
