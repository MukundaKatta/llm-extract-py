"""llm-extract-py — extract structured data from LLM output (JSON, code blocks, lists, etc.)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractResult:
    value: Any
    raw: str
    found: bool


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str, *, allow_partial: bool = False) -> ExtractResult:
    """
    Extract the first valid JSON object or array from text.
    Handles ```json ... ``` fences and bare JSON embedded in prose.

    Args:
        text: The LLM response text.
        allow_partial: If True, try to return partial JSON on parse failure.
    """
    # Try fenced code blocks first
    fence_patterns = [
        r"```json\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
    ]
    for pat in fence_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            try:
                return ExtractResult(value=json.loads(candidate), raw=candidate, found=True)
            except json.JSONDecodeError:
                pass

    # Try to find a JSON object or array in the raw text
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if not in_string:
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return ExtractResult(
                                value=json.loads(candidate),
                                raw=candidate,
                                found=True,
                            )
                        except json.JSONDecodeError:
                            break

    return ExtractResult(value=None, raw=text, found=False)


# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------

def extract_code(text: str, language: str | None = None) -> ExtractResult:
    """
    Extract code from a fenced code block.

    Args:
        text: The LLM response text.
        language: Optional language hint (e.g. 'python', 'typescript').
                  If None, accepts any language tag.
    """
    lang_pat = re.escape(language) if language else r"[a-zA-Z0-9_+-]*"
    pattern = rf"```{lang_pat}\s*([\s\S]*?)```"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return ExtractResult(value=m.group(1).strip(), raw=m.group(0), found=True)

    # Fall back: any code block if no language match
    if language:
        m2 = re.search(r"```\s*([\s\S]*?)```", text)
        if m2:
            return ExtractResult(value=m2.group(1).strip(), raw=m2.group(0), found=True)

    return ExtractResult(value=None, raw=text, found=False)


def extract_all_code_blocks(text: str) -> list[dict]:
    """
    Extract all fenced code blocks from text.
    Returns list of {"language": str, "code": str} dicts.
    """
    pattern = r"```([a-zA-Z0-9_+-]*)\s*([\s\S]*?)```"
    results = []
    for m in re.finditer(pattern, text):
        results.append({"language": m.group(1) or "", "code": m.group(2).strip()})
    return results


# ---------------------------------------------------------------------------
# List / bullet extraction
# ---------------------------------------------------------------------------

def extract_list(text: str) -> ExtractResult:
    """
    Extract a bulleted or numbered list from text.
    Returns a list of stripped strings.
    """
    pattern = r"^[ \t]*(?:[-*•]|\d+[.)]) +(.+)$"
    items = [m.group(1).strip() for m in re.finditer(pattern, text, re.MULTILINE)]
    if items:
        return ExtractResult(value=items, raw=text, found=True)
    return ExtractResult(value=[], raw=text, found=False)


# ---------------------------------------------------------------------------
# Thinking tag strip
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Strip <thinking>...</thinking> tags from Claude extended-thinking output."""
    return re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE).strip()


# ---------------------------------------------------------------------------
# Key-value extraction
# ---------------------------------------------------------------------------

def extract_key_value(text: str, key: str) -> ExtractResult:
    """
    Extract a value associated with a key in LLM output.
    Handles formats: "Key: value", "**Key**: value", "key = value".
    """
    patterns = [
        rf"\*{{0,2}}{re.escape(key)}\*{{0,2}}\s*:\s*(.+?)(?:\n|$)",
        rf"{re.escape(key)}\s*=\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return ExtractResult(value=m.group(1).strip(), raw=m.group(0), found=True)
    return ExtractResult(value=None, raw=text, found=False)


# ---------------------------------------------------------------------------
# Yes/No extraction
# ---------------------------------------------------------------------------

def extract_bool(text: str) -> ExtractResult:
    """
    Extract a boolean answer from LLM output.
    Returns True/False or None if ambiguous.
    """
    lowered = text.lower().strip()
    positive = ["yes", "true", "correct", "affirmative", "agree", "right", "confirmed"]
    negative = ["no", "false", "incorrect", "negative", "disagree", "wrong", "denied"]

    for word in positive:
        if re.search(rf"\b{word}\b", lowered):
            return ExtractResult(value=True, raw=text, found=True)
    for word in negative:
        if re.search(rf"\b{word}\b", lowered):
            return ExtractResult(value=False, raw=text, found=True)

    return ExtractResult(value=None, raw=text, found=False)


__all__ = [
    "extract_json",
    "extract_code",
    "extract_all_code_blocks",
    "extract_list",
    "strip_thinking",
    "extract_key_value",
    "extract_bool",
    "ExtractResult",
]
