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
                return ExtractResult(
                    value=json.loads(candidate), raw=candidate, found=True
                )
            except json.JSONDecodeError:
                pass

    # Try to find a JSON object or array in the raw text. Scan candidates of
    # both kinds and return whichever valid JSON value *starts earliest* in the
    # text, so the documented "first" contract is honored regardless of whether
    # an object or an array appears first.
    best: ExtractResult | None = None
    best_start = len(text)
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1 or start >= best_start:
            continue
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            best = ExtractResult(
                                value=json.loads(candidate),
                                raw=candidate,
                                found=True,
                            )
                            best_start = start
                        except json.JSONDecodeError:
                            pass
                        break

    if best is not None:
        return best
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
    return re.sub(
        r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE
    ).strip()


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

    The *earliest-occurring* sentiment word wins, so a reply like
    ``"No, that is not correct."`` resolves to ``False`` even though it also
    contains the positive word "correct". This mirrors how a human reads the
    leading word as the answer.
    """
    lowered = text.lower().strip()
    positive = ["yes", "true", "correct", "affirmative", "agree", "right", "confirmed"]
    negative = ["no", "false", "incorrect", "negative", "disagree", "wrong", "denied"]

    best_value: bool | None = None
    best_index = len(lowered) + 1
    for word in positive:
        m = re.search(rf"\b{word}\b", lowered)
        if m is not None and m.start() < best_index:
            best_value, best_index = True, m.start()
    for word in negative:
        m = re.search(rf"\b{word}\b", lowered)
        if m is not None and m.start() < best_index:
            best_value, best_index = False, m.start()

    if best_value is None:
        return ExtractResult(value=None, raw=text, found=False)
    return ExtractResult(value=best_value, raw=text, found=True)


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
