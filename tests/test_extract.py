import pytest
from llm_extract import (
    extract_json,
    extract_code,
    extract_all_code_blocks,
    extract_list,
    strip_thinking,
    extract_key_value,
    extract_bool,
    ExtractResult,
)


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def test_json_object_in_fence():
    text = '```json\n{"key": "value"}\n```'
    r = extract_json(text)
    assert r.found
    assert r.value == {"key": "value"}


def test_json_array_in_fence():
    text = '```json\n[1, 2, 3]\n```'
    r = extract_json(text)
    assert r.found
    assert r.value == [1, 2, 3]


def test_json_bare_object_in_prose():
    text = 'The result is {"status": "ok", "count": 5} as shown.'
    r = extract_json(text)
    assert r.found
    assert r.value["status"] == "ok"


def test_json_bare_array():
    text = 'Items: [1, 2, 3] end'
    r = extract_json(text)
    assert r.found
    assert r.value == [1, 2, 3]


def test_json_nested():
    text = '```json\n{"a": {"b": [1, 2]}}\n```'
    r = extract_json(text)
    assert r.found
    assert r.value["a"]["b"] == [1, 2]


def test_json_not_found():
    r = extract_json("no json here")
    assert not r.found
    assert r.value is None


def test_json_generic_fence_fallback():
    text = '```\n{"x": 1}\n```'
    r = extract_json(text)
    assert r.found
    assert r.value == {"x": 1}


def test_json_with_prose_around():
    text = "Here is the answer:\n```json\n{\"ok\": true}\n```\nDone."
    r = extract_json(text)
    assert r.found
    assert r.value["ok"] is True


# ---------------------------------------------------------------------------
# extract_code
# ---------------------------------------------------------------------------

def test_code_python_fence():
    text = '```python\nprint("hello")\n```'
    r = extract_code(text, language="python")
    assert r.found
    assert 'print("hello")' in r.value


def test_code_any_language():
    text = '```typescript\nconst x = 1;\n```'
    r = extract_code(text)
    assert r.found
    assert "const x = 1" in r.value


def test_code_wrong_language_falls_back():
    text = '```typescript\nconst x = 1;\n```'
    r = extract_code(text, language="python")
    assert r.found  # falls back to any block


def test_code_not_found():
    r = extract_code("no code here")
    assert not r.found
    assert r.value is None


def test_code_strips_whitespace():
    text = "```python\n   x = 1   \n```"
    r = extract_code(text, language="python")
    assert r.value == "x = 1"


# ---------------------------------------------------------------------------
# extract_all_code_blocks
# ---------------------------------------------------------------------------

def test_all_code_blocks_multiple():
    text = "```python\nx=1\n```\n\n```js\nconsole.log()\n```"
    blocks = extract_all_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0]["language"] == "python"
    assert blocks[1]["language"] == "js"


def test_all_code_blocks_no_language():
    text = "```\nsome code\n```"
    blocks = extract_all_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]["language"] == ""
    assert blocks[0]["code"] == "some code"


def test_all_code_blocks_empty():
    blocks = extract_all_code_blocks("no code blocks here")
    assert blocks == []


# ---------------------------------------------------------------------------
# extract_list
# ---------------------------------------------------------------------------

def test_list_bullet_dash():
    text = "Items:\n- apple\n- banana\n- cherry"
    r = extract_list(text)
    assert r.found
    assert r.value == ["apple", "banana", "cherry"]


def test_list_numbered():
    text = "Steps:\n1. First\n2. Second\n3. Third"
    r = extract_list(text)
    assert r.found
    assert r.value == ["First", "Second", "Third"]


def test_list_asterisk():
    text = "* one\n* two\n* three"
    r = extract_list(text)
    assert r.found
    assert len(r.value) == 3


def test_list_not_found():
    r = extract_list("just a paragraph with no bullets")
    assert not r.found
    assert r.value == []


def test_list_strips_items():
    text = "-   spaced item   "
    r = extract_list(text)
    assert r.found
    assert r.value[0] == "spaced item"


# ---------------------------------------------------------------------------
# strip_thinking
# ---------------------------------------------------------------------------

def test_strip_thinking_basic():
    text = "<thinking>This is my reasoning.</thinking>The answer is 42."
    assert strip_thinking(text) == "The answer is 42."


def test_strip_thinking_multiline():
    text = "<thinking>\nLine 1\nLine 2\n</thinking>\nFinal answer."
    assert strip_thinking(text) == "Final answer."


def test_strip_thinking_no_tags():
    text = "Just a plain response."
    assert strip_thinking(text) == "Just a plain response."


def test_strip_thinking_case_insensitive():
    text = "<THINKING>hidden</THINKING>visible"
    assert strip_thinking(text) == "visible"


def test_strip_thinking_multiple():
    text = "<thinking>a</thinking>middle<thinking>b</thinking>end"
    result = strip_thinking(text)
    assert "middle" in result
    assert "end" in result
    assert "a" not in result
    assert "b" not in result


# ---------------------------------------------------------------------------
# extract_key_value
# ---------------------------------------------------------------------------

def test_key_value_colon():
    r = extract_key_value("Status: active\nOther: stuff", "Status")
    assert r.found
    assert r.value == "active"


def test_key_value_bold():
    r = extract_key_value("**Score**: 95\n**Grade**: A", "Score")
    assert r.found
    assert r.value == "95"


def test_key_value_equals():
    r = extract_key_value("name = Alice\nage = 30", "name")
    assert r.found
    assert r.value == "Alice"


def test_key_value_case_insensitive():
    r = extract_key_value("status: done", "STATUS")
    assert r.found
    assert r.value == "done"


def test_key_value_not_found():
    r = extract_key_value("unrelated text", "missing_key")
    assert not r.found
    assert r.value is None


# ---------------------------------------------------------------------------
# extract_bool
# ---------------------------------------------------------------------------

def test_bool_yes():
    assert extract_bool("Yes, I agree.").value is True


def test_bool_no():
    assert extract_bool("No, that is incorrect.").value is False


def test_bool_true_word():
    assert extract_bool("The statement is true.").value is True


def test_bool_false_word():
    assert extract_bool("The statement is false.").value is False


def test_bool_correct():
    assert extract_bool("That is correct.").value is True


def test_bool_wrong():
    assert extract_bool("That is wrong.").value is False


def test_bool_ambiguous():
    r = extract_bool("Maybe, it depends on context.")
    assert r.value is None
    assert not r.found


def test_bool_found_flag():
    assert extract_bool("yes").found is True
    assert extract_bool("hmm").found is False
