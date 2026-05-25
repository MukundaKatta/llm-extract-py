# llm-extract-py

Extract structured data from LLM output: JSON, code blocks, lists, booleans, and more.

```bash
pip install llm-extract-py
```

## Quick start

```python
from llm_extract import extract_json, extract_code, extract_list, extract_bool, strip_thinking

response = '```json\n{"name": "Alice", "score": 95}\n```'

r = extract_json(response)
if r.found:
    print(r.value["name"])  # Alice
```

## Functions

### `extract_json(text)`
Finds the first valid JSON object or array. Handles ` ```json ` fences and bare JSON in prose.

```python
r = extract_json('Result: {"ok": true}')
# r.value == {"ok": True}
```

### `extract_code(text, language=None)`
Extracts code from a fenced code block.

```python
r = extract_code(response, language="python")
# r.value == 'def hello():\n    ...'
```

### `extract_all_code_blocks(text)`
Returns all fenced blocks as `[{"language": str, "code": str}, ...]`.

### `extract_list(text)`
Extracts bulleted (`-`, `*`, `•`) or numbered (`1.`, `2)`) lists.

```python
r = extract_list("- apple\n- banana\n- cherry")
# r.value == ["apple", "banana", "cherry"]
```

### `strip_thinking(text)`
Strips `<thinking>...</thinking>` tags from Claude extended-thinking output.

```python
clean = strip_thinking("<thinking>reasoning</thinking>The answer is 42.")
# "The answer is 42."
```

### `extract_key_value(text, key)`
Extracts a value from `Key: value`, `**Key**: value`, or `key = value` patterns.

### `extract_bool(text)`
Returns `True`, `False`, or `None` from yes/no LLM answers.

```python
extract_bool("Yes, that is correct.").value  # True
extract_bool("No, that is wrong.").value     # False
extract_bool("It depends.").value            # None
```

## ExtractResult

All functions return `ExtractResult(value, raw, found)`.

```python
r = extract_json(text)
r.found  # bool
r.value  # parsed value or None
r.raw    # original matched text
```

## Zero dependencies
