"""Standard-library unittest suite for llm_extract.

Runnable without any third-party dependencies::

    python3 -m unittest discover -s tests
"""

import os
import sys
import unittest

# Make the package importable when running directly from a checkout
# (i.e. without an editable install), so ``unittest discover`` works as-is.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_extract import (  # noqa: E402
    ExtractResult,
    extract_all_code_blocks,
    extract_bool,
    extract_code,
    extract_json,
    extract_key_value,
    extract_list,
    strip_thinking,
)


class TestExtractJson(unittest.TestCase):
    def test_json_object_in_fence(self):
        r = extract_json('```json\n{"key": "value"}\n```')
        self.assertTrue(r.found)
        self.assertEqual(r.value, {"key": "value"})

    def test_json_array_in_fence(self):
        r = extract_json("```json\n[1, 2, 3]\n```")
        self.assertTrue(r.found)
        self.assertEqual(r.value, [1, 2, 3])

    def test_json_bare_object_in_prose(self):
        r = extract_json('The result is {"status": "ok", "count": 5} as shown.')
        self.assertTrue(r.found)
        self.assertEqual(r.value["status"], "ok")

    def test_json_bare_array(self):
        r = extract_json("Items: [1, 2, 3] end")
        self.assertTrue(r.found)
        self.assertEqual(r.value, [1, 2, 3])

    def test_json_nested(self):
        r = extract_json('```json\n{"a": {"b": [1, 2]}}\n```')
        self.assertTrue(r.found)
        self.assertEqual(r.value["a"]["b"], [1, 2])

    def test_json_not_found(self):
        r = extract_json("no json here")
        self.assertFalse(r.found)
        self.assertIsNone(r.value)

    def test_json_generic_fence_fallback(self):
        r = extract_json('```\n{"x": 1}\n```')
        self.assertTrue(r.found)
        self.assertEqual(r.value, {"x": 1})

    def test_json_with_prose_around(self):
        text = 'Here is the answer:\n```json\n{"ok": true}\n```\nDone.'
        r = extract_json(text)
        self.assertTrue(r.found)
        self.assertIs(r.value["ok"], True)

    def test_json_array_before_object_returns_array(self):
        # The array appears first in the text, so it should be returned even
        # though an object follows.
        r = extract_json('First an array [1, 2, 3] then object {"k": 1}')
        self.assertTrue(r.found)
        self.assertEqual(r.value, [1, 2, 3])

    def test_json_object_before_array_returns_object(self):
        r = extract_json('Object {"k": 1} comes before array [1, 2, 3]')
        self.assertTrue(r.found)
        self.assertEqual(r.value, {"k": 1})

    def test_json_string_containing_brace(self):
        r = extract_json('{"msg": "a } b", "n": 1}')
        self.assertTrue(r.found)
        self.assertEqual(r.value, {"msg": "a } b", "n": 1})

    def test_json_invalid_fence_falls_back_to_bare(self):
        # A fence whose contents are not valid JSON should not stop a valid bare
        # JSON value later in the text from being found.
        r = extract_json('```json\nnot valid\n```\n{"a": 1}')
        self.assertTrue(r.found)
        self.assertEqual(r.value, {"a": 1})

    def test_json_result_type(self):
        r = extract_json('{"a": 1}')
        self.assertIsInstance(r, ExtractResult)
        self.assertEqual(r.raw, '{"a": 1}')


class TestExtractCode(unittest.TestCase):
    def test_code_python_fence(self):
        r = extract_code('```python\nprint("hello")\n```', language="python")
        self.assertTrue(r.found)
        self.assertIn('print("hello")', r.value)

    def test_code_any_language(self):
        r = extract_code("```typescript\nconst x = 1;\n```")
        self.assertTrue(r.found)
        self.assertIn("const x = 1", r.value)

    def test_code_wrong_language_falls_back(self):
        r = extract_code("```typescript\nconst x = 1;\n```", language="python")
        self.assertTrue(r.found)  # falls back to any block

    def test_code_not_found(self):
        r = extract_code("no code here")
        self.assertFalse(r.found)
        self.assertIsNone(r.value)

    def test_code_strips_whitespace(self):
        r = extract_code("```python\n   x = 1   \n```", language="python")
        self.assertEqual(r.value, "x = 1")


class TestExtractAllCodeBlocks(unittest.TestCase):
    def test_multiple(self):
        text = "```python\nx=1\n```\n\n```js\nconsole.log()\n```"
        blocks = extract_all_code_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["language"], "python")
        self.assertEqual(blocks[1]["language"], "js")

    def test_no_language(self):
        blocks = extract_all_code_blocks("```\nsome code\n```")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["language"], "")
        self.assertEqual(blocks[0]["code"], "some code")

    def test_empty(self):
        self.assertEqual(extract_all_code_blocks("no code blocks here"), [])


class TestExtractList(unittest.TestCase):
    def test_bullet_dash(self):
        r = extract_list("Items:\n- apple\n- banana\n- cherry")
        self.assertTrue(r.found)
        self.assertEqual(r.value, ["apple", "banana", "cherry"])

    def test_numbered(self):
        r = extract_list("Steps:\n1. First\n2. Second\n3. Third")
        self.assertTrue(r.found)
        self.assertEqual(r.value, ["First", "Second", "Third"])

    def test_asterisk(self):
        r = extract_list("* one\n* two\n* three")
        self.assertTrue(r.found)
        self.assertEqual(len(r.value), 3)

    def test_not_found(self):
        r = extract_list("just a paragraph with no bullets")
        self.assertFalse(r.found)
        self.assertEqual(r.value, [])

    def test_strips_items(self):
        r = extract_list("-   spaced item   ")
        self.assertTrue(r.found)
        self.assertEqual(r.value[0], "spaced item")


class TestStripThinking(unittest.TestCase):
    def test_basic(self):
        text = "<thinking>This is my reasoning.</thinking>The answer is 42."
        self.assertEqual(strip_thinking(text), "The answer is 42.")

    def test_multiline(self):
        text = "<thinking>\nLine 1\nLine 2\n</thinking>\nFinal answer."
        self.assertEqual(strip_thinking(text), "Final answer.")

    def test_no_tags(self):
        self.assertEqual(strip_thinking("Just a plain response."), "Just a plain response.")

    def test_case_insensitive(self):
        self.assertEqual(strip_thinking("<THINKING>hidden</THINKING>visible"), "visible")

    def test_multiple(self):
        text = "<thinking>a</thinking>middle<thinking>b</thinking>end"
        result = strip_thinking(text)
        self.assertIn("middle", result)
        self.assertIn("end", result)
        self.assertNotIn("a", result)
        self.assertNotIn("b", result)


class TestExtractKeyValue(unittest.TestCase):
    def test_colon(self):
        r = extract_key_value("Status: active\nOther: stuff", "Status")
        self.assertTrue(r.found)
        self.assertEqual(r.value, "active")

    def test_bold(self):
        r = extract_key_value("**Score**: 95\n**Grade**: A", "Score")
        self.assertTrue(r.found)
        self.assertEqual(r.value, "95")

    def test_equals(self):
        r = extract_key_value("name = Alice\nage = 30", "name")
        self.assertTrue(r.found)
        self.assertEqual(r.value, "Alice")

    def test_case_insensitive(self):
        r = extract_key_value("status: done", "STATUS")
        self.assertTrue(r.found)
        self.assertEqual(r.value, "done")

    def test_not_found(self):
        r = extract_key_value("unrelated text", "missing_key")
        self.assertFalse(r.found)
        self.assertIsNone(r.value)


class TestExtractBool(unittest.TestCase):
    def test_yes(self):
        self.assertIs(extract_bool("Yes, I agree.").value, True)

    def test_no(self):
        self.assertIs(extract_bool("No, that is incorrect.").value, False)

    def test_true_word(self):
        self.assertIs(extract_bool("The statement is true.").value, True)

    def test_false_word(self):
        self.assertIs(extract_bool("The statement is false.").value, False)

    def test_correct(self):
        self.assertIs(extract_bool("That is correct.").value, True)

    def test_wrong(self):
        self.assertIs(extract_bool("That is wrong.").value, False)

    def test_ambiguous(self):
        r = extract_bool("Maybe, it depends on context.")
        self.assertIsNone(r.value)
        self.assertFalse(r.found)

    def test_found_flag(self):
        self.assertTrue(extract_bool("yes").found)
        self.assertFalse(extract_bool("hmm").found)

    def test_leading_negative_wins_over_later_positive(self):
        # Regression: "No, that is not correct." used to return True because
        # every positive word was checked before any negative word, so the
        # later "correct" masked the leading "No". The earliest sentiment word
        # must win.
        r = extract_bool("No, that is not correct.")
        self.assertIs(r.value, False)
        self.assertTrue(r.found)

    def test_leading_positive_wins_over_later_negative(self):
        r = extract_bool("Yes, although it is not wrong to ask.")
        self.assertIs(r.value, True)


if __name__ == "__main__":
    unittest.main()
