import unittest
from unittest.mock import patch

import prompt_enhancer
from prompt_enhancer import OUTPUT_SCHEMA, choose_task, parse_answer


class PromptEnhancerTests(unittest.TestCase):
    def test_routes_reference_requests_to_i2i(self):
        self.assertEqual(choose_task([]), "t2i")
        self.assertEqual(choose_task(["identity.png"]), "i2i")

    def test_parses_t2i_answer(self):
        answer = 'notes\n{"rewritten_prompt":"A detailed scene","wh_ratio":"16:9"}'
        parsed = parse_answer(answer, "t2i")
        self.assertTrue(parsed["parse_ok"])
        self.assertEqual(parsed["rewritten_prompt"], "A detailed scene")
        self.assertEqual(parsed["wh_ratio"], "16:9")
        self.assertEqual(parsed["ratio_follow"], "")

    def test_parses_i2i_answer_and_legacy_prompt_key(self):
        answer = '{"rewrited_prompt":"Keep <image1> identity","wh_ratio":"","ratio_follow":"<image1>"}'
        parsed = parse_answer(answer, "i2i")
        self.assertTrue(parsed["parse_ok"])
        self.assertEqual(parsed["rewritten_prompt"], "Keep <image1> identity")
        self.assertEqual(parsed["ratio_follow"], "<image1>")

    def test_rejects_unparseable_answer(self):
        parsed = parse_answer("not json", "t2i")
        self.assertFalse(parsed["parse_ok"])
        self.assertEqual(parsed["rewritten_prompt"], "")

    def test_structured_output_schema_requires_expected_fields(self):
        self.assertEqual(
            set(OUTPUT_SCHEMA["required"]),
            {"rewritten_prompt", "wh_ratio", "ratio_follow"},
        )
        self.assertFalse(OUTPUT_SCHEMA["additionalProperties"])

    def test_vllm_parse_failure_retries_once_with_larger_budget(self):
        valid = '{"rewritten_prompt":"A stable scene","wh_ratio":"3:2","ratio_follow":""}'
        calls = []

        def fake_request(profile, prompt, references, seed, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return "not json", 1.0, "length"
            return valid, 2.0, "stop"

        with (patch.object(prompt_enhancer, "PROMPT_ENHANCER_BACKEND", "vllm"),
              patch.object(prompt_enhancer, "_vllm_request", side_effect=fake_request)):
            result = prompt_enhancer.enhance_prompt("test", [], seed=7)

        self.assertTrue(result["retried"])
        self.assertEqual(result["rewritten_prompt"], "A stable scene")
        self.assertEqual(result["elapsed_seconds"], 3.0)
        self.assertEqual(calls[1]["max_tokens"], 1536)
        self.assertEqual(calls[1]["temperature"], 0.2)
        self.assertTrue(calls[1]["structured"])
        self.assertEqual(result["finish_reason"], "stop")

    def test_vllm_length_finish_retries_even_when_json_is_parseable(self):
        valid = '{"rewritten_prompt":"A scene","wh_ratio":"1:1","ratio_follow":""}'
        calls = []

        def fake_request(profile, prompt, references, seed, **kwargs):
            calls.append(kwargs)
            return valid, 1.0, "length" if len(calls) == 1 else "stop"

        with (patch.object(prompt_enhancer, "PROMPT_ENHANCER_BACKEND", "vllm"),
              patch.object(prompt_enhancer, "_vllm_request", side_effect=fake_request)):
            result = prompt_enhancer.enhance_prompt("test", [], seed=7)

        self.assertEqual(len(calls), 2)
        self.assertTrue(result["retried"])


if __name__ == "__main__":
    unittest.main()
