import unittest

from prompt_enhancer import choose_task, parse_answer


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


if __name__ == "__main__":
    unittest.main()
