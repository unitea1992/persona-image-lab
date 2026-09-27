import unittest

from lab import preview_steps_for


class PreviewScheduleTests(unittest.TestCase):
    def test_forty_steps_preview_only_near_completion(self):
        self.assertEqual(preview_steps_for(40), {36, 39})

    def test_short_runs_do_not_preview_the_final_step(self):
        self.assertEqual(preview_steps_for(6), {5})
        self.assertEqual(preview_steps_for(1), set())


if __name__ == "__main__":
    unittest.main()
