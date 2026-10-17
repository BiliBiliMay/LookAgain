import unittest
from evaluation.qwen_transfer_control import fixed_windows, run_fixed_search
from evaluation.timelens_search import SearchConfig
from tests.evaluation.test_timelens_search import CriticEventRunner, EmptyRunner


class ControlTests(unittest.TestCase):
    def test_windows_cover_ends_and_deduplicate_short_video(self):
        windows = fixed_windows(100)
        self.assertEqual(len(windows), 12)
        self.assertEqual(min(w.start for w in windows), 0)
        self.assertEqual(max(w.end for w in windows), 100)
        self.assertTrue(all(w.is_valid(100) for w in windows))
        self.assertEqual(len(fixed_windows(3)), 1)

    def test_accepts_agreed_challenger_within_fifteen_calls(self):
        runner = CriticEventRunner([42, 47])
        result = run_fixed_search({'duration': 100},
                                 {'timestamps': [[2, 8]], 'answers': 'baseline'},
                                 runner, SearchConfig())
        self.assertEqual(result.interval.as_list(), [42, 47])
        self.assertEqual(len(runner.calls), 12)
        self.assertEqual(len(runner.critic_calls), 2)
        self.assertEqual(result.search_cost['total_model_calls'], 15)
        self.assertFalse(result.graph)

    def test_order_disagreement_keeps_baseline(self):
        runner = CriticEventRunner([42, 47], forced_choices=['B', 'B'])
        result = run_fixed_search({'duration': 100}, {'timestamps': [[2, 8]]},
                                 runner, SearchConfig())
        self.assertEqual(result.interval.as_list(), [2, 8])
        self.assertEqual(result.fallback_reason, 'critic_order_disagreement')

    def test_invalid_proposals_keep_baseline_without_verifier(self):
        result = run_fixed_search({'duration': 100}, {'timestamps': [[2, 8]]},
                                 EmptyRunner(), SearchConfig())
        self.assertEqual(result.interval.as_list(), [2, 8])
        self.assertEqual(result.search_cost['total_model_calls'], 13)

    def test_invalid_baseline_retains_failure_and_counts_attempt(self):
        runner = EmptyRunner()
        result = run_fixed_search({'duration': 100}, {'timestamps': [[110, 120]]},
                                 runner, SearchConfig())
        self.assertEqual(result.interval.as_list(), [110, 120])
        self.assertEqual(result.search_cost['total_model_calls'], 1)
        self.assertFalse(runner.calls)


if __name__ == '__main__':
    unittest.main()
