import math
import unittest

from evaluation.activezoom import (
    Interval,
    Particle,
    ZoomConfig,
    fallback_grid_proposals,
    generate_proposals,
    normalize_subclip_prediction,
    run_activezoom_sample,
    score_particles,
    select_final_candidate,
)


class ActiveZoomTests(unittest.TestCase):
    def test_proposals_are_clamped_and_deduplicated(self):
        config = ZoomConfig(min_window_seconds=8.0)
        proposals = generate_proposals(Interval(1.0, 4.0), 10.0, 1.0, config)
        keys = {(round(p.start), round(p.end)) for p in proposals}

        self.assertEqual(len(keys), len(proposals))
        self.assertTrue(proposals)
        for proposal in proposals:
            self.assertGreaterEqual(proposal.start, 0.0)
            self.assertLessEqual(proposal.end, 10.0)
            self.assertGreater(proposal.end, proposal.start)

    def test_invalid_baseline_uses_deterministic_grid(self):
        config = ZoomConfig(min_window_seconds=8.0, fallback_grid_size=4)
        first = fallback_grid_proposals(20.0, 1.0, config)
        second = fallback_grid_proposals(20.0, 1.0, config)

        self.assertEqual(first, second)
        self.assertEqual(first[0], Interval(0.0, 8.0))
        self.assertEqual(first[-1], Interval(12.0, 20.0))

    def test_relative_subclip_timestamps_are_shifted_to_global_time(self):
        raw, normalized, valid, mode = normalize_subclip_prediction(
            [[1.0, 4.0]], Interval(10.0, 20.0), 30.0, 1.0
        )

        self.assertEqual(raw, Interval(1.0, 4.0))
        self.assertEqual(normalized, Interval(11.0, 14.0))
        self.assertTrue(valid)
        self.assertEqual(mode, "relative")

    def test_absolute_subclip_timestamps_are_not_double_offset(self):
        _, normalized, valid, mode = normalize_subclip_prediction(
            [[20.0, 25.0]], Interval(10.0, 40.0), 60.0, 1.0
        )

        self.assertEqual(normalized, Interval(20.0, 25.0))
        self.assertTrue(valid)
        self.assertEqual(mode, "absolute")

    def test_invalid_intervals_are_rejected(self):
        _, normalized, valid, mode = normalize_subclip_prediction(
            [[5.0, 2.0]], Interval(0.0, 10.0), 10.0, 1.0
        )
        self.assertIsNone(normalized)
        self.assertFalse(valid)
        self.assertEqual(mode, "invalid")

        _, normalized, valid, _ = normalize_subclip_prediction(
            [[math.nan, 2.0]], Interval(0.0, 10.0), 10.0, 1.0
        )
        self.assertIsNone(normalized)
        self.assertFalse(valid)

    def test_posterior_prefers_candidate_with_strongest_agreement(self):
        config = ZoomConfig(min_window_seconds=8.0)
        baseline = Interval(10.0, 20.0)
        particles = [
            Particle(
                proposal=Interval(8.0, 24.0),
                prediction=Interval(11.0, 21.0),
                source="iter1_a",
                valid=True,
            ),
            Particle(
                proposal=Interval(9.0, 25.0),
                prediction=Interval(12.0, 22.0),
                source="iter1_b",
                valid=True,
            ),
            Particle(
                proposal=Interval(60.0, 90.0),
                prediction=Interval(70.0, 80.0),
                source="iter1_outlier",
                valid=True,
            ),
        ]

        score_particles(particles, baseline, 100.0, 1.0, config)
        outlier = particles[2]
        best = max(particles, key=lambda p: p.score_components["selection_posterior"])

        self.assertIn(best, particles[:2])
        self.assertLess(
            outlier.score_components["selection_posterior"],
            best.score_components["selection_posterior"],
        )

    def test_fallback_returns_baseline_when_posterior_is_diffuse(self):
        config = ZoomConfig(
            min_window_seconds=8.0,
            map_threshold=0.35,
            containment_weight=0.0,
            prior_weight=0.0,
            edge_penalty_weight=0.0,
        )
        baseline = Interval(10.0, 20.0)
        particles = [
            Particle(
                proposal=Interval(0.0, 15.0),
                prediction=Interval(1.0, 4.0),
                source="iter1_a",
                valid=True,
            ),
            Particle(
                proposal=Interval(25.0, 40.0),
                prediction=Interval(28.0, 31.0),
                source="iter1_b",
                valid=True,
            ),
            Particle(
                proposal=Interval(70.0, 85.0),
                prediction=Interval(75.0, 79.0),
                source="iter1_c",
                valid=True,
            ),
        ]
        score_particles(particles, baseline, 100.0, 1.0, config)
        result = select_final_candidate(
            particles, baseline, "baseline answer", 100.0, 1.0, config
        )

        self.assertTrue(result.fallback_used)
        self.assertEqual(result.selected_source, "baseline")
        self.assertEqual(result.interval, baseline)
        self.assertEqual(result.fallback_reason, "posterior_diffuse")

    def test_run_activezoom_sample_with_fake_runner(self):
        class FakeRunner:
            def predict_interval(self, anno, proposal):
                del anno, proposal
                return {
                    "answer": "The event happens in 12 - 18 seconds.",
                    "timestamps": [[12.0, 18.0]],
                }

        anno = {
            "video_path": "video.mp4",
            "query": "open the door",
            "duration": 40.0,
            "span": [[11.0, 19.0]],
        }
        baseline = {
            "timestamps": [[8.0, 22.0]],
            "answers": "The event happens in 8 - 22 seconds.",
        }

        result = run_activezoom_sample(
            anno,
            baseline,
            FakeRunner(),
            ZoomConfig(iterations=1, min_window_seconds=8.0),
            unit=1.0,
        )

        self.assertFalse(result.fallback_used)
        self.assertEqual(result.interval, Interval(12.0, 18.0))
        self.assertEqual(result.answer, "The event happens in 12 - 18 seconds.")
        self.assertGreaterEqual(len(result.particles), 2)


if __name__ == "__main__":
    unittest.main()
