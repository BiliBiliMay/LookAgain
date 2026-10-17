import unittest

from evaluation.timelens_search import (
    CertificateState,
    Interval,
    SearchCandidate,
    SearchConfig,
    TemporalHypothesisGraph,
    generate_coarse_windows,
    normalize_timestamp_prediction,
    run_timelens_search_sample,
    update_certificate_state,
)


class FakePrediction:
    def __init__(self, answer, timestamps):
        self.answer = answer
        self.timestamps = timestamps


class EventRunner:
    def __init__(self, event):
        self.event = Interval(*event)
        self.calls = []

    def predict_interval(self, anno, clip, fps_override=None):
        self.calls.append((clip, fps_override))
        if clip.start <= self.event.start and clip.end >= self.event.end:
            return FakePrediction(
                "The event happens in 42.0 - 47.0 seconds.",
                [[self.event.start, self.event.end]],
            )
        return FakePrediction("No matching event.", [])


class EmptyRunner:
    def __init__(self):
        self.calls = []

    def predict_interval(self, anno, clip, fps_override=None):
        self.calls.append((clip, fps_override))
        return FakePrediction("No timestamps.", [])


class CriticEventRunner(EventRunner):
    def __init__(self, event, forced_choices=None):
        super().__init__(event)
        self.critic_calls = []
        self.forced_choices = list(forced_choices or [])

    def verify_pair(self, anno, candidate_a, candidate_b, fps_override=None):
        self.critic_calls.append((candidate_a, candidate_b, fps_override))
        if self.forced_choices:
            choice = self.forced_choices.pop(0)
        else:
            iou_a = self._iou(candidate_a, self.event)
            iou_b = self._iou(candidate_b, self.event)
            choice = "A" if iou_a > iou_b else "B"
        return {
            "choice": choice,
            "confidence": 0.9,
            "failure_tag": "too_early",
            "answer": "synthetic critic verdict",
        }

    @staticmethod
    def _iou(a, b):
        intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
        union = max(a.end, b.end) - min(a.start, b.start)
        return intersection / union if union > 0 else 0.0


class TimeLensSearchTests(unittest.TestCase):
    def test_certificate_updates_are_focal_and_monotone(self):
        config = SearchConfig(
            certificate_boundary_iou=0.55,
            certificate_support_iou=0.55,
            certificate_contradiction_weight=0.45,
        )
        focal = SearchCandidate(
            proposal=Interval(30.0, 60.0),
            prediction=Interval(42.0, 47.0),
            source="focal",
            stage="lookagain",
            valid=True,
        )
        start_probe = SearchCandidate(
            proposal=Interval(37.0, 47.0),
            prediction=Interval(42.0, 47.0),
            source="start_probe",
            stage="lookagain",
            valid=True,
        )
        end_probe = SearchCandidate(
            proposal=Interval(42.0, 52.0),
            prediction=Interval(42.0, 47.0),
            source="end_probe",
            stage="lookagain",
            valid=True,
        )

        state = CertificateState(focal_source=focal.source)
        state = update_certificate_state(
            state, focal, start_probe, "check_start", 1.0, config
        )
        self.assertTrue(state.left_boundary)
        self.assertTrue(state.support)
        self.assertFalse(state.right_boundary)

        state = update_certificate_state(
            state, focal, end_probe, "check_end", 1.0, config
        )
        self.assertTrue(state.is_certified(focal))

        contradictory = SearchCandidate(
            proposal=Interval(30.0, 60.0),
            prediction=Interval(75.0, 85.0),
            source="contradictory",
            stage="lookagain",
            valid=True,
        )
        state = update_certificate_state(
            state, focal, contradictory, "support_probe", 1.0, config
        )
        self.assertTrue(state.contradiction)
        self.assertFalse(state.is_certified(focal))

    def test_graph_merges_transposition_and_records_second_parent(self):
        config = SearchConfig(graph_merge_iou=0.90)
        graph = TemporalHypothesisGraph(config, duration=100.0, unit=1.0)
        first = SearchCandidate(
            proposal=Interval(20.0, 40.0),
            prediction=Interval(24.0, 31.0),
            source="first",
            stage="graph",
            valid=True,
        )
        graph.add_candidate(first, parent_source="baseline", action="move_left")

        equivalent = graph.find_equivalent_proposal(Interval(20.5, 40.5))
        self.assertIs(equivalent, first)
        graph.add_transition("second_parent", first.source, "shrink_right")

        self.assertEqual(graph.transposition_hits, 1)
        self.assertIn("second_parent", first.parent_sources)
        self.assertTrue(
            any(
                edge.relation == "edit" and edge.action == "shrink_right"
                for edge in graph.edges
            )
        )

    def test_generate_coarse_windows_evenly_covers_duration(self):
        config = SearchConfig(coarse_grid_size=5, coarse_window_fraction=0.2)

        windows = generate_coarse_windows(100.0, config)

        self.assertEqual([w.as_list() for w in windows], [
            [0.0, 20.0],
            [20.0, 40.0],
            [40.0, 60.0],
            [60.0, 80.0],
            [80.0, 100.0],
        ])

    def test_generate_coarse_windows_uses_full_window_for_short_videos(self):
        config = SearchConfig(
            coarse_grid_size=9,
            coarse_window_fraction=0.25,
            min_window_seconds=12.0,
        )

        windows = generate_coarse_windows(8.0, config)

        self.assertEqual([w.as_list() for w in windows], [[0.0, 8.0]])

    def test_normalize_timestamp_prediction_accepts_absolute_and_relative(self):
        proposal = Interval(20.0, 40.0)

        raw, normalized, valid, mode = normalize_timestamp_prediction(
            [[24.0, 31.0]], proposal, duration=100.0, unit=1.0
        )
        self.assertEqual(raw, Interval(24.0, 31.0))
        self.assertEqual(normalized, Interval(24.0, 31.0))
        self.assertTrue(valid)
        self.assertEqual(mode, "absolute")

        raw, normalized, valid, mode = normalize_timestamp_prediction(
            [[2.0, 8.0]], proposal, duration=100.0, unit=1.0
        )
        self.assertEqual(raw, Interval(2.0, 8.0))
        self.assertEqual(normalized, Interval(22.0, 28.0))
        self.assertTrue(valid)
        self.assertEqual(mode, "relative")

    def test_search_refines_event_and_records_cost(self):
        anno = {
            "video_path": "/videos/video.mp4",
            "query": "the visible event",
            "duration": 100.0,
        }
        baseline = {
            "timestamps": [[0.0, 100.0]],
            "answers": "The event happens in 0.0 - 100.0 seconds.",
        }
        config = SearchConfig(
            coarse_grid_size=9,
            coarse_window_fraction=0.25,
            refine_rounds=2,
            top_k_stage1=2,
            top_k_refine=2,
            fps=2,
            zoom_fps=4,
            dense_zoom_fps=8,
        )
        runner = EventRunner((42.0, 47.0))

        result = run_timelens_search_sample(
            anno=anno,
            baseline_pred=baseline,
            runner=runner,
            config=config,
            unit=1.0,
        )

        self.assertEqual(result.interval, Interval(42.0, 47.0))
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.selected_source.startswith("refine_round"))
        self.assertEqual(result.search_cost["model_calls"], len(runner.calls))
        self.assertIn(2, {fps for _, fps in runner.calls})
        self.assertIn(4, {fps for _, fps in runner.calls})
        self.assertIn(8, {fps for _, fps in runner.calls})

        payload = result.to_prediction_value(duration=100.0, config=config)
        self.assertEqual(payload["timestamps"], [[42.0, 47.0]])
        self.assertEqual(payload["duration"], 100.0)
        self.assertIn("search_trace", payload)
        self.assertIn("search_cost", payload)
        self.assertEqual(payload["selected_source"], result.selected_source)
        self.assertFalse(payload["fallback_used"])

    def test_search_falls_back_to_baseline_when_all_candidates_invalid(self):
        anno = {
            "video_path": "/videos/video.mp4",
            "query": "the visible event",
            "duration": 120.0,
        }
        baseline = {
            "timestamps": [[12.0, 19.0]],
            "answers": "The event happens in 12.0 - 19.0 seconds.",
        }

        result = run_timelens_search_sample(
            anno=anno,
            baseline_pred=baseline,
            runner=EmptyRunner(),
            config=SearchConfig(coarse_grid_size=3, refine_rounds=1),
            unit=1.0,
        )

        self.assertEqual(result.interval, Interval(12.0, 19.0))
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.selected_source, "baseline")
        self.assertEqual(result.fallback_reason, "all_search_candidates_invalid")

    def test_budgeted_graph_counts_baseline_and_accepts_order_swapped_critic(self):
        anno = {
            "video_path": "/videos/video.mp4",
            "query": "the visible event",
            "duration": 100.0,
        }
        baseline = {
            "timestamps": [[30.0, 40.0]],
            "answers": "The event happens in 30.0 - 40.0 seconds.",
        }
        runner = CriticEventRunner((42.0, 47.0))
        config = SearchConfig(
            search_mode="budgeted_graph",
            max_total_calls=10,
            critic_mode="self",
            critic_calls=2,
            graph_seed_count=5,
            graph_beam_width=3,
        )

        result = run_timelens_search_sample(
            anno=anno,
            baseline_pred=baseline,
            runner=runner,
            config=config,
            unit=1.0,
        )

        self.assertEqual(result.interval, Interval(42.0, 47.0))
        self.assertFalse(result.fallback_used)
        self.assertEqual(len(runner.critic_calls), 2)
        self.assertLessEqual(result.search_cost["total_model_calls"], 10)
        self.assertEqual(
            result.search_cost["total_model_calls"],
            1 + len(runner.calls) + len(runner.critic_calls),
        )
        self.assertEqual(
            [item["normalized_choice"] for item in result.verification_trace],
            ["challenger", "challenger"],
        )
        self.assertGreater(result.graph["node_count"], 1)

    def test_budgeted_graph_falls_back_when_critic_order_disagrees(self):
        anno = {
            "video_path": "/videos/video.mp4",
            "query": "the visible event",
            "duration": 100.0,
        }
        baseline = {
            "timestamps": [[30.0, 40.0]],
            "answers": "The event happens in 30.0 - 40.0 seconds.",
        }
        runner = CriticEventRunner((42.0, 47.0), forced_choices=["B", "B"])
        config = SearchConfig(
            search_mode="budgeted_graph",
            max_total_calls=10,
            critic_mode="self",
            critic_calls=2,
        )

        result = run_timelens_search_sample(
            anno=anno,
            baseline_pred=baseline,
            runner=runner,
            config=config,
            unit=1.0,
        )

        self.assertEqual(result.interval, Interval(30.0, 40.0))
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "critic_order_disagreement")

    def test_lookagain_astar_b5_respects_budget_and_falls_back_without_certificate(self):
        anno = {
            "video_path": "/videos/video.mp4",
            "query": "the visible event",
            "duration": 100.0,
        }
        baseline = {
            "timestamps": [[30.0, 40.0]],
            "answers": "The event happens in 30.0 - 40.0 seconds.",
        }
        runner = CriticEventRunner((42.0, 47.0))
        result = run_timelens_search_sample(
            anno=anno,
            baseline_pred=baseline,
            runner=runner,
            config=SearchConfig(
                search_mode="lookagain_astar",
                lookagain_controller="guided",
                max_total_calls=5,
                critic_calls=2,
                graph_seed_count=5,
            ),
            unit=1.0,
        )

        self.assertEqual(result.interval, Interval(30.0, 40.0))
        self.assertTrue(result.fallback_used)
        self.assertEqual(
            result.fallback_reason, "certificate_not_found_within_budget"
        )
        self.assertFalse(result.search_cost["certificate_found"])
        self.assertEqual(result.search_cost["search_call_budget"], 2)
        self.assertLessEqual(result.search_cost["total_model_calls"], 5)
        self.assertEqual(len(runner.critic_calls), 0)

    def test_lookagain_astar_b10_returns_first_anchor_certified_challenger(self):
        anno = {
            "video_path": "/videos/video.mp4",
            "query": "the visible event",
            "duration": 100.0,
        }
        baseline = {
            "timestamps": [[30.0, 40.0]],
            "answers": "The event happens in 30.0 - 40.0 seconds.",
        }
        runner = CriticEventRunner((42.0, 47.0))
        result = run_timelens_search_sample(
            anno=anno,
            baseline_pred=baseline,
            runner=runner,
            config=SearchConfig(
                search_mode="lookagain_astar",
                lookagain_controller="guided",
                lookagain_anchor_period=3,
                max_total_calls=10,
                critic_calls=2,
                graph_seed_count=5,
            ),
            unit=1.0,
        )

        self.assertEqual(result.interval, Interval(42.0, 47.0))
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.search_cost["certificate_found"])
        self.assertEqual(result.search_cost["certificate_state"]["deficit"], 0)
        self.assertEqual(len(runner.critic_calls), 2)
        self.assertEqual(
            [item["normalized_choice"] for item in result.verification_trace],
            ["challenger", "challenger"],
        )
        self.assertEqual(
            result.search_cost["total_model_calls"],
            1 + len(runner.calls) + len(runner.critic_calls),
        )
        self.assertLessEqual(result.search_cost["total_model_calls"], 10)
        certificate_events = [
            item for item in result.controller_trace if item["event"] == "certificate"
        ]
        self.assertEqual(len(certificate_events), 1)
        self.assertEqual(certificate_events[0]["queue"], "anchor")

    def test_lookagain_astar_ucs_b15_is_supported(self):
        anno = {
            "video_path": "/videos/video.mp4",
            "query": "the visible event",
            "duration": 100.0,
        }
        baseline = {
            "timestamps": [[30.0, 40.0]],
            "answers": "The event happens in 30.0 - 40.0 seconds.",
        }
        runner = CriticEventRunner((42.0, 47.0))
        result = run_timelens_search_sample(
            anno=anno,
            baseline_pred=baseline,
            runner=runner,
            config=SearchConfig(
                search_mode="lookagain_astar",
                lookagain_controller="ucs",
                max_total_calls=15,
                critic_calls=2,
                graph_seed_count=5,
            ),
            unit=1.0,
        )

        self.assertEqual(result.interval, Interval(42.0, 47.0))
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.search_cost["controller"], "ucs")
        self.assertEqual(result.search_cost["guided_pops"], 0)
        self.assertLessEqual(result.search_cost["total_model_calls"], 15)


if __name__ == "__main__":
    unittest.main()
