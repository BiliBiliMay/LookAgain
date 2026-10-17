"""Fixed-window consensus control; no graph, certificate, or ground-truth selection."""
from evaluation.timelens_search import (
    Interval, SearchCandidate, baseline_interval_from_prediction,
    _evaluate_proposals, _certificate_fallback_result, _verify_certified_challenger,
    interval_iou,
)


def fixed_windows(duration, min_window=4.0):
    # Twelve fixed views: three half-video, four third-video, five quarter-video.
    # End-aligned evenly spaced starts cover the full video at every scale.
    windows = []
    seen = set()
    for fraction, count in ((0.5, 3), (1 / 3, 4), (0.25, 5)):
        width = min(duration, max(min_window, duration * fraction))
        for index in range(count):
            start = (duration - width) * index / (count - 1)
            window = Interval(start, start + width)
            key = (round(window.start, 6), round(window.end, 6))
            if key not in seen:
                seen.add(key)
                windows.append(window)
    return windows


def consensus_candidate(candidates):
    valid = [c for c in candidates if c.valid and c.prediction is not None]
    if not valid:
        return None
    # Include baseline in consensus; ties preserve proposal order (baseline first).
    for candidate in valid:
        peers = [c for c in valid if c is not candidate]
        candidate.score = sum(interval_iou(candidate.prediction, p.prediction)
                              for p in peers) / max(len(peers), 1)
        candidate.score_components['mean_pairwise_iou'] = candidate.score
    return max(valid, key=lambda c: c.score)


def run_fixed_search(anno, baseline_pred, runner, config, unit=1.0):
    duration = float(anno['duration'])
    baseline = baseline_interval_from_prediction(baseline_pred, duration)
    answer = baseline_pred.get('answers', '')
    cost = dict(model_calls=0, baseline_calls=1, critic_calls=0,
                max_total_calls=config.max_total_calls, reserved_critic_calls=2,
                clip_seconds=0.0, frame_seconds=0.0, calls_by_stage={},
                control='fixed_multiscale_consensus_v1')
    candidates = []
    if baseline is not None:
        candidates.append(SearchCandidate(
            proposal=Interval(0, duration), prediction=baseline, raw_prediction=baseline,
            answer=answer, source='baseline', stage='baseline', valid=True,
            normalized_mode='baseline', fps=config.fps,
        ))
        allowance = max(config.max_total_calls - 1 - 2, 0)
        proposals = fixed_windows(duration, config.min_window_seconds)[:allowance]
        candidates.extend(_evaluate_proposals(
            anno, runner, proposals, duration, unit, 'coarse', 0,
            config.zoom_fps, cost,
        ))
    challenger = consensus_candidate(candidates)
    if challenger is None or challenger.source == 'baseline':
        result = _certificate_fallback_result(
            candidates, baseline, answer, duration, unit, cost,
            'baseline_missing' if baseline is None else 'consensus_prefers_baseline',
        )
        verification = []
    else:
        # This helper only implements the same final two-order gate. The control
        # does not claim its challenger has a LookAgain certificate.
        result, verification = _verify_certified_challenger(
            anno, runner, challenger, candidates, baseline, answer,
            duration, unit, config, cost,
        )
    cost['total_model_calls'] = 1 + cost['model_calls']
    assert cost['total_model_calls'] <= config.max_total_calls
    result.search_cost = cost
    result.verification_trace = verification
    return result
