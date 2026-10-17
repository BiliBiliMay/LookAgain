"""ActiveZoom-VTG interval logic for TimeLens inference.

This module is intentionally model-free. It proposes temporal subclips,
normalizes subclip predictions back to global video time, scores candidate
intervals, and decides whether to keep an active prediction or fall back to
the one-shot TimeLens result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EPS = 1e-6


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.end)

    def is_finite(self) -> bool:
        return math.isfinite(self.start) and math.isfinite(self.end)

    def is_valid(self, duration: Optional[float] = None, min_length: float = EPS) -> bool:
        if not self.is_finite() or self.length <= min_length:
            return False
        if duration is None:
            return True
        return self.end > 0.0 and self.start < duration

    def clamp(self, duration: float) -> "Interval":
        return Interval(
            min(max(self.start, 0.0), duration),
            min(max(self.end, 0.0), duration),
        )

    def shifted(self, offset: float) -> "Interval":
        return Interval(self.start + offset, self.end + offset)

    def rounded(self, unit: float) -> "Interval":
        if unit <= 0:
            return self
        return Interval(round(self.start / unit) * unit, round(self.end / unit) * unit)

    def as_list(self) -> List[float]:
        return [float(self.start), float(self.end)]


@dataclass
class ZoomConfig:
    iterations: int = 2
    top_k: int = 3
    min_window_seconds: float = 8.0
    fallback_grid_size: int = 5
    center_factors: Tuple[float, ...] = (0.75, 1.5, 3.0)
    shifted_factor: float = 1.5
    boundary_factor: float = 1.5
    map_threshold: float = 0.35
    broad_baseline_fraction: float = 0.6
    broad_replace_threshold: float = 0.55
    agreement_weight: float = 1.4
    containment_weight: float = 0.6
    prior_weight: float = 0.7
    edge_penalty_weight: float = 0.45
    posterior_temperature: float = 1.0


@dataclass
class Particle:
    proposal: Interval
    prediction: Optional[Interval]
    raw_prediction: Optional[Interval] = None
    answer: str = ""
    source: str = ""
    iteration: int = 0
    valid: bool = False
    normalized_mode: str = ""
    posterior: float = 0.0
    score: float = float("-inf")
    score_components: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "iteration": self.iteration,
            "proposal": self.proposal.as_list(),
            "raw_prediction": self.raw_prediction.as_list()
            if self.raw_prediction is not None
            else None,
            "prediction": self.prediction.as_list()
            if self.prediction is not None
            else None,
            "valid": self.valid,
            "normalized_mode": self.normalized_mode,
            "posterior": self.posterior,
            "score": self.score,
            "score_components": dict(self.score_components),
            "answer": self.answer,
        }


@dataclass
class CandidateResult:
    interval: Interval
    answer: str
    particles: List[Particle]
    selected_source: str
    fallback_used: bool
    fallback_reason: str = ""
    selected_particle_index: Optional[int] = None
    selected_confidence: float = 0.0

    def to_metadata(self, config: ZoomConfig) -> Dict[str, Any]:
        return {
            "particles": [particle.to_json() for particle in self.particles],
            "scores": {
                "selected_confidence": self.selected_confidence,
                "map_threshold": config.map_threshold,
                "broad_replace_threshold": config.broad_replace_threshold,
            },
            "selected_source": self.selected_source,
            "selected_particle_index": self.selected_particle_index,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


def coerce_interval(value: Any) -> Optional[Interval]:
    """Convert a timestamp pair or nested timestamp list into an Interval."""
    if value is None:
        return None
    if isinstance(value, Interval):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        first = value[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return coerce_interval(first)
        if len(value) >= 2:
            try:
                return Interval(float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                return None
    return None


def baseline_interval_from_prediction(
    baseline_pred: Optional[Dict[str, Any]], duration: float
) -> Optional[Interval]:
    if not isinstance(baseline_pred, dict):
        return None
    interval = coerce_interval(baseline_pred.get("timestamps"))
    if interval is None or not interval.is_valid(duration):
        return None
    clamped = interval.clamp(duration)
    if not clamped.is_valid(duration):
        return None
    return clamped


def round_interval(interval: Interval, duration: float, unit: float) -> Interval:
    rounded = interval.rounded(unit).clamp(duration)
    if rounded.is_valid(duration):
        return rounded
    return interval.clamp(duration)


def proposal_min_window(duration: float, unit: float, config: ZoomConfig) -> float:
    if duration <= 0:
        return 0.0
    unit_floor = 2.0 * unit if unit > 0 else 0.0
    return min(duration, max(config.min_window_seconds, unit_floor))


def make_window(center: float, width: float, duration: float) -> Interval:
    if duration <= 0:
        return Interval(0.0, 0.0)
    width = min(max(width, EPS), duration)
    start = center - 0.5 * width
    end = center + 0.5 * width
    if start < 0.0:
        end -= start
        start = 0.0
    if end > duration:
        start -= end - duration
        end = duration
    return Interval(max(start, 0.0), min(end, duration))


def dedupe_intervals(
    intervals: Iterable[Interval], duration: float, unit: float
) -> List[Interval]:
    seen = set()
    deduped: List[Interval] = []
    for interval in intervals:
        clamped = interval.clamp(duration)
        if not clamped.is_valid(duration):
            continue
        rounded = clamped.rounded(unit)
        key = (round(rounded.start, 6), round(rounded.end, 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clamped)
    return deduped


def fallback_grid_proposals(
    duration: float, unit: float, config: ZoomConfig
) -> List[Interval]:
    min_window = proposal_min_window(duration, unit, config)
    if duration <= 0:
        return []
    if duration <= min_window + EPS or config.fallback_grid_size <= 1:
        return [Interval(0.0, duration)]

    count = max(config.fallback_grid_size, 2)
    stride = (duration - min_window) / float(count - 1)
    intervals = [
        Interval(i * stride, i * stride + min_window)
        for i in range(count)
    ]
    intervals[-1] = Interval(duration - min_window, duration)
    return dedupe_intervals(intervals, duration, unit)


def generate_proposals(
    anchor: Optional[Interval],
    duration: float,
    unit: float,
    config: ZoomConfig,
) -> List[Interval]:
    """Generate ActiveZoom subclip proposals from a baseline or particle interval."""
    if anchor is None or not anchor.is_valid(duration):
        return fallback_grid_proposals(duration, unit, config)

    min_window = proposal_min_window(duration, unit, config)
    anchor = anchor.clamp(duration)
    length = max(anchor.length, unit if unit > 0 else EPS, EPS)

    intervals: List[Interval] = []
    for factor in config.center_factors:
        intervals.append(make_window(anchor.center, max(factor * length, min_window), duration))

    shifted_width = max(config.shifted_factor * length, min_window)
    shift = 0.5 * length
    intervals.append(make_window(anchor.center - shift, shifted_width, duration))
    intervals.append(make_window(anchor.center + shift, shifted_width, duration))

    boundary_width = max(config.boundary_factor * length, min_window)
    intervals.append(make_window(anchor.start, boundary_width, duration))
    intervals.append(make_window(anchor.end, boundary_width, duration))

    return dedupe_intervals(intervals, duration, unit)


def normalize_subclip_prediction(
    raw_prediction: Any,
    proposal: Interval,
    duration: float,
    unit: float = 1.0,
) -> Tuple[Optional[Interval], Optional[Interval], bool, str]:
    """Normalize TimeLens output from a subclip back to global video seconds.

    Absolute outputs that already fall inside the proposal are kept as-is.
    Otherwise, outputs in [0, clip_duration] are treated as relative to the
    subclip and shifted by proposal.start.
    """
    raw = coerce_interval(raw_prediction)
    if raw is None or not raw.is_valid(None):
        return raw, None, False, "invalid"

    tolerance = max(unit * 0.5, 1e-4) if unit > 0 else 1e-4
    if raw.start >= proposal.start - tolerance and raw.end <= proposal.end + tolerance:
        normalized = raw
        mode = "absolute"
    elif raw.start >= -tolerance and raw.end <= proposal.length + tolerance:
        normalized = raw.shifted(proposal.start)
        mode = "relative"
    else:
        normalized = raw
        mode = "absolute_outside_proposal"

    clamped = normalized.clamp(duration)
    valid = clamped.is_valid(duration)
    return raw, clamped if valid else None, valid, mode


def interval_iou(a: Interval, b: Interval) -> float:
    if not a.is_valid(None) or not b.is_valid(None):
        return 0.0
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    if union <= 0:
        return 0.0
    return inter / union


def _median(values: Sequence[float], default: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return default
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def _agreement_score(
    particle: Particle,
    valid_particles: Sequence[Particle],
    duration: float,
    unit: float,
    config: ZoomConfig,
) -> float:
    assert particle.prediction is not None
    lengths = [p.prediction.length for p in valid_particles if p.prediction is not None]
    min_window = proposal_min_window(duration, unit, config)
    median_length = _median(lengths, min_window)
    center_scale = max(0.5 * min_window, 0.5 * median_length, unit, EPS)

    density = 0.0
    for other in valid_particles:
        if other.prediction is None:
            continue
        center_delta = abs(particle.prediction.center - other.prediction.center) / center_scale
        length_ratio = max(particle.prediction.length, EPS) / max(other.prediction.length, EPS)
        log_length_delta = abs(math.log(length_ratio))
        density += math.exp(-0.5 * (center_delta * center_delta + log_length_delta * log_length_delta))
    return density / max(len(valid_particles), 1)


def _containment_score(prediction: Interval, proposal: Interval, unit: float) -> float:
    tolerance = max(unit, EPS)
    if prediction.start >= proposal.start - tolerance and prediction.end <= proposal.end + tolerance:
        return 1.0
    return interval_iou(prediction, proposal)


def _prior_score(
    prediction: Interval,
    baseline: Optional[Interval],
    duration: float,
    unit: float,
    config: ZoomConfig,
) -> float:
    if baseline is None or not baseline.is_valid(duration):
        return 0.5
    min_window = proposal_min_window(duration, unit, config)
    center_scale = max(baseline.length, min_window, unit, EPS)
    center_score = math.exp(-abs(prediction.center - baseline.center) / center_scale)
    length_ratio = max(prediction.length, EPS) / max(baseline.length, EPS)
    length_score = math.exp(-abs(math.log(length_ratio)))
    return 0.5 * interval_iou(prediction, baseline) + 0.25 * center_score + 0.25 * length_score


def _edge_penalty(prediction: Interval, proposal: Interval, unit: float) -> float:
    margin = max(unit, 0.05 * proposal.length, EPS)
    penalty = 0.0
    if abs(prediction.start - proposal.start) <= margin:
        penalty += 0.5
    if abs(prediction.end - proposal.end) <= margin:
        penalty += 0.5
    return min(penalty, 1.0)


def score_particles(
    particles: Sequence[Particle],
    baseline: Optional[Interval],
    duration: float,
    unit: float,
    config: ZoomConfig,
) -> None:
    valid_particles = [
        p for p in particles
        if p.valid and p.prediction is not None and p.source != "baseline"
    ]
    if not valid_particles:
        for particle in particles:
            particle.posterior = 0.0
            particle.score = float("-inf")
            particle.score_components = {"selection_posterior": 0.0}
        return

    for particle in particles:
        if particle not in valid_particles or particle.prediction is None:
            particle.posterior = 0.0
            particle.score = float("-inf")
            particle.score_components = {"selection_posterior": 0.0}
            continue
        agreement = _agreement_score(particle, valid_particles, duration, unit, config)
        containment = _containment_score(particle.prediction, particle.proposal, unit)
        prior = _prior_score(particle.prediction, baseline, duration, unit, config)
        edge = _edge_penalty(particle.prediction, particle.proposal, unit)
        score = (
            config.agreement_weight * agreement
            + config.containment_weight * containment
            + config.prior_weight * prior
            - config.edge_penalty_weight * edge
        )
        particle.score = score
        particle.score_components = {
            "agreement": agreement,
            "containment": containment,
            "prior": prior,
            "edge_penalty": edge,
        }

    max_score = max(p.score for p in valid_particles)
    temperature = max(config.posterior_temperature, EPS)
    exp_scores = [math.exp((p.score - max_score) / temperature) for p in valid_particles]
    denom = sum(exp_scores)
    for particle, exp_score in zip(valid_particles, exp_scores):
        particle.posterior = exp_score / denom if denom > 0 else 0.0

    for particle in valid_particles:
        cluster_mass = 0.0
        for other in valid_particles:
            if other.prediction is None or particle.prediction is None:
                continue
            if _candidate_similarity(particle.prediction, other.prediction, duration, unit, config) >= 0.55:
                cluster_mass += other.posterior
        particle.score_components["selection_posterior"] = min(cluster_mass, 1.0)


def _candidate_similarity(
    a: Interval,
    b: Interval,
    duration: float,
    unit: float,
    config: ZoomConfig,
) -> float:
    min_window = proposal_min_window(duration, unit, config)
    center_scale = max(0.5 * min_window, 0.5 * max(a.length, b.length), unit, EPS)
    center_delta = abs(a.center - b.center) / center_scale
    length_ratio = max(a.length, EPS) / max(b.length, EPS)
    length_delta = abs(math.log(length_ratio))
    return math.exp(-0.5 * (center_delta * center_delta + length_delta * length_delta))


def _selection_confidence(particle: Particle) -> float:
    return particle.score_components.get("selection_posterior", particle.posterior)


def select_final_candidate(
    particles: Sequence[Particle],
    baseline: Optional[Interval],
    baseline_answer: str,
    duration: float,
    unit: float,
    config: ZoomConfig,
) -> CandidateResult:
    active_valid = [
        (idx, p)
        for idx, p in enumerate(particles)
        if p.source != "baseline" and p.valid and p.prediction is not None
    ]
    if not active_valid:
        if baseline is not None and baseline.is_valid(duration):
            return CandidateResult(
                interval=round_interval(baseline, duration, unit),
                answer=baseline_answer,
                particles=list(particles),
                selected_source="baseline",
                fallback_used=True,
                fallback_reason="all_active_predictions_invalid",
            )
        return CandidateResult(
            interval=Interval(duration + 10.0, duration + 20.0),
            answer="ACTIVEZOOM_NO_VALID_PREDICTION",
            particles=list(particles),
            selected_source="invalid_placeholder",
            fallback_used=True,
            fallback_reason="all_predictions_invalid",
        )

    best_idx, best = max(
        active_valid,
        key=lambda item: (_selection_confidence(item[1]), item[1].score, item[1].posterior),
    )
    confidence = _selection_confidence(best)

    if baseline is not None and baseline.is_valid(duration):
        if confidence < config.map_threshold:
            return CandidateResult(
                interval=round_interval(baseline, duration, unit),
                answer=baseline_answer,
                particles=list(particles),
                selected_source="baseline",
                fallback_used=True,
                fallback_reason="posterior_diffuse",
                selected_particle_index=best_idx,
                selected_confidence=confidence,
            )

        if (
            baseline.length >= config.broad_baseline_fraction * duration
            and confidence < config.broad_replace_threshold
        ):
            return CandidateResult(
                interval=round_interval(baseline, duration, unit),
                answer=baseline_answer,
                particles=list(particles),
                selected_source="baseline",
                fallback_used=True,
                fallback_reason="broad_baseline_low_active_confidence",
                selected_particle_index=best_idx,
                selected_confidence=confidence,
            )

    assert best.prediction is not None
    return CandidateResult(
        interval=round_interval(best.prediction, duration, unit),
        answer=best.answer,
        particles=list(particles),
        selected_source=best.source,
        fallback_used=False,
        selected_particle_index=best_idx,
        selected_confidence=confidence,
    )


def _prediction_payload(result: Any) -> Tuple[str, Any]:
    if isinstance(result, dict):
        return str(result.get("answer", result.get("answers", ""))), result.get("timestamps")
    if isinstance(result, tuple) and len(result) >= 2:
        return str(result[0]), result[1]
    answer = getattr(result, "answer", getattr(result, "answers", ""))
    timestamps = getattr(result, "timestamps", None)
    return str(answer), timestamps


def _runner_predict(runner: Any, anno: Dict[str, Any], proposal: Interval, config: ZoomConfig) -> Tuple[str, Any]:
    if hasattr(runner, "predict_interval"):
        return _prediction_payload(runner.predict_interval(anno, proposal))
    if hasattr(runner, "predict"):
        return _prediction_payload(runner.predict(anno, proposal))
    return _prediction_payload(runner(anno, proposal))


def _proposal_key(interval: Interval, unit: float) -> Tuple[float, float]:
    rounded = interval.rounded(unit)
    return (round(rounded.start, 6), round(rounded.end, 6))


def run_activezoom_sample(
    anno: Dict[str, Any],
    baseline_pred: Optional[Dict[str, Any]],
    runner: Any,
    config: Optional[ZoomConfig] = None,
    unit: float = 1.0,
) -> CandidateResult:
    config = config or ZoomConfig()
    duration = float(anno["duration"])
    baseline = baseline_interval_from_prediction(baseline_pred, duration)
    baseline_answer = ""
    if isinstance(baseline_pred, dict):
        baseline_answer = str(baseline_pred.get("answers", baseline_pred.get("answer", "")))

    particles: List[Particle] = []
    if baseline is not None:
        particles.append(
            Particle(
                proposal=Interval(0.0, duration),
                prediction=baseline,
                raw_prediction=baseline,
                answer=baseline_answer,
                source="baseline",
                iteration=-1,
                valid=True,
                normalized_mode="baseline",
            )
        )

    proposals = generate_proposals(baseline, duration, unit, config)
    seen_proposals = {_proposal_key(proposal, unit) for proposal in proposals}

    for iteration in range(max(config.iterations, 0)):
        if not proposals:
            break

        wave_particles: List[Particle] = []
        for proposal_index, proposal in enumerate(proposals):
            source = f"iter{iteration + 1}_proposal{proposal_index}"
            answer, timestamps = _runner_predict(runner, anno, proposal, config)
            raw, normalized, valid, mode = normalize_subclip_prediction(
                timestamps, proposal, duration, unit
            )
            wave_particles.append(
                Particle(
                    proposal=proposal,
                    prediction=normalized,
                    raw_prediction=raw,
                    answer=answer,
                    source=source,
                    iteration=iteration + 1,
                    valid=valid,
                    normalized_mode=mode,
                )
            )

        particles.extend(wave_particles)
        score_particles(particles, baseline, duration, unit, config)

        if iteration >= config.iterations - 1:
            break

        ranked = sorted(
            [p for p in particles if p.source != "baseline" and p.valid and p.prediction is not None],
            key=lambda p: (_selection_confidence(p), p.score, p.posterior),
            reverse=True,
        )[: max(config.top_k, 0)]

        next_proposals: List[Interval] = []
        for particle in ranked:
            assert particle.prediction is not None
            for proposal in generate_proposals(particle.prediction, duration, unit, config):
                key = _proposal_key(proposal, unit)
                if key in seen_proposals:
                    continue
                seen_proposals.add(key)
                next_proposals.append(proposal)
        proposals = next_proposals

    score_particles(particles, baseline, duration, unit, config)
    return select_final_candidate(particles, baseline, baseline_answer, duration, unit, config)
