"""Performance-first temporal search wrapper for TimeLens inference.

This module is intentionally model-free. It proposes coarse temporal windows,
normalizes TimeLens subclip predictions back into global video seconds, scores
candidate intervals, and decides whether to use the searched interval or fall
back to the one-shot baseline prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import inspect
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
class SearchConfig:
    search_mode: str = "exhaustive_refine"
    coarse_grid_size: int = 9
    coarse_window_fraction: float = 0.25
    refine_rounds: int = 2
    top_k_stage1: int = 5
    top_k_refine: int = 5
    min_window_seconds: float = 4.0
    tiny_event_window_seconds: float = 8.0
    fps: int = 2
    zoom_fps: int = 4
    dense_zoom_fps: int = 8
    refine_window_factors: Tuple[float, ...] = (4.0, 2.0, 1.0)
    diffuse_confidence_threshold: float = 0.30
    broad_baseline_fraction: float = 0.60
    broad_replace_threshold: float = 0.55
    agreement_weight: float = 1.4
    containment_weight: float = 0.8
    prior_weight: float = 0.6
    focus_weight: float = 0.25
    edge_penalty_weight: float = 0.35
    posterior_temperature: float = 1.0
    max_trace_answer_chars: int = 256
    max_total_calls: int = 15
    graph_seed_count: int = 5
    graph_beam_width: int = 3
    graph_merge_iou: float = 0.92
    graph_support_iou: float = 0.55
    graph_contradiction_proposal_iou: float = 0.50
    graph_contradiction_prediction_iou: float = 0.10
    graph_support_weight: float = 0.65
    graph_contradiction_weight: float = 0.55
    lookagain_controller: str = "guided"
    lookagain_anchor_period: int = 3
    certificate_boundary_iou: float = 0.55
    certificate_support_iou: float = 0.55
    certificate_contradiction_weight: float = 0.45
    max_controller_trace_steps: int = 256
    critic_mode: str = "self"
    critic_calls: int = 2
    critic_fps: int = 4
    critic_min_confidence: float = 0.55


@dataclass
class SearchCandidate:
    proposal: Interval
    prediction: Optional[Interval]
    raw_prediction: Optional[Interval] = None
    answer: str = ""
    source: str = ""
    stage: str = ""
    round_index: int = 0
    candidate_index: int = 0
    fps: Optional[int] = None
    valid: bool = False
    normalized_mode: str = ""
    posterior: float = 0.0
    confidence: float = 0.0
    score: float = float("-inf")
    score_components: Dict[str, float] = field(default_factory=dict)
    parent_sources: List[str] = field(default_factory=list)
    action: str = ""

    def to_json(self, max_answer_chars: int = 256) -> Dict[str, Any]:
        answer = self.answer
        if max_answer_chars > 0 and len(answer) > max_answer_chars:
            answer = answer[:max_answer_chars] + "...<truncated>"
        return {
            "source": self.source,
            "stage": self.stage,
            "round_index": self.round_index,
            "candidate_index": self.candidate_index,
            "fps": self.fps,
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
            "confidence": self.confidence,
            "score": self.score,
            "score_components": dict(self.score_components),
            "parent_sources": list(self.parent_sources),
            "action": self.action,
            "answer": answer,
        }


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    action: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": float(self.weight),
            "action": self.action,
        }


@dataclass(frozen=True)
class CertificateState:
    """Four-bit evidence certificate attached to one canonical focal node."""

    focal_source: str
    left_boundary: bool = False
    right_boundary: bool = False
    support: bool = False
    contradiction: bool = False

    @property
    def deficit(self) -> int:
        return int(not self.left_boundary) + int(not self.right_boundary) + int(not self.support)

    def is_certified(self, focal: Optional[SearchCandidate]) -> bool:
        return bool(
            focal is not None
            and focal.stage != "baseline"
            and focal.valid
            and focal.prediction is not None
            and self.left_boundary
            and self.right_boundary
            and self.support
            and not self.contradiction
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "focal_source": self.focal_source,
            "left_boundary": self.left_boundary,
            "right_boundary": self.right_boundary,
            "support": self.support,
            "contradiction": self.contradiction,
            "deficit": self.deficit,
        }


class TemporalHypothesisGraph:
    """Evaluated interval states plus typed cross-hypothesis relations."""

    def __init__(self, config: SearchConfig, duration: float, unit: float):
        self.config = config
        self.duration = duration
        self.unit = unit
        self.nodes: Dict[str, SearchCandidate] = {}
        self.edges: List[GraphEdge] = []
        self._edge_keys = set()
        self.transposition_hits = 0

    def _add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        action: str = "",
        directed: bool = False,
    ) -> None:
        if source == target:
            return
        endpoints = (source, target) if directed else tuple(sorted((source, target)))
        key = (endpoints[0], endpoints[1], relation, action)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(GraphEdge(source, target, relation, weight, action))

    def find_equivalent_proposal(self, proposal: Interval) -> Optional[SearchCandidate]:
        for candidate in self.nodes.values():
            if interval_iou(candidate.proposal, proposal) >= self.config.graph_merge_iou:
                return candidate
        return None

    def add_candidate(
        self,
        candidate: SearchCandidate,
        parent_source: str = "",
        action: str = "",
    ) -> SearchCandidate:
        equivalent = self.find_equivalent_proposal(candidate.proposal)
        if equivalent is not None:
            self.transposition_hits += 1
            if parent_source and parent_source not in equivalent.parent_sources:
                equivalent.parent_sources.append(parent_source)
                self._add_edge(
                    parent_source,
                    equivalent.source,
                    "edit",
                    action=action,
                    directed=True,
                )
            return equivalent

        candidate.action = action
        if parent_source:
            candidate.parent_sources.append(parent_source)
        self.nodes[candidate.source] = candidate
        if parent_source:
            self._add_edge(
                parent_source,
                candidate.source,
                "edit",
                action=action,
                directed=True,
            )

        if not candidate.valid or candidate.prediction is None:
            return candidate
        for other in list(self.nodes.values()):
            if other.source == candidate.source or not other.valid or other.prediction is None:
                continue
            prediction_iou = interval_iou(candidate.prediction, other.prediction)
            proposal_iou = interval_iou(candidate.proposal, other.proposal)
            if prediction_iou >= self.config.graph_merge_iou:
                self._add_edge(candidate.source, other.source, "equivalent", prediction_iou)
            elif prediction_iou >= self.config.graph_support_iou:
                self._add_edge(candidate.source, other.source, "support", prediction_iou)
            elif prediction_iou > 0.0:
                self._add_edge(candidate.source, other.source, "overlap", prediction_iou)
            if (
                proposal_iou >= self.config.graph_contradiction_proposal_iou
                and prediction_iou <= self.config.graph_contradiction_prediction_iou
            ):
                self._add_edge(
                    candidate.source,
                    other.source,
                    "contradiction",
                    proposal_iou * (1.0 - prediction_iou),
                )
        return candidate

    def add_transition(self, parent_source: str, target_source: str, action: str) -> None:
        self.transposition_hits += 1
        target = self.nodes.get(target_source)
        if target is not None and parent_source not in target.parent_sources:
            target.parent_sources.append(parent_source)
        self._add_edge(
            parent_source,
            target_source,
            "edit",
            action=action,
            directed=True,
        )

    def evidence(self, source: str) -> Tuple[float, float]:
        support = 0.0
        contradiction = 0.0
        support_count = 0
        contradiction_count = 0
        for edge in self.edges:
            if source not in (edge.source, edge.target):
                continue
            if edge.relation in ("equivalent", "support"):
                support += edge.weight
                support_count += 1
            elif edge.relation == "contradiction":
                contradiction += edge.weight
                contradiction_count += 1
        if support_count:
            support /= support_count
        if contradiction_count:
            contradiction /= contradiction_count
        return min(support, 1.0), min(contradiction, 1.0)

    def to_json(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for edge in self.edges:
            counts[edge.relation] = counts.get(edge.relation, 0) + 1
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "edge_counts": counts,
            "transposition_hits": self.transposition_hits,
            "edges": [edge.to_json() for edge in self.edges],
        }


@dataclass
class SearchResult:
    interval: Interval
    answer: str
    candidates: List[SearchCandidate]
    selected_source: str
    fallback_used: bool
    fallback_reason: str = ""
    selected_candidate_index: Optional[int] = None
    selected_confidence: float = 0.0
    search_cost: Dict[str, Any] = field(default_factory=dict)
    graph: Dict[str, Any] = field(default_factory=dict)
    verification_trace: List[Dict[str, Any]] = field(default_factory=list)
    controller_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_prediction_value(
        self,
        duration: float,
        config: Optional[SearchConfig] = None,
    ) -> Dict[str, Any]:
        max_answer_chars = (
            config.max_trace_answer_chars if config is not None else SearchConfig().max_trace_answer_chars
        )
        return {
            "timestamps": [self.interval.as_list()],
            "answers": self.answer or format_answer(self.interval),
            "duration": float(duration),
            "search_trace": [
                candidate.to_json(max_answer_chars=max_answer_chars)
                for candidate in self.candidates
            ],
            "search_cost": dict(self.search_cost),
            "selected_source": self.selected_source,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "selected_candidate_index": self.selected_candidate_index,
            "selected_confidence": self.selected_confidence,
            "hypothesis_graph": dict(self.graph),
            "verification_trace": list(self.verification_trace),
            "controller_trace": list(self.controller_trace),
        }


def format_answer(interval: Interval) -> str:
    return f"The event happens in {interval.start:.3f} - {interval.end:.3f} seconds."


def coerce_interval(value: Any) -> Optional[Interval]:
    if value is None:
        return None
    if isinstance(value, Interval):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
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
    interval = interval.clamp(duration)
    return interval if interval.is_valid(duration) else None


def round_interval(interval: Interval, duration: float, unit: float) -> Interval:
    clamped = interval.clamp(duration)
    rounded = clamped.rounded(unit).clamp(duration)
    if rounded.is_valid(duration):
        return rounded
    return clamped


def _min_window(duration: float, config: SearchConfig) -> float:
    if duration <= 0:
        return 0.0
    return min(duration, max(config.min_window_seconds, EPS))


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


def _proposal_key(interval: Interval, unit: float = 1.0) -> Tuple[float, float]:
    rounded = interval.rounded(unit)
    return (round(rounded.start, 6), round(rounded.end, 6))


def dedupe_intervals(
    intervals: Iterable[Interval],
    duration: float,
    unit: float = 1.0,
) -> List[Interval]:
    seen = set()
    deduped: List[Interval] = []
    for interval in intervals:
        clamped = interval.clamp(duration)
        if not clamped.is_valid(duration):
            continue
        key = _proposal_key(clamped, unit)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clamped)
    return deduped


def generate_coarse_windows(
    duration: float,
    config: Optional[SearchConfig] = None,
    unit: float = 1.0,
) -> List[Interval]:
    config = config or SearchConfig()
    if duration <= 0:
        return []

    window_width = max(_min_window(duration, config), duration * config.coarse_window_fraction)
    window_width = min(window_width, duration)
    if duration <= window_width + EPS or config.coarse_grid_size <= 1:
        return [Interval(0.0, duration)]

    count = max(config.coarse_grid_size, 2)
    stride = (duration - window_width) / float(count - 1)
    windows = [
        Interval(i * stride, i * stride + window_width)
        for i in range(count)
    ]
    windows[-1] = Interval(duration - window_width, duration)
    return dedupe_intervals(windows, duration, unit)


def generate_refinement_windows(
    anchor: Interval,
    duration: float,
    config: Optional[SearchConfig] = None,
    unit: float = 1.0,
    round_index: int = 1,
) -> List[Interval]:
    config = config or SearchConfig()
    anchor = anchor.clamp(duration)
    if not anchor.is_valid(duration):
        return []

    round_scale = 0.5 ** max(round_index - 1, 0)
    event_floor = config.tiny_event_window_seconds if anchor.length <= config.tiny_event_window_seconds else anchor.length
    event_floor *= round_scale
    min_width = min(duration, max(config.min_window_seconds, event_floor))
    intervals: List[Interval] = []

    for factor in config.refine_window_factors:
        width = max(min_width, anchor.length * factor * round_scale)
        intervals.append(make_window(anchor.center, width, duration))

    narrow_width = max(config.min_window_seconds, min_width)
    shift = max(0.5 * anchor.length, 0.25 * narrow_width)
    intervals.append(make_window(anchor.center - shift, narrow_width, duration))
    intervals.append(make_window(anchor.center + shift, narrow_width, duration))
    intervals.append(make_window(anchor.start, narrow_width, duration))
    intervals.append(make_window(anchor.end, narrow_width, duration))

    return dedupe_intervals(intervals, duration, unit)


def normalize_timestamp_prediction(
    raw_prediction: Any,
    proposal: Interval,
    duration: float,
    unit: float = 1.0,
) -> Tuple[Optional[Interval], Optional[Interval], bool, str]:
    raw = coerce_interval(raw_prediction)
    if raw is None or not raw.is_valid(None):
        return raw, None, False, "invalid"

    tolerance = max(0.5 * unit, 1e-4) if unit > 0 else 1e-4
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
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return default
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def _candidate_similarity(
    a: Interval,
    b: Interval,
    duration: float,
    unit: float,
    config: SearchConfig,
) -> float:
    scale_floor = max(_min_window(duration, config), unit, EPS)
    center_scale = max(0.5 * scale_floor, 0.5 * max(a.length, b.length), EPS)
    center_delta = abs(a.center - b.center) / center_scale
    length_ratio = max(a.length, EPS) / max(b.length, EPS)
    length_delta = abs(math.log(length_ratio))
    return math.exp(-0.5 * (center_delta * center_delta + length_delta * length_delta))


def _agreement_score(
    candidate: SearchCandidate,
    valid_candidates: Sequence[SearchCandidate],
    duration: float,
    unit: float,
    config: SearchConfig,
) -> float:
    assert candidate.prediction is not None
    lengths = [
        item.prediction.length
        for item in valid_candidates
        if item.prediction is not None
    ]
    median_length = _median(lengths, _min_window(duration, config))
    center_scale = max(0.5 * median_length, 0.5 * _min_window(duration, config), unit, EPS)

    density = 0.0
    for other in valid_candidates:
        if other.prediction is None:
            continue
        center_delta = abs(candidate.prediction.center - other.prediction.center) / center_scale
        length_ratio = max(candidate.prediction.length, EPS) / max(other.prediction.length, EPS)
        length_delta = abs(math.log(length_ratio))
        density += math.exp(-0.5 * (center_delta * center_delta + length_delta * length_delta))
    return density / max(len(valid_candidates), 1)


def _containment_score(prediction: Interval, proposal: Interval, unit: float) -> float:
    tolerance = max(unit, EPS)
    if prediction.start >= proposal.start - tolerance and prediction.end <= proposal.end + tolerance:
        return 1.0
    return interval_iou(prediction, proposal)


def _prior_score(
    prediction: Interval,
    baseline: Optional[Interval],
    duration: float,
    config: SearchConfig,
) -> float:
    if baseline is None or not baseline.is_valid(duration):
        return 0.5
    center_scale = max(baseline.length, _min_window(duration, config), EPS)
    center_score = math.exp(-abs(prediction.center - baseline.center) / center_scale)
    length_ratio = max(prediction.length, EPS) / max(baseline.length, EPS)
    length_score = math.exp(-abs(math.log(length_ratio)))
    return 0.5 * interval_iou(prediction, baseline) + 0.25 * center_score + 0.25 * length_score


def _focus_score(proposal: Interval, duration: float) -> float:
    if duration <= 0:
        return 0.0
    return max(0.0, 1.0 - (proposal.length / duration))


def _edge_penalty(prediction: Interval, proposal: Interval, unit: float) -> float:
    margin = max(unit, 0.05 * proposal.length, EPS)
    penalty = 0.0
    if abs(prediction.start - proposal.start) <= margin:
        penalty += 0.5
    if abs(prediction.end - proposal.end) <= margin:
        penalty += 0.5
    return min(penalty, 1.0)


def score_candidates(
    candidates: Sequence[SearchCandidate],
    baseline: Optional[Interval],
    duration: float,
    unit: float,
    config: SearchConfig,
    graph: Optional[TemporalHypothesisGraph] = None,
) -> None:
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate.stage != "baseline" and candidate.valid and candidate.prediction is not None
    ]
    if not valid_candidates:
        for candidate in candidates:
            candidate.posterior = 0.0
            candidate.confidence = 0.0
            candidate.score = float("-inf")
            candidate.score_components = {"selection_posterior": 0.0}
        return

    for candidate in candidates:
        if candidate not in valid_candidates or candidate.prediction is None:
            candidate.posterior = 0.0
            candidate.confidence = 0.0
            candidate.score = float("-inf")
            candidate.score_components = {"selection_posterior": 0.0}
            continue
        agreement = _agreement_score(candidate, valid_candidates, duration, unit, config)
        containment = _containment_score(candidate.prediction, candidate.proposal, unit)
        prior = _prior_score(candidate.prediction, baseline, duration, config)
        focus = _focus_score(candidate.proposal, duration)
        edge = _edge_penalty(candidate.prediction, candidate.proposal, unit)
        graph_support = 0.0
        graph_contradiction = 0.0
        if graph is not None:
            graph_support, graph_contradiction = graph.evidence(candidate.source)
        score = (
            config.agreement_weight * agreement
            + config.containment_weight * containment
            + config.prior_weight * prior
            + config.focus_weight * focus
            - config.edge_penalty_weight * edge
            + config.graph_support_weight * graph_support
            - config.graph_contradiction_weight * graph_contradiction
        )
        candidate.score = score
        candidate.score_components = {
            "agreement": agreement,
            "containment": containment,
            "prior": prior,
            "focus": focus,
            "edge_penalty": edge,
            "graph_support": graph_support,
            "graph_contradiction": graph_contradiction,
        }

    max_score = max(candidate.score for candidate in valid_candidates)
    temperature = max(config.posterior_temperature, EPS)
    exp_scores = [
        math.exp((candidate.score - max_score) / temperature)
        for candidate in valid_candidates
    ]
    denom = sum(exp_scores)
    for candidate, exp_score in zip(valid_candidates, exp_scores):
        candidate.posterior = exp_score / denom if denom > 0 else 0.0

    for candidate in valid_candidates:
        assert candidate.prediction is not None
        cluster_mass = 0.0
        for other in valid_candidates:
            if other.prediction is None:
                continue
            if _candidate_similarity(candidate.prediction, other.prediction, duration, unit, config) >= 0.55:
                cluster_mass += other.posterior
        candidate.confidence = min(cluster_mass, 1.0)
        candidate.score_components["selection_posterior"] = candidate.confidence


def _prediction_payload(result: Any) -> Tuple[str, Any]:
    if isinstance(result, dict):
        return str(result.get("answer", result.get("answers", ""))), result.get("timestamps")
    if isinstance(result, tuple) and len(result) >= 2:
        return str(result[0]), result[1]
    answer = getattr(result, "answer", getattr(result, "answers", ""))
    timestamps = getattr(result, "timestamps", None)
    return str(answer), timestamps


def _runner_accepts_fps_override(method: Any) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "fps_override" in signature.parameters


def _runner_predict(
    runner: Any,
    anno: Dict[str, Any],
    proposal: Interval,
    fps_override: Optional[int],
) -> Tuple[str, Any]:
    if hasattr(runner, "predict_interval"):
        method = runner.predict_interval
        if _runner_accepts_fps_override(method):
            return _prediction_payload(method(anno, proposal, fps_override=fps_override))
        return _prediction_payload(method(anno, proposal))
    if hasattr(runner, "predict"):
        method = runner.predict
        if _runner_accepts_fps_override(method):
            return _prediction_payload(method(anno, proposal, fps_override=fps_override))
        return _prediction_payload(method(anno, proposal))
    return _prediction_payload(runner(anno, proposal))


def _evaluate_proposals(
    anno: Dict[str, Any],
    runner: Any,
    proposals: Sequence[Interval],
    duration: float,
    unit: float,
    stage: str,
    round_index: int,
    fps_override: int,
    search_cost: Dict[str, Any],
) -> List[SearchCandidate]:
    candidates: List[SearchCandidate] = []
    for candidate_index, proposal in enumerate(proposals):
        if stage == "coarse":
            source = f"coarse_candidate{candidate_index}"
        elif stage in ("graph", "lookagain"):
            sequence = search_cost.get("graph_node_sequence", 0)
            source = f"graph_node{sequence}"
            search_cost["graph_node_sequence"] = sequence + 1
        else:
            source = f"refine_round{round_index}_candidate{candidate_index}"
        answer, timestamps = _runner_predict(runner, anno, proposal, fps_override=fps_override)
        raw, normalized, valid, mode = normalize_timestamp_prediction(
            timestamps, proposal, duration, unit
        )
        candidates.append(
            SearchCandidate(
                proposal=proposal,
                prediction=normalized,
                raw_prediction=raw,
                answer=answer,
                source=source,
                stage=stage,
                round_index=round_index,
                candidate_index=candidate_index,
                fps=fps_override,
                valid=valid,
                normalized_mode=mode,
            )
        )
        search_cost["model_calls"] = search_cost.get("model_calls", 0) + 1
        search_cost["clip_seconds"] = search_cost.get("clip_seconds", 0.0) + proposal.length
        search_cost["frame_seconds"] = search_cost.get("frame_seconds", 0.0) + proposal.length * fps_override
        by_stage = search_cost.setdefault("calls_by_stage", {})
        by_stage[stage] = by_stage.get(stage, 0) + 1
    return candidates


def _ranked_candidates(
    candidates: Sequence[SearchCandidate],
    top_k: int,
) -> List[SearchCandidate]:
    ranked = [
        candidate
        for candidate in candidates
        if candidate.stage != "baseline" and candidate.valid and candidate.prediction is not None
    ]
    ranked.sort(
        key=lambda candidate: (
            candidate.confidence,
            candidate.score,
            candidate.posterior,
            -candidate.proposal.length,
        ),
        reverse=True,
    )
    return ranked[: max(top_k, 0)]


def select_final_candidate(
    candidates: Sequence[SearchCandidate],
    baseline: Optional[Interval],
    baseline_answer: str,
    duration: float,
    unit: float,
    config: SearchConfig,
    search_cost: Dict[str, Any],
) -> SearchResult:
    active_valid = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate.stage != "baseline" and candidate.valid and candidate.prediction is not None
    ]

    if not active_valid:
        if baseline is not None and baseline.is_valid(duration):
            rounded = round_interval(baseline, duration, unit)
            return SearchResult(
                interval=rounded,
                answer=baseline_answer or format_answer(rounded),
                candidates=list(candidates),
                selected_source="baseline",
                fallback_used=True,
                fallback_reason="all_search_candidates_invalid",
                search_cost=search_cost,
            )
        placeholder = Interval(duration + 10.0, duration + 20.0)
        return SearchResult(
            interval=placeholder,
            answer="TIMELENS_SEARCH_NO_VALID_PREDICTION",
            candidates=list(candidates),
            selected_source="invalid_placeholder",
            fallback_used=True,
            fallback_reason="all_predictions_invalid",
            search_cost=search_cost,
        )

    best_index, best = max(
        active_valid,
        key=lambda item: (
            item[1].confidence,
            item[1].score,
            item[1].posterior,
            item[1].round_index,
        ),
    )
    confidence = best.confidence

    if baseline is not None and baseline.is_valid(duration):
        if confidence < config.diffuse_confidence_threshold:
            rounded = round_interval(baseline, duration, unit)
            return SearchResult(
                interval=rounded,
                answer=baseline_answer or format_answer(rounded),
                candidates=list(candidates),
                selected_source="baseline",
                fallback_used=True,
                fallback_reason="posterior_diffuse",
                selected_candidate_index=best_index,
                selected_confidence=confidence,
                search_cost=search_cost,
            )

        if (
            baseline.length >= config.broad_baseline_fraction * duration
            and confidence < config.broad_replace_threshold
        ):
            rounded = round_interval(baseline, duration, unit)
            return SearchResult(
                interval=rounded,
                answer=baseline_answer or format_answer(rounded),
                candidates=list(candidates),
                selected_source="baseline",
                fallback_used=True,
                fallback_reason="broad_baseline_low_search_confidence",
                selected_candidate_index=best_index,
                selected_confidence=confidence,
                search_cost=search_cost,
            )

    assert best.prediction is not None
    rounded = round_interval(best.prediction, duration, unit)
    return SearchResult(
        interval=rounded,
        answer=best.answer or format_answer(rounded),
        candidates=list(candidates),
        selected_source=best.source,
        fallback_used=False,
        selected_candidate_index=best_index,
        selected_confidence=confidence,
        search_cost=search_cost,
    )


def _dedupe_labeled_intervals(
    proposals: Sequence[Tuple[Interval, str]],
    duration: float,
    unit: float,
) -> List[Tuple[Interval, str]]:
    seen = set()
    result: List[Tuple[Interval, str]] = []
    for proposal, action in proposals:
        proposal = proposal.clamp(duration)
        if not proposal.is_valid(duration):
            continue
        key = _proposal_key(proposal, unit)
        if key in seen:
            continue
        seen.add(key)
        result.append((proposal, action))
    return result


def generate_graph_seed_windows(
    baseline: Optional[Interval],
    duration: float,
    config: SearchConfig,
    unit: float = 1.0,
) -> List[Tuple[Interval, str]]:
    """Mix global escape hypotheses with baseline-centred repair hypotheses."""
    if duration <= 0:
        return []
    proposals: List[Tuple[Interval, str]] = []
    coarse = generate_coarse_windows(duration, config, unit)
    if coarse:
        indices = sorted(set((0, len(coarse) // 2, len(coarse) - 1)))
        proposals.extend((coarse[index], "global_retrieve") for index in indices)

    if baseline is not None and baseline.is_valid(duration):
        width = min(
            duration,
            max(2.0 * baseline.length, config.min_window_seconds),
        )
        proposals.append((make_window(baseline.center, width, duration), "zoom_baseline"))
        proposals.append(
            (make_window(baseline.center - 0.75 * baseline.length, width, duration), "move_left")
        )
        proposals.append(
            (make_window(baseline.center + 0.75 * baseline.length, width, duration), "move_right")
        )
    proposals = _dedupe_labeled_intervals(proposals, duration, unit)
    return proposals[: max(config.graph_seed_count, 1)]


def generate_graph_neighbors(
    candidate: SearchCandidate,
    baseline: Optional[Interval],
    duration: float,
    config: SearchConfig,
    unit: float = 1.0,
) -> List[Tuple[Interval, str]]:
    """Create reversible, failure-conditioned neighbours of one hypothesis."""
    prediction = candidate.prediction
    if prediction is None or not prediction.is_valid(duration):
        return []
    scale = max(prediction.length, config.min_window_seconds)
    context_width = min(duration, max(2.0 * scale, config.min_window_seconds))
    margin = max(unit, 0.08 * candidate.proposal.length)
    touches_left = abs(prediction.start - candidate.proposal.start) <= margin
    touches_right = abs(prediction.end - candidate.proposal.end) <= margin

    proposals: List[Tuple[Interval, str]] = []
    if touches_left:
        proposals.append(
            (make_window(prediction.center - 0.5 * scale, 2.5 * scale, duration), "expand_left")
        )
    if touches_right:
        proposals.append(
            (make_window(prediction.center + 0.5 * scale, 2.5 * scale, duration), "expand_right")
        )

    if baseline is not None and baseline.is_valid(duration):
        center_delta = prediction.center - baseline.center
        if center_delta < -0.25 * max(baseline.length, unit):
            proposals.append(
                (make_window(prediction.center - 0.5 * scale, context_width, duration), "move_left")
            )
        elif center_delta > 0.25 * max(baseline.length, unit):
            proposals.append(
                (make_window(prediction.center + 0.5 * scale, context_width, duration), "move_right")
            )
        if prediction.length < 0.60 * baseline.length:
            proposals.append((make_window(prediction.center, 3.0 * scale, duration), "expand_both"))
        elif prediction.length > 1.50 * baseline.length:
            proposals.append((make_window(prediction.center, 1.5 * scale, duration), "shrink_both"))

    proposals.extend(
        [
            (make_window(prediction.center, context_width, duration), "zoom_in"),
            (make_window(prediction.center - 0.5 * scale, context_width, duration), "move_left"),
            (make_window(prediction.center + 0.5 * scale, context_width, duration), "move_right"),
            (make_window(prediction.start, context_width, duration), "check_start"),
            (make_window(prediction.end, context_width, duration), "check_end"),
        ]
    )
    return _dedupe_labeled_intervals(proposals, duration, unit)


def _frontier_priority(
    parent: Optional[SearchCandidate],
    proposal: Interval,
    action: str,
    graph: TemporalHypothesisGraph,
) -> float:
    parent_score = 0.0
    if parent is not None and math.isfinite(parent.score):
        parent_score = parent.score
    action_bonus = {
        "zoom_baseline": 0.45,
        "zoom_in": 0.40,
        "check_start": 0.34,
        "check_end": 0.34,
        "expand_left": 0.38,
        "expand_right": 0.38,
        "expand_both": 0.32,
        "shrink_both": 0.30,
        "move_left": 0.28,
        "move_right": 0.28,
        "global_retrieve": 0.18,
    }.get(action, 0.0)
    maximum_overlap = 0.0
    for node in graph.nodes.values():
        maximum_overlap = max(maximum_overlap, interval_iou(node.proposal, proposal))
    novelty = 1.0 - maximum_overlap
    return parent_score + action_bonus + 0.25 * novelty


def generate_lookagain_probe(
    state: CertificateState,
    focal: SearchCandidate,
    duration: float,
    config: SearchConfig,
    unit: float = 1.0,
) -> Optional[Tuple[Interval, str]]:
    """Return the next deterministic observation required by the certificate."""
    prediction = focal.prediction
    if prediction is None or not focal.valid or not prediction.is_valid(duration):
        return None

    scale = max(prediction.length, config.min_window_seconds, unit)
    context_width = min(duration, max(2.0 * scale, config.min_window_seconds))
    if not state.left_boundary:
        return make_window(prediction.start, context_width, duration), "check_start"
    if not state.right_boundary:
        return make_window(prediction.end, context_width, duration), "check_end"
    if not state.support:
        support_width = min(duration, max(3.0 * scale, config.min_window_seconds))
        return make_window(prediction.center, support_width, duration), "support_probe"
    return None


def update_certificate_state(
    state: CertificateState,
    focal: SearchCandidate,
    evidence: SearchCandidate,
    action: str,
    unit: float,
    config: SearchConfig,
) -> CertificateState:
    """Apply one cached neural observation to a focal candidate's four-bit state."""
    if (
        focal.prediction is None
        or evidence.prediction is None
        or not focal.valid
        or not evidence.valid
        or evidence.source == focal.source
    ):
        return state

    prediction_iou = interval_iou(focal.prediction, evidence.prediction)
    proposal_iou = interval_iou(focal.proposal, evidence.proposal)
    boundary_tolerance = max(unit, 0.10 * focal.prediction.length)

    left_boundary = state.left_boundary
    right_boundary = state.right_boundary
    support = state.support
    contradiction = state.contradiction

    if (
        action == "check_start"
        and abs(evidence.prediction.start - focal.prediction.start) <= boundary_tolerance
        and prediction_iou >= config.certificate_boundary_iou
    ):
        left_boundary = True
    if (
        action == "check_end"
        and abs(evidence.prediction.end - focal.prediction.end) <= boundary_tolerance
        and prediction_iou >= config.certificate_boundary_iou
    ):
        right_boundary = True

    independently_framed = proposal_iou < config.graph_merge_iou
    if independently_framed and prediction_iou >= config.certificate_support_iou:
        support = True

    contradiction_weight = proposal_iou * (1.0 - prediction_iou)
    if (
        proposal_iou >= config.graph_contradiction_proposal_iou
        and prediction_iou <= config.graph_contradiction_prediction_iou
        and contradiction_weight > config.certificate_contradiction_weight
    ):
        contradiction = True

    return CertificateState(
        focal_source=state.focal_source,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        support=support,
        contradiction=contradiction,
    )


def _guidance_priority(
    state: CertificateState,
    graph: TemporalHypothesisGraph,
) -> Tuple[float, float, float]:
    support, contradiction = graph.evidence(state.focal_source)
    return float(state.deficit), 1.0 - support, contradiction


def _critic_payload(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {
            "choice": getattr(result, "choice", "uncertain"),
            "confidence": getattr(result, "confidence", 0.0),
            "failure_tag": getattr(result, "failure_tag", "uncertain"),
            "answer": getattr(result, "answer", ""),
        }
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    payload["confidence"] = min(max(confidence, 0.0), 1.0)
    payload["choice"] = str(payload.get("choice", "uncertain")).strip().upper()
    payload["failure_tag"] = str(payload.get("failure_tag", "uncertain"))
    payload["answer"] = str(payload.get("answer", ""))
    return payload


def _runner_verify_pair(
    runner: Any,
    anno: Dict[str, Any],
    candidate_a: Interval,
    candidate_b: Interval,
    fps_override: int,
) -> Dict[str, Any]:
    method = getattr(runner, "verify_pair", None)
    if method is None:
        return {
            "choice": "UNCERTAIN",
            "confidence": 0.0,
            "failure_tag": "critic_unavailable",
            "answer": "Runner does not implement verify_pair.",
        }
    return _critic_payload(
        method(
            anno,
            candidate_a,
            candidate_b,
            fps_override=fps_override,
        )
    )


def _finish_graph_result(
    result: SearchResult,
    graph: TemporalHypothesisGraph,
    search_cost: Dict[str, Any],
    verification_trace: Optional[List[Dict[str, Any]]] = None,
    controller_trace: Optional[List[Dict[str, Any]]] = None,
) -> SearchResult:
    graph_payload = graph.to_json()
    search_cost["graph_nodes"] = graph_payload["node_count"]
    search_cost["graph_edges"] = graph_payload["edge_count"]
    search_cost["transposition_hits"] = graph_payload["transposition_hits"]
    search_cost["total_model_calls"] = (
        search_cost.get("baseline_calls", 0) + search_cost.get("model_calls", 0)
    )
    result.graph = graph_payload
    result.search_cost = search_cost
    result.verification_trace = verification_trace or []
    result.controller_trace = controller_trace or []
    return result


def run_budgeted_graph_search_sample(
    anno: Dict[str, Any],
    baseline_pred: Optional[Dict[str, Any]],
    runner: Any,
    config: SearchConfig,
    unit: float = 1.0,
) -> SearchResult:
    duration = float(anno["duration"])
    baseline = baseline_interval_from_prediction(baseline_pred, duration)
    baseline_answer = ""
    if isinstance(baseline_pred, dict):
        baseline_answer = str(baseline_pred.get("answers", baseline_pred.get("answer", "")))

    max_total_calls = max(int(config.max_total_calls), 1)
    baseline_calls = 1 if baseline is not None else 0
    requested_critic_calls = max(int(config.critic_calls), 0)
    reserve_critic_calls = 0
    if config.critic_mode == "self" and baseline is not None:
        reserve_critic_calls = min(
            requested_critic_calls,
            max(max_total_calls - baseline_calls - 1, 0),
        )
    search_budget = max(max_total_calls - baseline_calls - reserve_critic_calls, 0)
    search_cost: Dict[str, Any] = {
        "model_calls": 0,
        "baseline_calls": baseline_calls,
        "critic_calls": 0,
        "clip_seconds": 0.0,
        "frame_seconds": 0.0,
        "calls_by_stage": {},
        "max_total_calls": max_total_calls,
        "search_call_budget": search_budget,
        "reserved_critic_calls": reserve_critic_calls,
        "config": {
            "search_mode": config.search_mode,
            "max_total_calls": max_total_calls,
            "graph_seed_count": config.graph_seed_count,
            "graph_beam_width": config.graph_beam_width,
            "critic_mode": config.critic_mode,
            "critic_calls": requested_critic_calls,
            "fps": config.fps,
            "zoom_fps": config.zoom_fps,
            "critic_fps": config.critic_fps,
        },
    }

    candidates: List[SearchCandidate] = []
    graph = TemporalHypothesisGraph(config, duration, unit)
    if baseline is not None:
        baseline_candidate = SearchCandidate(
            proposal=Interval(0.0, duration),
            prediction=baseline,
            raw_prediction=baseline,
            answer=baseline_answer,
            source="baseline",
            stage="baseline",
            round_index=-1,
            candidate_index=-1,
            valid=True,
            normalized_mode="baseline",
        )
        candidates.append(baseline_candidate)
        graph.add_candidate(baseline_candidate)

    if search_budget <= 0:
        result = select_final_candidate(
            candidates, baseline, baseline_answer, duration, unit, config, search_cost
        )
        return _finish_graph_result(result, graph, search_cost)

    frontier: List[Tuple[float, int, Interval, str, str]] = []
    frontier_sequence = 0
    for proposal, action in generate_graph_seed_windows(
        baseline, duration, config, unit
    ):
        parent_source = "baseline" if baseline is not None else ""
        parent = graph.nodes.get(parent_source)
        priority = _frontier_priority(parent, proposal, action, graph)
        frontier.append((priority, frontier_sequence, proposal, parent_source, action))
        frontier_sequence += 1

    expanded_sources = set()
    while frontier and search_cost["model_calls"] < search_budget:
        frontier.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        _, _, proposal, parent_source, action = frontier.pop(0)
        equivalent = graph.find_equivalent_proposal(proposal)
        if equivalent is not None:
            if parent_source and parent_source != equivalent.source:
                graph.add_transition(parent_source, equivalent.source, action)
            continue

        evaluated = _evaluate_proposals(
            anno=anno,
            runner=runner,
            proposals=[proposal],
            duration=duration,
            unit=unit,
            stage="graph",
            round_index=0,
            fps_override=(
                config.fps if action == "global_retrieve" else config.zoom_fps
            ),
            search_cost=search_cost,
        )[0]
        evaluated.action = action
        candidates.append(evaluated)
        graph.add_candidate(evaluated, parent_source=parent_source, action=action)
        score_candidates(candidates, baseline, duration, unit, config, graph=graph)

        ranked = _ranked_candidates(candidates, max(config.graph_beam_width, 1))
        ranked_sources = {candidate.source for candidate in ranked}
        if evaluated.source not in ranked_sources or evaluated.source in expanded_sources:
            continue
        expanded_sources.add(evaluated.source)
        for neighbour, neighbour_action in generate_graph_neighbors(
            evaluated, baseline, duration, config, unit
        ):
            priority = _frontier_priority(evaluated, neighbour, neighbour_action, graph)
            frontier.append(
                (
                    priority,
                    frontier_sequence,
                    neighbour,
                    evaluated.source,
                    neighbour_action,
                )
            )
            frontier_sequence += 1

    score_candidates(candidates, baseline, duration, unit, config, graph=graph)
    ranked = _ranked_candidates(candidates, 1)
    if not ranked:
        result = select_final_candidate(
            candidates, baseline, baseline_answer, duration, unit, config, search_cost
        )
        return _finish_graph_result(result, graph, search_cost)

    challenger = ranked[0]
    if (
        baseline is None
        or config.critic_mode == "none"
        or reserve_critic_calls <= 0
        or challenger.prediction is None
    ):
        result = select_final_candidate(
            candidates, baseline, baseline_answer, duration, unit, config, search_cost
        )
        return _finish_graph_result(result, graph, search_cost)

    if interval_iou(baseline, challenger.prediction) >= 0.98:
        rounded = round_interval(baseline, duration, unit)
        result = SearchResult(
            interval=rounded,
            answer=baseline_answer or format_answer(rounded),
            candidates=candidates,
            selected_source="baseline",
            fallback_used=True,
            fallback_reason="challenger_equivalent_to_baseline",
            selected_confidence=challenger.confidence,
            search_cost=search_cost,
        )
        return _finish_graph_result(result, graph, search_cost)

    verification_trace: List[Dict[str, Any]] = []
    normalized_choices: List[str] = []
    for critic_index in range(reserve_critic_calls):
        swapped = critic_index % 2 == 1
        candidate_a = challenger.prediction if swapped else baseline
        candidate_b = baseline if swapped else challenger.prediction
        verdict = _runner_verify_pair(
            runner,
            anno,
            candidate_a,
            candidate_b,
            fps_override=config.critic_fps,
        )
        search_cost["model_calls"] += 1
        search_cost["critic_calls"] += 1
        critic_margin = max(
            2.0,
            0.25 * max(candidate_a.length, candidate_b.length),
        )
        critic_clip = Interval(
            min(candidate_a.start, candidate_b.start) - critic_margin,
            max(candidate_a.end, candidate_b.end) + critic_margin,
        ).clamp(duration)
        search_cost["clip_seconds"] += critic_clip.length
        search_cost["frame_seconds"] += critic_clip.length * config.critic_fps
        calls_by_stage = search_cost.setdefault("calls_by_stage", {})
        calls_by_stage["critic"] = calls_by_stage.get("critic", 0) + 1
        choice = verdict["choice"]
        if choice == "A":
            normalized = "challenger" if swapped else "baseline"
        elif choice == "B":
            normalized = "baseline" if swapped else "challenger"
        else:
            normalized = "uncertain"
        if verdict["confidence"] < config.critic_min_confidence:
            normalized = "uncertain"
        normalized_choices.append(normalized)
        verification_trace.append(
            {
                "call": critic_index + 1,
                "swapped": swapped,
                "candidate_a": candidate_a.as_list(),
                "candidate_b": candidate_b.as_list(),
                "normalized_choice": normalized,
                **verdict,
            }
        )

    accept_challenger = bool(normalized_choices) and all(
        choice == "challenger" for choice in normalized_choices
    )
    if accept_challenger:
        rounded = round_interval(challenger.prediction, duration, unit)
        result = SearchResult(
            interval=rounded,
            answer=challenger.answer or format_answer(rounded),
            candidates=candidates,
            selected_source=challenger.source,
            fallback_used=False,
            selected_candidate_index=candidates.index(challenger),
            selected_confidence=challenger.confidence,
            search_cost=search_cost,
        )
    else:
        rounded = round_interval(baseline, duration, unit)
        if all(choice == "baseline" for choice in normalized_choices):
            fallback_reason = "critic_prefers_baseline"
        elif "uncertain" in normalized_choices:
            fallback_reason = "critic_uncertain"
        else:
            fallback_reason = "critic_order_disagreement"
        result = SearchResult(
            interval=rounded,
            answer=baseline_answer or format_answer(rounded),
            candidates=candidates,
            selected_source="baseline",
            fallback_used=True,
            fallback_reason=fallback_reason,
            selected_candidate_index=candidates.index(challenger),
            selected_confidence=challenger.confidence,
            search_cost=search_cost,
        )
    return _finish_graph_result(
        result, graph, search_cost, verification_trace=verification_trace
    )


def _certificate_fallback_result(
    candidates: Sequence[SearchCandidate],
    baseline: Optional[Interval],
    baseline_answer: str,
    duration: float,
    unit: float,
    search_cost: Dict[str, Any],
    reason: str,
    challenger: Optional[SearchCandidate] = None,
) -> SearchResult:
    if baseline is not None and baseline.is_valid(duration):
        rounded = round_interval(baseline, duration, unit)
        selected_index = None
        selected_confidence = 0.0
        if challenger is not None and challenger in candidates:
            selected_index = list(candidates).index(challenger)
            selected_confidence = challenger.confidence
        return SearchResult(
            interval=rounded,
            answer=baseline_answer or format_answer(rounded),
            candidates=list(candidates),
            selected_source="baseline",
            fallback_used=True,
            fallback_reason=reason,
            selected_candidate_index=selected_index,
            selected_confidence=selected_confidence,
            search_cost=search_cost,
        )
    placeholder = Interval(duration + 10.0, duration + 20.0)
    return SearchResult(
        interval=placeholder,
        answer="LOOKAGAIN_NO_VALID_BASELINE",
        candidates=list(candidates),
        selected_source="invalid_placeholder",
        fallback_used=True,
        fallback_reason=reason,
        search_cost=search_cost,
    )


def _accept_certified_challenger(
    challenger: SearchCandidate,
    candidates: Sequence[SearchCandidate],
    duration: float,
    unit: float,
    search_cost: Dict[str, Any],
) -> SearchResult:
    assert challenger.prediction is not None
    rounded = round_interval(challenger.prediction, duration, unit)
    return SearchResult(
        interval=rounded,
        answer=challenger.answer or format_answer(rounded),
        candidates=list(candidates),
        selected_source=challenger.source,
        fallback_used=False,
        selected_candidate_index=list(candidates).index(challenger),
        selected_confidence=challenger.confidence,
        search_cost=search_cost,
    )


def _verify_certified_challenger(
    anno: Dict[str, Any],
    runner: Any,
    challenger: SearchCandidate,
    candidates: Sequence[SearchCandidate],
    baseline: Optional[Interval],
    baseline_answer: str,
    duration: float,
    unit: float,
    config: SearchConfig,
    search_cost: Dict[str, Any],
) -> Tuple[SearchResult, List[Dict[str, Any]]]:
    """Apply the existing order-swapped verifier to one certified challenger."""
    if challenger.prediction is None:
        return (
            _certificate_fallback_result(
                candidates,
                baseline,
                baseline_answer,
                duration,
                unit,
                search_cost,
                "certified_challenger_invalid",
                challenger,
            ),
            [],
        )
    if baseline is None or not baseline.is_valid(duration):
        return (
            _certificate_fallback_result(
                candidates,
                baseline,
                baseline_answer,
                duration,
                unit,
                search_cost,
                "baseline_missing",
                challenger,
            ),
            [],
        )
    if interval_iou(baseline, challenger.prediction) >= 0.98:
        return (
            _certificate_fallback_result(
                candidates,
                baseline,
                baseline_answer,
                duration,
                unit,
                search_cost,
                "challenger_equivalent_to_baseline",
                challenger,
            ),
            [],
        )

    requested_critic_calls = max(int(config.critic_calls), 0)
    reserve_critic_calls = int(search_cost.get("reserved_critic_calls", 0))
    if config.critic_mode == "none" or requested_critic_calls == 0:
        return _accept_certified_challenger(
            challenger, candidates, duration, unit, search_cost
        ), []
    if reserve_critic_calls < requested_critic_calls:
        return (
            _certificate_fallback_result(
                candidates,
                baseline,
                baseline_answer,
                duration,
                unit,
                search_cost,
                "critic_budget_unavailable",
                challenger,
            ),
            [],
        )

    verification_trace: List[Dict[str, Any]] = []
    normalized_choices: List[str] = []
    for critic_index in range(reserve_critic_calls):
        swapped = critic_index % 2 == 1
        candidate_a = challenger.prediction if swapped else baseline
        candidate_b = baseline if swapped else challenger.prediction
        verdict = _runner_verify_pair(
            runner,
            anno,
            candidate_a,
            candidate_b,
            fps_override=config.critic_fps,
        )
        search_cost["model_calls"] += 1
        search_cost["critic_calls"] += 1
        critic_margin = max(2.0, 0.25 * max(candidate_a.length, candidate_b.length))
        critic_clip = Interval(
            min(candidate_a.start, candidate_b.start) - critic_margin,
            max(candidate_a.end, candidate_b.end) + critic_margin,
        ).clamp(duration)
        search_cost["clip_seconds"] += critic_clip.length
        search_cost["frame_seconds"] += critic_clip.length * config.critic_fps
        calls_by_stage = search_cost.setdefault("calls_by_stage", {})
        calls_by_stage["critic"] = calls_by_stage.get("critic", 0) + 1
        choice = verdict["choice"]
        if choice == "A":
            normalized = "challenger" if swapped else "baseline"
        elif choice == "B":
            normalized = "baseline" if swapped else "challenger"
        else:
            normalized = "uncertain"
        if verdict["confidence"] < config.critic_min_confidence:
            normalized = "uncertain"
        normalized_choices.append(normalized)
        verification_trace.append(
            {
                "call": critic_index + 1,
                "swapped": swapped,
                "candidate_a": candidate_a.as_list(),
                "candidate_b": candidate_b.as_list(),
                "normalized_choice": normalized,
                **verdict,
            }
        )

    if normalized_choices and all(choice == "challenger" for choice in normalized_choices):
        return (
            _accept_certified_challenger(
                challenger, candidates, duration, unit, search_cost
            ),
            verification_trace,
        )
    if normalized_choices and all(choice == "baseline" for choice in normalized_choices):
        reason = "critic_prefers_baseline"
    elif "uncertain" in normalized_choices:
        reason = "critic_uncertain"
    else:
        reason = "critic_order_disagreement"
    return (
        _certificate_fallback_result(
            candidates,
            baseline,
            baseline_answer,
            duration,
            unit,
            search_cost,
            reason,
            challenger,
        ),
        verification_trace,
    )


def run_lookagain_astar_search_sample(
    anno: Dict[str, Any],
    baseline_pred: Optional[Dict[str, Any]],
    runner: Any,
    config: SearchConfig,
    unit: float = 1.0,
) -> SearchResult:
    """Certificate-guided graph search with a UCS anchor and optional guidance."""
    controller = str(config.lookagain_controller).strip().lower()
    if controller not in ("ucs", "guided"):
        raise ValueError("lookagain_controller must be 'ucs' or 'guided'")

    duration = float(anno["duration"])
    baseline = baseline_interval_from_prediction(baseline_pred, duration)
    baseline_answer = ""
    if isinstance(baseline_pred, dict):
        baseline_answer = str(baseline_pred.get("answers", baseline_pred.get("answer", "")))

    max_total_calls = max(int(config.max_total_calls), 1)
    baseline_calls = 1 if baseline is not None else 0
    requested_critic_calls = max(int(config.critic_calls), 0)
    reserve_critic_calls = 0
    if config.critic_mode == "self" and baseline is not None:
        reserve_critic_calls = min(
            requested_critic_calls,
            max(max_total_calls - baseline_calls - 1, 0),
        )
    search_budget = max(max_total_calls - baseline_calls - reserve_critic_calls, 0)
    search_cost: Dict[str, Any] = {
        "model_calls": 0,
        "baseline_calls": baseline_calls,
        "critic_calls": 0,
        "clip_seconds": 0.0,
        "frame_seconds": 0.0,
        "calls_by_stage": {},
        "max_total_calls": max_total_calls,
        "search_call_budget": search_budget,
        "reserved_critic_calls": reserve_critic_calls,
        "controller": controller,
        "controller_expansions": 0,
        "anchor_pops": 0,
        "guided_pops": 0,
        "certificate_found": False,
        "config": {
            "search_mode": config.search_mode,
            "lookagain_controller": controller,
            "lookagain_anchor_period": max(int(config.lookagain_anchor_period), 1),
            "max_total_calls": max_total_calls,
            "graph_seed_count": config.graph_seed_count,
            "certificate_boundary_iou": config.certificate_boundary_iou,
            "certificate_support_iou": config.certificate_support_iou,
            "certificate_contradiction_weight": config.certificate_contradiction_weight,
            "critic_mode": config.critic_mode,
            "critic_calls": requested_critic_calls,
            "fps": config.fps,
            "zoom_fps": config.zoom_fps,
            "critic_fps": config.critic_fps,
        },
    }

    graph = TemporalHypothesisGraph(config, duration, unit)
    candidates: List[SearchCandidate] = []
    controller_trace: List[Dict[str, Any]] = []

    def append_trace(payload: Dict[str, Any]) -> None:
        if len(controller_trace) < max(int(config.max_controller_trace_steps), 0):
            controller_trace.append(payload)
        else:
            search_cost["controller_trace_truncated"] = True

    if baseline is None:
        result = _certificate_fallback_result(
            candidates,
            baseline,
            baseline_answer,
            duration,
            unit,
            search_cost,
            "baseline_missing",
        )
        return _finish_graph_result(
            result, graph, search_cost, controller_trace=controller_trace
        )

    baseline_candidate = SearchCandidate(
        proposal=Interval(0.0, duration),
        prediction=baseline,
        raw_prediction=baseline,
        answer=baseline_answer,
        source="baseline",
        stage="baseline",
        round_index=-1,
        candidate_index=-1,
        valid=True,
        normalized_mode="baseline",
    )
    candidates.append(baseline_candidate)
    graph.add_candidate(baseline_candidate)

    anchor_heap: List[Tuple[int, int, CertificateState]] = []
    guided_heap: List[Tuple[float, float, float, int, int, Tuple[int, int], CertificateState]] = []
    best_cost: Dict[CertificateState, int] = {}
    expanded_cost: Dict[CertificateState, int] = {}
    queue_sequence = 0

    def graph_version() -> Tuple[int, int]:
        return len(graph.nodes), len(graph.edges)

    def push_state(state: CertificateState, path_cost: int) -> None:
        nonlocal queue_sequence
        previous = best_cost.get(state)
        if previous is not None and previous <= path_cost:
            return
        best_cost[state] = path_cost
        sequence = queue_sequence
        queue_sequence += 1
        heapq.heappush(anchor_heap, (path_cost, sequence, state))
        if controller == "guided":
            deficit, support_deficit, contradiction = _guidance_priority(state, graph)
            heapq.heappush(
                guided_heap,
                (
                    deficit,
                    support_deficit,
                    contradiction,
                    path_cost,
                    sequence,
                    graph_version(),
                    state,
                ),
            )

    def pop_anchor() -> Optional[Tuple[int, CertificateState]]:
        while anchor_heap:
            path_cost, _, state = heapq.heappop(anchor_heap)
            if best_cost.get(state) == path_cost:
                return path_cost, state
        return None

    def pop_guided() -> Optional[Tuple[int, CertificateState]]:
        while guided_heap:
            (
                old_deficit,
                old_support_deficit,
                old_contradiction,
                path_cost,
                sequence,
                old_version,
                state,
            ) = heapq.heappop(guided_heap)
            if best_cost.get(state) != path_cost:
                continue
            current_priority = _guidance_priority(state, graph)
            current_version = graph_version()
            old_priority = (old_deficit, old_support_deficit, old_contradiction)
            if old_version != current_version or old_priority != current_priority:
                heapq.heappush(
                    guided_heap,
                    (*current_priority, path_cost, sequence, current_version, state),
                )
                continue
            return path_cost, state
        return None

    def observe(
        proposal: Interval,
        parent_source: str,
        action: str,
    ) -> Tuple[SearchCandidate, int, bool]:
        equivalent = graph.find_equivalent_proposal(proposal)
        if equivalent is not None:
            if parent_source and parent_source != equivalent.source:
                graph.add_transition(parent_source, equivalent.source, action)
            return equivalent, 0, True
        evaluated = _evaluate_proposals(
            anno=anno,
            runner=runner,
            proposals=[proposal],
            duration=duration,
            unit=unit,
            stage="lookagain",
            round_index=0,
            fps_override=config.fps if action == "global_retrieve" else config.zoom_fps,
            search_cost=search_cost,
        )[0]
        evaluated.action = action
        candidates.append(evaluated)
        canonical = graph.add_candidate(
            evaluated, parent_source=parent_source, action=action
        )
        score_candidates(candidates, baseline, duration, unit, config, graph=graph)
        return canonical, 1, False

    push_state(CertificateState(focal_source="baseline"), 0)
    certified_state: Optional[CertificateState] = None
    scheduler_steps = 0
    anchor_period = max(int(config.lookagain_anchor_period), 1)

    while anchor_heap:
        queue_name = "anchor"
        item: Optional[Tuple[int, CertificateState]]
        if controller == "guided" and scheduler_steps % anchor_period != 0:
            queue_name = "guided"
            item = pop_guided()
            if item is None:
                queue_name = "anchor"
                item = pop_anchor()
        else:
            item = pop_anchor()
        if item is None:
            break
        path_cost, state = item
        focal = graph.nodes.get(state.focal_source)

        if state.is_certified(focal):
            scheduler_steps += 1
            search_cost[f"{queue_name}_pops"] += 1
            if queue_name == "anchor":
                certified_state = state
                append_trace(
                    {
                        "event": "certificate",
                        "queue": queue_name,
                        "path_cost": path_cost,
                        "state": state.to_json(),
                    }
                )
                break
            append_trace(
                {
                    "event": "certificate_waiting_for_anchor",
                    "queue": queue_name,
                    "path_cost": path_cost,
                    "state": state.to_json(),
                }
            )
            continue
        if focal is None or not focal.valid or focal.prediction is None:
            continue
        previous_expansion = expanded_cost.get(state)
        if previous_expansion is not None and previous_expansion <= path_cost:
            continue
        if search_cost["model_calls"] >= search_budget:
            continue
        scheduler_steps += 1
        search_cost[f"{queue_name}_pops"] += 1
        expanded_cost[state] = path_cost
        search_cost["controller_expansions"] += 1
        append_trace(
            {
                "event": "expand",
                "queue": queue_name,
                "path_cost": path_cost,
                "state": state.to_json(),
            }
        )

        if focal.stage == "baseline":
            seed_actions = generate_graph_seed_windows(
                baseline, duration, config, unit
            )
            for proposal, action in seed_actions:
                if search_cost["model_calls"] >= search_budget:
                    break
                evidence, edge_cost, cache_hit = observe(
                    proposal, focal.source, action
                )
                append_trace(
                    {
                        "event": "observe",
                        "queue": queue_name,
                        "path_cost": path_cost + edge_cost,
                        "focal_source": focal.source,
                        "evidence_source": evidence.source,
                        "action": action,
                        "cache_hit": cache_hit,
                    }
                )
                if evidence.valid and evidence.prediction is not None and evidence.stage != "baseline":
                    push_state(
                        CertificateState(focal_source=evidence.source),
                        path_cost + edge_cost,
                    )
            continue

        if state.contradiction:
            append_trace(
                {
                    "event": "contradicted_dead_end",
                    "queue": queue_name,
                    "path_cost": path_cost,
                    "state": state.to_json(),
                }
            )
            continue
        probe = generate_lookagain_probe(
            state, focal, duration, config, unit
        )
        if probe is None:
            continue
        proposal, action = probe
        evidence, edge_cost, cache_hit = observe(proposal, focal.source, action)
        successor_cost = path_cost + edge_cost
        successor = update_certificate_state(
            state, focal, evidence, action, unit, config
        )
        append_trace(
            {
                "event": "probe",
                "queue": queue_name,
                "path_cost": successor_cost,
                "focal_source": focal.source,
                "evidence_source": evidence.source,
                "action": action,
                "cache_hit": cache_hit,
                "before": state.to_json(),
                "after": successor.to_json(),
            }
        )
        if evidence.valid and evidence.prediction is not None and evidence.stage != "baseline":
            push_state(
                CertificateState(focal_source=evidence.source),
                successor_cost,
            )
        if successor != state:
            push_state(successor, successor_cost)

    score_candidates(candidates, baseline, duration, unit, config, graph=graph)
    search_cost["search_model_calls"] = search_cost["model_calls"]
    search_cost["lifted_states"] = len(best_cost)
    search_cost["closed_states"] = len(expanded_cost)
    search_cost["certificate_found"] = certified_state is not None

    if certified_state is None:
        reason = (
            "certificate_not_found_within_budget"
            if search_cost["model_calls"] >= search_budget
            else "certificate_frontier_exhausted"
        )
        result = _certificate_fallback_result(
            candidates,
            baseline,
            baseline_answer,
            duration,
            unit,
            search_cost,
            reason,
        )
        return _finish_graph_result(
            result,
            graph,
            search_cost,
            controller_trace=controller_trace,
        )

    challenger = graph.nodes[certified_state.focal_source]
    search_cost["certificate_source"] = challenger.source
    search_cost["certificate_state"] = certified_state.to_json()
    result, verification_trace = _verify_certified_challenger(
        anno,
        runner,
        challenger,
        candidates,
        baseline,
        baseline_answer,
        duration,
        unit,
        config,
        search_cost,
    )
    return _finish_graph_result(
        result,
        graph,
        search_cost,
        verification_trace=verification_trace,
        controller_trace=controller_trace,
    )


def run_timelens_search_sample(
    anno: Dict[str, Any],
    baseline_pred: Optional[Dict[str, Any]],
    runner: Any,
    config: Optional[SearchConfig] = None,
    unit: float = 1.0,
) -> SearchResult:
    config = config or SearchConfig()
    if config.search_mode == "lookagain_astar":
        return run_lookagain_astar_search_sample(
            anno=anno,
            baseline_pred=baseline_pred,
            runner=runner,
            config=config,
            unit=unit,
        )
    if config.search_mode == "budgeted_graph":
        return run_budgeted_graph_search_sample(
            anno=anno,
            baseline_pred=baseline_pred,
            runner=runner,
            config=config,
            unit=unit,
        )
    duration = float(anno["duration"])
    baseline = baseline_interval_from_prediction(baseline_pred, duration)
    baseline_answer = ""
    if isinstance(baseline_pred, dict):
        baseline_answer = str(baseline_pred.get("answers", baseline_pred.get("answer", "")))

    candidates: List[SearchCandidate] = []
    if baseline is not None:
        candidates.append(
            SearchCandidate(
                proposal=Interval(0.0, duration),
                prediction=baseline,
                raw_prediction=baseline,
                answer=baseline_answer,
                source="baseline",
                stage="baseline",
                round_index=-1,
                candidate_index=-1,
                valid=True,
                normalized_mode="baseline",
            )
        )

    search_cost: Dict[str, Any] = {
        "model_calls": 0,
        "clip_seconds": 0.0,
        "frame_seconds": 0.0,
        "calls_by_stage": {},
        "config": {
            "search_mode": config.search_mode,
            "coarse_grid_size": config.coarse_grid_size,
            "coarse_window_fraction": config.coarse_window_fraction,
            "refine_rounds": config.refine_rounds,
            "top_k_stage1": config.top_k_stage1,
            "top_k_refine": config.top_k_refine,
            "fps": config.fps,
            "zoom_fps": config.zoom_fps,
            "dense_zoom_fps": config.dense_zoom_fps,
        },
    }

    seen_proposals = set()
    coarse_proposals = generate_coarse_windows(duration, config, unit)
    coarse_proposals = [
        proposal
        for proposal in coarse_proposals
        if not (key := _proposal_key(proposal, unit)) in seen_proposals
        and not seen_proposals.add(key)
    ]
    candidates.extend(
        _evaluate_proposals(
            anno=anno,
            runner=runner,
            proposals=coarse_proposals,
            duration=duration,
            unit=unit,
            stage="coarse",
            round_index=0,
            fps_override=config.fps,
            search_cost=search_cost,
        )
    )
    score_candidates(candidates, baseline, duration, unit, config)

    if config.search_mode == "coarse_only":
        return select_final_candidate(
            candidates, baseline, baseline_answer, duration, unit, config, search_cost
        )
    if config.search_mode != "exhaustive_refine":
        raise ValueError(f"Unsupported search_mode: {config.search_mode}")

    for round_index in range(1, max(config.refine_rounds, 0) + 1):
        top_k = config.top_k_stage1 if round_index == 1 else config.top_k_refine
        anchors = _ranked_candidates(candidates, top_k)
        if not anchors:
            break

        next_proposals: List[Interval] = []
        for anchor in anchors:
            assert anchor.prediction is not None
            for proposal in generate_refinement_windows(
                anchor.prediction,
                duration,
                config,
                unit,
                round_index=round_index,
            ):
                key = _proposal_key(proposal, unit)
                if key in seen_proposals:
                    continue
                seen_proposals.add(key)
                next_proposals.append(proposal)
        if not next_proposals:
            break

        fps_override = config.zoom_fps if round_index == 1 else config.dense_zoom_fps
        candidates.extend(
            _evaluate_proposals(
                anno=anno,
                runner=runner,
                proposals=next_proposals,
                duration=duration,
                unit=unit,
                stage="refine",
                round_index=round_index,
                fps_override=fps_override,
                search_cost=search_cost,
            )
        )
        score_candidates(candidates, baseline, duration, unit, config)

    score_candidates(candidates, baseline, duration, unit, config)
    return select_final_candidate(
        candidates, baseline, baseline_answer, duration, unit, config, search_cost
    )
