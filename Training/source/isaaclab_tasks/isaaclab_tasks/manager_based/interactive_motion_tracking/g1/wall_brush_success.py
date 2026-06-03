from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import os
from typing import Tuple


Vec3 = Tuple[float, float, float]
VISUAL_REVIEW_VIDEO_FILE = "rl-video-step-0.mp4"


@dataclass(frozen=True)
class WallBrushSuccessThresholds:
    contact_ratio: float
    row_ratio: float
    combined_ratio: float
    coverage: float
    anchor_radius: float
    row_radius: float | None = None

    def __post_init__(self):
        if self.row_radius is None:
            object.__setattr__(self, "row_radius", self.anchor_radius)


@dataclass(frozen=True)
class WallBrushStepSample:
    brush_tip: Vec3
    wall_start: Vec3
    wall_mid: Vec3
    wall_end: Vec3
    active: bool
    reset_boundary: bool
    legal_wall_contact: bool


@dataclass(frozen=True)
class WallBrushMilestoneReward:
    anchor_hit_bonus: float = 0.0
    all_anchor_hit_bonus: float = 0.0
    canceled_success: bool = False


@dataclass(frozen=True)
class WallBrushPriorEvaluation:
    successful_prior_count: int
    total_prior_count: int
    success_rate: float
    training_milestone_pass: bool
    acceptance_target_pass: bool


@dataclass(frozen=True)
class WallBrushPriorMetricSummary:
    expected_prior_count: int
    evaluated_prior_count: int
    dreamcontrol_style_success_count: int
    dreamcontrol_style_success_rate: float
    training_milestone_pass_count: int
    training_milestone_required_count: int
    training_milestone_pass_fixed: bool
    training_milestone_pass_rate: float
    acceptance_target_pass_count: int
    acceptance_target_required_count: int
    acceptance_target_pass_fixed: bool
    acceptance_target_pass_rate: float


@dataclass(frozen=True)
class WallBrushEpisodeSuccess:
    active_steps: int = 0
    contact_steps: int = 0
    row_steps: int = 0
    combined_steps: int = 0
    min_combined_phase: float | None = None
    max_combined_phase: float | None = None
    next_anchor_index: int = 0
    pending_success: bool = False
    countable_success: bool = False
    invalidated: bool = False

    @classmethod
    def start(cls) -> "WallBrushEpisodeSuccess":
        return cls()


def _yz_distance(a: Vec3, b: Vec3) -> float:
    return sqrt((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _phase_on_row(point: Vec3, start: Vec3, end: Vec3) -> float:
    line_y = end[1] - start[1]
    line_z = end[2] - start[2]
    denom = line_y * line_y + line_z * line_z
    if denom <= 1e-12:
        return 0.0
    phase = ((point[1] - start[1]) * line_y + (point[2] - start[2]) * line_z) / denom
    return max(0.0, min(1.0, phase))


def _distance_to_row(point: Vec3, start: Vec3, end: Vec3) -> float:
    phase = _phase_on_row(point, start, end)
    projected = (point[0], start[1] + (end[1] - start[1]) * phase, start[2] + (end[2] - start[2]) * phase)
    return _yz_distance(point, projected)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _coverage(state: WallBrushEpisodeSuccess) -> float:
    if state.min_combined_phase is None or state.max_combined_phase is None:
        return 0.0
    return max(0.0, state.max_combined_phase - state.min_combined_phase)


def _meets_success_thresholds(state: WallBrushEpisodeSuccess, thresholds: WallBrushSuccessThresholds) -> bool:
    return (
        not state.invalidated
        and state.next_anchor_index >= 3
        and _ratio(state.contact_steps, state.active_steps) >= thresholds.contact_ratio
        and _ratio(state.row_steps, state.active_steps) >= thresholds.row_ratio
        and _ratio(state.combined_steps, state.active_steps) >= thresholds.combined_ratio
        and _coverage(state) >= thresholds.coverage
    )


def update_wall_brush_episode_success(
    state: WallBrushEpisodeSuccess,
    sample: WallBrushStepSample,
    thresholds: WallBrushSuccessThresholds,
) -> tuple[WallBrushEpisodeSuccess, WallBrushMilestoneReward]:
    if sample.reset_boundary:
        canceled = state.pending_success or state.countable_success
        return (
            WallBrushEpisodeSuccess(
                active_steps=state.active_steps,
                contact_steps=state.contact_steps,
                row_steps=state.row_steps,
                combined_steps=state.combined_steps,
                min_combined_phase=state.min_combined_phase,
                max_combined_phase=state.max_combined_phase,
                next_anchor_index=state.next_anchor_index,
                pending_success=False,
                countable_success=False,
                invalidated=True,
            ),
            WallBrushMilestoneReward(canceled_success=canceled),
        )

    if not sample.active or state.invalidated:
        return state, WallBrushMilestoneReward()

    active_steps = state.active_steps + 1
    legal_contact = sample.legal_wall_contact
    row_valid = _distance_to_row(sample.brush_tip, sample.wall_start, sample.wall_end) <= float(thresholds.row_radius)
    combined_valid = legal_contact and row_valid
    phase = _phase_on_row(sample.brush_tip, sample.wall_start, sample.wall_end)

    min_phase = state.min_combined_phase
    max_phase = state.max_combined_phase
    if combined_valid:
        min_phase = phase if min_phase is None else min(min_phase, phase)
        max_phase = phase if max_phase is None else max(max_phase, phase)

    anchors = (sample.wall_start, sample.wall_mid, sample.wall_end)
    next_anchor_index = state.next_anchor_index
    anchor_bonus = 0.0
    all_anchor_bonus = 0.0
    if legal_contact and next_anchor_index < len(anchors):
        if _yz_distance(sample.brush_tip, anchors[next_anchor_index]) <= thresholds.anchor_radius:
            next_anchor_index += 1
            anchor_bonus = 1.0
            if next_anchor_index == len(anchors):
                all_anchor_bonus = 3.0

    next_state = WallBrushEpisodeSuccess(
        active_steps=active_steps,
        contact_steps=state.contact_steps + int(legal_contact),
        row_steps=state.row_steps + int(row_valid),
        combined_steps=state.combined_steps + int(combined_valid),
        min_combined_phase=min_phase,
        max_combined_phase=max_phase,
        next_anchor_index=next_anchor_index,
        pending_success=False,
        countable_success=False,
        invalidated=False,
    )
    success = _meets_success_thresholds(next_state, thresholds)
    next_state = WallBrushEpisodeSuccess(
        active_steps=next_state.active_steps,
        contact_steps=next_state.contact_steps,
        row_steps=next_state.row_steps,
        combined_steps=next_state.combined_steps,
        min_combined_phase=next_state.min_combined_phase,
        max_combined_phase=next_state.max_combined_phase,
        next_anchor_index=next_state.next_anchor_index,
        pending_success=success,
        countable_success=success,
        invalidated=False,
    )
    return next_state, WallBrushMilestoneReward(anchor_hit_bonus=anchor_bonus, all_anchor_hit_bonus=all_anchor_bonus)


def evaluate_27_prior_successes(
    prior_successes,
    *,
    expected_prior_count: int = 27,
    training_milestone_count: int = 24,
) -> WallBrushPriorEvaluation:
    prior_successes = list(prior_successes)
    if len(prior_successes) != expected_prior_count:
        raise ValueError(f"Expected {expected_prior_count} prior results, got {len(prior_successes)}")
    successful = sum(1 for success in prior_successes if bool(success))
    return WallBrushPriorEvaluation(
        successful_prior_count=successful,
        total_prior_count=expected_prior_count,
        success_rate=successful / expected_prior_count,
        training_milestone_pass=successful >= training_milestone_count,
        acceptance_target_pass=successful == expected_prior_count,
    )


def summarize_27_prior_metrics(
    prior_metrics,
    training_passes,
    acceptance_passes,
    *,
    expected_prior_count: int = 27,
    training_milestone_count: int = 24,
) -> WallBrushPriorMetricSummary:
    prior_metrics = list(prior_metrics)
    training_passes = [bool(item) for item in training_passes]
    acceptance_passes = [bool(item) for item in acceptance_passes]
    if len(prior_metrics) != expected_prior_count:
        raise ValueError(f"Expected {expected_prior_count} prior metrics, got {len(prior_metrics)}")
    if len(training_passes) != expected_prior_count:
        raise ValueError(f"Expected {expected_prior_count} training pass results, got {len(training_passes)}")
    if len(acceptance_passes) != expected_prior_count:
        raise ValueError(f"Expected {expected_prior_count} acceptance pass results, got {len(acceptance_passes)}")

    dreamcontrol_successes = sum(1 for row in prior_metrics if int(row.get("n_successes", 0)) > 0)
    training_count = sum(1 for item in training_passes if item)
    acceptance_count = sum(1 for item in acceptance_passes if item)
    return WallBrushPriorMetricSummary(
        expected_prior_count=expected_prior_count,
        evaluated_prior_count=len(prior_metrics),
        dreamcontrol_style_success_count=dreamcontrol_successes,
        dreamcontrol_style_success_rate=dreamcontrol_successes / expected_prior_count,
        training_milestone_pass_count=training_count,
        training_milestone_required_count=training_milestone_count,
        training_milestone_pass_fixed=training_count >= training_milestone_count,
        training_milestone_pass_rate=training_count / expected_prior_count,
        acceptance_target_pass_count=acceptance_count,
        acceptance_target_required_count=expected_prior_count,
        acceptance_target_pass_fixed=acceptance_count == expected_prior_count,
        acceptance_target_pass_rate=acceptance_count / expected_prior_count,
    )


def build_wall_brush_visual_review_entries(
    checkpoint_path,
    prior_id: int,
    view_prefix: str,
    *,
    video_length: int = 220,
):
    checkpoint_dir = os.path.dirname(os.fspath(checkpoint_path))
    prior_id = int(prior_id)
    clean_prefix = str(view_prefix).strip().strip("/") or "wall_brush_visual_review"
    views = [
        {
            "view": "side",
            "view_name": f"{clean_prefix}_prior{prior_id}_side_wall_marker",
            "camera_eye": "0.08,-1.90,1.02",
            "camera_lookat": "0.45,0.00,0.92",
            "camera_resolution": "1600,1000",
        },
        {
            "view": "oblique",
            "view_name": f"{clean_prefix}_prior{prior_id}_oblique_wall_marker",
            "camera_eye": "-0.85,-1.65,1.22",
            "camera_lookat": "0.36,0.00,0.92",
            "camera_resolution": "1600,1000",
        },
    ]
    for item in views:
        item["prior_id"] = prior_id
        item["video_length"] = int(video_length)
        item["video_path"] = os.path.join(checkpoint_dir, "videos", item["view_name"], VISUAL_REVIEW_VIDEO_FILE)
    return views


def select_suspicious_prior_for_visual_review(prior_metrics):
    prior_metrics = list(prior_metrics)
    if not prior_metrics:
        raise ValueError("Cannot select a suspicious prior from an empty metric list")

    def risk_key(metric):
        self_collision_resets = int(
            metric.get("self_collision_resets", metric.get("self_collision_termination_count", 0))
        )
        self_collision_violation = float(metric.get("max_self_collision_proxy_violation", 0.0))
        self_collision_margin = float(metric.get("min_self_collision_margin_m", 1e6))
        self_collision_invalid = (
            self_collision_resets > 0 or self_collision_violation > 1e-6 or self_collision_margin < 0.0
        )

        illegal_resets = int(metric.get("illegal_contact_resets", 0))
        illegal_ratio = float(metric.get("illegal_contact_ratio", 0.0))
        n_successes = int(metric.get("n_successes", 0))
        failed_success = 1 if n_successes <= 0 else 0
        clearance = float(metric.get("min_nonbrush_clearance_m", 1e6))
        pose_prior_score = max(
            float(metric.get("mean_joint_prior_error_rad", 0.0)) / 0.45,
            float(metric.get("mean_right_arm_joint_prior_error_rad", 0.0)) / 0.35,
            float(metric.get("mean_root_orientation_error_deg", 0.0)) / 18.0,
            float(metric.get("mean_root_position_error_m", 0.0)) / 0.12,
        )
        pose_prior_invalid = pose_prior_score > 1.0
        return (
            self_collision_invalid,
            self_collision_resets,
            self_collision_violation,
            -self_collision_margin,
            illegal_resets > 0 or illegal_ratio > 0.0,
            illegal_resets,
            illegal_ratio,
            pose_prior_invalid,
            pose_prior_score,
            failed_success,
            -clearance,
        )

    return max(prior_metrics, key=risk_key)
