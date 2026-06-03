import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks"
    / "manager_based"
    / "interactive_motion_tracking"
    / "g1"
    / "wall_brush_success.py"
)

spec = importlib.util.spec_from_file_location("wall_brush_success", MODULE_PATH)
wall_brush_success = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wall_brush_success
spec.loader.exec_module(wall_brush_success)


def _sample(y, legal=True, active=True, reset=False):
    return wall_brush_success.WallBrushStepSample(
        brush_tip=(0.45, y, 0.90),
        wall_start=(0.45, 0.00, 0.90),
        wall_mid=(0.45, 0.50, 0.90),
        wall_end=(0.45, 1.00, 0.90),
        active=active,
        reset_boundary=reset,
        legal_wall_contact=legal,
    )


class WallBrushSuccessTest(unittest.TestCase):
    def test_episode_success_requires_ordered_anchor_hits_and_is_canceled_by_reset_boundary(self):
        thresholds = wall_brush_success.WallBrushSuccessThresholds(
            contact_ratio=0.60,
            row_ratio=0.60,
            combined_ratio=0.50,
            coverage=0.70,
            anchor_radius=0.05,
        )
        state = wall_brush_success.WallBrushEpisodeSuccess.start()

        for y in (0.0, 0.25, 0.50, 0.75, 1.0):
            state, reward = wall_brush_success.update_wall_brush_episode_success(state, _sample(y), thresholds)

        self.assertGreater(reward.anchor_hit_bonus, 0.0)
        self.assertTrue(state.pending_success)
        self.assertTrue(state.countable_success)

        state, reward = wall_brush_success.update_wall_brush_episode_success(state, _sample(1.0, reset=True), thresholds)

        self.assertTrue(reward.canceled_success)
        self.assertFalse(state.pending_success)
        self.assertFalse(state.countable_success)

    def test_success_rate_uses_fixed_27_prior_denominator(self):
        result = wall_brush_success.evaluate_27_prior_successes([True] * 24 + [False] * 3)

        self.assertEqual(result.successful_prior_count, 24)
        self.assertEqual(result.total_prior_count, 27)
        self.assertAlmostEqual(result.success_rate, 24 / 27)
        self.assertTrue(result.training_milestone_pass)
        self.assertFalse(result.acceptance_target_pass)

        final_result = wall_brush_success.evaluate_27_prior_successes([True] * 27)

        self.assertTrue(final_result.training_milestone_pass)
        self.assertTrue(final_result.acceptance_target_pass)

    def test_metric_summary_rates_use_fixed_27_prior_denominator_even_when_some_priors_are_inactive(self):
        metrics = [{"prior_id": idx, "active_steps": 52, "n_successes": 1} for idx in range(24)]
        metrics.extend({"prior_id": idx, "active_steps": 0, "n_successes": 0} for idx in range(24, 27))
        training_passes = [True] * 24 + [False] * 3
        acceptance_passes = [True] * 26 + [False]

        summary = wall_brush_success.summarize_27_prior_metrics(metrics, training_passes, acceptance_passes)

        self.assertEqual(summary.expected_prior_count, 27)
        self.assertEqual(summary.evaluated_prior_count, 27)
        self.assertEqual(summary.dreamcontrol_style_success_count, 24)
        self.assertAlmostEqual(summary.dreamcontrol_style_success_rate, 24 / 27)
        self.assertEqual(summary.training_milestone_pass_count, 24)
        self.assertAlmostEqual(summary.training_milestone_pass_rate, 24 / 27)
        self.assertTrue(summary.training_milestone_pass_fixed)
        self.assertEqual(summary.acceptance_target_pass_count, 26)
        self.assertAlmostEqual(summary.acceptance_target_pass_rate, 26 / 27)
        self.assertFalse(summary.acceptance_target_pass_fixed)

    def test_suspicious_prior_selection_prefers_invalid_contact_over_pretty_video(self):
        metrics = [
            {"prior_id": 0, "n_successes": 1, "illegal_contact_resets": 0, "min_nonbrush_clearance_m": 0.18},
            {"prior_id": 6, "n_successes": 1, "illegal_contact_resets": 0, "min_nonbrush_clearance_m": 0.021},
            {"prior_id": 9, "n_successes": 0, "illegal_contact_resets": 1, "min_nonbrush_clearance_m": 0.11},
        ]

        selected = wall_brush_success.select_suspicious_prior_for_visual_review(metrics)

        self.assertEqual(selected["prior_id"], 9)

    def test_suspicious_prior_selection_uses_illegal_contact_ratio_without_reset(self):
        metrics = [
            {"prior_id": 0, "n_successes": 1, "illegal_contact_ratio": 0.0, "min_nonbrush_clearance_m": 0.02},
            {"prior_id": 4, "n_successes": 1, "illegal_contact_ratio": 0.2, "min_nonbrush_clearance_m": 0.12},
        ]

        selected = wall_brush_success.select_suspicious_prior_for_visual_review(metrics)

        self.assertEqual(selected["prior_id"], 4)

    def test_suspicious_prior_selection_prefers_self_collision_over_low_wall_clearance(self):
        metrics = [
            {"prior_id": 0, "n_successes": 1, "illegal_contact_resets": 0, "min_nonbrush_clearance_m": 0.01},
            {
                "prior_id": 5,
                "n_successes": 1,
                "illegal_contact_resets": 0,
                "self_collision_resets": 1,
                "max_self_collision_proxy_violation": 0.25,
                "min_self_collision_margin_m": -0.018,
                "min_nonbrush_clearance_m": 0.12,
            },
        ]

        selected = wall_brush_success.select_suspicious_prior_for_visual_review(metrics)

        self.assertEqual(selected["prior_id"], 5)

    def test_suspicious_prior_selection_prefers_pose_prior_drift_when_contacts_are_clean(self):
        metrics = [
            {
                "prior_id": 1,
                "n_successes": 1,
                "illegal_contact_resets": 0,
                "self_collision_resets": 0,
                "mean_joint_prior_error_rad": 0.12,
                "mean_right_arm_joint_prior_error_rad": 0.08,
                "mean_root_orientation_error_deg": 4.0,
                "min_nonbrush_clearance_m": 0.03,
            },
            {
                "prior_id": 8,
                "n_successes": 1,
                "illegal_contact_resets": 0,
                "self_collision_resets": 0,
                "mean_joint_prior_error_rad": 0.72,
                "mean_right_arm_joint_prior_error_rad": 0.58,
                "mean_root_orientation_error_deg": 23.0,
                "min_nonbrush_clearance_m": 0.15,
            },
        ]

        selected = wall_brush_success.select_suspicious_prior_for_visual_review(metrics)

        self.assertEqual(selected["prior_id"], 8)

    def test_visual_review_entries_target_suspicious_prior_with_side_and_oblique_views(self):
        entries = wall_brush_success.build_wall_brush_visual_review_entries(
            "/root/autodl-tmp/IsaacLab/logs/rsl_rl/g1/run/model_149.pt",
            prior_id=17,
            view_prefix="eval_model_149",
            video_length=96,
        )

        self.assertEqual([entry["view"] for entry in entries], ["side", "oblique"])
        self.assertEqual([entry["prior_id"] for entry in entries], [17, 17])
        self.assertTrue(entries[0]["view_name"].endswith("_prior17_side_wall_marker"))
        self.assertTrue(entries[1]["view_name"].endswith("_prior17_oblique_wall_marker"))
        self.assertEqual(entries[0]["video_length"], 96)
        self.assertEqual(
            entries[0]["video_path"],
            "/root/autodl-tmp/IsaacLab/logs/rsl_rl/g1/run/videos/eval_model_149_prior17_side_wall_marker/rl-video-step-0.mp4",
        )

    def test_anchor_hits_must_be_legal_and_ordered(self):
        thresholds = wall_brush_success.WallBrushSuccessThresholds(
            contact_ratio=0.60,
            row_ratio=0.60,
            combined_ratio=0.50,
            coverage=0.70,
            anchor_radius=0.05,
        )
        state = wall_brush_success.WallBrushEpisodeSuccess.start()

        for sample in (
            _sample(0.50),  # middle before start should not count
            _sample(0.00, legal=False),  # start without legal wall contact should not count
            _sample(1.00),  # end before ordered start/middle should not count
            _sample(0.25),
            _sample(0.75),
        ):
            state, _ = wall_brush_success.update_wall_brush_episode_success(state, sample, thresholds)

        self.assertEqual(state.next_anchor_index, 0)
        self.assertFalse(state.pending_success)
        self.assertFalse(state.countable_success)


if __name__ == "__main__":
    unittest.main()
