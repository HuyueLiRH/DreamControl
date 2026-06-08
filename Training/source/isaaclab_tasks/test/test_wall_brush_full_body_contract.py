import re
import tempfile
import unittest
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "scripts").exists() and (candidate / "rl_references").exists():
            return candidate
        if (
            (candidate / "scripts" / "reinforcement_learning" / "rsl_rl").exists()
            and (candidate / "source" / "isaaclab_tasks").exists()
        ):
            return candidate
        if (candidate / "Training").exists() and (candidate / "TrajGen").exists():
            return candidate
    return start


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


LOCAL_REPO_ROOT = _find_repo_root(ARTIFACT_ROOT)
LOCAL_SCRIPTS_ROOT = _first_existing(LOCAL_REPO_ROOT / "Training" / "scripts", LOCAL_REPO_ROOT / "scripts")

TASK_SOURCE = _first_existing(
    SOURCE_ROOT
    / "isaaclab_tasks"
    / "manager_based"
    / "interactive_motion_tracking"
    / "g1"
    / "motion_tracking_wall_brush_env.py",
    ARTIFACT_ROOT / "motion_tracking_wall_brush_env.py",
)
INIT_SOURCE = _first_existing(
    SOURCE_ROOT / "isaaclab_tasks" / "manager_based" / "interactive_motion_tracking" / "g1" / "__init__.py",
    ARTIFACT_ROOT / "g1_init.py",
)
AGENT_SOURCE = _first_existing(INIT_SOURCE.parent / "agents" / "rsl_rl_ppo_cfg.py", ARTIFACT_ROOT / "rsl_rl_ppo_cfg.py")
TRAIN_WRAPPER = _first_existing(
    Path("/root/autodl-tmp/remote_dreamcontrol_wall_brush_staged_train.sh"),
    LOCAL_SCRIPTS_ROOT / "remote_dreamcontrol_wall_brush_staged_train.sh",
    ARTIFACT_ROOT / "remote_dreamcontrol_wall_brush_staged_train.sh",
)
EVAL_WRAPPER = _first_existing(
    Path("/root/autodl-tmp/remote_dreamcontrol_wall_brush_staged_eval.sh"),
    LOCAL_SCRIPTS_ROOT / "remote_dreamcontrol_wall_brush_staged_eval.sh",
    ARTIFACT_ROOT / "remote_dreamcontrol_wall_brush_staged_eval.sh",
)
BUTTONPRESS_TRAIN_WRAPPER = _first_existing(
    Path("/root/autodl-tmp/remote_wall_brush_buttonpress_aligned_train.sh"),
    LOCAL_SCRIPTS_ROOT / "remote_wall_brush_buttonpress_aligned_train.sh",
    ARTIFACT_ROOT / "remote_wall_brush_buttonpress_aligned_train.sh",
)
BUTTONPRESS_EVAL_WRAPPER = _first_existing(
    Path("/root/autodl-tmp/remote_wall_brush_buttonpress_aligned_eval.sh"),
    LOCAL_SCRIPTS_ROOT / "remote_wall_brush_buttonpress_aligned_eval.sh",
    ARTIFACT_ROOT / "remote_wall_brush_buttonpress_aligned_eval.sh",
)
BUTTONPRESS_SMOKE_WRAPPER = _first_existing(
    Path("/root/autodl-tmp/remote_wall_brush_buttonpress_aligned_smoke.sh"),
    LOCAL_SCRIPTS_ROOT / "remote_wall_brush_buttonpress_aligned_smoke.sh",
    ARTIFACT_ROOT / "remote_wall_brush_buttonpress_aligned_smoke.sh",
)
EVAL_SCRIPT = _first_existing(
    Path("/root/autodl-tmp/IsaacLab/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py"),
    LOCAL_SCRIPTS_ROOT / "reinforcement_learning" / "rsl_rl" / "eval_wall_brush_policy.py",
    ARTIFACT_ROOT / "eval_wall_brush_policy.py",
)
PLAY_SCRIPT = _first_existing(
    Path("/root/autodl-tmp/IsaacLab/scripts/reinforcement_learning/rsl_rl/play_wall_brush_fixed_view.py"),
    LOCAL_SCRIPTS_ROOT / "reinforcement_learning" / "rsl_rl" / "play_wall_brush_fixed_view.py",
    ARTIFACT_ROOT / "play_wall_brush_fixed_view.py",
)
CLASS_NAME = "G1WallBrushNoWallCollisionDreamControlEnvCfg"
TASK_ID = "Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-v0"
WARMSTART_TASK_ID = "Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-Warmstart-v0"
BALANCE_WARMSTART_TASK_ID = "Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-BalanceWarmstart-v0"
BUTTONPRESS_BALANCE_TASK_ID = (
    "Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressBalance-v0"
)
BUTTONPRESS_ALIGNED_TASK_ID = (
    "Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAligned-v0"
)
BUTTONPRESS_ALIGNED_ANTIJITTER_TASK_ID = (
    "Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-v0"
)
BUTTONPRESS_ANTIJITTER_TRAIN_WRAPPER = _first_existing(
    LOCAL_SCRIPTS_ROOT / "remote_wall_brush_buttonpress_aligned_antijitter_train.sh",
    ARTIFACT_ROOT / "remote_wall_brush_buttonpress_aligned_antijitter_train.sh",
)
BUTTONPRESS_ANTIJITTER_EVAL_WRAPPER = _first_existing(
    LOCAL_SCRIPTS_ROOT / "remote_wall_brush_buttonpress_aligned_antijitter_eval.sh",
    ARTIFACT_ROOT / "remote_wall_brush_buttonpress_aligned_antijitter_eval.sh",
)
BUTTONPRESS_ANTIJITTER_SWEEP_WRAPPER = _first_existing(
    LOCAL_SCRIPTS_ROOT / "remote_wall_brush_antijitter_sweep.sh",
    ARTIFACT_ROOT / "remote_wall_brush_antijitter_sweep.sh",
)
STANDSTILL_ROW_CONTACT_TASK_ID = (
    "Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-StandStillRowContact-v0"
)
AGILE_BASE_CLASS_NAME = "G1WallBrushNoWallCollisionDreamControlAgileBaseEnvCfg"
AGILE_BASE_TASK_ID = "Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-AgileBase-v0"
FORBIDDEN_TASK_ID = "Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-StandPrepIK-UpperBodyResidualAnchor-v0"
STAGED250_REF = (
    "../TrajGen/sample/Wall_Brush_27/"
    "wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"
)
AGILE_TEACHER_POLICY = (
    "/root/autodl-tmp/WBC-AGILE/agile/data/policy/velocity_height_g1/"
    "unitree_g1_velocity_height_teacher.pt"
)
FINGERTIP_BRUSH_LINK = "right_hand_index_1_link"
WRIST_BRUSH_LINK = "right_wrist_yaw_link"


def _source() -> str:
    return TASK_SOURCE.read_text(encoding="utf-8")


def _class_block(source: str, name: str) -> str:
    match = re.search(rf"^class {name}(?:\([^)]+\))?:\n", source, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"{name} is not defined")
    next_match = re.search(r"^class \w+(?:\([^)]+\))?:\n", source[match.end() :], flags=re.MULTILINE)
    end = match.end() + next_match.start() if next_match else len(source)
    return source[match.start() : end]


class WallBrushFullBodyContractTest(unittest.TestCase):
    def test_find_repo_root_recognizes_remote_isaaclab_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "IsaacLab"
            test_dir = root / "source" / "isaaclab_tasks" / "test"
            test_dir.mkdir(parents=True)
            (root / "scripts" / "reinforcement_learning" / "rsl_rl").mkdir(parents=True)

            self.assertEqual(_find_repo_root(test_dir), root)

    def test_full_body_task_class_is_unlocked_27dof_and_uses_staged250_prior(self):
        block = _class_block(_source(), CLASS_NAME)
        normalized_block = block.replace('"\n            "', "")

        self.assertIn(f"class {CLASS_NAME}(G1WallBrushNoWallCollisionEnvCfg):", block)
        self.assertIn("actions: ActionsCfg = ActionsCfg()", block)
        self.assertIn(STAGED250_REF, normalized_block)
        self.assertIn("G1_MINIMAL_CFG.replace", block)
        self.assertIn("fix_root_link = False", block)
        self.assertNotIn("G1_MINIMAL_CFG_FIXED_BASE", block)
        self.assertNotIn("JointNamesOrder_UB", block)
        self.assertNotIn("WallBrushUpperBody", block)

    def test_full_body_task_uses_10s_50hz_virtual_wall_contract(self):
        block = _class_block(_source(), CLASS_NAME)

        self.assertIn("self.decimation = 4", block)
        self.assertIn("self.episode_length_s = 10.0", block)
        self.assertIn("self.sim.dt = 0.005", block)
        self.assertIn("G1WallBrushNoWallCollisionEnvCfg", block)
        self.assertIn("terminations: WallBrushDreamControlTerminationsCfg", block)

    def test_full_body_task_reset_contract_matches_button_press_style(self):
        source = _source()
        term_block = _class_block(source, "WallBrushDreamControlTerminationsCfg")
        event_block = _class_block(source, "WallBrushDreamControlEventCfg")
        task_block = _class_block(source, CLASS_NAME)

        self.assertIn("class WallBrushDreamControlTerminationsCfg(WallBrushNoWallCollisionTerminationsCfg):", term_block)
        self.assertIn('torso_below_threshold = DoneTerm(func=root_below_threshold, params={"thres": 0.3})', term_block)
        self.assertIn('torso_angle_below_threshold = DoneTerm(func=root_angle_below_threshold, params={"thres": 0.5})', term_block)
        self.assertIn("self_collision_proxy = None", term_block)
        self.assertIn("func=reset_root_state_for_motion", event_block)
        self.assertIn('"offset_z": 0.0', event_block)
        self.assertIn("events: WallBrushDreamControlEventCfg", task_block)
        self.assertIn("torso_below_threshold", source)
        self.assertIn("torso_angle_below_threshold", source)

    def test_full_body_task_is_registered_and_forbidden_task_remains_non_default(self):
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        self.assertIn(f'id="{TASK_ID}"', init_source)
        self.assertIn(
            f"motion_tracking_wall_brush_env:{CLASS_NAME}",
            init_source,
        )

    def test_staged_wrappers_default_to_full_body_task_without_unconditional_resume(self):
        train_source = TRAIN_WRAPPER.read_text(encoding="utf-8")
        eval_source = EVAL_WRAPPER.read_text(encoding="utf-8")

        self.assertIn(f'TASK="${{1:-{BUTTONPRESS_ALIGNED_TASK_ID}}}"', train_source)
        self.assertIn(f'TASK="${{1:-{BUTTONPRESS_ALIGNED_TASK_ID}}}"', eval_source)
        self.assertNotIn(f'TASK="${{1:-{TASK_ID}}}"', train_source)
        self.assertNotIn(f'TASK="${{1:-{TASK_ID}}}"', eval_source)
        self.assertNotIn(f'TASK="${{1:-{FORBIDDEN_TASK_ID}}}"', train_source)
        self.assertNotIn(f'TASK="${{1:-{FORBIDDEN_TASK_ID}}}"', eval_source)
        self.assertIn("RESUME_ARGS=()", train_source)
        self.assertIn('"${RESUME_ARGS[@]}"', train_source)
        self.assertNotIn("  --resume \\\n", train_source)
        self.assertNotIn("reference_time_offset", train_source)
        self.assertNotIn("reference_time_offset", eval_source)
        self.assertNotIn("env.actions.joint_pos.scale", train_source)
        self.assertNotIn("env.actions.joint_pos.scale", eval_source)

    def test_main_policy_observes_five_future_reference_snapshots(self):
        obs_block = _class_block(_source(), "ObservationsCfg")

        self.assertIn("target_ref_curr = ObsTerm(func=target_ref", obs_block)
        self.assertIn("target_ref_t_plus_0p1 = ObsTerm(func=target_ref", obs_block)
        self.assertIn('params={"time_offset": 0.1, "visualize_markers": VISUALIZE_MARKERS}', obs_block)
        self.assertIn("target_ref_t_plus_0p2 = ObsTerm(func=target_ref", obs_block)
        self.assertIn('params={"time_offset": 0.2, "visualize_markers": VISUALIZE_MARKERS}', obs_block)
        self.assertIn("target_ref_t_plus_0p3 = ObsTerm(func=target_ref", obs_block)
        self.assertIn('params={"time_offset": 0.3, "visualize_markers": VISUALIZE_MARKERS}', obs_block)
        self.assertIn("target_ref_t_plus_0p4 = ObsTerm(func=target_ref", obs_block)
        self.assertIn('params={"time_offset": 0.4, "visualize_markers": VISUALIZE_MARKERS}', obs_block)
        self.assertNotIn("target_ref_next =", obs_block)
        self.assertNotIn("target_ref_next_next =", obs_block)

    def test_buttonpress_aligned_dedicated_wrappers_exist_and_force_main_contract(self):
        for wrapper in (BUTTONPRESS_TRAIN_WRAPPER, BUTTONPRESS_EVAL_WRAPPER, BUTTONPRESS_SMOKE_WRAPPER):
            self.assertTrue(wrapper.exists(), f"{wrapper} is missing")
            source = wrapper.read_text(encoding="utf-8")
            self.assertIn(BUTTONPRESS_ALIGNED_TASK_ID, source)
            self.assertIn(STAGED250_REF, source)
            self.assertIn("env.episode_length_s=10.0", source)
            self.assertNotIn(FORBIDDEN_TASK_ID, source)
            self.assertNotIn("AgileBase", source)
            self.assertNotIn("UpperBody", source)

        eval_source = BUTTONPRESS_EVAL_WRAPPER.read_text(encoding="utf-8")
        self.assertRegex(eval_source, r'NUM_STEPS="\$\{\d+:-500\}"')
        self.assertRegex(eval_source, r'VIDEO_LENGTH="\$\{\d+:-500\}"')
        self.assertIn('--num_steps "$NUM_STEPS"', eval_source)
        self.assertIn('--video_length "$VIDEO_LENGTH"', eval_source)

    def test_eval_reports_fall_reset_causes_and_first_active_step(self):
        eval_source = EVAL_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("torso_below_termination_count", eval_source)
        self.assertIn("torso_angle_termination_count", eval_source)
        self.assertIn("first_active_step", eval_source)
        self.assertIn("min_root_z_m", eval_source)
        self.assertIn("min_root_cos_z", eval_source)
        self.assertIn("--trace_prior_ids", eval_source)
        self.assertIn("trace_rows", eval_source)
        self.assertIn("mean_active_action_delta_l2", eval_source)
        self.assertIn("mean_active_right_arm_action_delta_l2", eval_source)
        self.assertIn("mean_brush_tip_accel_mps2", eval_source)

    def test_eval_and_video_support_reference_action_mode(self):
        eval_source = EVAL_SCRIPT.read_text(encoding="utf-8")
        play_source = PLAY_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--reference_actions", eval_source)
        self.assertIn("action_mode_reference", eval_source)
        self.assertIn("--action_smoothing_alpha", eval_source)
        self.assertIn("smooth_actions", eval_source)
        self.assertIn("--reference_actions", play_source)
        self.assertIn("action_mode_reference", play_source)
        self.assertIn("--action_smoothing_alpha", play_source)
        self.assertIn("smooth_actions", play_source)

    def test_warmstart_task_stays_full_body_unlocked_and_softens_early_collision_penalty(self):
        source = _source()
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        warm_block = _class_block(source, "G1WallBrushNoWallCollisionDreamControlWarmstartEnvCfg")
        rewards_block = _class_block(source, "G1WallBrushDreamControlWarmstartRewards")

        self.assertIn("actions: ActionsCfg = ActionsCfg()", warm_block)
        self.assertIn("G1WallBrushDreamControlWarmstartRewards", warm_block)
        self.assertIn("fix_root_link = False", warm_block)
        self.assertIn("weight=-80.0", rewards_block)
        self.assertIn("brush_tip_contact = None", rewards_block)
        self.assertIn("brush_tip_contact_band = None", rewards_block)
        self.assertNotIn("G1_MINIMAL_CFG_FIXED_BASE", warm_block)
        self.assertNotIn("WallBrushUpperBody", warm_block)
        self.assertIn(f'id="{WARMSTART_TASK_ID}"', init_source)
        self.assertIn(
            "motion_tracking_wall_brush_env:G1WallBrushNoWallCollisionDreamControlWarmstartEnvCfg",
            init_source,
        )

    def test_warmstart_uses_low_initial_action_noise_runner(self):
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        agent_source = AGENT_SOURCE.read_text(encoding="utf-8")
        runner_block = _class_block(agent_source, "G1WallBrushWarmstartPPORunnerCfg")

        self.assertIn("self.policy.init_noise_std = 0.08", runner_block)
        self.assertIn("self.algorithm.entropy_coef = 0.0001", runner_block)
        self.assertIn("G1WallBrushWarmstartPPORunnerCfg", init_source)

    def test_balance_warmstart_is_full_body_short_horizon_standing_curriculum(self):
        source = _source()
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        balance_block = _class_block(source, "G1WallBrushNoWallCollisionDreamControlBalanceWarmstartEnvCfg")
        rewards_block = _class_block(source, "G1WallBrushDreamControlBalanceWarmstartRewards")

        self.assertIn("actions: ActionsCfg = ActionsCfg()", balance_block)
        self.assertIn("G1WallBrushDreamControlBalanceWarmstartRewards", balance_block)
        self.assertIn("fix_root_link = False", balance_block)
        self.assertIn("self.episode_length_s = 4.0", balance_block)
        self.assertIn("brush_tip_contact = None", rewards_block)
        self.assertIn("brush_tip_reference = None", rewards_block)
        self.assertIn("brush_tip_progress = None", rewards_block)
        self.assertIn("right_arm_action_reference = None", rewards_block)
        self.assertIn("lower_body_action_reference = None", rewards_block)
        self.assertNotIn("G1_MINIMAL_CFG_FIXED_BASE", balance_block)
        self.assertIn(f'id="{BALANCE_WARMSTART_TASK_ID}"', init_source)
        self.assertIn("G1WallBrushWarmstartPPORunnerCfg", init_source)

    def test_buttonpress_aligned_curricula_use_official_runner_and_light_rewards(self):
        source = _source()
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        aligned_block = _class_block(source, "G1WallBrushNoWallCollisionDreamControlButtonPressAlignedEnvCfg")
        balance_block = _class_block(source, "G1WallBrushNoWallCollisionDreamControlButtonPressBalanceEnvCfg")
        aligned_rewards = _class_block(source, "G1WallBrushDreamControlButtonPressAlignedRewards")
        balance_rewards = _class_block(source, "G1WallBrushDreamControlButtonPressBalanceRewards")

        self.assertIn("actions: ActionsCfg = ActionsCfg()", aligned_block)
        self.assertIn("actions: ActionsCfg = ActionsCfg()", balance_block)
        self.assertIn("fix_root_link = False", aligned_block)
        self.assertIn("fix_root_link = False", balance_block)
        self.assertIn("self.episode_length_s = 10.0", aligned_block)
        self.assertIn("self.episode_length_s = 4.0", balance_block)
        self.assertNotIn("G1_MINIMAL_CFG_FIXED_BASE", aligned_block)
        self.assertNotIn("G1_MINIMAL_CFG_FIXED_BASE", balance_block)

        self.assertIn("termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)", aligned_rewards)
        self.assertIn("alive_reward = RewTerm(func=mdp.is_alive, weight=1.0)", aligned_rewards)
        self.assertIn("joint_deviation_ref", aligned_rewards)
        self.assertIn("weight=-0.2", aligned_rewards)
        self.assertIn("keypts_deviation_ref", aligned_rewards)
        self.assertIn("weight=-0.05", aligned_rewards)
        self.assertIn("position_tracking_error", aligned_rewards)
        self.assertIn("orientation_tracking_error", aligned_rewards)
        self.assertIn("brush_tip_reference", aligned_rewards)
        self.assertIn("brush_tip_contact", aligned_rewards)
        self.assertNotIn("brush_tip_anchor_approach", aligned_rewards)
        self.assertNotIn("brush_tip_late_progress", aligned_rewards)
        self.assertIn("brush_tip_anchor_milestones", aligned_rewards)
        self.assertIn("root_upright", aligned_rewards)
        self.assertIn("root_height_floor", aligned_rewards)
        self.assertIn("root_upright", balance_rewards)
        self.assertIn("root_height_floor", balance_rewards)
        self.assertIn("joint_deviation_ref = None", balance_rewards)
        self.assertIn("brush_tip_reference = None", balance_rewards)

        self.assertIn(f'id="{BUTTONPRESS_BALANCE_TASK_ID}"', init_source)
        self.assertIn(f'id="{BUTTONPRESS_ALIGNED_TASK_ID}"', init_source)
        self.assertIn(
            "motion_tracking_wall_brush_env:G1WallBrushNoWallCollisionDreamControlButtonPressBalanceEnvCfg",
            init_source,
        )
        self.assertIn(
            "motion_tracking_wall_brush_env:G1WallBrushNoWallCollisionDreamControlButtonPressAlignedEnvCfg",
            init_source,
        )
        self.assertIn("rsl_rl_ppo_cfg:G1FlatPPORunnerCfg", init_source)

    def test_brush_proxy_is_fingertip_link_without_legacy_brush_offset(self):
        source = _source()
        eval_source = EVAL_SCRIPT.read_text(encoding="utf-8")
        play_source = PLAY_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(f'BRUSH_LINK = "{FINGERTIP_BRUSH_LINK}"', source)
        self.assertNotIn(f'BRUSH_LINK = "{WRIST_BRUSH_LINK}"', source)
        self.assertIn(f'default="{FINGERTIP_BRUSH_LINK}"', eval_source)
        self.assertIn(f'BRUSH_LINK = "{FINGERTIP_BRUSH_LINK}"', play_source)
        self.assertNotIn(f'default="{WRIST_BRUSH_LINK}"', eval_source)
        self.assertNotIn(f'BRUSH_LINK = "{WRIST_BRUSH_LINK}"', play_source)

        brush_tip_fn = re.search(
            r"def _brush_tip_pos\(env, asset_cfg: SceneEntityCfg, motion_res: dict\) -> torch.Tensor:\n(?P<body>.*?)(?=\n\n)",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(brush_tip_fn)
        self.assertIn("return link_pos", brush_tip_fn.group("body"))
        self.assertNotIn("motion_res[\"brush_offset\"]", brush_tip_fn.group("body"))
        self.assertNotIn("+ motion_res[\"brush_offset\"]", eval_source)
        self.assertNotIn("+ motion_res[\"brush_offset\"]", play_source)

    def test_standstill_row_contact_curriculum_keeps_warmstart_stability_and_adds_contact(self):
        source = _source()
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        env_block = _class_block(source, "G1WallBrushNoWallCollisionDreamControlStandStillRowContactEnvCfg")
        rewards_block = _class_block(source, "G1WallBrushDreamControlStandStillRowContactRewards")

        self.assertIn("actions: ActionsCfg = ActionsCfg()", env_block)
        self.assertIn("G1WallBrushDreamControlStandStillRowContactRewards", env_block)
        self.assertIn("fix_root_link = False", env_block)
        self.assertIn("self.episode_length_s = 10.0", env_block)
        self.assertNotIn("G1_MINIMAL_CFG_FIXED_BASE", env_block)

        self.assertIn("termination_penalty = RewTerm(func=mdp.is_terminated, weight=-8000.0)", rewards_block)
        self.assertIn("alive_reward = RewTerm(func=mdp.is_alive, weight=12.0)", rewards_block)
        self.assertIn("brush_tip_contact = RewTerm", rewards_block)
        self.assertIn("brush_tip_contact_band = RewTerm", rewards_block)
        self.assertIn("brush_tip_row = RewTerm", rewards_block)
        self.assertIn("brush_tip_progress = RewTerm", rewards_block)
        self.assertIn("stability_action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)", rewards_block)

        self.assertIn(f'id="{STANDSTILL_ROW_CONTACT_TASK_ID}"', init_source)
        self.assertIn(
            "motion_tracking_wall_brush_env:G1WallBrushNoWallCollisionDreamControlStandStillRowContactEnvCfg",
            init_source,
        )
        self.assertIn("G1WallBrushWarmstartPPORunnerCfg", init_source)

    def test_agile_base_task_is_root_unlocked_10s_and_registered(self):
        source = _source()
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        env_block = _class_block(source, AGILE_BASE_CLASS_NAME)

        self.assertIn(
            f"class {AGILE_BASE_CLASS_NAME}(G1WallBrushNoWallCollisionDreamControlEnvCfg):",
            env_block,
        )
        self.assertIn("actions: WallBrushAgileBaseActionsCfg = WallBrushAgileBaseActionsCfg()", env_block)
        self.assertIn("rewards: G1WallBrushDreamControlAgileBaseRewards", env_block)
        self.assertIn("G1_MINIMAL_CFG.replace", env_block)
        self.assertIn("fix_root_link = False", env_block)
        self.assertNotIn("G1_MINIMAL_CFG_FIXED_BASE", env_block)
        self.assertIn("self.decimation = 4", env_block)
        self.assertIn("self.episode_length_s = 10.0", env_block)
        self.assertIn("self.sim.dt = 0.005", env_block)
        self.assertIn(f'id="{AGILE_BASE_TASK_ID}"', init_source)
        self.assertIn(
            f"motion_tracking_wall_brush_env:{AGILE_BASE_CLASS_NAME}",
            init_source,
        )

    def test_buttonpress_aligned_antijitter_success_route_is_fixed(self):
        source = _source()
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        rewards_block = _class_block(source, "G1WallBrushDreamControlButtonPressAlignedAntiJitterRewards")
        env_block = _class_block(source, "G1WallBrushNoWallCollisionDreamControlButtonPressAlignedAntiJitterEnvCfg")

        self.assertIn(
            "class G1WallBrushDreamControlButtonPressAlignedAntiJitterRewards("
            "G1WallBrushDreamControlButtonPressAlignedRewards):",
            rewards_block,
        )
        self.assertIn("action_accel_l2 = RewTerm(func=action_accel_l2, weight=-0.006)", rewards_block)
        self.assertIn("brush_tip_smoothness = RewTerm", rewards_block)
        self.assertIn("self_collision_proxy = RewTerm", rewards_block)
        self.assertIn("nonbrush_wall_clearance = RewTerm", rewards_block)
        self.assertIn("RIGHT_HAND_NONBRUSH_LINKS = [", source)
        self.assertIn("for link in RIGHT_HAND_LINKS if link != BRUSH_LINK", source)
        self.assertIn("right_hand_nonbrush_wall_clearance = RewTerm", rewards_block)

        self.assertIn(
            "class G1WallBrushNoWallCollisionDreamControlButtonPressAlignedAntiJitterEnvCfg("
            "G1WallBrushNoWallCollisionDreamControlButtonPressAlignedEnvCfg",
            env_block,
        )
        self.assertIn("G1WallBrushDreamControlButtonPressAlignedAntiJitterRewards()", env_block)
        self.assertNotIn("G1_MINIMAL_CFG_FIXED_BASE", env_block)
        self.assertNotIn("fix_root_link = True", env_block)

        aligned_block = _class_block(source, "G1WallBrushNoWallCollisionDreamControlButtonPressAlignedEnvCfg")
        self.assertIn("actions: ActionsCfg = ActionsCfg()", aligned_block)
        self.assertIn("fix_root_link = False", aligned_block)
        self.assertIn("self.decimation = 4", aligned_block)
        self.assertIn("self.episode_length_s = 10.0", aligned_block)
        self.assertIn("self.sim.dt = 0.005", aligned_block)

        self.assertIn(f'id="{BUTTONPRESS_ALIGNED_ANTIJITTER_TASK_ID}"', init_source)
        self.assertIn(
            "motion_tracking_wall_brush_env:G1WallBrushNoWallCollisionDreamControlButtonPressAlignedAntiJitterEnvCfg",
            init_source,
        )
        self.assertNotIn("ButtonPressAlignedBodyGroupAntiJitter", source + init_source)

    def test_buttonpress_aligned_antijitter_wrappers_use_official_contract(self):
        train_source = BUTTONPRESS_ANTIJITTER_TRAIN_WRAPPER.read_text(encoding="utf-8")
        eval_source = BUTTONPRESS_ANTIJITTER_EVAL_WRAPPER.read_text(encoding="utf-8")
        sweep_source = BUTTONPRESS_ANTIJITTER_SWEEP_WRAPPER.read_text(encoding="utf-8")

        for script_source in (train_source, eval_source):
            self.assertIn(BUTTONPRESS_ALIGNED_ANTIJITTER_TASK_ID, script_source)
            self.assertIn(STAGED250_REF, script_source)
            self.assertIn("env.episode_length_s=10.0", script_source)
            self.assertIn("/root/autodl-tmp/envs/isaaclab", script_source)
            self.assertIn("/root/autodl-tmp/IsaacLab", script_source)
            self.assertNotIn("FIXED_BASE", script_source)
            self.assertNotIn("UpperBody", script_source)
            self.assertNotIn("BodyGroupAntiJitter", script_source)

        self.assertIn('MAX_ITERATIONS="${2:-300}"', train_source)
        self.assertIn('RESUME_ACTION_STD="${6:-0.0015}"', train_source)
        self.assertIn('NUM_ENVS="${2:-27}"', eval_source)
        self.assertIn('NUM_STEPS="${3:-500}"', eval_source)
        self.assertIn('VIDEO_LENGTH="${6:-500}"', eval_source)
        self.assertIn("remote_wall_brush_buttonpress_aligned_antijitter_eval.sh", sweep_source)

    def test_agile_base_action_freezes_lower_body_and_trains_right_arm_residual(self):
        source = _source()
        action_block = _class_block(source, "WallBrushAgileLowerBodyAction")
        action_cfg_block = _class_block(source, "WallBrushAgileLowerBodyActionCfg")
        actions_cfg_block = _class_block(source, "WallBrushAgileBaseActionsCfg")
        rewards_block = _class_block(source, "G1WallBrushDreamControlAgileBaseRewards")

        self.assertIn("class WallBrushAgileLowerBodyAction(ActionTerm):", action_block)
        self.assertIn("return 0", action_block)
        self.assertIn("fixed_command", action_block)
        self.assertIn("torch.jit.load", action_block)
        self.assertIn("_AGILE_TEACHER_INPUT_DIM = 83", action_block)
        self.assertIn("_AGILE_TEACHER_OUTPUT_DIM = 12", action_block)
        self.assertIn("set_joint_position_target", action_block)

        self.assertIn("class_type: type[ActionTerm] = WallBrushAgileLowerBodyAction", action_cfg_block)
        self.assertIn(AGILE_TEACHER_POLICY, action_cfg_block)
        self.assertIn("fixed_command: tuple[float, float, float, float] = (0.0, 0.05, 0.0, 0.70)", action_cfg_block)
        self.assertIn("joint_names: list[str] = AGILE_LOWER_BODY_JOINTS", action_cfg_block)

        self.assertIn("joint_pos = WallBrushMotionResidualJointPositionActionCfg", actions_cfg_block)
        self.assertIn("joint_names=[", actions_cfg_block)
        self.assertIn('"waist_yaw_joint"', actions_cfg_block)
        self.assertIn('"right_wrist_yaw_joint"', actions_cfg_block)
        self.assertIn("agile_lower_body = WallBrushAgileLowerBodyActionCfg", actions_cfg_block)

        self.assertIn("class G1WallBrushDreamControlAgileBaseRewards(G1WallBrushResidualAnchorRegularizedRewards):", rewards_block)
        self.assertIn("lower_body_action_reference = None", rewards_block)
        self.assertIn("right_arm_action_reference = None", rewards_block)


if __name__ == "__main__":
    unittest.main()
