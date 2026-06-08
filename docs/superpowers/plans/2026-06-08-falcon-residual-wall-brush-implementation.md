# FALCON Residual Wall-Brush Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new DreamControl/IsaacLab `FalconResidual8D` wall-brushing task that composes an existing AGILE lower-body teacher with an 8D waist-yaw/right-arm residual policy and validates it through local contracts plus remote zero-residual and short-train smoke gates.

**Architecture:** Keep the existing successful AntiJitter task unchanged. Add a new task ID, action/reward/env config, wrappers, and tests in the current DreamControl task package. Use virtual wall force shaping only; no physical wall collision and no long training.

**Tech Stack:** Python `unittest`, IsaacLab manager-based task configs, RSL-RL train/eval scripts, AutoDL helper `scripts/autodl_remote.py`.

---

### Task 0: Dirty Worktree Separation

**Files:**
- Inspect: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py`
- Inspect: `Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py`
- Inspect: `docs/superpowers/plans/2026-06-08-wall-brush-paper-guided-smoke-gate.md`

- [ ] **Step 1: Confirm current branch and diff scope**

Run:

```bash
git status --short --branch
git diff -- Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py
```

Expected: branch is `experiment/wall-brush-paper-guided-task`. Any existing diff is limited to the prior smoke-gate fix: `RIGHT_HAND_NONBRUSH_LINKS`, remote IsaacLab test-root detection, and the AntiJitter import contract assertion.

- [ ] **Step 2: Commit the existing smoke-gate fix separately**

Run:

```bash
git add Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py docs/superpowers/plans/2026-06-08-wall-brush-paper-guided-smoke-gate.md
git commit -m "test: add wall brush smoke gate fixes"
```

Expected: one commit containing only the previous smoke-gate fix and smoke-gate plan. Do not include the FALCON implementation plan or any new FALCON task changes in this commit.

- [ ] **Step 3: Verify clean feature starting point**

Run:

```bash
git status --short --branch
```

Expected: no modified task/test files remain except this implementation plan if it is intentionally uncommitted.

### Task 1: Virtual Force Formula

**Files:**
- Modify: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py`
- Test: `Training/source/isaaclab_tasks/test/test_wall_brush_success.py`

- [ ] **Step 1: Write failing tests for deterministic virtual force math**

Append these tests to `WallBrushSuccessTest` in `Training/source/isaaclab_tasks/test/test_wall_brush_success.py`:

```python
    def test_virtual_wall_normal_force_uses_proximity_band_without_physical_contact(self):
        self.assertAlmostEqual(
            wall_brush_success.virtual_wall_normal_force_n(
                wall_x_error_m=0.20,
                contact_band_m=0.04,
                max_force_n=4.0,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            wall_brush_success.virtual_wall_normal_force_n(
                wall_x_error_m=0.02,
                contact_band_m=0.04,
                max_force_n=4.0,
            ),
            2.0,
        )
        self.assertAlmostEqual(
            wall_brush_success.virtual_wall_normal_force_n(
                wall_x_error_m=0.0,
                contact_band_m=0.04,
                max_force_n=4.0,
            ),
            4.0,
        )

    def test_virtual_force_band_violation_is_zero_inside_target_band(self):
        self.assertAlmostEqual(
            wall_brush_success.virtual_wall_force_band_violation(
                force_n=2.0,
                target_force_n=2.0,
                tolerance_n=0.5,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            wall_brush_success.virtual_wall_force_band_violation(
                force_n=3.0,
                target_force_n=2.0,
                tolerance_n=0.5,
            ),
            1.0,
        )
        self.assertAlmostEqual(
            wall_brush_success.virtual_wall_force_band_violation(
                force_n=0.0,
                target_force_n=2.0,
                tolerance_n=0.5,
            ),
            9.0,
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_success.WallBrushSuccessTest.test_virtual_wall_normal_force_uses_proximity_band_without_physical_contact -v
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_success.WallBrushSuccessTest.test_virtual_force_band_violation_is_zero_inside_target_band -v
```

Expected: both fail with `AttributeError` because the helper functions do not exist.

- [ ] **Step 3: Add the pure helper functions**

Add these functions near `_ratio()` in `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py`:

```python
def virtual_wall_normal_force_n(
    wall_x_error_m: float,
    *,
    contact_band_m: float,
    max_force_n: float,
) -> float:
    contact_band_m = max(float(contact_band_m), 1e-6)
    max_force_n = max(float(max_force_n), 0.0)
    proximity = max(0.0, contact_band_m - abs(float(wall_x_error_m))) / contact_band_m
    return min(max_force_n, max_force_n * proximity)


def virtual_wall_force_band_violation(
    force_n: float,
    *,
    target_force_n: float,
    tolerance_n: float,
) -> float:
    tolerance_n = max(float(tolerance_n), 1e-6)
    error = max(0.0, abs(float(force_n) - float(target_force_n)) - tolerance_n)
    return (error / tolerance_n) ** 2
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run:

```bash
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_success.py -v
```

Expected: all wall-brush success tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py Training/source/isaaclab_tasks/test/test_wall_brush_success.py
git commit -m "test: add virtual wall force helpers"
```

### Task 2: FALCON Task Contract Tests

**Files:**
- Modify: `Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py`

- [ ] **Step 1: Add constants for the new task and wrappers**

Add these constants after `AGILE_BASE_TASK_ID`:

```python
FALCON_RESIDUAL8D_CLASS_NAME = "G1WallBrushNoWallCollisionDreamControlFalconResidual8DEnvCfg"
FALCON_RESIDUAL8D_REWARDS_CLASS_NAME = "G1WallBrushDreamControlFalconResidual8DRewards"
FALCON_RESIDUAL8D_TASK_ID = (
    "Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-FalconResidual8D-v0"
)
FALCON_RESIDUAL8D_TRAIN_WRAPPER = _first_existing(
    LOCAL_SCRIPTS_ROOT / "remote_wall_brush_falcon_residual8d_train.sh",
    ARTIFACT_ROOT / "remote_wall_brush_falcon_residual8d_train.sh",
)
FALCON_RESIDUAL8D_EVAL_WRAPPER = _first_existing(
    LOCAL_SCRIPTS_ROOT / "remote_wall_brush_falcon_residual8d_eval.sh",
    ARTIFACT_ROOT / "remote_wall_brush_falcon_residual8d_eval.sh",
)
FALCON_RESIDUAL8D_SMOKE_WRAPPER = _first_existing(
    LOCAL_SCRIPTS_ROOT / "remote_wall_brush_falcon_residual8d_smoke.sh",
    ARTIFACT_ROOT / "remote_wall_brush_falcon_residual8d_smoke.sh",
)
FALCON_RESIDUAL8D_JOINTS = [
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
```

- [ ] **Step 2: Add failing contract test for registration, action composition, rewards, and wrappers**

Append this test method to `WallBrushFullBodyContractTest`:

```python
    def test_falcon_residual8d_task_is_new_root_unlocked_agile_teacher_task(self):
        source = _source()
        init_source = INIT_SOURCE.read_text(encoding="utf-8")
        env_block = _class_block(source, FALCON_RESIDUAL8D_CLASS_NAME)
        rewards_block = _class_block(source, FALCON_RESIDUAL8D_REWARDS_CLASS_NAME)
        actions_block = _class_block(source, "WallBrushFalconResidual8DActionsCfg")

        self.assertIn(
            f"class {FALCON_RESIDUAL8D_CLASS_NAME}(G1WallBrushNoWallCollisionDreamControlEnvCfg):",
            env_block,
        )
        self.assertIn("actions: WallBrushFalconResidual8DActionsCfg = WallBrushFalconResidual8DActionsCfg()", env_block)
        self.assertIn("rewards: G1WallBrushDreamControlFalconResidual8DRewards", env_block)
        self.assertIn("G1_MINIMAL_CFG.replace", env_block)
        self.assertIn("fix_root_link = False", env_block)
        self.assertIn("self.decimation = 4", env_block)
        self.assertIn("self.episode_length_s = 10.0", env_block)
        self.assertIn("self.sim.dt = 0.005", env_block)
        self.assertNotIn("G1_MINIMAL_CFG_FIXED_BASE", env_block)
        self.assertNotIn("fix_root_link = True", env_block)

        self.assertIn("joint_pos = WallBrushMotionResidualJointPositionActionCfg", actions_block)
        self.assertIn("agile_lower_body = WallBrushAgileLowerBodyActionCfg", actions_block)
        for joint_name in FALCON_RESIDUAL8D_JOINTS:
            self.assertIn(f'"{joint_name}"', actions_block)
        self.assertNotIn('"waist_roll_joint"', actions_block)
        self.assertNotIn('"waist_pitch_joint"', actions_block)
        self.assertIn("scale=0.06", actions_block)
        self.assertIn("reference_mode=\"current\"", actions_block)
        self.assertIn(AGILE_TEACHER_POLICY, source)

        self.assertIn("virtual_wall_force_band = RewTerm", rewards_block)
        self.assertIn("brush_normal_alignment = RewTerm", rewards_block)
        self.assertIn("residual_action_l2 = RewTerm(func=mdp.action_l2, weight=-0.025)", rewards_block)
        self.assertIn("action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.012)", rewards_block)
        self.assertIn("action_accel_l2 = RewTerm(func=action_accel_l2, weight=-0.004)", rewards_block)
        self.assertIn("left_arm_wall_clearance = RewTerm", rewards_block)
        self.assertIn("self_collision_proxy = RewTerm", rewards_block)

        self.assertIn(f'id="{FALCON_RESIDUAL8D_TASK_ID}"', init_source)
        self.assertIn(f"motion_tracking_wall_brush_env:{FALCON_RESIDUAL8D_CLASS_NAME}", init_source)
        self.assertIn("rsl_rl_ppo_cfg:G1WallBrushPPORunnerCfg", init_source)
        self.assertNotIn("ButtonPressAlignedBodyGroupAntiJitter", source + init_source)
```

- [ ] **Step 3: Add failing contract test for wrappers and eval skip-checkpoint support**

Append this test method:

```python
    def test_falcon_residual8d_wrappers_use_existing_env_and_no_checkpoint_zero_residual(self):
        eval_script = EVAL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--skip_checkpoint", eval_script)
        self.assertIn('resume_path = "zero_residual_no_checkpoint"', eval_script)
        self.assertIn("args_cli.skip_checkpoint", eval_script)

        for wrapper in (
            FALCON_RESIDUAL8D_TRAIN_WRAPPER,
            FALCON_RESIDUAL8D_EVAL_WRAPPER,
            FALCON_RESIDUAL8D_SMOKE_WRAPPER,
        ):
            self.assertTrue(wrapper.exists(), f"{wrapper} is missing")
            source = wrapper.read_text(encoding="utf-8")
            self.assertIn(FALCON_RESIDUAL8D_TASK_ID, source)
            self.assertIn(STAGED250_REF, source)
            self.assertIn("/root/autodl-tmp/envs/isaaclab", source)
            self.assertIn("/root/autodl-tmp/IsaacLab", source)
            self.assertIn("env.episode_length_s=10.0", source)
            self.assertNotIn("ButtonPressAlignedAntiJitter", source)
            self.assertNotIn("BodyGroupAntiJitter", source)
            self.assertNotIn("FIXED_BASE", source)
            self.assertNotIn("UpperBody", source)

        eval_source = FALCON_RESIDUAL8D_EVAL_WRAPPER.read_text(encoding="utf-8")
        smoke_source = FALCON_RESIDUAL8D_SMOKE_WRAPPER.read_text(encoding="utf-8")
        train_source = FALCON_RESIDUAL8D_TRAIN_WRAPPER.read_text(encoding="utf-8")
        self.assertIn('ZERO_ACTIONS="${5:-1}"', eval_source)
        self.assertIn("--skip_checkpoint", eval_source)
        self.assertIn("--zero_actions", eval_source)
        self.assertIn("MAX_ITERATIONS=\"${2:-1}\"", smoke_source)
        self.assertIn("bash scripts/remote_wall_brush_falcon_residual8d_eval.sh", smoke_source)
        self.assertIn("MAX_ITERATIONS=\"${2:-5}\"", train_source)
        self.assertIn("RESUME_ARGS=()", train_source)
        self.assertNotIn("--resume", train_source)
```

- [ ] **Step 4: Run the tests and verify RED**

Run:

```bash
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_full_body_contract.WallBrushFullBodyContractTest.test_falcon_residual8d_task_is_new_root_unlocked_agile_teacher_task -v
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_full_body_contract.WallBrushFullBodyContractTest.test_falcon_residual8d_wrappers_use_existing_env_and_no_checkpoint_zero_residual -v
```

Expected: first test fails because `G1WallBrushNoWallCollisionDreamControlFalconResidual8DEnvCfg` is not defined. Second test fails because wrappers and `--skip_checkpoint` do not exist.

- [ ] **Step 5: Commit the failing tests**

Run:

```bash
git add Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py
git commit -m "test: specify falcon residual wall brush contract"
```

### Task 3: Implement Task Config, Rewards, Registration, And Eval Skip-Checkpoint

**Files:**
- Modify: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py`
- Modify: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py`
- Modify: `Training/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py`

- [ ] **Step 1: Add FALCON residual constants**

Add after `RIGHT_ARM_JOINTS_MASK`:

```python
FALCON_RESIDUAL8D_JOINTS = [
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
```

- [ ] **Step 2: Add torch reward helpers for virtual force and brush normal alignment**

Add these functions after `wall_contact_force_above_threshold()`:

```python
def virtual_wall_normal_force_n(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    contact_band_m: float = 0.04,
    max_force_n: float = 4.0,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    tip = _brush_tip_pos(env, asset_cfg, motion_res)
    wall_x = motion_res["wall_mid"][:, 0]
    wall_x_error = tip[:, 0] - wall_x
    contact_band_m = max(float(contact_band_m), 1e-6)
    max_force_n = max(float(max_force_n), 0.0)
    proximity = (contact_band_m - torch.abs(wall_x_error)).clamp_min(0.0) / contact_band_m
    return torch.clamp(max_force_n * proximity, max=max_force_n)


def virtual_wall_force_band_violation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    contact_band_m: float = 0.04,
    max_force_n: float = 4.0,
    target_force_n: float = 2.0,
    tolerance_n: float = 0.6,
    active_only: bool = True,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    force = virtual_wall_normal_force_n(env, asset_cfg, contact_band_m, max_force_n)
    tolerance_n = max(float(tolerance_n), 1e-6)
    error = (torch.abs(force - float(target_force_n)) - tolerance_n).clamp_min(0.0)
    penalty = torch.square(error / tolerance_n)
    if active_only:
        penalty = penalty * motion_res["stroke_active"].float()
    return penalty


def brush_normal_alignment_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
    local_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    active_only: bool = True,
) -> torch.Tensor:
    motion_res = _motion_state(env)
    if motion_res is None:
        return _zero_reward(env)
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    axis = torch.tensor(local_axis, device=env.device, dtype=quat.dtype).view(1, 3).repeat(env.scene.num_envs, 1)
    brush_normal = math_utils.quat_apply(quat, axis)
    wall_normal = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=quat.dtype).view(1, 3)
    alignment = torch.abs(torch.sum(brush_normal * wall_normal, dim=1)).clamp(0.0, 1.0)
    penalty = torch.square(1.0 - alignment)
    if active_only:
        penalty = penalty * motion_res["stroke_active"].float()
    return penalty
```

- [ ] **Step 3: Add FALCON action config**

Add after `WallBrushAgileBaseActionsCfg`:

```python
@configclass
class WallBrushFalconResidual8DActionsCfg(ActionsCfgBase):
    joint_pos = WallBrushMotionResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=FALCON_RESIDUAL8D_JOINTS,
        preserve_order=True,
        scale=0.06,
        use_default_offset=False,
        reference_mode="current",
    )
    agile_lower_body = WallBrushAgileLowerBodyActionCfg(asset_name="robot")
```

- [ ] **Step 4: Add FALCON reward config**

Add after `G1WallBrushDreamControlAgileBaseRewards`:

```python
@configclass
class G1WallBrushDreamControlFalconResidual8DRewards(G1WallBrushDreamControlButtonPressAlignedRewards):
    """FALCON-style dual-agent residual rewards with virtual wall-force shaping."""

    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.012)
    action_accel_l2 = RewTerm(func=action_accel_l2, weight=-0.004)
    residual_action_l2 = RewTerm(func=mdp.action_l2, weight=-0.025)
    brush_tip_contact = RewTerm(
        func=brush_tip_wall_contact,
        weight=2.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.12, "fine_std": 0.04},
    )
    brush_tip_contact_band = RewTerm(
        func=brush_tip_wall_band_violation,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "target_band": 0.04, "scale": 0.08},
    )
    brush_tip_row = RewTerm(
        func=brush_tip_row_accuracy,
        weight=1.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.24, "fine_std": 0.08},
    )
    brush_tip_progress = RewTerm(
        func=brush_tip_line_progress,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "broad_std": 0.30, "fine_std": 0.12},
    )
    virtual_wall_force_band = RewTerm(
        func=virtual_wall_force_band_violation,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]),
            "contact_band_m": 0.04,
            "max_force_n": 4.0,
            "target_force_n": 2.0,
            "tolerance_n": 0.6,
            "active_only": True,
        },
    )
    brush_normal_alignment = RewTerm(
        func=brush_normal_alignment_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[BRUSH_LINK]), "active_only": True},
    )
    left_arm_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-6.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_ARM_LINKS, preserve_order=True),
            "min_clearance": 0.08,
            "active_only": False,
        },
    )
    self_collision_proxy = RewTerm(
        func=self_collision_proxy_violation,
        weight=-45.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    nonbrush_wall_clearance = RewTerm(
        func=body_wall_clearance_violation,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=NONBRUSH_LINKS, preserve_order=True),
            "min_clearance": 0.08,
            "active_only": False,
        },
    )
```

- [ ] **Step 5: Add FALCON env config**

Add after `G1WallBrushNoWallCollisionDreamControlAgileBaseEnvCfg`:

```python
@configclass
class G1WallBrushNoWallCollisionDreamControlFalconResidual8DEnvCfg(G1WallBrushNoWallCollisionDreamControlEnvCfg):
    """Root-unlocked FALCON-style wall-brush task with AGILE legs and 8D upper residuals."""

    actions: WallBrushFalconResidual8DActionsCfg = WallBrushFalconResidual8DActionsCfg()
    rewards: G1WallBrushDreamControlFalconResidual8DRewards = G1WallBrushDreamControlFalconResidual8DRewards()
    terminations: WallBrushDreamControlTerminationsCfg = WallBrushDreamControlTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
```

- [ ] **Step 6: Register the task**

Add after the AgileBase registration in `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py`:

```python
gym.register(
    id="Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-FalconResidual8D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.motion_tracking_wall_brush_env:G1WallBrushNoWallCollisionDreamControlFalconResidual8DEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WallBrushPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
```

- [ ] **Step 7: Add skip-checkpoint support to eval**

Add this parser argument after `--checkpoint` is inherited from `cli_args.add_rsl_rl_args(parser)` usage is available in the script; place it before `cli_args.add_rsl_rl_args(parser)`:

```python
parser.add_argument(
    "--skip_checkpoint",
    action="store_true",
    help="Create the environment and evaluate zero/reference actions without resolving or loading a checkpoint.",
)
```

Change the checkpoint resolution block to:

```python
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.skip_checkpoint:
        resume_path = "zero_residual_no_checkpoint"
    elif args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise FileNotFoundError(f"No published pretrained checkpoint for {train_task_name}")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = log_root_path if args_cli.skip_checkpoint else os.path.dirname(resume_path)
```

Change the runner branch to:

```python
    if args_cli.zero_actions or args_cli.reference_actions or args_cli.skip_checkpoint:
        runner = None
        policy = None
    else:
        runner = _build_runner(env, agent_cfg)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
```

- [ ] **Step 8: Run targeted contract tests and py_compile**

Run:

```bash
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_success.py -v
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_full_body_contract.WallBrushFullBodyContractTest.test_falcon_residual8d_task_is_new_root_unlocked_agile_teacher_task -v
PYTHONPYCACHEPREFIX=/tmp/dreamcontrol_pycache python3 -m py_compile Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py Training/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py
```

Expected: the success tests, the targeted FALCON task contract, and py_compile pass. Full contract tests remain intentionally red until Task 4 creates the wrapper files.

- [ ] **Step 9: Commit**

Run:

```bash
git add Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py Training/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py
git commit -m "feat: add falcon residual wall brush task"
```

### Task 4: Add Remote Wrappers

**Files:**
- Create: `Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh`
- Create: `Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh`
- Create: `Training/scripts/remote_wall_brush_falcon_residual8d_train.sh`

- [ ] **Step 1: Create eval wrapper**

Create `Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

TASK="Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-FalconResidual8D-v0"
NUM_ENVS="${1:-27}"
NUM_STEPS="${2:-500}"
OUTPUT="${3:-/root/autodl-tmp/wall_brush_falcon_residual8d_zero_eval.json}"
VISUAL_REVIEW="${4:-0}"
ZERO_ACTIONS="${5:-1}"
VIDEO_LENGTH="${6:-500}"
ACTION_SMOOTHING_ALPHA="${7:-1.0}"

REF_MOTIONS_PATH="../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"

source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab
cd /root/autodl-tmp/IsaacLab

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/isaac_utils:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic"

EXTRA_EVAL_ARGS=(--skip_checkpoint)
if [[ "$ZERO_ACTIONS" != "0" ]]; then
  EXTRA_EVAL_ARGS+=(--zero_actions)
fi

TERM=xterm python scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py \
  --task="$TASK" \
  --headless \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --num_steps "$NUM_STEPS" \
  --output "$OUTPUT" \
  --action_smoothing_alpha "$ACTION_SMOOTHING_ALPHA" \
  "${EXTRA_EVAL_ARGS[@]}" \
  env.episode_length_s=10.0 \
  env.ref_motions_path="$REF_MOTIONS_PATH"

if [[ "$VISUAL_REVIEW" != "0" ]]; then
  echo "Visual review for skip-checkpoint FALCON zero-residual eval is deferred to the existing play script after smoke metrics pass." >&2
fi
```

- [ ] **Step 2: Create train wrapper**

Create `Training/scripts/remote_wall_brush_falcon_residual8d_train.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

TASK="Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-FalconResidual8D-v0"
NUM_ENVS="${1:-1024}"
MAX_ITERATIONS="${2:-5}"
RUN_NAME="${3:-falcon_residual8d_short_smoke}"
RESUME_ACTION_STD="${4:-0.003}"

REF_MOTIONS_PATH="../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"

source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab
cd /root/autodl-tmp/IsaacLab

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/isaac_utils:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic"

RESUME_ARGS=()

TERM=xterm python scripts/reinforcement_learning/rsl_rl/train.py \
  --task="$TASK" \
  --headless \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --resume_action_std "$RESUME_ACTION_STD" \
  "${RESUME_ARGS[@]}" \
  --run_name "$RUN_NAME" \
  env.episode_length_s=10.0 \
  env.ref_motions_path="$REF_MOTIONS_PATH"
```

- [ ] **Step 3: Create smoke wrapper**

Create `Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${1:-256}"
MAX_ITERATIONS="${2:-1}"
RUN_NAME="${3:-falcon_residual8d_smoke}"
ZERO_OUTPUT="${4:-/root/autodl-tmp/wall_brush_falcon_residual8d_zero_smoke.json}"

REF_MOTIONS_ABS="/root/autodl-tmp/TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"

source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab
cd /root/autodl-tmp/IsaacLab

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/isaac_utils:$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic"

if ps -eo pid,ppid,stat,etime,cmd | grep -E "[i]saac|[k]it|[t]rain.py|[p]lay.py|[e]val_wall_brush" >/tmp/wall_brush_active_jobs.txt; then
  cat /tmp/wall_brush_active_jobs.txt
  echo "Refusing to start FALCON smoke while another Isaac/Kit train/eval/video job is active." >&2
  exit 2
fi

python -m unittest source/isaaclab_tasks/test/test_wall_brush_success.py source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v

python - <<'PY'
from pathlib import Path
import numpy as np

path = Path("/root/autodl-tmp/TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz")
if not path.exists():
    raise SystemExit(f"missing staged250 prior: {path}")
data = np.load(path, allow_pickle=True)
motion_arrays = [
    value for value in (data[key] for key in data.files)
    if hasattr(value, "ndim") and value.ndim >= 2 and value.shape[:2] == (27, 250)
]
if not motion_arrays:
    raise SystemExit(f"no array with leading shape (27, 250) in {path}; keys={sorted(data.files)}")
print("staged250 prior ok: motions=27 frames=250")
PY

bash scripts/remote_wall_brush_falcon_residual8d_eval.sh 27 500 "$ZERO_OUTPUT" 0 1 500 1.0
bash scripts/remote_wall_brush_falcon_residual8d_train.sh "$NUM_ENVS" "$MAX_ITERATIONS" "$RUN_NAME" 0.003
```

- [ ] **Step 4: Run wrapper syntax checks and FALCON contract tests**

Run:

```bash
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_train.sh
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v
```

Expected: wrapper syntax checks exit 0 and full-body contract tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh Training/scripts/remote_wall_brush_falcon_residual8d_train.sh Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py
git commit -m "test: add falcon residual smoke wrappers"
```

### Task 5: Local Verification

**Files:**
- Verify: all modified task/test/script files

- [ ] **Step 1: Run all local wall-brush tests**

Run:

```bash
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_success.py Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run local static checks**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/dreamcontrol_pycache python3 -m py_compile \
  Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py \
  Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py \
  Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py \
  Training/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_train.sh
```

Expected: all commands exit 0.

- [ ] **Step 3: Confirm successful AntiJitter contract remains protected**

Run:

```bash
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_full_body_contract.WallBrushFullBodyContractTest.test_buttonpress_aligned_antijitter_success_route_is_fixed -v
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_full_body_contract.WallBrushFullBodyContractTest.test_buttonpress_aligned_antijitter_wrappers_use_official_contract -v
```

Expected: both tests pass.

### Task 6: Remote Sync And Smoke Eval

**Files:**
- Sync: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py`
- Sync: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py`
- Sync: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py`
- Sync: `Training/source/isaaclab_tasks/test/test_wall_brush_success.py`
- Sync: `Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py`
- Sync: `Training/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py`
- Sync: `Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh`
- Sync: `Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh`
- Sync: `Training/scripts/remote_wall_brush_falcon_residual8d_train.sh`

- [ ] **Step 1: Remote preflight**

Run from `/Users/huyue/Projects/codex_migration_required`:

```bash
python3 scripts/autodl_remote.py exec "test -d /root/autodl-tmp/envs/isaaclab"
python3 scripts/autodl_remote.py exec "test -d /root/autodl-tmp/IsaacLab"
python3 scripts/autodl_remote.py exec "test -f /root/autodl-tmp/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_teacher.pt"
python3 scripts/autodl_remote.py exec "test -f /root/autodl-tmp/TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz"
python3 scripts/autodl_remote.py exec "ps -eo pid,ppid,stat,etime,cmd | grep -E '[i]saac|[k]it|[t]rain.py|[p]lay.py|[e]val_wall_brush' || true"
```

Expected: paths exist and no active Isaac/Kit/train/eval process is listed.

- [ ] **Step 2: Back up remote files before syncing**

Run:

```bash
python3 scripts/autodl_remote.py exec 'backup="/root/autodl-tmp/wall_brush_falcon_residual8d_backup_$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$backup/g1" "$backup/rsl_rl" "$backup/test" "$backup/scripts"; cd /root/autodl-tmp/IsaacLab; cp -a source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py "$backup/g1/"; cp -a source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py "$backup/g1/"; cp -a source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py "$backup/g1/"; cp -a scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py "$backup/rsl_rl/"; cp -a source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py "$backup/test/"; cp -a source/isaaclab_tasks/test/test_wall_brush_success.py "$backup/test/"; printf "%s\n" "$backup"'
```

Expected: prints a backup directory under `/root/autodl-tmp`.

- [ ] **Step 3: Sync only required files**

Run one `put` command per file:

```bash
python3 scripts/autodl_remote.py put vendor/DreamControl-antijitter-publish/Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py /root/autodl-tmp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py
python3 scripts/autodl_remote.py put vendor/DreamControl-antijitter-publish/Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py /root/autodl-tmp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py
python3 scripts/autodl_remote.py put vendor/DreamControl-antijitter-publish/Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py /root/autodl-tmp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py
python3 scripts/autodl_remote.py put vendor/DreamControl-antijitter-publish/Training/source/isaaclab_tasks/test/test_wall_brush_success.py /root/autodl-tmp/IsaacLab/source/isaaclab_tasks/test/test_wall_brush_success.py
python3 scripts/autodl_remote.py put vendor/DreamControl-antijitter-publish/Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py /root/autodl-tmp/IsaacLab/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py
python3 scripts/autodl_remote.py put vendor/DreamControl-antijitter-publish/Training/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py /root/autodl-tmp/IsaacLab/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py
python3 scripts/autodl_remote.py put vendor/DreamControl-antijitter-publish/Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh /root/autodl-tmp/IsaacLab/scripts/remote_wall_brush_falcon_residual8d_eval.sh
python3 scripts/autodl_remote.py put vendor/DreamControl-antijitter-publish/Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh /root/autodl-tmp/IsaacLab/scripts/remote_wall_brush_falcon_residual8d_smoke.sh
python3 scripts/autodl_remote.py put vendor/DreamControl-antijitter-publish/Training/scripts/remote_wall_brush_falcon_residual8d_train.sh /root/autodl-tmp/IsaacLab/scripts/remote_wall_brush_falcon_residual8d_train.sh
```

Expected: all uploads exit 0.

- [ ] **Step 4: Remote static tests**

Run:

```bash
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/IsaacLab && source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab && PYTHONPYCACHEPREFIX=/tmp/dreamcontrol_pycache python -m py_compile source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py"
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/IsaacLab && source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab && python -m unittest source/isaaclab_tasks/test/test_wall_brush_success.py source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v"
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/IsaacLab && bash -n scripts/remote_wall_brush_falcon_residual8d_eval.sh && bash -n scripts/remote_wall_brush_falcon_residual8d_smoke.sh && bash -n scripts/remote_wall_brush_falcon_residual8d_train.sh"
```

Expected: py_compile, unittest, and bash syntax checks all pass.

- [ ] **Step 5: Run zero-residual 27-prior eval**

Run:

```bash
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/IsaacLab && bash scripts/remote_wall_brush_falcon_residual8d_eval.sh 27 500 /root/autodl-tmp/wall_brush_falcon_residual8d_zero_eval.json 0 1 500 1.0"
```

Expected: output JSON exists. Pass target is 27 evaluated priors, preferably 27/27 survival, `min_root_z > 0.70`, and `min_root_cos_z > 0.98`. Brush success is diagnostic.

- [ ] **Step 6: Parse zero-residual metrics**

Run:

```bash
python3 scripts/autodl_remote.py get /root/autodl-tmp/wall_brush_falcon_residual8d_zero_eval.json /private/tmp/wall_brush_falcon_residual8d_zero_eval.json
python3 -c 'import json; d=json.load(open("/private/tmp/wall_brush_falcon_residual8d_zero_eval.json")); s=d["summary"]; print({k:s.get(k) for k in ["evaluated_prior_count","survival_rate","dreamcontrol_style_success_count","min_root_z_m","min_root_cos_z","mean_brush_tip_jerk_mps3","mean_root_position_error_m","mean_root_orientation_error_deg","max_foot_slip_m","min_left_arm_clearance_m","min_self_collision_margin_m"]})'
```

Expected: command prints the smoke metrics. If root survival fails, stop and debug AGILE teacher reset/action composition before running train smoke.

### Task 7: Remote Short Train Smoke

**Files:**
- Use: `Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh`
- Use: `Training/scripts/remote_wall_brush_falcon_residual8d_train.sh`

- [ ] **Step 1: Run 1-iteration smoke only if zero-residual passed root gates**

Run:

```bash
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/IsaacLab && bash scripts/remote_wall_brush_falcon_residual8d_smoke.sh 256 1 falcon_residual8d_plan_smoke /root/autodl-tmp/wall_brush_falcon_residual8d_plan_zero_smoke.json"
```

Expected: wrapper runs local remote tests, prior check, zero-residual eval, and 1 training iteration. It writes a log/checkpoint directory under `/root/autodl-tmp/IsaacLab/logs/rsl_rl/g1`.

- [ ] **Step 2: Verify no remote Isaac jobs remain active**

Run:

```bash
python3 scripts/autodl_remote.py exec "ps -eo pid,ppid,stat,etime,cmd | grep -E '[i]saac|[k]it|[t]rain.py|[p]lay.py|[e]val_wall_brush' || true"
```

Expected: no active matching process after the smoke command exits.

- [ ] **Step 3: Commit smoke wrapper adjustments if needed**

If the remote smoke reveals only wrapper argument issues and those are fixed locally, run:

```bash
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_success.py Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_train.sh
git add Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh Training/scripts/remote_wall_brush_falcon_residual8d_train.sh
git commit -m "fix: stabilize falcon residual smoke wrappers"
```

Expected: use this step only for wrapper fixes discovered during smoke. If the failure is task/action/reward logic, return to systematic debugging before modifying wrappers.

### Task 8: Completion Verification

**Files:**
- Verify: all changed files

- [ ] **Step 1: Final local verification**

Run:

```bash
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_success.py Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v
PYTHONPYCACHEPREFIX=/tmp/dreamcontrol_pycache python3 -m py_compile \
  Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py \
  Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py \
  Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py \
  Training/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh
bash -n Training/scripts/remote_wall_brush_falcon_residual8d_train.sh
```

Expected: every command exits 0.

- [ ] **Step 2: Final remote verification**

Run:

```bash
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/IsaacLab && source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab && python -m unittest source/isaaclab_tasks/test/test_wall_brush_success.py source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v"
python3 scripts/autodl_remote.py exec "ps -eo pid,ppid,stat,etime,cmd | grep -E '[i]saac|[k]it|[t]rain.py|[p]lay.py|[e]val_wall_brush' || true"
```

Expected: remote tests pass and no Isaac/Kit/train/eval process remains active.

- [ ] **Step 3: Summarize evidence**

Report:

```text
Local tests: exact passing test count from unittest output
Remote tests: exact passing test count from unittest output
Zero-residual eval JSON: /root/autodl-tmp/wall_brush_falcon_residual8d_zero_eval.json
Zero-residual key metrics: evaluated priors, survival_rate, min_root_z_m, min_root_cos_z, brush jerk, root errors, foot slip, left-arm clearance, self-collision margin
Short smoke train: run name and checkpoint/log path
Long training: not started
```

Expected: the user can decide whether to authorize longer training based on the smoke metrics and videos, not on a hidden training run.
