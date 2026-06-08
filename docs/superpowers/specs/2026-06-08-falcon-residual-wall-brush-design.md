# FALCON-Style Residual Wall-Brush Design

## Goal

Add a new DreamControl/IsaacLab wall-brushing task for Unitree G1 that tests whether a FALCON-style dual-agent residual controller can reduce the visible jitter seen in the current full-body AntiJitter route without contaminating the fixed successful route.

The v0 task must use the current remote IsaacLab environment and the existing staged250 27-prior wall-brush bundle. It must not introduce a new Python environment, migrate the training stack to HumanoidVerse, use fixed-base/root-locked/upper-body-only final behavior, or start long training without an explicit user command.

## Confirmed Decisions

- Implementation route: add a new task inside the existing DreamControl IsaacLab task code.
- Lower-body teacher: use the existing AGILE lower-body teacher at `/root/autodl-tmp/WBC-AGILE/agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_teacher.pt`.
- Residual action size: 8D residual policy controlling `waist_yaw_joint` plus the 7 right-arm joints.
- Wall/force model: virtual wall with virtual force curriculum, not physical wall collision in v0.
- Initialization: do not resume the current 27D AntiJitter full-body checkpoint into the 8D policy.
- First verification: zero-residual/reference replay smoke, then a 1-5 iteration short training smoke only after the task pipeline is registered and testable.

## Reference Framing

FALCON motivates a dual-agent humanoid control pattern: a lower-body policy maintains locomotion or balance while an upper-body policy adapts end-effector behavior under force interaction. HumanoidVerse motivates keeping simulator, task, and algorithm boundaries explicit. Residual policy learning motivates using a residual policy on top of an imperfect controller rather than replacing the whole controller.

For this codebase, those ideas map to a conservative IsaacLab task variant:

- teacher lower body supplies the 12 leg joint targets;
- the wall-brush motion prior supplies the upper reference;
- the learned residual policy only changes waist yaw and right arm;
- stability rewards and metrics ensure jitter is not merely moved into root, legs, or left arm.

## Non-Goals

- Do not port FALCON or HumanoidVerse as the runtime training stack.
- Do not vendor external simulator runtimes or create a new conda environment.
- Do not modify the existing `Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-v0` task except where shared tests need to protect the new task boundary.
- Do not add physical wall collision for v0.
- Do not force a 27/27 brush success requirement for zero-residual replay; first prove balance, action composition, and metric visibility.
- Do not start long training during implementation.

## New Task

Register a new task ID:

```text
Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-FalconResidual8D-v0
```

The environment class should inherit from the current root-unlocked DreamControl wall-brush config:

```text
G1WallBrushNoWallCollisionDreamControlFalconResidual8DEnvCfg
```

The task keeps the core contract:

- `G1_MINIMAL_CFG`
- `fix_root_link=False`
- root-unlocked full-body simulation
- `sim.dt = 0.005`
- `decimation = 4`
- `episode_length_s = 10.0`
- staged250 27-prior bundle:

```text
../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz
```

The task should be isolated from existing successful tasks so it can fail or regress without changing the known successful AntiJitter path.

## Action Composition

The final target sent to the robot should be interpreted as:

```text
lower_body_final = a_teacher_lower
upper_final = a_ref_upper + residual_policy_8d
```

The lower-body teacher controls:

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
```

The residual policy controls:

```text
waist_yaw_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

The left arm is not a learned residual channel in v0. It should remain constrained by prior/default tracking and stability rewards so it cannot become the main jitter compensation channel.

The existing `WallBrushAgileLowerBodyAction` and `WallBrushMotionResidualJointPositionAction` are the preferred implementation base. If import or runtime tests show the existing composition does not actually drive both action terms correctly, the implementation must stop and debug the action manager boundary before adding rewards.

## Observations

The new task should initially reuse the current 992D policy observation contract rather than designing a new observation space. That keeps compatibility with the existing motion reference terms and eval tooling.

Additional observation terms are allowed only if the first smoke tests show a concrete need. Candidate future terms are virtual normal force target, force curriculum phase, and brush normal alignment error.

## Rewards

The v0 reward should prioritize jitter isolation over aggressive brush-force maximization.

Required reward groups:

- Brush-wall distance: reward small brush-tip x-axis error to the virtual wall plane.
- Stroke tracking: reward row accuracy, phase/progress, ordered anchor milestones, and coverage.
- Virtual normal force: reward a curriculum target band represented as a virtual force derived from brush penetration/proximity, not from physical contact impulse.
- Brush normal alignment: penalize misalignment between the brush/right-hand proxy normal and the wall normal.
- Residual smoothness: penalize residual action L2, action rate, and action acceleration.
- Stability: penalize root orientation/height deviations, feet slide, self-collision proxy violations, and illegal non-brush wall proximity.
- Left-arm quietness: keep left-arm wall clearance and self-collision terms visible so jitter does not move there.

The reward should not make force/contact terms dominate the survival and stability terms during smoke testing. Early smoke should be able to pass with conservative residual behavior.

## Virtual Force Curriculum

The v0 curriculum should be represented through task parameters and reward weights, not a physical wall.

Curriculum phases:

1. No force phase: brush row and coverage tracking with no virtual normal force requirement.
2. Small force phase: add a low virtual normal force target band.
3. Randomized virtual interaction phase: randomize target normal force, virtual friction coefficient, contact point offset along the brush row, and wall reaction scale.

Virtual force can be modeled from brush-tip wall error and approach velocity. The exact scalar formula belongs in the implementation plan and must be covered by unit tests before it is used in reward code.

## Evaluation And Smoke Gates

The v0 smoke gate is metrics-first.

Required local tests:

- Contract test proves the new task is registered.
- Contract test proves the new task is root-unlocked, full-body, 10s, 50Hz control, and staged250-based.
- Contract test proves the action structure uses AGILE lower-body teacher plus 8D residual policy.
- Contract test proves the task does not mention fixed-base, root-locked, upper-body-only, or body-group AntiJitter failed-route task IDs.
- Success/eval tests continue to pass for the existing 27-prior metrics.

Required remote tests:

- Use `/root/autodl-tmp/envs/isaaclab`.
- Use `/root/autodl-tmp/IsaacLab`.
- Sync only files needed for the new task and smoke wrappers.
- Run remote unittest and py_compile before running Isaac.
- Run zero-residual/reference replay smoke.
- Run a 1-5 iteration short train smoke only after the zero-residual task pipeline works.
- Do not start long training without an explicit user command.

Zero-residual smoke target:

- 27 evaluated priors.
- 500 steps per prior.
- 27/27 survival preferred.
- `min_root_z > 0.70m`.
- `min_root_cos_z > 0.98`.
- Brush success is diagnostic, not the pass/fail gate for v0.

Short-train smoke target:

- training command starts and exits normally;
- checkpoint/log directory is written;
- follow-up eval JSON is written;
- body-group stability metrics are present.

## Metrics To Preserve

The eval output must keep reporting:

- DreamControl-style success count and acceptance count.
- Survival rate and done step.
- Contact, row, combined, ordered anchor, and coverage metrics.
- Brush-tip speed, acceleration, and jerk.
- Action delta and right-arm action delta.
- Root z, root upright, root position error, and root orientation error.
- Foot slip.
- Left-arm clearance.
- Right-hand clearance.
- Self-collision margin and proxy violation.
- Suspicious prior selection and visual review entries.

The main research question is whether the new task reduces visible and measured jitter without moving it into root, legs, or left arm.

## Files Expected To Change

Primary task file:

```text
Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py
```

Registration file:

```text
Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py
```

Contract tests:

```text
Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py
```

Optional eval/smoke wrapper files:

```text
Training/scripts/remote_wall_brush_falcon_residual8d_eval.sh
Training/scripts/remote_wall_brush_falcon_residual8d_smoke.sh
Training/scripts/remote_wall_brush_falcon_residual8d_train.sh
```

The implementation plan should decide whether existing eval tooling can run zero-residual/reference replay without a new Python script. If not, add the smallest wrapper or flag needed to express zero residual.

## Risks And Mitigations

Risk: AGILE teacher state/action assumptions do not match the wall-brush reset state.

Mitigation: zero-residual smoke is mandatory before any train smoke. If the robot falls with zero residual, debug teacher observation and reset compatibility before reward tuning.

Risk: 8D residual is not expressive enough for some wall-brush rows.

Mitigation: keep success diagnostic in v0. Only consider 10D residual after video and metrics show reachability is the bottleneck rather than jitter.

Risk: virtual force rewards encourage high-frequency contact oscillation.

Mitigation: begin with no-force and small-force phases, keep residual/action acceleration penalties active, and track brush jerk in every eval.

Risk: lower-body teacher hides root or leg instability.

Mitigation: keep root, foot slip, and self-collision metrics in the gate. Do not judge by brush success alone.

Risk: implementation accidentally contaminates the successful AntiJitter task.

Mitigation: task ID, reward class, env class, wrappers, and tests must be new. Existing AntiJitter smoke tests continue to pass.

## Decisions Reserved For Implementation Plan

These decisions do not need more user input before planning. The implementation plan must make them explicit and cover them with tests or smoke checks:

- Define the exact virtual force scalar formula and thresholds.
- Decide whether zero-residual replay is a flag on existing eval/play scripts or a dedicated wrapper.
- Verify whether `WallBrushAgileLowerBodyAction.action_dim == 0` composes correctly with the residual action term in RSL-RL train/eval.
- Decide whether left-arm default/prior holding needs a dedicated action term or can remain reward-only in v0.
