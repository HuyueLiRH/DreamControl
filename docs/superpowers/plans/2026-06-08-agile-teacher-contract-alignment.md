# AGILE Teacher Contract Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the wall-brush AGILE lower-body action term with the exported Unitree G1 velocity-height teacher contract before any long residual-RL training.

**Architecture:** Keep the existing full-body wall-brush tasks intact. Patch only the new AGILE/FALCON action-term constants and isolation eval support, then validate with short no-checkpoint evals against the current remote IsaacLab environment.

**Tech Stack:** Python, IsaacLab manager-based environments, RSL-RL eval script, TorchScript AGILE teacher exported by WBC-AGILE.

---

### Task 1: Lock The Exported Teacher Contract In Tests

**Files:**
- Modify: `Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py`

- [ ] **Step 1: Write the failing test**

Add a contract test that asserts:
- `AGILE_LOWER_BODY_JOINTS` matches the teacher action order:
  `left_hip_pitch`, `right_hip_pitch`, `left_hip_roll`, `right_hip_roll`, `left_hip_yaw`, `right_hip_yaw`, `left_knee`, `right_knee`, `left_ankle_pitch`, `right_ankle_pitch`, `left_ankle_roll`, `right_ankle_roll`.
- `AGILE_TEACHER_JOINT_OBS_ORDER` matches the exported 29-joint observation order, including zero-filled `waist_roll_joint` and `waist_pitch_joint` slots when the 27-DoF asset lacks them.
- `AGILE_LOWER_BODY_POLICY_OUTPUT_SCALE_ORDERED` matches the exported scale vector.
- the fixed stance command is `(0.0, 0.0, 0.0, 0.70)`.

- [ ] **Step 2: Run test to verify it fails**

Run:
`python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_full_body_contract.WallBrushFullBodyContractTest.test_agile_teacher_contract_matches_exported_velocity_height_policy -v`

Expected: FAIL because the current joint order and scales are not aligned.

### Task 2: Align The Action Term Constants

**Files:**
- Modify: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py`

- [ ] **Step 1: Implement minimal contract alignment**

Change the AGILE constants to the exported teacher order and scale. Keep `action_dim=0` for this isolation task so the upper-body residual policy still owns only the 8D residual and the lower-body teacher receives a fixed standing command.

- [ ] **Step 2: Run local tests**

Run:
`python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_success.py Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v`

Expected: all tests pass.

- [ ] **Step 3: Compile touched files**

Run:
`PYTHONPYCACHEPREFIX=/tmp/dreamcontrol_pycache python3 -m py_compile Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py`

Expected: exit 0.

### Task 3: Remote Smoke Eval

**Files synced to remote only:**
- `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py`
- `Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py`

- [ ] **Step 1: Back up remote files**

Create a timestamped backup under `/root/autodl-tmp`.

- [ ] **Step 2: Upload only touched files**

Use `python3 scripts/autodl_remote.py put`.

- [ ] **Step 3: Verify remote tests**

Run the new contract test plus `py_compile` inside `/root/autodl-tmp/envs/isaaclab`.

- [ ] **Step 4: Run no-training default-stand eval**

Run 500-step no-checkpoint evals for:
- FALCON residual task with zero residual.
- FALCON residual task with default upper-body targets.
- AgileBase task with zero residual.
- AgileBase task with default upper-body targets.

Expected: survival should improve materially from the previous 0/27 at step 42-43 if the contract mismatch was the root cause. If it still fails, collect the new failure mode before proposing another fix.

### Task 4: Commit And Report

**Files:**
- Commit only the plan, test, and action-term alignment if tests pass.

- [ ] **Step 1: Commit local changes**

Commit message:
`fix: align agile teacher wall brush contract`

- [ ] **Step 2: Report evidence**

Include local tests, remote tests, remote eval JSON paths, and whether any Isaac/eval/train process remains active.
