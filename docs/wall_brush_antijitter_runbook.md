# Wall Brush Anti-Jitter Runbook

This runbook fixes the previously successful full-body wall-brush route in the repository.

## Task

Primary task:

```text
Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-v0
```

Baseline task remains available and unchanged:

```text
Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAligned-v0
```

The AntiJitter task inherits the ButtonPressAligned full-body contract:

- `G1_MINIMAL_CFG`
- root unlocked
- 27 DoF full-body `JointPositionActionCfg`
- `sim.dt = 0.005`
- `decimation = 4`
- `episode_length_s = 10.0`
- virtual wall target, no physical wall collision core
- staged250 27-prior reference bundle

## Reference Bundle

Remote reference path:

```text
/root/autodl-tmp/TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz
```

Wrapper path relative to `/root/autodl-tmp/IsaacLab`:

```text
../TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz
```

## Known Successful Checkpoint

The previous successful checkpoint is stored on the remote training machine:

```text
/root/autodl-tmp/IsaacLab/logs/rsl_rl/g1/2026-06-03_11-28-21-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-a_buttonpress_aligned_antijitter_official_8192env_2000iter_from_fingertip_best/model_1999.pt
```

It should be treated as the current successful baseline: it completes the wall-brush task but still has visible jitter.

## Deploy Files

From the migration workspace:

```bash
python3 scripts/autodl_remote.py put vendor/DreamControl/Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py /root/autodl-tmp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py
python3 scripts/autodl_remote.py put vendor/DreamControl/Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py /root/autodl-tmp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py
python3 scripts/autodl_remote.py put vendor/DreamControl/Training/scripts/remote_wall_brush_buttonpress_aligned_antijitter_train.sh /root/autodl-tmp/IsaacLab/scripts/remote_wall_brush_buttonpress_aligned_antijitter_train.sh
python3 scripts/autodl_remote.py put vendor/DreamControl/Training/scripts/remote_wall_brush_buttonpress_aligned_antijitter_eval.sh /root/autodl-tmp/IsaacLab/scripts/remote_wall_brush_buttonpress_aligned_antijitter_eval.sh
python3 scripts/autodl_remote.py put vendor/DreamControl/Training/scripts/remote_wall_brush_antijitter_sweep.sh /root/autodl-tmp/IsaacLab/scripts/remote_wall_brush_antijitter_sweep.sh
```

## Static Checks

```bash
PYTHONPYCACHEPREFIX=/tmp/dreamcontrol_pycache python3 -m py_compile \
  Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py \
  Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py

bash -n Training/scripts/remote_wall_brush_buttonpress_aligned_antijitter_train.sh
bash -n Training/scripts/remote_wall_brush_buttonpress_aligned_antijitter_eval.sh
bash -n Training/scripts/remote_wall_brush_antijitter_sweep.sh
```

## No-Training Sweep

Evaluate a checkpoint with playback smoothing only. This does not update weights.

```bash
python3 scripts/autodl_remote.py exec 'cd /root/autodl-tmp/IsaacLab && bash scripts/remote_wall_brush_antijitter_sweep.sh /root/autodl-tmp/IsaacLab/logs/rsl_rl/g1/2026-06-03_11-28-21-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-a_buttonpress_aligned_antijitter_official_8192env_2000iter_from_fingertip_best/model_1999.pt /root/autodl-tmp/wall_brush_antijitter_sweep_current "1.00 0.85 0.70 0.55"'
```

## Training Command

Use this only after explicit approval. It resumes from the successful fingertip/ButtonPressAligned run and fine-tunes the AntiJitter task.

```bash
python3 scripts/autodl_remote.py exec 'cd /root/autodl-tmp/IsaacLab && bash scripts/remote_wall_brush_buttonpress_aligned_antijitter_train.sh 8192 300 2026-06-02_06-38-12-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAligned-a_buttonpress_aligned_fingertip_official_8192env_2000iter model_1999.pt buttonpress_aligned_antijitter_from_fingertip_best_8192env_300iter 0.0015'
```

## Evaluation Command

Replace `RUN_DIR/MODEL` with the selected checkpoint.

```bash
python3 scripts/autodl_remote.py exec 'cd /root/autodl-tmp/IsaacLab && bash scripts/remote_wall_brush_buttonpress_aligned_antijitter_eval.sh /root/autodl-tmp/IsaacLab/logs/rsl_rl/g1/RUN_DIR/MODEL 27 500 /root/autodl-tmp/wall_brush_antijitter_eval.json 1 500 0 1.0'
```
