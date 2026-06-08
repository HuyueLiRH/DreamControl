# Newton Wall-Brush Reproduction

This file records how to rebuild the Newton wall-brush sim-to-sim path if the remote machine is lost.

## Repositories

The environment is tracked through forks and pinned refs, not by vendoring all of Newton and IsaacLab into DreamControl.

| component | fork | upstream | ref |
| --- | --- | --- | --- |
| DreamControl task/scripts | `https://github.com/HuyueLiRH/DreamControl.git` | `https://github.com/GenRobo/DreamControl.git` | `codex/newton-wall-brush-reproduction-pins` |
| IsaacLab Newton integration | `https://github.com/HuyueLiRH/IsaacLab.git` | `https://github.com/isaac-sim/IsaacLab.git` | `v3.0.0-beta` |
| Newton | `https://github.com/HuyueLiRH/newton.git` | `https://github.com/newton-physics/newton.git` | `v1.2.1` |

The machine-readable version pins live in `configs/newton_wall_brush_versions.env`.

## Required Artifacts

These generated artifacts are not committed to git:

- checkpoint: `model_1999.pt`
- prior bundle: `TrajGen/sample/Wall_Brush_27/wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz`
- optional videos and eval JSON/CSV outputs

The last validated checkpoint path on the old remote was:

```text
/root/autodl-tmp/IsaacLab/logs/rsl_rl/g1/2026-06-03_11-28-21-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-a_buttonpress_aligned_antijitter_official_8192env_2000iter_from_fingertip_best/model_1999.pt
```

## Fresh Machine Setup

Clone DreamControl and check out this PR branch or the merge commit that contains it:

```bash
git clone https://github.com/HuyueLiRH/DreamControl.git /root/autodl-tmp/DreamControl
cd /root/autodl-tmp/DreamControl
git checkout codex/newton-wall-brush-reproduction-pins
```

Prepare the expected Training root used by the existing remote scripts:

```bash
ln -sfn /root/autodl-tmp/DreamControl/Training /root/autodl-tmp/IsaacLab
```

Clone and pin the IsaacLab/Newton source forks:

```bash
cd /root/autodl-tmp/DreamControl
bash Training/scripts/newton_wall_brush_setup_env.sh
```

For a fresh Python environment, run the same script with dependency installation enabled:

```bash
bash Training/scripts/newton_wall_brush_setup_env.sh --install
```

## Run Evaluation

Place the checkpoint and prior bundle at the paths above, then run:

```bash
cd /root/autodl-tmp/DreamControl
bash Training/scripts/newton_wall_brush_eval.sh \
  /root/autodl-tmp/IsaacLab/logs/rsl_rl/g1/2026-06-03_11-28-21-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-a_buttonpress_aligned_antijitter_official_8192env_2000iter_from_fingertip_best/model_1999.pt \
  27 \
  500 \
  /root/autodl-tmp/newton_wall_brush_eval.json \
  1
```

The script defaults to `physics=newton_mjwarp` through `configs/newton_wall_brush_versions.env`.

## Exact AntiJitter Task

The checkpoint was trained with, and the eval script defaults to:

```text
Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-v0
```

To run a different task variant, override this value in `configs/newton_wall_brush_versions.env` or in the shell environment:

```bash
NEWTON_WALL_BRUSH_TASK="Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAligned-v0"
```

## When To Patch The Forks

Do not copy Newton or IsaacLab into DreamControl. If the Newton backend or IsaacLab integration needs a local fix:

1. create a branch in `HuyueLiRH/IsaacLab` or `HuyueLiRH/newton`;
2. commit the upstream patch there;
3. update `ISAACLAB_NEWTON_REF` or `NEWTON_REF` in `configs/newton_wall_brush_versions.env`;
4. rerun this document from a clean machine.
