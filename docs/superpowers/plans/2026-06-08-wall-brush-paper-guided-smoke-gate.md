# Wall Brush Paper-Guided Smoke Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a clean smoke gate for the `experiment/wall-brush-paper-guided-task` branch before any paper-guided reward or task changes.

**Architecture:** Keep the successful ButtonPressAlignedAntiJitter route intact and validate changes through file-level contract tests plus a no-training remote IsaacLab eval/video run. Remote execution must use `/root/autodl-tmp/envs/isaaclab` and `/root/autodl-tmp/IsaacLab`; only files needed by tests/eval are synced.

**Tech Stack:** Python `unittest`, IsaacLab manager-based task configs, RSL-RL eval scripts, AutoDL SSH helper `scripts/autodl_remote.py`.

---

### Task 1: Verify Worktree And Baseline Contracts

**Files:**
- Read: `docs/wall_brush_antijitter_runbook.md`
- Test: `Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py`
- Test: `Training/source/isaaclab_tasks/test/test_wall_brush_success.py`

- [x] **Step 1: Confirm isolated worktree and branch**

Run:

```bash
git rev-parse --git-dir
git rev-parse --git-common-dir
git branch --show-current
git log -1 --oneline
git status --short --branch
```

Expected: linked worktree on `experiment/wall-brush-paper-guided-task`, HEAD `4b32e8d`, no unrelated local edits before starting.

- [x] **Step 2: Run local baseline tests**

Run:

```bash
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_success.py -v
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v
PYTHONPYCACHEPREFIX=/tmp/dreamcontrol_pycache python3 -m py_compile Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py
```

Expected: success tests pass, full-body contract passes, and py_compile exits 0.

### Task 2: Make Contract Tests Work In Remote IsaacLab Layout

**Files:**
- Modify: `Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py`

- [x] **Step 1: Write failing test for remote root detection**

Add:

```python
def test_find_repo_root_recognizes_remote_isaaclab_layout(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "IsaacLab"
        test_dir = root / "source" / "isaaclab_tasks" / "test"
        test_dir.mkdir(parents=True)
        (root / "scripts" / "reinforcement_learning" / "rsl_rl").mkdir(parents=True)

        self.assertEqual(_find_repo_root(test_dir), root)
```

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_full_body_contract.WallBrushFullBodyContractTest.test_find_repo_root_recognizes_remote_isaaclab_layout -v
```

Expected: FAIL because `_find_repo_root()` returns the test directory.

- [x] **Step 3: Implement root detection**

Add this condition inside `_find_repo_root()`:

```python
if (
    (candidate / "scripts" / "reinforcement_learning" / "rsl_rl").exists()
    and (candidate / "source" / "isaaclab_tasks").exists()
):
    return candidate
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_full_body_contract.WallBrushFullBodyContractTest.test_find_repo_root_recognizes_remote_isaaclab_layout -v
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v
```

Expected: both commands pass.

### Task 3: Fix AntiJitter Import NameError

**Files:**
- Modify: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py`
- Modify: `Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py`

- [x] **Step 1: Write failing contract assertion**

Add these assertions to `test_buttonpress_aligned_antijitter_success_route_is_fixed`:

```python
self.assertIn("RIGHT_HAND_NONBRUSH_LINKS = [", source)
self.assertIn("for link in RIGHT_HAND_LINKS if link != BRUSH_LINK", source)
```

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_full_body_contract.WallBrushFullBodyContractTest.test_buttonpress_aligned_antijitter_success_route_is_fixed -v
```

Expected: FAIL because `RIGHT_HAND_NONBRUSH_LINKS` is missing.

- [x] **Step 3: Define the missing constant**

Add after `RIGHT_HAND_LINKS`:

```python
RIGHT_HAND_NONBRUSH_LINKS = [
    link for link in RIGHT_HAND_LINKS if link != BRUSH_LINK
]
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
python3 -m unittest Training.source.isaaclab_tasks.test.test_wall_brush_full_body_contract.WallBrushFullBodyContractTest.test_buttonpress_aligned_antijitter_success_route_is_fixed -v
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_success.py -v
python3 -m unittest Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v
PYTHONPYCACHEPREFIX=/tmp/dreamcontrol_pycache python3 -m py_compile Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py
```

Expected: all pass.

### Task 4: Run Remote Smoke Gate Without Training

**Files:**
- Sync: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py`
- Sync: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py`
- Sync: `Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py`
- Sync: `Training/source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py`
- Sync: `Training/source/isaaclab_tasks/test/test_wall_brush_success.py`
- Sync: `Training/scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py`
- Sync: `Training/scripts/reinforcement_learning/rsl_rl/play_wall_brush_fixed_view.py`
- Sync: `Training/scripts/remote_wall_brush_buttonpress_aligned_antijitter_eval.sh`
- Sync: `Training/scripts/remote_wall_brush_buttonpress_aligned_antijitter_train.sh`
- Sync: `Training/scripts/remote_wall_brush_antijitter_sweep.sh`

- [x] **Step 1: Back up overwritten remote files**

Run:

```bash
python3 scripts/autodl_remote.py exec 'backup="/root/autodl-tmp/wall_brush_paper_guided_backup_$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$backup/g1" "$backup/rsl_rl" "$backup/test" "$backup/scripts"; cd /root/autodl-tmp/IsaacLab; cp -a source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_wall_brush_env.py "$backup/g1/"; cp -a source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/__init__.py "$backup/g1/"; cp -a source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/wall_brush_success.py "$backup/g1/"; cp -a scripts/reinforcement_learning/rsl_rl/eval_wall_brush_policy.py "$backup/rsl_rl/"; cp -a scripts/reinforcement_learning/rsl_rl/play_wall_brush_fixed_view.py "$backup/rsl_rl/"; cp -a source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py "$backup/test/"; cp -a source/isaaclab_tasks/test/test_wall_brush_success.py "$backup/test/"; printf "%s\n" "$backup"'
```

Expected: prints backup directory under `/root/autodl-tmp`.

- [x] **Step 2: Sync only test/eval files needed for the smoke gate**

Run one `python3 scripts/autodl_remote.py put <local> <remote>` per listed sync file.

Expected: all SFTP commands exit 0.

- [x] **Step 3: Run remote tests in existing IsaacLab environment**

Run:

```bash
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/IsaacLab && source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab && python -m unittest source/isaaclab_tasks/test/test_wall_brush_success.py -v"
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/IsaacLab && source /root/miniconda3/bin/activate /root/autodl-tmp/envs/isaaclab && python -m unittest source/isaaclab_tasks/test/test_wall_brush_full_body_contract.py -v"
```

Expected: 9 success tests pass and 20 full-body contract tests pass.

- [x] **Step 4: Run no-training 27-prior 500-step eval**

Run:

```bash
python3 scripts/autodl_remote.py exec "cd /root/autodl-tmp/IsaacLab && bash scripts/remote_wall_brush_buttonpress_aligned_antijitter_eval.sh /root/autodl-tmp/IsaacLab/logs/rsl_rl/g1/2026-06-03_11-28-21-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-a_buttonpress_aligned_antijitter_official_8192env_2000iter_from_fingertip_best/model_1999.pt 27 500 /root/autodl-tmp/wall_brush_paper_guided_smoke_20260608T1740Z.json 1 500 0 1.0"
```

Expected: JSON reports 27/27 DreamControl success and 27/27 acceptance target, and visual review videos are written for the suspicious prior.

### Task 5: Stop Before Paper-Guided Training

**Files:**
- None

- [x] **Step 1: Verify no remote Isaac/Kit train/eval/video job remains active**

Run:

```bash
python3 scripts/autodl_remote.py exec "ps -eo pid,ppid,stat,etime,cmd | grep -E '[i]saac|[k]it|[t]rain.py|[p]lay.py|[e]val_wall_brush' || true"
```

Expected: no active matching process after smoke eval/video exits.

- [x] **Step 2: Do not start long training**

Expected: no `train.py` long-run command is launched until the user explicitly requests it.
