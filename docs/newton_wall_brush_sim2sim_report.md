# Newton Wall-Brush Sim-to-Sim Report

## Scope

This records the first IsaacLab Newton sim-to-sim validation run for the G1 wall-brush DreamControl policy.

- Task: `Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAlignedAntiJitter-v0`
- Checkpoint: `model_1999.pt` from the wall-brush anti-jitter training run
- Reference bundle: `wall_brush_27_prior_references_dreamcontrol_staged250_standprefix_approach_stroke_recover.npz`
- PhysX baseline checkout: `/root/autodl-tmp/IsaacLab`
- Newton checkout used for validation: `/root/autodl-tmp/IsaacLab-newton`

For rebuild instructions from a fresh machine, see `docs/newton_wall_brush_reproduce.md`.

## Contract

The Newton validation path uses the same policy contract as the PhysX task.

- Policy observation dim: 992
- Action dim: 27
- Observation terms: `base_lin_vel`, `base_ang_vel`, `projected_gravity`, `joint_pos`, `joint_vel`, `actions`, five `target_ref_*` terms, `wall_brush_target`, `current_time`
- Action joints: the 27 policy-controlled G1 joints. The hand/finger joints exist in the asset but are not policy-controlled.

The Newton run required two important compatibility fixes:

- IsaacLab/Newton state tensors may be returned as `ProxyArray`, so the eval/play scripts convert those reads with `value.torch` when available.
- Newton/IsaacLab internals use xyzw quaternions, while this trained policy expects the `target_ref_*` observation quaternion slot in the original wxyz convention.

The exact checkpoint was trained under the AntiJitter task variant. This PR preserves the Newton eval/play compatibility patches and environment pins, but does not include the unmerged AntiJitter training-task implementation. Until that task variant is merged, the reproduction script defaults to the closest present task, `Isaac-Motion-Tracking-Wall-Brush-NoWallCollision-DreamControl-ButtonPressAligned-v0`, and exposes the exact task name in `configs/newton_wall_brush_versions.env`.

## Results

Short 80-step gate:

| prior | fall | NaN | min root z | max upright error | mean brush-tip error |
| --- | --- | --- | ---: | ---: | ---: |
| 4 | no | no | 0.7718 m | 3.47 deg | 0.0158 m |
| 16 | no | no | 0.7702 m | 5.83 deg | 0.0233 m |

All-27 500-step Newton evaluation:

| metric | result |
| --- | ---: |
| priors evaluated | 27 |
| survival | 27/27 |
| NaN failures | 0/27 |
| DreamControl-style success | 27/27 |
| training milestone pass | 27/27 |
| strict acceptance target pass | 19/27 |
| mean brush-tip error | 0.0132 m |
| mean wall-x error | 0.0086 m |
| mean row-yz error | 0.0088 m |
| mean root orientation error | 6.12 deg |

The 8 strict-target misses were foot-slip-only misses: priors 0, 1, 3, 6, 7, 9, 12, and 13 exceeded the 0.030 m target slightly, with max observed slip 0.0338 m. There were no wall-contact terminations and no self-collision terminations.

## Video Evidence

Inspection videos were generated for priors 4, 16, and 24. They were not committed because they are generated binary artifacts; the local validation copies lived under:

- `artifacts/newton_wall_brush_sim2sim/newton_prior4_policyquat.mp4`
- `artifacts/newton_wall_brush_sim2sim/newton_prior16_policyquat.mp4`
- `artifacts/newton_wall_brush_sim2sim/newton_prior24_policyquat.mp4`

The videos were recorded with marker overlays disabled because Isaac Sim 5.1 rejected `PreviewSurfaceCfg` marker material creation in this Newton setup.

## Conclusion

Newton is credible as second-simulator evidence for this policy after the observation/quaternion contract is fixed. The earlier large sim-to-sim mismatch should not be attributed only to hand mesh differences; the dominant issue in this Newton route was contract mismatch around MotionLib initialization and quaternion layout.

The remaining Newton-specific issue is a small foot-slip margin on 8/27 priors. This is not a fall, wall-contact, or row-tracking failure.
