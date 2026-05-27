# `sim/assets/` — SO-101 model assets

This directory ships with a calibrated SO-101 URDF and its STL meshes:

```
sim/assets/so101/
├── so101_new_calib.urdf       # Onshape-derived, joints already match SceneConfig
└── assets/                    # STL meshes referenced relatively by the URDF
    ├── base_motor_holder_so101_v1.stl
    ├── base_so101_v2.stl
    ├── ...
```

The URDF carries the 6 joint names the scene expects:
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.

## How it's loaded

`scene.py:build_scene()` ingests the URDF via `mujoco.MjSpec.from_file(urdf)`,
which converts URDF → MjSpec procedurally. We then add the world decor,
actuators, and joint damping/armature on top before calling `spec.compile()`.

Because the URDF references meshes as `assets/<file>.stl` (relative), keep the
`assets/` subdir alongside the URDF — don't flatten.

## Want a different SO-101 description?

Swap in any URDF or MJCF whose joints match the names above. Override the
default via:

```bash
uv run python -m sim.scripts.view_env --mjcf path/to/your_so101.xml
```

If the joint names differ, update `SceneConfig.arm_joint_names` and
`gripper_joint_name` to match.

## Mesh size

The meshes are small (~few MB total). They're checked in. If you want to keep
them out of git, add `sim/assets/so101/assets/` to `.gitignore` and fetch
them on demand.
