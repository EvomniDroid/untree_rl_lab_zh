# B2RM Minimal Velocity Task

`Unitree-B2RM-Velocity` is a deliberately small baseline for validating the
official Unitree RL Lab train/export workflow with the B2RM model.

- Flat terrain only; no cameras, depth, terrain curriculum, random pushes, or
  dynamics randomization.
- The policy has 12 outputs for the leg joints only.
- The six arm joints remain in the folded B2RM pose through a zero-dimensional
  `ArmHold` action term and their own PD gains.
- The reset state is upright, not prone. It uses the B2 SDK stand
  `targetPos_2`: hip `0.0`, thigh `0.67`, calf `-1.30` rad, with the same
  `Kp=1000`, `Kd=10` leg hold used by the official B2 stand example.

## Run

The task uses the B2RM URDF currently used by InstinctLab by default. To use a
copied asset tree instead, set `UNITREE_B2RM_URDF_PATH` to that URDF before
launching.

```bash
cd /home/zh/isaac/unitree_rl_lab

# First confirm the task is registered.
python scripts/list_envs.py

# Train the flat-ground velocity baseline.
python scripts/rsl_rl/train.py \
  --task Unitree-B2RM-Velocity \
  --headless \
  --num_envs 1024 \
  --max_iterations 5000

# Play the latest checkpoint after training.
python scripts/rsl_rl/play.py \
  --task Unitree-B2RM-Velocity \
  --num_envs 1
```

## Real Robot Bring-up

The learned policy is not a get-up controller. On hardware, first use a
separate guarded transition:

1. Passive/safe state.
2. Interpolate the legs and folded arm to the task default pose with fixed PD.
3. Verify upright orientation, foot contact, joint tracking error, and no
   fault for a short hold period.
4. Only then start the 12-dimensional leg policy while continuously holding
   the six arm targets.

The existing official `b2` deploy controller is 12-leg-DOF only. A B2RM
sim2sim/real bridge must therefore send the extra six fixed arm commands as
well; this task intentionally does not pretend the stock B2 controller can do
that unchanged.
