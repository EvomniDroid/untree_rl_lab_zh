# B2RM No-Vision Locomotion Task

`Unitree-B2RM-Velocity` is the no-vision locomotion stage for B2RM.  It is
intended to establish a real-robot-compatible trot and velocity controller
before adding the depth encoder used by the full Parkour task.

- Flat terrain only; no cameras, depth, terrain curriculum, gaps, or steps.
- Training commands are forward velocity only, uniformly sampled from
  `0.10` to `0.40 m/s`.  This makes learning a real walking gait the sole
  first-stage objective.
- The policy receives a sine/cosine gait phase and is rewarded for diagonal
  trot contacts.  It is penalized whenever measured forward velocity remains
  below a positive command, so a static target2 stance is not a good solution.
- No friction, mass, push, delay, or observation-noise randomization is used
  in this first walking stage.  Add those only after this policy walks in
  simulation and on the robot.
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

# Train the no-vision locomotion policy.
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

This task has a 50-dimensional policy observation: the 48 proprioceptive
values plus sine/cosine gait phase.  The real deployment program must load a
50-input ONNX policy and use the same 0.6 s gait period.  Do not load an older
48-input checkpoint into that program.
