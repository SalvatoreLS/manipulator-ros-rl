# Design notes

Four decisions in this environment are not obvious from the code, and each one exists
because the naive version was wrong in a specific way. They are written up here because
the reasoning is the interesting part, not the line count.

---

## 1. Two sources for the end-effector position

`DataHandler` gets the EE position from two different places, and prefers one:

- **On hardware**, `franka_robot_state_broadcaster` publishes `FrankaRobotState`, whose
  `O_T_EE` field is the column-major 4×4 base→EE transform. Translation is at indices
  12–14. This is the authoritative source.
- **In Gazebo**, that broadcaster does not exist: it requires the `fr3/robot_state`
  hardware interface, which the simulated hardware plugin does not provide. But
  `robot_state_publisher` consumes `/joint_states` and publishes the full TF tree, so the
  EE pose is available as a TF lookup of `fr3_link8` in `world`.

The node subscribes to both and keeps a `_ee_from_state_cb` flag. Once the hardware
broadcaster has fired even once, the TF fallback stops writing the EE position, so the two
sources can never fight. The same training code therefore runs unmodified in simulation
and on the real arm — the switch is automatic, not a flag.

*Files:* `environment/data_handler.py` — `state_callback`, `_joint_state_callback`.

---

## 2. Incremental joint deltas, not absolute positions

The action is a 7-vector in `[-1, 1]`. The obvious mapping is affine onto the full joint
range — action `-1` means "joint at its lower limit". That mapping breaks the MDP:

- A single action can command a motion of several radians. The controller cannot execute
  it within one environment step, so the state measured after the step does not reflect
  the action that was taken. The transition stored in the replay buffer is a lie.
- The dynamics stop being smooth: neighbouring actions produce wildly different outcomes.

Instead the action is a *delta*: `q_target = q_current + a · max_joint_delta`, with
`max_joint_delta = 0.1 rad`, then clipped to the FR3 joint limits. Every commanded move is
small enough to complete inside the settle window, so the reward genuinely measures the
consequence of the action, and the policy sees locally smooth dynamics. Clipping to the
real limits (rather than letting the controller saturate silently) keeps the commanded
target and the achievable target identical.

The cost is a horizon: reaching across the workspace now takes tens of steps rather than
one. That is the right trade — it is what makes the credit assignment learnable.

*Files:* `environment/data_handler.py` — `publish_command`.

---

## 3. Settle detection instead of a fixed sleep

After publishing a command the environment must wait before measuring. A fixed
`time.sleep(dt)` forces a choice between two bad options: too short, and the state is a
mid-motion transient; too long, and every step wastes wall-clock time that dominates
training duration.

`_wait_until_settled()` instead polls the measured joint velocities and returns as soon as
`max |dq| < 0.05 rad/s`, after a short mandatory dwell that gives the controller time to
start moving at all. A timeout bounds the worst case so a stuck joint cannot hang the run.

This makes the step duration adaptive: small moves return quickly, large ones take as long
as they need.

*Files:* `environment/arm_env.py` — `_wait_until_settled`.

---

## 4. Target sampling that is uniform in the *workspace*, not in the parameters

Targets are drawn in spherical coordinates and converted to Cartesian. Sampling `r`, `φ`
and `θ` uniformly would concentrate targets near the base and near the poles — the policy
would then be evaluated mostly on the easy region.

Two transforms fix the density:

- **Radius**: the volume element grows as `r²`, so uniform-in-volume requires sampling
  `r³` uniformly and taking the cube root.
- **Elevation**: the solid-angle element carries a `cos θ`, so uniform-in-solid-angle
  requires sampling `sin θ` uniformly and taking the arcsine.

The bounds themselves encode the robot, not the maths:

| Bound | Value | Why |
|---|---|---|
| radius | 0.2 – 0.75 m | inside the ~0.855 m FR3 reach, outside the near-base singularity |
| azimuth | ±150° | joint 1 is limited to ±166°; the rear 60° wedge is excluded |
| elevation | 22.5° – 70° | above the table (min z ≈ 0.19 m), below the overhead singularity |

Sampling draws from a `np.random.Generator` owned by the node, seeded from the
environment. This is what makes a run reproducible from its seed: the global `np.random`
state is shared with everything else in the process and is not a reliable channel.

*Files:* `environment/data_handler.py` — `set_random_target`.

---

## Reward

Potential-based shaping (Ng et al., 1999):

```
r = 10·(d_prev - d)  -  0.1·d  -  0.01  +  5·[reached]
```

terminating when `d < 10 cm`. All four weights are `FRANKA_*` environment overrides in
`environment/env_config.py`.

The dominant term is the *reduction* in distance over the step, not the absolute distance.
A plain `-distance` reward (what this was originally) encodes only where the arm is, never
whether the action helped: every action from a given state scores about the same, and the
learning signal has to come out of the value function alone. Rewarding progress leaves the
optimal policy unchanged — that is the point of potential-based shaping — while giving a
dense, per-step signal. Scales are chosen so a full-speed approach (~0.03 m/step) lands
around +0.3, keeping per-step reward roughly in `[-1, +1]`.

The residual `-0.1·d` keeps a gradient pointing at the target when progress stalls near
zero, and the `-0.01` step cost makes dawdling next to the target worse than reaching it.

The success bonus is **additive** rather than replacing the shaped terms, so the gradient
still exists on the terminal step — with a flat terminal reward, the last transition would
carry no information about *where* in the goal region the arm ended up. It is deliberately
small (5.0, previously 50.0): at 50 the terminal step was a ~150x discontinuity, and the
critic spent most of its capacity fitting that cliff rather than the shaped landscape.

Termination at 10 cm is a deliberately generous criterion. Because it is generous, success
is *reported* at 10, 5 and 2 cm (`info["success_at"]`), so the headline number cannot be
read as tighter than it is.

This is a position-only reward: no orientation term, no collision penalty, no effort or
action-rate regularisation. That is a limitation, not an oversight — see the limitations
section of [RESULTS.md](RESULTS.md).

---

## Concurrency

ROS 2 callbacks and the RL loop run in different threads:

- The node is spun by a `MultiThreadedExecutor` on a daemon thread, so subscriptions keep
  receiving while the main thread blocks in `step()`.
- Every read and write of the shared state goes through a single `threading.Lock`; the
  state subscription and the `/joint_states` subscription sit in separate callback groups
  so neither starves the other.
- Gradient updates run on a `ThreadPoolExecutor` worker, deliberately overlapping the
  ~50 ms that `step()` spends waiting on the controller. `sleep()` and CUDA kernels both
  release the GIL, so this is real parallelism rather than bookkeeping: the otherwise-idle
  settle window pays for the optimisation. The future is drained before the next
  `select_action`, so the policy is never read mid-update.

*Files:* `environment/arm_env.py` — `__init__`; `agent/train.py` — `_run_gradient_updates`
and the rollout loop.
