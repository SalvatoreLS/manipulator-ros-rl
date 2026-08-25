"""
Measure the wall-clock cost of a single SACAgent.optimize() call.

Why this exists: gradient_steps was sized (agent/config.py) on the assumption that each
optimize() costs ~8 ms and therefore that 6 of them hide inside the environment's
~50 ms per-step wait. Once the simulator was unthrottled and the settle window shortened,
the environment stopped being the bottleneck and throughput collapsed — which only makes
sense if optimize() is far more expensive than 8 ms. This measures it directly so
gradient_steps can be chosen from data rather than from that stale assumption.

Run inside the container:
    python3 scripts/bench_optimize.py [--device cuda|cpu] [--iters 200]
"""

import argparse
import time

import numpy as np
import torch

from distance_based_rl.agent.sac_agent import SACAgent

STATE_DIM = 28   # State.vector_dim(): 3+3+7+7+7+1
ACTION_DIM = 7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, help='cuda / cpu (default: auto)')
    parser.add_argument('--iters', type=int, default=200, help='timed optimize() calls')
    parser.add_argument('--warmup', type=int, default=30, help='untimed calls first')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--threads', type=int, default=1, help='torch CPU threads')
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))

    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        buffer_size=50000,
        device=args.device,
    )
    print(f"device      : {agent.device}")
    print(f"torch threads: {torch.get_num_threads()}")

    # Fill the buffer so sample() never underflows.
    rng = np.random.default_rng(0)
    for _ in range(args.batch_size * 4):
        agent.replay_buffer.push(
            rng.standard_normal(STATE_DIM).astype(np.float32),
            rng.standard_normal(ACTION_DIM).astype(np.float32),
            float(rng.standard_normal()),
            rng.standard_normal(STATE_DIM).astype(np.float32),
            False,
        )

    for _ in range(args.warmup):
        agent.optimize(batch_size=args.batch_size)
    if agent.device.type == 'cuda':
        torch.cuda.synchronize(agent.device)

    per_call = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        agent.optimize(batch_size=args.batch_size)
        if agent.device.type == 'cuda':
            torch.cuda.synchronize(agent.device)
        per_call.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(per_call)
    print(f"\noptimize() over {args.iters} calls, batch={args.batch_size}:")
    print(f"  mean   : {a.mean():7.2f} ms")
    print(f"  median : {np.median(a):7.2f} ms")
    print(f"  p90    : {np.percentile(a, 90):7.2f} ms")
    print(f"  min/max: {a.min():.2f} / {a.max():.2f} ms")

    print("\nImplied per-env-step gradient cost and ceiling on steps/s")
    print("(env cost excluded — add it to the per-step total):")
    for gs in (1, 2, 3, 4, 6):
        ms = a.mean() * gs
        print(f"  gradient_steps={gs}: {ms:7.1f} ms/step  ->  {1000.0 / ms:5.2f} step/s ceiling")


if __name__ == '__main__':
    main()
