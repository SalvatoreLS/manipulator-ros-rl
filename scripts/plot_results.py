#!/usr/bin/env python3
"""Turn the TensorBoard event files of a set of runs into the figures used in RESULTS.md.

Reads every run directory under ``--runs-dir``, groups them by the name of the parent
directory (one group per agent, e.g. ``sac`` and ``sb3``), and plots the mean ± std over
seeds for each metric.

    python3 scripts/plot_results.py --runs-dir output/runs --out-dir docs/figures

Layout expected (created by scripts/run_experiments.sh):

    output/runs/sac/seed0/logs/<timestamp>/events.out.tfevents.*
    output/runs/sac/seed1/...
    output/runs/sb3/seed0/...
"""

import argparse
import os
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')  # headless container: no display available
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Metric → (tensorboard tag, axis label, output filename)
METRICS = [
    ('rewards/episode',              'Episode return',            'episode_return'),
    ('train/success_rate_10',        'Success rate (10-ep mean)', 'success_rate'),
    ('train/final_distance',         'Final distance to target (m)', 'final_distance'),
    ('train/steps_to_goal',          'Steps to goal',             'steps_to_goal'),
]


def load_scalars(event_dir, tag):
    """Return (steps, values) for *tag* found anywhere under *event_dir*."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    steps, values = [], []
    for root, _dirs, files in os.walk(event_dir):
        if not any(f.startswith('events.out.tfevents') for f in files):
            continue
        acc = EventAccumulator(root, size_guidance={'scalars': 0})
        acc.Reload()
        if tag not in acc.Tags().get('scalars', []):
            continue
        for event in acc.Scalars(tag):
            steps.append(event.step)
            values.append(event.value)
    if not steps:
        return np.array([]), np.array([])
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(values)[order]


def collect(runs_dir, tag):
    """Map group name → list of per-seed (steps, values) arrays for *tag*."""
    groups = defaultdict(list)
    if not os.path.isdir(runs_dir):
        return groups
    for group in sorted(os.listdir(runs_dir)):
        group_path = os.path.join(runs_dir, group)
        if not os.path.isdir(group_path):
            continue
        for seed_dir in sorted(os.listdir(group_path)):
            seed_path = os.path.join(group_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue
            steps, values = load_scalars(seed_path, tag)
            if steps.size:
                groups[group].append((steps, values))
    return groups


def smooth(values, window):
    """Centred moving average; returns *values* unchanged when it is too short."""
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='same')


def plot_metric(groups, tag, ylabel, out_path, smooth_window):
    """Plot mean ± std across seeds for every group; return True if anything was drawn."""
    if not groups:
        return False

    fig, ax = plt.subplots(figsize=(7, 4.2))
    drew = False

    for name, runs in sorted(groups.items()):
        # Seeds can end at different episode counts (e.g. an interrupted run) — truncate
        # to the shortest so the mean is over a consistent set of seeds at every x.
        length = min(len(v) for _s, v in runs)
        if length == 0:
            continue
        stacked = np.vstack([v[:length] for _s, v in runs])
        steps = runs[0][0][:length]

        mean = smooth(stacked.mean(axis=0), smooth_window)
        std = smooth(stacked.std(axis=0), smooth_window)

        label = f'{name} (n={len(runs)} seed{"s" if len(runs) > 1 else ""})'
        ax.plot(steps, mean, label=label, linewidth=1.8)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.18)
        drew = True

    if not drew:
        plt.close(fig)
        return False

    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  wrote {out_path}')
    return True


def print_success_table(runs_dir):
    """Print the final success rate at each reporting threshold, per group."""
    thresholds = ['0.10', '0.05', '0.02']
    print('\nFinal success rate (last logged value, mean over seeds)')
    print(f'{"run":<12}' + ''.join(f'{t + " m":>12}' for t in thresholds))

    rows = {}
    for threshold in thresholds:
        groups = collect(runs_dir, f'train/success_rate_10_at_{threshold}m')
        for name, runs in groups.items():
            finals = [v[-1] for _s, v in runs if v.size]
            if finals:
                rows.setdefault(name, {})[threshold] = float(np.mean(finals))

    for name, values in sorted(rows.items()):
        cells = ''.join(
            f'{values.get(t, float("nan")):>11.0%} ' if t in values else f'{"—":>12}'
            for t in thresholds
        )
        print(f'{name:<12}{cells}')
    if not rows:
        print('  (no success-rate scalars found)')


def main():
    """Parse arguments and render every figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', default='output/runs', help='Directory of run groups (default: output/runs)')
    parser.add_argument('--out-dir',  default='docs/figures', help='Where to write PNGs (default: docs/figures)')
    parser.add_argument('--smooth',   type=int, default=5, help='Moving-average window in episodes (default: 5)')
    args = parser.parse_args()

    print(f'Reading runs from {args.runs_dir}')
    any_written = False
    for tag, ylabel, filename in METRICS:
        groups = collect(args.runs_dir, tag)
        out_path = os.path.join(args.out_dir, f'{filename}.png')
        if plot_metric(groups, tag, ylabel, out_path, args.smooth):
            any_written = True
        else:
            print(f'  skipped {filename} (tag "{tag}" not found in any run)')

    print_success_table(args.runs_dir)

    if not any_written:
        raise SystemExit(
            f'No TensorBoard scalars found under {args.runs_dir}. '
            'Run scripts/run_experiments.sh first.'
        )


if __name__ == '__main__':
    main()
