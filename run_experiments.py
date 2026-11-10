"""
Task V - Hyperparameter tuning and evaluation for Q-Learning and DQN.

Usage:
    python run_experiments.py                 # everything (sweeps + combined check, both methods)
    python run_experiments.py ql              # Q-Learning only
    python run_experiments.py dqn sweeps      # DQN one-at-a-time sweeps only
    python run_experiments.py combined        # only the combined-best confirmation runs

Every setting is changed one at a time from a fixed starting point and trained with
3 seeds. The maze is built exactly like main.py (np.random.seed(26), 4 walls, 5x5);
the training seeds are set AFTER the maze is built, so the maze never changes.
Drawing is off (pygame_delay = 0, no window updates) to make the runs fast.

After the one-at-a-time sweeps, a short "combined" check trains the clear winner of the sweeps
together with the candidates for the one setting whose result was mixed (see COMBINED).

Outputs (in this folder): results_summary.csv, one PNG per tested setting, maze_layout.png.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import sys
import csv
import time
import random
from types import SimpleNamespace
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn

from DQN import QNetwork
from QLearning import QTable
from utils import bfs_shortest_path

SEEDS = [0, 1, 2]
EPISODES = {"QLearning": 500, "DQN": 500}
MAXIMUM_STEPS = 500           # same as main.py
TEST_MAX_STEPS = 100          # step limit for the greedy (epsilon = 0) test episode
N_WORKERS = 4

# Starting point (baseline) for each method
BASE = {
    "QLearning": dict(q_learning_rate=0.1, discount_factor=0.99, epsilon=0.9),
    "DQN": dict(learning_rate=1e-4, discount_factor=0.99, epsilon=0.9, batch_size=64, capacity=1000,
                target_network_update_frequency=10, hidden_size=64),
}

# One-at-a-time sweeps: (file/plot key, config field, label, values)
SWEEPS = {
    "QLearning": [
        ("ql_alpha", "q_learning_rate", "Learning rate (alpha)", [0.01, 0.1, 0.5]),
        ("ql_gamma", "discount_factor", "Discount factor (gamma)", [0.5, 0.9, 0.99]),
        ("ql_epsilon", "epsilon", "Starting epsilon", [0.1, 0.5, 0.9]),
    ],
    "DQN": [
        ("dqn_lr", "learning_rate", "Learning rate", [1e-4, 1e-3, 5e-3]),
        ("dqn_gamma", "discount_factor", "Discount factor (gamma)", [0.5, 0.9, 0.99]),
        ("dqn_epsilon", "epsilon", "Starting epsilon", [0.1, 0.5, 0.9]),
        ("dqn_target_update", "target_network_update_frequency", "Target update frequency", [1, 10, 100]),
        ("dqn_batch", "batch_size", "Batch size", [32, 64, 128]),
        ("dqn_hidden", "hidden_size", "Hidden size", [32, 64, 128]),
    ],
}

# Confirmation step: the clear winner of the sweeps (starting epsilon 0.1) combined with the
# candidates for the one setting whose sweep result was mixed (alpha for Q-Learning, lr for DQN).
COMBINED = {
    "QLearning": [("eps0.1+alpha0.1", dict(epsilon=0.1, q_learning_rate=0.1)),
                  ("eps0.1+alpha0.5", dict(epsilon=0.1, q_learning_rate=0.5))],
    "DQN": [("eps0.1+lr1e-4", dict(epsilon=0.1, learning_rate=1e-4)),
            ("eps0.1+lr1e-3", dict(epsilon=0.1, learning_rate=1e-3))],
}

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3 (colour-blind safe as a set)


# ----------------------------------------------------------------------------------
# Environment and training (the training loop mirrors main.py step for step)
# ----------------------------------------------------------------------------------
def build_config(method, maze_size=5, wall_number=4, wall_seed=26, start=(0, 0), end=None, **overrides):
    """Build the maze exactly as main.py does, then add the learning settings."""
    config = SimpleNamespace()
    config.Learning_method = method
    config.MAZE_SIZE = maze_size
    config.Start_point = list(start)
    config.End_point = list(end) if end is not None else [maze_size - 1, maze_size - 1]
    config.maze_setting = np.zeros((maze_size, maze_size), dtype=int)
    config.wall_number = wall_number
    np.random.seed(wall_seed)
    for _ in range(wall_number):
        x = np.random.randint(1, maze_size)
        y = np.random.randint(1, maze_size)
        config.maze_setting[x, y] = 1
    if (config.maze_setting[tuple(config.Start_point)] == 1 or config.maze_setting[tuple(config.End_point)] == 1):
        raise ValueError("The start or end point coincides with the obstacle.")
    config.pygame_delay = 0
    config.loss_fn = nn.MSELoss()
    for key, value in BASE[method].items():
        setattr(config, key, value)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def perfect_score(config):
    path = bfs_shortest_path(config.maze_setting, config.Start_point, config.End_point)
    return 10 - (len(path) - 2), path


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_agent(config, agent_cls=None):
    if agent_cls is not None:
        return agent_cls(config)
    return QNetwork(config) if config.Learning_method == "DQN" else QTable(config)


def train(config, seed, episodes, agent_cls=None, epsilon_update=None, on_episode_end=None,
          max_steps=MAXIMUM_STEPS):
    """Train one agent. Returns (agent, rewards, wall_hits, steps_per_episode, seconds)."""
    torch.set_num_threads(1)
    set_seeds(seed)                         # seeds are set after the maze was built
    agent = make_agent(config, agent_cls)
    is_dqn = hasattr(agent, "replay_buffer")
    rewards, wall_hits, ep_steps = [], [], []
    learning_step = 1
    t0 = time.perf_counter()
    for episode in range(episodes):
        steps, total_reward, hits, done_flags = 1, 0, 0, False
        pos = list(config.Start_point)      # maze.reset()
        agent.robot_pos = list(config.Start_point)
        if hasattr(agent, "reset_episode"):
            agent.reset_episode()
        while not done_flags and steps < max_steps:
            steps += 1
            old_pos, action, new_pos, reward, done_flags = agent.step(pos)
            total_reward += reward
            hits += reward == -10
            if is_dqn:
                agent.replay_buffer.push(old_pos, action, new_pos, reward, done_flags)
                if len(agent.replay_buffer) > config.batch_size:
                    learning_step += 1
                    agent.learn()
                if learning_step % config.target_network_update_frequency == 0:
                    agent.target_update()
            else:
                agent.update_q_table(old_pos, action, new_pos, reward, done_flags)
            pos = new_pos                   # maze.move_Robot(new_pos)
        if epsilon_update is None:
            agent.epsilon_decline()
        else:
            agent.epsilon = epsilon_update(episode, agent.epsilon)
        rewards.append(total_reward)
        wall_hits.append(hits)
        ep_steps.append(steps - 1)
        if on_episode_end is not None:
            on_episode_end(agent, episode)
    seconds = time.perf_counter() - t0
    return agent, rewards, wall_hits, ep_steps, seconds


def greedy_test(agent, config, max_steps=TEST_MAX_STEPS):
    """One episode with epsilon = 0. Returns (reached_goal, steps, total_reward, path)."""
    saved_eps = agent.epsilon
    agent.epsilon = 0
    if hasattr(agent, "reset_episode"):
        agent.reset_episode()
    pos = list(config.Start_point)
    agent.robot_pos = list(config.Start_point)
    path, total, done_flags = [pos], 0, False
    while not done_flags and len(path) <= max_steps:
        _, _, pos, reward, done_flags = agent.step(pos)
        total += reward
        path.append(list(pos))
    agent.epsilon = saved_eps
    return done_flags, len(path) - 1, total, path


def episodes_to_learn(rewards, best, window=10):
    """First episode (1-based) where the 10-episode average is within 1 of the best reward."""
    if len(rewards) < window:
        return None
    avg = np.convolve(rewards, np.ones(window), "valid") / window
    hit = np.flatnonzero(avg >= best - 1)
    return int(hit[0]) + window if len(hit) else None


def moving_average(x, window=10):
    return np.convolve(x, np.ones(window), "valid") / window


# ----------------------------------------------------------------------------------
# One job = one (method, setting, seed) run, executed in a worker process
# ----------------------------------------------------------------------------------
def run_job(job):
    method, overrides, seed = job
    config = build_config(method, **overrides)
    best, _ = perfect_score(config)
    agent, rewards, wall_hits, ep_steps, seconds = train(config, seed, EPISODES[method])
    reached, steps, test_reward, path = greedy_test(agent, config)
    return dict(method=method, overrides=overrides, seed=seed, rewards=rewards,
                final_avg50=float(np.mean(rewards[-50:])), learn_ep=episodes_to_learn(rewards, best),
                time_s=seconds, reached=bool(reached), test_steps=steps, test_reward=test_reward,
                path=path, total_steps=int(np.sum(ep_steps)))


def key_of(method, overrides):
    full = dict(BASE[method])
    full.update(overrides)
    return method, tuple(sorted(full.items()))


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def summarise(runs):
    learned = [r["learn_ep"] for r in runs if r["learn_ep"] is not None]
    goal_steps = [r["test_steps"] for r in runs if r["reached"]]
    return dict(
        final=mean_std([r["final_avg50"] for r in runs]),
        learn=mean_std(learned), n_learned=len(learned),
        time=mean_std([r["time_s"] for r in runs]),
        n_goal=sum(r["reached"] for r in runs), steps=mean_std(goal_steps),
        n=len(runs),
    )


def plot_sweep(name, label, values, runs_by_value, best, method):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
                         "xtick.color": "#444444", "ytick.color": "#444444"})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ax = axes[0]
    for i, v in enumerate(values):
        curves = np.array([moving_average(r["rewards"]) for r in runs_by_value[v]])
        x = np.arange(10, curves.shape[1] + 10)
        m, s = curves.mean(0), curves.std(0)
        ax.plot(x, m, color=COLORS[i], linewidth=1.2, label=f"{v:g}")
        ax.fill_between(x, m - s, m + s, color=COLORS[i], alpha=0.15, linewidth=0)
    ax.axhline(best, color="#333333", linestyle="--", linewidth=1, label=f"best possible ({best})")
    ax.set_ylim(-60, best + 4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("10-episode average reward")
    ax.set_title(f"{method}: {label}", fontsize=7.5)
    ax.legend(title=label, fontsize=6, title_fontsize=6, loc="lower right", frameon=False)
    for j, (metric, ylabel) in enumerate([("final_avg50", "Avg reward, last 50 episodes"), ("time_s", "Training time (s)")]):
        ax = axes[j + 1]
        ms = [mean_std([r[metric] for r in runs_by_value[v]]) for v in values]
        ax.bar(range(len(values)), [m for m, _ in ms], yerr=[s for _, s in ms], color=COLORS[:len(values)],
               width=0.6, capsize=2, error_kw={"elinewidth": 0.7, "ecolor": "#333333"})
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v:g}" for v in values])
        ax.set_xlabel(label)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="#888888", linewidth=0.6)
    for ax in axes:
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{name}.png", dpi=300)
    plt.close(fig)


def count_shortest_paths(config):
    """Number of different shortest paths from start to goal (counting over BFS layers)."""
    n, maze = config.MAZE_SIZE, config.maze_setting
    start, end = tuple(config.Start_point), tuple(config.End_point)
    dist, ways, queue = {start: 0}, {start: 1}, [start]
    for x, y in queue:
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nxt = (x + dx, y + dy)
            if 0 <= nxt[0] < n and 0 <= nxt[1] < n and maze[nxt] == 0:
                if nxt not in dist:
                    dist[nxt], ways[nxt] = dist[(x, y)] + 1, 0
                    queue.append(nxt)
                if dist[nxt] == dist[(x, y)] + 1:
                    ways[nxt] += ways[(x, y)]
    return ways.get(end, 0)


def plot_maze_layout():
    """Figure of the maze: coordinates, state numbers, walls, start, goal and one BFS shortest path."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    config = build_config("QLearning")
    n, maze = config.MAZE_SIZE, config.maze_setting
    best, path = perfect_score(config)
    fig, ax = plt.subplots(figsize=(2.6, 2.75))
    for x in range(n):
        for y in range(n):
            face = "#222222" if maze[x, y] else "white"
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=face, edgecolor="#555555", linewidth=0.8))
            if maze[x, y]:
                ax.text(x, y, "wall", fontsize=6, color="white", ha="center", va="center")
            else:
                ax.text(x - 0.42, y - 0.42, f"s{x * n + y}", fontsize=5, color="#777777", ha="left", va="top")
    ax.add_patch(Rectangle((-0.45, -0.45), 0.9, 0.9, fill=False, edgecolor="#e34948", linewidth=2.2))
    ax.add_patch(Circle(tuple(config.End_point), 0.42, facecolor="#f2c200", edgecolor="none"))
    ax.plot([p[0] for p in path], [p[1] for p in path], color="#2a78d6", linewidth=1.6, marker="o", markersize=3,
            label=f"a BFS shortest path ({len(path) - 1} moves)")
    ax.text(0, 0.28, "start", fontsize=6, ha="center", color="#e34948")
    ax.text(4, 4.32, "goal", fontsize=6, ha="center", color="#222222")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("x  (right = x+1)", fontsize=7)
    ax.set_ylabel("y  (down = y+1)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig("maze_layout.png", dpi=300)
    plt.close(fig)
    walls = [tuple(int(v) for v in w) for w in np.argwhere(maze == 1)]
    print(f"Saved maze_layout.png  (walls: {walls}, shortest path {len(path) - 1} moves, "
          f"best reward {best}, {count_shortest_paths(config)} different shortest paths)")


def make_rows(method, parameter, value, runs):
    rows = []
    for r in runs:
        rows.append(dict(row_type="seed", method=method, parameter=parameter, value=value, seed=r["seed"],
                         episodes=EPISODES[method],
                         final_avg50=f"{r['final_avg50']:.2f}", final_avg50_std="",
                         episodes_to_learn=r["learn_ep"] if r["learn_ep"] is not None else "did not learn",
                         episodes_to_learn_std="", n_learned=int(r["learn_ep"] is not None),
                         train_time_s=f"{r['time_s']:.2f}", train_time_s_std="",
                         goal_reached=int(r["reached"]), greedy_steps=r["test_steps"] if r["reached"] else "",
                         greedy_steps_std="", n_seeds=1, total_env_steps=r["total_steps"]))
    s = summarise(runs)
    rows.append(dict(row_type="mean_std", method=method, parameter=parameter, value=value, seed="0,1,2",
                     episodes=EPISODES[method],
                     final_avg50=f"{s['final'][0]:.2f}", final_avg50_std=f"{s['final'][1]:.2f}",
                     episodes_to_learn=f"{s['learn'][0]:.1f}" if s["learn"][0] is not None else "did not learn",
                     episodes_to_learn_std=f"{s['learn'][1]:.1f}" if s["learn"][0] is not None else "",
                     n_learned=s["n_learned"],
                     train_time_s=f"{s['time'][0]:.2f}", train_time_s_std=f"{s['time'][1]:.2f}",
                     goal_reached=s["n_goal"],
                     greedy_steps=f"{s['steps'][0]:.1f}" if s["steps"][0] is not None else "",
                     greedy_steps_std=f"{s['steps'][1]:.1f}" if s["steps"][0] is not None else "",
                     n_seeds=s["n"], total_env_steps=int(np.mean([r["total_steps"] for r in runs]))))
    return rows


def main(methods, parts):
    jobs, seen = [], set()

    def add(method, overrides):
        k = key_of(method, overrides)
        if k not in seen:
            seen.add(k)
            jobs.extend((method, overrides, seed) for seed in SEEDS)

    for method in methods:
        if "sweeps" in parts:
            for _, field, _, values in SWEEPS[method]:
                for v in values:
                    add(method, {field: v})
        if "combined" in parts:
            for _, overrides in COMBINED[method]:
                add(method, overrides)
    jobs.sort(key=lambda j: j[0] != "DQN")      # longest jobs (DQN) first so the pool stays busy
    print(f"Running {len(jobs)} training runs on {N_WORKERS} workers ...", flush=True)
    t0 = time.time()
    results = {}
    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(run_job, jobs), 1):
            results.setdefault(key_of(res["method"], res["overrides"]), []).append(res)
            print(f"[{i}/{len(jobs)}] {res['method']} {res['overrides']} seed={res['seed']}: "
                  f"last50={res['final_avg50']:.2f} learn_ep={res['learn_ep']} time={res['time_s']:.1f}s "
                  f"goal={res['reached']} steps={res['test_steps']}", flush=True)
    print(f"All runs finished in {time.time() - t0:.0f} s")

    rows, rerun = [], set()
    for method in methods:
        best, _ = perfect_score(build_config(method))
        if "sweeps" in parts:
            for name, field, label, values in SWEEPS[method]:
                runs_by_value = {}
                for v in values:
                    runs_by_value[v] = sorted(results[key_of(method, {field: v})], key=lambda r: r["seed"])
                    rows += make_rows(method, field, v, runs_by_value[v])
                rerun.add((method, field))
                plot_sweep(name, label, values, runs_by_value, best, method)
                print(f"Saved {name}.png")
        if "combined" in parts:
            for label, overrides in COMBINED[method]:
                runs = sorted(results[key_of(method, overrides)], key=lambda r: r["seed"])
                rows += make_rows(method, "combined", label, runs)
            rerun.add((method, "combined"))
    if "sweeps" in parts:
        plot_maze_layout()

    # Keep the rows of the CSV that were not re-run in this call
    out = "results_summary.csv"
    fields = list(rows[0].keys())
    if os.path.exists(out):
        with open(out, newline="") as f:
            rows = [r for r in csv.DictReader(f) if (r["method"], r["parameter"]) not in rerun] + rows
    order = {"QLearning": 0, "DQN": 1}
    rows.sort(key=lambda r: (order[r["method"]], r["parameter"] == "combined"))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    methods = [m for a, m in [("ql", "QLearning"), ("dqn", "DQN")] if a in args] or ["QLearning", "DQN"]
    parts = [p for p in ("sweeps", "combined") if p in args] or ["sweeps", "combined"]
    main(methods, parts)
