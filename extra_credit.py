"""
Extra credit experiments EC1-EC6 (all extra credit code lives in this one file).

Usage:
    python extra_credit.py                 # run all six items
    python extra_credit.py ec2 ec5         # run only some items

    EC1  Policy arrows + value heatmaps (Q-Learning and DQN)    -> ec_policy_qlearning.png, ec_policy_dqn.png
    EC2  Double DQN vs DQN                                      -> ec_double_dqn.png
    EC3  SARSA vs Q-Learning                                    -> ec_sarsa.png
    EC4  Epsilon schedules (x0.995, x0.98, linear over 200 ep.) -> ec_epsilon_schedule.png
    EC5  Bigger 10x10 maze (scaling test)                       -> ec_big_maze.png
    EC6  Learning curves with +/- std bands                     -> ec_learning_curves_band.png

Every experiment uses 3 seeds (0, 1, 2); the seeds are set after the maze is built.
All numbers are written to extra_credit_summary.csv (one row per metric).
main.py is not changed by this file; Double DQN and SARSA are subclasses defined here.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import sys
import csv
import time
from multiprocessing import Pool

import numpy as np
import torch

import run_experiments as R
from DQN import QNetwork
from QLearning import QTable
from utils import bfs_shortest_path

SEEDS = R.SEEDS
EPISODES = 500
N_WORKERS = 4
COLORS = R.COLORS
ACTION_MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # up, right, down, left  (x, y)
ARROWS = ["↑", "→", "↓", "←"]

# Best settings found in Task V (results_summary.csv, "combined" rows); the same values are in main.py
BEST_QL = dict(q_learning_rate=0.5, discount_factor=0.99, epsilon=0.1)
BEST_DQN = dict(learning_rate=1e-3, discount_factor=0.99, epsilon=0.1, batch_size=64, capacity=1000,
                target_network_update_frequency=10, hidden_size=64)

# EC5: bigger maze (fixed seed, checked with BFS below)
BIG_SIZE, BIG_WALLS, BIG_SEED = 10, 30, 11    # 30 random walls -> 26 distinct cells; BFS path = 18 moves
BIG_EPISODES = 600


# ----------------------------------------------------------------------------------
# New agents
# ----------------------------------------------------------------------------------
class DoubleQNetwork(QNetwork):
    """Double DQN: the online network chooses the next action, the target network scores it."""
    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        current_q = self.run_state(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.run_state(next_states).argmax(1, keepdim=True)
            next_q = self.evaluate_target_net(next_states).gather(1, next_actions).squeeze(1)
            expected_q = rewards + self.discount_factor * next_q * (1 - dones)
        loss = self.config.loss_fn(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SarsaTable(QTable):
    """SARSA: the target uses the action the agent will really take next (on-policy)."""
    def __init__(self, config):
        super().__init__(config)
        self.next_action = None

    def reset_episode(self):
        self.next_action = None

    def choose_action(self, state):
        if self.next_action is not None:          # the action already picked for the update
            action, self.next_action = self.next_action, None
            return action
        return super().choose_action(state)

    def update_q_table(self, pos, action, new_pos, reward, done_flags):
        state = pos[0] * self.MAZE_SIZE + pos[1]
        next_state = new_pos[0] * self.MAZE_SIZE + new_pos[1]
        predict = self.q_table[state, action]
        if done_flags:
            target = reward
        else:
            self.next_action = QTable.choose_action(self, next_state)   # epsilon-greedy a'
            target = reward + self.discount_factor * self.q_table[next_state, self.next_action]
        self.q_table[state, action] = predict + self.learning_rate * (target - predict)


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
def optimal_q(config):
    """True Q* of the maze by value iteration (known, deterministic environment)."""
    n, gamma, maze = config.MAZE_SIZE, config.discount_factor, config.maze_setting
    q = np.zeros((n * n, 4))
    goal = tuple(config.End_point)
    for _ in range(2000):
        v = q.max(1)
        new_q = np.zeros_like(q)
        for x in range(n):
            for y in range(n):
                if maze[x, y] == 1 or (x, y) == goal:
                    continue
                for a, (dx, dy) in enumerate(ACTION_MOVES):
                    nx, ny = min(max(x + dx, 0), n - 1), min(max(y + dy, 0), n - 1)
                    if (nx, ny) == goal:
                        new_q[x * n + y, a] = 10
                    elif maze[nx, ny] == 1:
                        new_q[x * n + y, a] = -10 + gamma * v[x * n + y]
                    else:
                        new_q[x * n + y, a] = -1 + gamma * v[nx * n + ny]
        if np.max(np.abs(new_q - q)) < 1e-9:
            break
        q = new_q
    return new_q


def free_states(config):
    n, goal = config.MAZE_SIZE, tuple(config.End_point)
    return [x * n + y for x in range(n) for y in range(n)
            if config.maze_setting[x, y] == 0 and (x, y) != goal]


def all_q(agent, config):
    n = config.MAZE_SIZE
    if isinstance(agent, QTable):
        return agent.q_table.copy()
    with torch.no_grad():
        return agent.run_state(torch.eye(n * n)).numpy()


def policy_accuracy(q_learned, q_star, states):
    """Share of free cells whose greedy action is optimal under Q*."""
    good = [q_star[s, np.argmax(q_learned[s])] >= q_star[s].max() - 1e-6 for s in states]
    return float(np.mean(good))


def wall_adjacent_cells(path, maze):
    n = maze.shape[0]
    count = 0
    for x, y in path[1:-1]:
        if any(0 <= x + dx < n and 0 <= y + dy < n and maze[x + dx, y + dy] == 1 for dx, dy in ACTION_MOVES):
            count += 1
    return count


def style(ax):
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def band(ax, curves, label, color, x0=10):
    curves = np.array([R.moving_average(c) for c in curves])
    x = np.arange(x0, curves.shape[1] + x0)
    m, s = curves.mean(0), curves.std(0)
    ax.plot(x, m, color=color, linewidth=1.2, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, linewidth=0)


def plt_setup():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
                         "xtick.color": "#444444", "ytick.color": "#444444"})
    return plt


ROWS = []


def record(experiment, variant, metric, values):
    vals = [v for v in values if v is not None and not isinstance(v, str)]   # text (paths) has no mean
    missing = "did not learn" if metric == "episodes_to_learn" else "goal not reached"
    ROWS.append(dict(experiment=experiment, variant=variant, metric=metric,
                     seed0=_f(values[0], missing), seed1=_f(values[1], missing), seed2=_f(values[2], missing),
                     mean=_f(np.mean(vals)) if vals else "", std=_f(np.std(vals)) if vals else "",
                     n_valid=len(vals)))


def _f(v, missing="did not learn"):
    if v is None:
        return missing
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    return f"{v:.4f}" if isinstance(v, float) else v


def basic_metrics(experiment, variant, runs, best):
    record(experiment, variant, "avg_reward_last50", [float(np.mean(r["rewards"][-50:])) for r in runs])
    record(experiment, variant, "episodes_to_learn", [R.episodes_to_learn(r["rewards"], best) for r in runs])
    record(experiment, variant, "train_time_s", [r["time_s"] for r in runs])
    record(experiment, variant, "greedy_reached_goal", [r["reached"] for r in runs])
    record(experiment, variant, "greedy_steps", [r["steps"] for r in runs])


# ----------------------------------------------------------------------------------
# Generic job runner (one process per run)
# ----------------------------------------------------------------------------------
def linear_schedule(episode, eps, start=0.9, end=0.01, length=200):
    return max(end, start - (start - end) * (episode + 1) / length)


SCHEDULES = {
    "x0.995 per episode (default)": lambda ep, eps: max(0.01, 0.995 * eps),
    "x0.98 per episode (faster)": lambda ep, eps: max(0.01, 0.98 * eps),
    "linear to 0.01 over 200 episodes": linear_schedule,
}

AGENTS = {"QTable": None, "QNetwork": None, "DoubleQNetwork": DoubleQNetwork, "SarsaTable": SarsaTable}


def run_job(job):
    """job = dict(method, agent, settings, seed, episodes, maze, schedule, track_q)."""
    maze_kw = job.get("maze", {})
    config = R.build_config(job["method"], **maze_kw, **job["settings"])
    q_star = optimal_q(config) if job.get("track_q") else None
    states = free_states(config)
    q_trace = []

    def on_end(agent, episode):
        if q_star is not None and (episode + 1) % 10 == 0:
            q = all_q(agent, config)
            diff = q[states] - q_star[states]
            q_trace.append((episode + 1, float(diff.mean()), float(np.abs(diff).mean())))

    schedule = SCHEDULES.get(job.get("schedule"))
    agent, rewards, hits, steps, seconds = R.train(config, job["seed"], job["episodes"],
                                                   agent_cls=AGENTS[job["agent"]], epsilon_update=schedule,
                                                   on_episode_end=on_end)
    reached, n_steps, test_reward, path = R.greedy_test(agent, config, max_steps=4 * config.MAZE_SIZE ** 2)
    q = all_q(agent, config)
    q_s = optimal_q(config)
    return dict(job=job, rewards=rewards, wall_hits=hits, time_s=seconds, reached=bool(reached),
                steps=n_steps if reached else None, path=path, q=q, q_trace=q_trace,
                policy_acc=policy_accuracy(q, q_s, states),
                value_mae=float(np.abs(q[states].max(1) - q_s[states].max(1)).mean()),
                total_steps=int(np.sum(steps)))


def run_many(jobs):
    with Pool(N_WORKERS) as pool:
        results = pool.map(run_job, jobs)
    return results


def jobs_for(method, agent, settings, episodes=EPISODES, **extra):
    return [dict(method=method, agent=agent, settings=settings, seed=s, episodes=episodes, **extra) for s in SEEDS]


# ----------------------------------------------------------------------------------
# EC1 - policy and value pictures
# ----------------------------------------------------------------------------------
def draw_policy(ax, q, config, title, vmin, vmax, cmap):
    n, maze, goal = config.MAZE_SIZE, config.maze_setting, tuple(config.End_point)
    values = np.full((n, n), np.nan)          # image[row=y, col=x]
    for x in range(n):
        for y in range(n):
            if maze[x, y] == 0 and (x, y) != goal:
                values[y, x] = q[x * n + y].max()
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    for x in range(n):
        for y in range(n):
            if maze[x, y] == 1:
                ax.add_patch(plt_rect(x, y, "#222222"))
            elif (x, y) == goal:
                ax.add_patch(plt_rect(x, y, "#f2c200"))
                ax.text(x, y, "G", ha="center", va="center", fontsize=8, fontweight="bold", color="#222222")
            else:
                v = values[y, x]
                light = (v - vmin) / (vmax - vmin) > 0.55
                ax.text(x, y - 0.12, ARROWS[int(np.argmax(q[x * n + y]))], ha="center", va="center",
                        fontsize=9 if n <= 5 else 6, color="white" if light else "#111111")
                if n <= 5:
                    ax.text(x, y + 0.3, f"{v:.1f}", ha="center", va="center", fontsize=5,
                            color="white" if light else "#111111")
    sx, sy = config.Start_point
    ax.add_patch(plt_rect(sx, sy, "none", edge="#e34948"))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=6.5)


def plt_rect(x, y, face, edge=None):
    from matplotlib.patches import Rectangle
    return Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=face, edgecolor=edge or face, linewidth=2 if edge else 0)


def ec1(results_ql, results_dqn):
    plt = plt_setup()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    for name, results, label in [("ec_policy_qlearning", results_ql, "Q-Learning"), ("ec_policy_dqn", results_dqn, "DQN")]:
        config = R.build_config(results[0]["job"]["method"], **results[0]["job"]["settings"])
        q_star = optimal_q(config)
        states = free_states(config)
        vmin, vmax = min(q_star[states].max(1).min(), min(r["q"][states].max(1).min() for r in results)), 10
        fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.3))
        fig.subplots_adjust(wspace=0.38)
        draw_policy(axes[0], q_star, config, "Optimal (value iteration)\n ", vmin, vmax, cmap)
        for ax, r in zip(axes[1:], results):
            draw_policy(ax, r["q"], config, f"{label}, seed {r['job']['seed']}\n"
                        f"{100 * r['policy_acc']:.0f}% optimal, error {r['value_mae']:.2f}",
                        vmin, vmax, cmap)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.03).set_label("max Q(s, a)", fontsize=6.5)
        fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        record("EC1", label, "policy_optimal_action_share", [r["policy_acc"] for r in results])
        record("EC1", label, "value_mae_vs_optimal", [r["value_mae"] for r in results])
        print(f"Saved {name}.png")


# ----------------------------------------------------------------------------------
# EC2 - Double DQN
# ----------------------------------------------------------------------------------
def ec2():
    plt = plt_setup()
    runs = {"DQN": run_many(jobs_for("DQN", "QNetwork", BEST_DQN, track_q=True)),
            "Double DQN": run_many(jobs_for("DQN", "DoubleQNetwork", BEST_DQN, track_q=True))}
    best, _ = R.perfect_score(R.build_config("DQN"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
    for i, (name, res) in enumerate(runs.items()):
        band(axes[0], [r["rewards"] for r in res], name, COLORS[i])
        ep = np.array([t[0] for t in res[0]["q_trace"]])
        bias = np.array([[t[1] for t in r["q_trace"]] for r in res])
        mae = np.array([[t[2] for t in r["q_trace"]] for r in res])
        for k, seed_bias in enumerate(bias):                  # one thin line per seed (shows short spikes)
            axes[1].plot(ep, seed_bias, color=COLORS[i], linewidth=0.8, alpha=0.8,
                         label=f"{name} (each seed)" if k == 0 else None)
        axes[2].plot(ep, mae.mean(0), color=COLORS[i], linewidth=1.2, label=name)
        axes[2].fill_between(ep, mae.mean(0) - mae.std(0), mae.mean(0) + mae.std(0), color=COLORS[i],
                             alpha=0.18, linewidth=0)
        basic_metrics("EC2", name, res, best)
        record("EC2", name, "q_bias_final(mean Qpred-Q*)", [r["q_trace"][-1][1] for r in res])
        record("EC2", name, "q_abs_error_final", [r["q_trace"][-1][2] for r in res])
        record("EC2", name, "q_abs_error_avg_over_training", [float(np.mean([t[2] for t in r["q_trace"]])) for r in res])
        record("EC2", name, "q_bias_max_over_training", [float(np.max([t[1] for t in r["q_trace"]])) for r in res])
    axes[0].axhline(best, color="#333333", linestyle="--", linewidth=1, label=f"best possible ({best})")
    axes[0].set_ylim(-60, best + 4)
    axes[0].set(xlabel="Episode", ylabel="10-episode average reward", title="Learning curve (mean ± std)")
    axes[1].axhline(0, color="#888888", linewidth=0.8)
    axes[1].set_ylim(-12, 5)
    axes[1].set(xlabel="Episode", ylabel="mean of Q_pred − Q*", title="Bias of predicted Q-values (zoom)")
    axes[2].set(xlabel="Episode", ylabel="mean |Q_pred − Q*|", title="Error of predicted Q-values")
    for ax in axes:
        style(ax)
        ax.title.set_fontsize(7.5)
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig("ec_double_dqn.png", dpi=300)
    plt.close(fig)
    print("Saved ec_double_dqn.png")


# ----------------------------------------------------------------------------------
# EC3 - SARSA vs Q-Learning
# ----------------------------------------------------------------------------------
def ec3():
    plt = plt_setup()
    settings = dict(BEST_QL, epsilon=0.9)     # keep exploration on so the two methods can differ
    runs = {"Q-Learning": run_many(jobs_for("QLearning", "QTable", settings)),
            "SARSA": run_many(jobs_for("QLearning", "SarsaTable", settings))}
    config = R.build_config("QLearning", **settings)
    best, _ = R.perfect_score(config)
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.3), gridspec_kw={"width_ratios": [2, 1.6, 1, 1]})
    for i, (name, res) in enumerate(runs.items()):
        band(axes[0], [r["rewards"] for r in res], name, COLORS[i])
        band(axes[1], [r["wall_hits"] for r in res], name, COLORS[i])
        basic_metrics("EC3", name, res, best)
        record("EC3", name, "wall_hits_total_training", [int(np.sum(r["wall_hits"])) for r in res])
        record("EC3", name, "wall_adjacent_cells_on_greedy_path",
               [wall_adjacent_cells(r["path"], config.maze_setting) if r["reached"] else None for r in res])
        record("EC3", name, "mean_maxQ_wall_adjacent_cells", [mean_q_near_walls(r["q"], config) for r in res])
        record("EC3", name, "mean_maxQ_other_cells", [mean_q_near_walls(r["q"], config, near=False) for r in res])
        record("EC3", name, "greedy_path", [path_str(r["path"]) for r in res])
    axes[0].axhline(best, color="#333333", linestyle="--", linewidth=1, label=f"best possible ({best})")
    axes[0].set_ylim(-60, best + 4)
    axes[0].set(xlabel="Episode", ylabel="10-episode average reward", title="Learning curve (mean ± std)")
    axes[1].set(xlabel="Episode", ylabel="wall hits / episode (10-ep avg)", title="Wall hits during training")
    axes[1].set_ylim(0, None)
    for ax in axes[:2]:
        style(ax)
        ax.title.set_fontsize(7.5)
        ax.legend(fontsize=6, frameon=False)
    for ax, (name, res), color in zip(axes[2:], runs.items(), COLORS):
        draw_paths(ax, config, [r["path"] for r in res], color, f"{name} paths\n(3 seeds)")
    fig.tight_layout()
    fig.savefig("ec_sarsa.png", dpi=300)
    plt.close(fig)
    print("Saved ec_sarsa.png")


def mean_q_near_walls(q, config, near=True):
    n, maze = config.MAZE_SIZE, config.maze_setting
    vals = []
    for s in free_states(config):
        x, y = divmod(s, n)
        adj = any(0 <= x + dx < n and 0 <= y + dy < n and maze[x + dx, y + dy] == 1 for dx, dy in ACTION_MOVES)
        if adj == near:
            vals.append(q[s].max())
    return float(np.mean(vals))


def path_str(path):
    return " ".join(f"({x},{y})" for x, y in path)


def draw_paths(ax, config, paths, color, title):
    n, maze = config.MAZE_SIZE, config.maze_setting
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    for x in range(n):
        for y in range(n):
            if maze[x, y] == 1:
                ax.add_patch(plt_rect(x, y, "#222222"))
    gx, gy = config.End_point
    ax.add_patch(plt_rect(gx, gy, "#f2c200"))
    sx, sy = config.Start_point
    ax.add_patch(plt_rect(sx, sy, "none", edge="#e34948"))
    offsets = [-0.12, 0, 0.12]
    for k, path in enumerate(paths):
        xs = [p[0] + offsets[k % 3] for p in path]
        ys = [p[1] + offsets[k % 3] for p in path]
        ax.plot(xs, ys, color=color, linewidth=1.1, alpha=0.85, marker="o", markersize=1.8)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(color="#dddddd", linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=7)


# ----------------------------------------------------------------------------------
# EC4 - epsilon schedules
# ----------------------------------------------------------------------------------
def ec4():
    plt = plt_setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
    best, _ = R.perfect_score(R.build_config("QLearning"))
    for i, sched in enumerate(SCHEDULES):
        eps, trace = 0.9, []
        for ep in range(EPISODES):
            trace.append(eps)
            eps = SCHEDULES[sched](ep, eps)
        axes[0].plot(range(1, EPISODES + 1), trace, color=COLORS[i], linewidth=1.2, label=sched)
    for ax, method, agent, base, title in [(axes[1], "QLearning", "QTable", BEST_QL, "Q-Learning"),
                                           (axes[2], "DQN", "QNetwork", BEST_DQN, "DQN")]:
        settings = dict(base, epsilon=0.9)
        for i, sched in enumerate(SCHEDULES):
            res = run_many(jobs_for(method, agent, settings, schedule=sched))
            band(ax, [r["rewards"] for r in res], sched, COLORS[i])
            basic_metrics("EC4", f"{title} | {sched}", res, best)
        ax.axhline(best, color="#333333", linestyle="--", linewidth=1)
        ax.set_ylim(-60, best + 4)
        ax.set(xlabel="Episode", ylabel="10-episode average reward", title=f"{title} (mean ± std)")
    axes[0].set(xlabel="Episode", ylabel="epsilon", title="Epsilon schedules (start 0.9)")
    for ax in axes:
        style(ax)
        ax.title.set_fontsize(7.5)
    axes[0].legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    fig.savefig("ec_epsilon_schedule.png", dpi=300)
    plt.close(fig)
    print("Saved ec_epsilon_schedule.png")


# ----------------------------------------------------------------------------------
# EC5 - bigger maze
# ----------------------------------------------------------------------------------
BIG_MAZE = dict(maze_size=BIG_SIZE, wall_number=BIG_WALLS, wall_seed=BIG_SEED)


def ec5():
    plt = plt_setup()
    config = R.build_config("QLearning", **BIG_MAZE)
    path = bfs_shortest_path(config.maze_setting, config.Start_point, config.End_point)
    assert path is not None, "The goal of the big maze cannot be reached"
    best = 10 - (len(path) - 2)
    n_walls = int(config.maze_setting.sum())
    print(f"Big maze {BIG_SIZE}x{BIG_SIZE}: {n_walls} distinct walls, BFS path {len(path) - 1} moves, best reward {best}")
    runs = {"Q-Learning": run_many(jobs_for("QLearning", "QTable", BEST_QL, episodes=BIG_EPISODES, maze=BIG_MAZE)),
            "DQN": run_many(jobs_for("DQN", "QNetwork", BEST_DQN, episodes=BIG_EPISODES, maze=BIG_MAZE))}
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.4), gridspec_kw={"width_ratios": [1.15, 2, 0.9, 0.9]})
    draw_paths(axes[0], config, [path], "#2a78d6", f"{BIG_SIZE}x{BIG_SIZE} maze, BFS path\n({len(path) - 1} moves)")
    for i, (name, res) in enumerate(runs.items()):
        for k, r in enumerate(res):            # one line per seed: a failed seed would hide inside a mean
            sm = R.moving_average(r["rewards"])
            axes[1].plot(np.arange(10, len(sm) + 10), sm, color=COLORS[i], linewidth=0.9, alpha=0.85,
                         label=f"{name} (each seed)" if k == 0 else None)
        basic_metrics("EC5", name, res, best)
        record("EC5", name, "total_env_steps", [r["total_steps"] for r in res])
        record("EC5", name, "policy_optimal_action_share", [r["policy_acc"] for r in res])
    for metric, value in [("distinct_walls", n_walls), ("bfs_shortest_moves", len(path) - 1), ("best_reward", best)]:
        record("EC5", "maze (same for all seeds)", metric, [value] * 3)
    axes[1].axhline(best, color="#333333", linestyle="--", linewidth=1, label=f"best possible ({best})")
    axes[1].set_ylim(-700, best + 30)
    axes[1].set(xlabel="Episode", ylabel="10-episode average reward", title="Learning curves (each seed)")
    axes[1].title.set_fontsize(7.5)
    axes[1].legend(fontsize=6, frameon=False, loc="center right")
    for j, (metric, ylabel) in enumerate([("ep", "Episodes to learn"), ("time_s", "Training time (s)")]):
        ax = axes[2 + j]
        for i, (name, res) in enumerate(runs.items()):
            if metric == "ep":
                vals = [R.episodes_to_learn(r["rewards"], best) for r in res]
                ok = [v for v in vals if v is not None]
                m, s = (np.mean(ok), np.std(ok)) if ok else (0, 0)
                ax.text(i, m + s + 5, f"{len(ok)}/3 learned", ha="center", fontsize=6)
            else:
                vals = [r["time_s"] for r in res]
                m, s = np.mean(vals), np.std(vals)
                ax.text(i, m + s + 5, f"{m:.2f} s" if m < 10 else f"{m:.0f} s", ha="center", fontsize=6)
            ax.bar(i, m, yerr=s, color=COLORS[i], width=0.6, capsize=2, error_kw={"elinewidth": 0.7})
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Q-Learn.", "DQN"])
        ax.set_ylabel(ylabel)
    for ax in axes[1:]:
        style(ax)
    fig.tight_layout()
    fig.savefig("ec_big_maze.png", dpi=300)
    plt.close(fig)
    print("Saved ec_big_maze.png")


# ----------------------------------------------------------------------------------
# EC6 - learning curves with error bands
# ----------------------------------------------------------------------------------
def ec6(results_ql, results_dqn):
    plt = plt_setup()
    best, _ = R.perfect_score(R.build_config("QLearning"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.3))
    for ax, ylim, title in [(axes[0], None, "Full range"), (axes[1], (-30, best + 3), "Zoom near the best reward")]:
        for i, (name, res) in enumerate([("Q-Learning", results_ql), ("DQN", results_dqn)]):
            band(ax, [r["rewards"] for r in res], name, COLORS[i])
        ax.axhline(best, color="#333333", linestyle="--", linewidth=1, label=f"best possible ({best})")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set(xlabel="Episode", ylabel="10-episode average reward", title=f"{title} (mean ± std, 3 seeds)")
        style(ax)
        ax.title.set_fontsize(7.5)
        ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("ec_learning_curves_band.png", dpi=300)
    plt.close(fig)
    for name, res in [("Q-Learning", results_ql), ("DQN", results_dqn)]:
        basic_metrics("EC6", name, res, best)
    print("Saved ec_learning_curves_band.png")


def main(items):
    t0 = time.time()
    results_ql = results_dqn = None
    if "ec1" in items or "ec6" in items:
        results_ql = run_many(jobs_for("QLearning", "QTable", BEST_QL))
        results_dqn = run_many(jobs_for("DQN", "QNetwork", BEST_DQN))
    if "ec1" in items:
        ec1(results_ql, results_dqn)
    if "ec2" in items:
        ec2()
    if "ec3" in items:
        ec3()
    if "ec4" in items:
        ec4()
    if "ec5" in items:
        ec5()
    if "ec6" in items:
        ec6(results_ql, results_dqn)

    out = "extra_credit_summary.csv"
    rows = ROWS
    fields = ["experiment", "variant", "metric", "seed0", "seed1", "seed2", "mean", "std", "n_valid"]
    done = {r["experiment"] for r in rows}
    if os.path.exists(out):
        with open(out, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["experiment"] not in done] + rows
    rows.sort(key=lambda r: r["experiment"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    chosen = [a.lower() for a in sys.argv[1:]] or ["ec1", "ec2", "ec3", "ec4", "ec5", "ec6"]
    main(chosen)
