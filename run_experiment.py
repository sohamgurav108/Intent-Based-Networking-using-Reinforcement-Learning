"""
run_experiment.py
Runs every strategy on the simulated network and draws the graphs
for the report. It repeats the experiment with 20 different random
seeds, so the result does not depend on luck.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from network import PATHS
from simulator import Simulator
from intents import INTENTS, reward, best_path
from agent import RandomStrategy, ShortestPathStrategy, LowestLatencyStrategy, QLearningAgent

NUM_SEEDS = 20
STEPS = 600
FAILURE_START, FAILURE_END = 300, 450
STRATEGY_NAMES = ["Random", "Shortest path", "Lowest latency", "RL agent (Q-learning)"]
COLORS = {"Random": "#9e9e9e", "Shortest path": "#e69f00", "Lowest latency": "#009e73",
          "RL agent (Q-learning)": "#d62728", "Perfect (oracle)": "#000000"}
OUT_DIR = "results"


# ---------------------------------------------------------------
# One complete run (one seed). Returns the reward of every
# strategy, for every intent, at every step.
# ---------------------------------------------------------------
def run_once(seed):
    sim = Simulator(seed=seed, total_steps=STEPS,
                    failure_start=FAILURE_START, failure_end=FAILURE_END)
    strategies = [RandomStrategy(seed), ShortestPathStrategy(),
                  LowestLatencyStrategy(), QLearningAgent(list(INTENTS), seed=seed)]

    names = STRATEGY_NAMES + ["Perfect (oracle)"]
    rewards = {n: {i: [] for i in INTENTS} for n in names}
    rl_choice = {i: [] for i in INTENTS}      # path chosen by the RL agent
    best_choice = {i: [] for i in INTENTS}    # the perfect path

    while not sim.done():
        sim.step()
        info = sim.true_metrics()
        for intent in INTENTS:
            best_name, best_value = best_path(intent, info)
            rewards["Perfect (oracle)"][intent].append(best_value)
            best_choice[intent].append(best_name)

            for s in strategies:
                path = s.choose(intent, info)
                s.update(intent, path, reward(intent, sim.measure(path)))   # learns from noisy data
                rewards[s.name][intent].append(reward(intent, info[path]))  # scored on the truth
                if s.name == "RL agent (Q-learning)":
                    rl_choice[intent].append(path)
    return rewards, rl_choice, best_choice


def rolling(x, window=15):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def shade_failure(ax, offset=0):
    ax.axvspan(FAILURE_START - offset, FAILURE_END - offset, color="red", alpha=0.08)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Running {NUM_SEEDS} experiments of {STEPS} steps each ...")

    all_rewards = []            # one entry per seed
    first_rl_choice = first_best_choice = None
    for seed in range(1, NUM_SEEDS + 1):
        rewards, rl_choice, best_choice = run_once(seed)
        all_rewards.append(rewards)
        if seed == 1:
            first_rl_choice, first_best_choice = rl_choice, best_choice

    names = STRATEGY_NAMES + ["Perfect (oracle)"]
    intent_list = list(INTENTS)

    # mean reward per (strategy, intent) for each seed
    per_seed = {n: np.array([[np.mean(r[n][i]) for i in intent_list] for r in all_rewards])
                for n in names}                      # shape: seeds x intents
    overall = {n: per_seed[n].mean(axis=1) for n in names}   # per seed

    # ---------------- table in the terminal ----------------
    print("\nAverage reward (mean over %d seeds, higher is better)\n" % NUM_SEEDS)
    print(f"{'Strategy':<24}" + "".join(f"{i[:12]:>14}" for i in intent_list) + f"{'OVERALL':>16}")
    for n in names:
        m = per_seed[n].mean(axis=0)
        row = f"{n:<24}" + "".join(f"{v:>14.3f}" for v in m)
        row += f"{overall[n].mean():>10.3f} ±{overall[n].std():.3f}"
        print(row)

    oracle = overall["Perfect (oracle)"].mean()
    print("\nShare of the perfect score reached:")
    for n in STRATEGY_NAMES:
        print(f"  {n:<24}{100 * overall[n].mean() / oracle:>6.1f} %")

    # ---------------- Graph 1: reward over time ----------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in names:
        # average over seeds and intents at every step
        series = np.mean([[r[n][i] for i in intent_list] for r in all_rewards], axis=(0, 1))
        smooth = rolling(series)
        x = np.arange(len(smooth)) + 15
        ax.plot(x, smooth, label=n, color=COLORS[n], linewidth=2,
                linestyle="--" if n.startswith("Perfect") else "-")
    shade_failure(ax)
    ax.text((FAILURE_START + FAILURE_END) / 2, ax.get_ylim()[1] * 0.97, "link R1-R3 down",
            ha="center", va="top", fontsize=9, color="red")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Average reward (smoothed)")
    ax.set_title("Reward over time (mean of %d runs and 5 intents)" % NUM_SEEDS)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "1_reward_over_time.png"), dpi=150)

    # ---------------- Graph 2: bar chart per intent ----------------
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [i.replace("_", "\n") for i in intent_list] + ["OVERALL"]
    x = np.arange(len(labels))
    width = 0.16
    for k, n in enumerate(names):
        means = list(per_seed[n].mean(axis=0)) + [overall[n].mean()]
        stds = list(per_seed[n].std(axis=0)) + [overall[n].std()]
        ax.bar(x + (k - 2) * width, means, width, yerr=stds, capsize=2,
               label=n, color=COLORS[n])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average reward")
    ax.set_title("Average reward per intent (error bars: spread across seeds)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "2_reward_per_intent.png"), dpi=150)

    # ---------------- Graph 3: which path did the agent choose ----------------
    show = ["gaming", "file_transfer"]
    path_names = list(PATHS)
    fig, axes = plt.subplots(len(show), 1, figsize=(10, 6), sharex=True)
    for ax, intent in zip(axes, show):
        t = np.arange(1, STEPS + 1)
        best_y = [path_names.index(p) for p in first_best_choice[intent]]
        rl_y = [path_names.index(p) for p in first_rl_choice[intent]]
        ax.step(t, best_y, where="mid", color="black", alpha=0.5, label="Perfect choice")
        ax.scatter(t, rl_y, s=6, color=COLORS["RL agent (Q-learning)"], label="RL agent's choice")
        shade_failure(ax)
        ax.set_yticks(range(len(path_names)))
        ax.set_yticklabels(path_names)
        ax.set_ylabel(intent)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Path chosen over time (shaded = link R1-R3 down)")
    axes[-1].set_xlabel("Time step")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "3_path_choices.png"), dpi=150)

    print(f"\nGraphs saved in the '{OUT_DIR}' folder:")
    for f in sorted(os.listdir(OUT_DIR)):
        print("  ", f)
    plt.show()