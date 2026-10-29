"""
agent.py
The learning agent (Q-learning) and the simple strategies
we compare it against.

Every strategy has the same two methods:
  choose(intent, info)  -> returns the name of a path
  update(intent, path, reward)  -> learns from the result (only the agent uses it)
'info' is the true metrics of all paths. Only the rule-based baselines
are allowed to look at it. The learning agent never does.
"""

import random
from network import PATHS


# ---------------------------------------------------------------
# BASELINE 1: Random
# Picks any path. This is the "no intelligence" reference.
# ---------------------------------------------------------------
class RandomStrategy:
    name = "Random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def choose(self, intent, info):
        return self.rng.choice(list(PATHS))

    def update(self, intent, path, reward):
        pass


# ---------------------------------------------------------------
# BASELINE 2: Shortest path (fewest hops)
# A fixed rule, like classic routing. It never changes its mind.
# ---------------------------------------------------------------
class ShortestPathStrategy:
    name = "Shortest path"

    def choose(self, intent, info):
        return min(PATHS, key=lambda p: (len(PATHS[p]), p))

    def update(self, intent, path, reward):
        pass


# ---------------------------------------------------------------
# BASELINE 3: Lowest latency
# A smarter rule: look at the current numbers and take the working
# path with the lowest delay. It ignores what the user wants.
# ---------------------------------------------------------------
class LowestLatencyStrategy:
    name = "Lowest latency"

    def choose(self, intent, info):
        working = {p: m for p, m in info.items() if m is not None}
        if not working:
            return "P1"
        return min(working, key=lambda p: working[p]["latency"])

    def update(self, intent, path, reward):
        pass


# ---------------------------------------------------------------
# THE LEARNING AGENT: Q-learning (a bandit style)
#
# It keeps a table Q[intent][path] = "how good has this path been
# for this intent". It never sees the network numbers. It only
# tries a path and gets a reward.
#
#   choose : usually the path with the highest Q (exploit),
#            sometimes a random path (explore), with chance epsilon
#   update : Q = Q + alpha * (reward - Q)
#            moves Q a little toward the reward just received
#
# Because alpha stays constant, old results slowly fade, so the
# agent can follow a network that keeps changing.
# ---------------------------------------------------------------
class QLearningAgent:
    name = "RL agent (Q-learning)"

    def __init__(self, intents, alpha=0.2, epsilon_start=0.5,
                 epsilon_decay=0.98, epsilon_min=0.05, seed=0):
        self.rng = random.Random(seed)
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # optimistic start (1.0): every path looks good until tried,
        # so the agent is tempted to try them all
        self.q = {i: {p: 1.0 for p in PATHS} for i in intents}
        self.epsilon = {i: epsilon_start for i in intents}

    def choose(self, intent, info):
        # explore: try a random path
        if self.rng.random() < self.epsilon[intent]:
            path = self.rng.choice(list(PATHS))
        # exploit: take the path with the best Q so far
        else:
            best = max(self.q[intent].values())
            top = [p for p, v in self.q[intent].items() if v == best]
            path = self.rng.choice(top)

        # explore a little less each time
        self.epsilon[intent] = max(self.epsilon_min,
                                   self.epsilon[intent] * self.epsilon_decay)
        return path

    def update(self, intent, path, reward):
        old = self.q[intent][path]
        self.q[intent][path] = old + self.alpha * (reward - old)


# ---------------------------------------------------------------
# TEST: a quick run of all strategies on the simulator
# (the full experiment with graphs comes in the next step)
# ---------------------------------------------------------------
if __name__ == "__main__":
    from simulator import Simulator
    from intents import INTENTS, reward, best_path

    strategies = [
        RandomStrategy(seed=1),
        ShortestPathStrategy(),
        LowestLatencyStrategy(),
        QLearningAgent(list(INTENTS), seed=1),
    ]

    sim = Simulator(seed=1)
    totals = {s.name: {i: 0.0 for i in INTENTS} for s in strategies}
    totals["Perfect (oracle)"] = {i: 0.0 for i in INTENTS}

    while not sim.done():
        sim.step()
        info = sim.true_metrics()
        for intent in INTENTS:
            # the perfect answer, for reference
            _, best_value = best_path(intent, info)
            totals["Perfect (oracle)"][intent] += best_value

            for s in strategies:
                path = s.choose(intent, info)
                # the agent learns from a noisy measurement ...
                s.update(intent, path, reward(intent, sim.measure(path)))
                # ... but is scored on the true result
                totals[s.name][intent] += reward(intent, info[path])

    print("Average reward per request (higher is better)\n")
    print(f"{'Strategy':<24}" + "".join(f"{i[:12]:>14}" for i in INTENTS) + f"{'OVERALL':>10}")
    for name, per_intent in totals.items():
        avgs = [v / sim.total_steps for v in per_intent.values()]
        print(f"{name:<24}" + "".join(f"{a:>14.3f}" for a in avgs) + f"{sum(avgs)/len(avgs):>10.3f}")