"""
simulator.py
Makes the network change over time, so that the best path is not
always the same. It models:
  1. Daily traffic: load rises and falls in a wave.
  2. Random congestion spikes on single links.
  3. One link failure (and recovery) at a fixed time.
  4. Measurement noise (a real measurement is never exact).
"""

import math
import random
from network import get_base_links, PATHS, path_metrics


class Simulator:
    def __init__(self, seed=0, total_steps=600,
                 failure_link=("R1", "R3"), failure_start=300, failure_end=450,
                 spike_chance=0.03, wave_length=200):
        self.rng = random.Random(seed)
        self.total_steps = total_steps
        self.failure_link = failure_link
        self.failure_start = failure_start
        self.failure_end = failure_end
        self.spike_chance = spike_chance
        self.wave_length = wave_length

        self.base = get_base_links()
        self.t = 0

        # every link starts with its own normal level of use (15% to 35%)
        self.base_util = {link: self.rng.uniform(0.15, 0.35) for link in self.base}
        # active congestion spikes: link -> [extra_utilisation, steps_left]
        self.spikes = {}

    # -----------------------------------------------------------
    # Move time forward by one step
    # -----------------------------------------------------------
    def step(self):
        self.t += 1

        # old spikes fade away
        for link in list(self.spikes):
            self.spikes[link][1] -= 1
            if self.spikes[link][1] <= 0:
                del self.spikes[link]

        # a new spike may start on a random link
        if self.rng.random() < self.spike_chance:
            link = self.rng.choice(list(self.base))
            self.spikes[link] = [self.rng.uniform(0.3, 0.55), self.rng.randint(10, 30)]

    def done(self):
        return self.t >= self.total_steps

    # -----------------------------------------------------------
    # How busy is a link right now? (0 = empty, 0.95 = almost full)
    # -----------------------------------------------------------
    def utilisation(self, link):
        # daily wave between 0 and 1
        wave = 0.5 + 0.5 * math.sin(2 * math.pi * self.t / self.wave_length)
        u = self.base_util[link] + 0.25 * wave
        if link in self.spikes:
            u += self.spikes[link][0]
        return min(u, 0.95)

    def is_failed(self, link):
        same = link == self.failure_link or link == self.failure_link[::-1]
        return same and self.failure_start <= self.t < self.failure_end

    # -----------------------------------------------------------
    # The real state of every link right now.
    # A busier link has more delay, more jitter, more loss and
    # less free bandwidth.
    # -----------------------------------------------------------
    def current_links(self):
        links = {}
        for link, b in self.base.items():
            u = self.utilisation(link)
            links[link] = {
                "latency": b["latency"] * (1 + 2 * u ** 2),
                "jitter": b["jitter"] * (1 + 3 * u ** 2),
                "loss": b["loss"] + 8 * max(0.0, u - 0.7),
                "bandwidth": b["bandwidth"] * (1 - u),
                "cost": b["cost"],
                "up": not self.is_failed(link),
            }
        return links

    # -----------------------------------------------------------
    # True metrics of every path (used for the "perfect" answer)
    # -----------------------------------------------------------
    def true_metrics(self):
        links = self.current_links()
        return {name: path_metrics(nodes, links) for name, nodes in PATHS.items()}

    # -----------------------------------------------------------
    # What the agent actually sees when it sends traffic on one
    # path: the true values plus a little measurement noise.
    # -----------------------------------------------------------
    def measure(self, path_name):
        m = self.true_metrics()[path_name]
        if m is None:
            return None
        noisy = dict(m)
        noisy["latency"] = max(0.0, m["latency"] * self.rng.gauss(1, 0.05))
        noisy["jitter"] = max(0.0, m["jitter"] * self.rng.gauss(1, 0.10))
        noisy["loss"] = max(0.0, m["loss"] + self.rng.gauss(0, 0.2))
        noisy["bandwidth"] = max(0.0, m["bandwidth"] * self.rng.gauss(1, 0.05))
        return noisy


# ---------------------------------------------------------------
# TEST: run this file directly
# Shows which path is the best one at different moments.
# ---------------------------------------------------------------
if __name__ == "__main__":
    from intents import INTENTS, best_path

    sim = Simulator(seed=1)
    print("Best path at different times (the answer keeps changing)\n")
    print(f"{'Step':<6}" + "".join(f"{i[:14]:>16}" for i in INTENTS))

    counts = {i: {} for i in INTENTS}
    while not sim.done():
        sim.step()
        metrics = sim.true_metrics()
        row = ""
        for i in INTENTS:
            best, _ = best_path(i, metrics)
            counts[i][best] = counts[i].get(best, 0) + 1
            row += f"{best:>16}"
        if sim.t % 50 == 0:
            print(f"{sim.t:<6}" + row)

    print("\nHow many steps each path was the best (out of 600):")
    for i in INTENTS:
        print(f"{i:<20}", dict(sorted(counts[i].items())))