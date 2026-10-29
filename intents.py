"""
intents.py
What the user wants (the intent), and how a path is scored for it.
"""

# ---------------------------------------------------------------
# 1. INTENTS
# 'weights' say how much each metric matters (each row adds up to 1).
# 'limits' are hard requirements. If a path breaks a limit, its
# reward becomes 0 (the path is not acceptable for this intent).
# These values are our own design choices.
# ---------------------------------------------------------------
INTENTS = {
    "video_conferencing": {
        "weights": {"latency": 0.25, "jitter": 0.25, "loss": 0.30, "bandwidth": 0.10, "cost": 0.10},
        "limits": {"max_latency": 100, "max_loss": 3},
    },
    "gaming": {
        "weights": {"latency": 0.45, "jitter": 0.30, "loss": 0.15, "bandwidth": 0.05, "cost": 0.05},
        "limits": {"max_latency": 50},
    },
    "file_transfer": {
        "weights": {"latency": 0.05, "jitter": 0.00, "loss": 0.10, "bandwidth": 0.65, "cost": 0.20},
        "limits": {"min_bandwidth": 100},
    },
    "video_streaming": {
        "weights": {"latency": 0.10, "jitter": 0.10, "loss": 0.25, "bandwidth": 0.45, "cost": 0.10},
        "limits": {"min_bandwidth": 100},
    },
    "cheap_backup": {
        "weights": {"latency": 0.00, "jitter": 0.00, "loss": 0.10, "bandwidth": 0.20, "cost": 0.70},
        "limits": {},
    },
}

# ---------------------------------------------------------------
# 2. NORMALISATION
# Each metric is converted to a score between 0 and 1,
# where 1 is always "better". We use fixed reference values:
#   latency  : 0 ms is perfect, 150 ms or more is worst
#   jitter   : 0 ms is perfect, 60 ms or more is worst
#   loss     : 0 %  is perfect, 10 % or more is worst
#   bandwidth: 500 Mbps or more is perfect, 0 is worst
#   cost     : 0 is perfect, 25 or more is worst
# ---------------------------------------------------------------
WORST_LATENCY = 150.0
WORST_JITTER = 60.0
WORST_LOSS = 10.0
BEST_BANDWIDTH = 500.0
WORST_COST = 25.0


def clip01(x):
    """Keep a number between 0 and 1."""
    return max(0.0, min(1.0, x))


def metric_scores(m):
    """Convert raw metrics (ms, %, Mbps, cost) into 0..1 scores."""
    return {
        "latency": clip01(1 - m["latency"] / WORST_LATENCY),
        "jitter": clip01(1 - m["jitter"] / WORST_JITTER),
        "loss": clip01(1 - m["loss"] / WORST_LOSS),
        "bandwidth": clip01(m["bandwidth"] / BEST_BANDWIDTH),
        "cost": clip01(1 - m["cost"] / WORST_COST),
    }


def meets_limits(m, limits):
    """Check the hard requirements of an intent."""
    if "max_latency" in limits and m["latency"] > limits["max_latency"]:
        return False
    if "max_loss" in limits and m["loss"] > limits["max_loss"]:
        return False
    if "min_bandwidth" in limits and m["bandwidth"] < limits["min_bandwidth"]:
        return False
    return True


# ---------------------------------------------------------------
# 3. REWARD
# reward = sum of (weight x score) for every metric.
# A broken path (None) or a path that breaks a limit gets 0.
# ---------------------------------------------------------------
def reward(intent_name, m):
    if m is None:
        return 0.0

    intent = INTENTS[intent_name]
    if not meets_limits(m, intent["limits"]):
        return 0.0

    scores = metric_scores(m)
    total = 0.0
    for metric, weight in intent["weights"].items():
        total += weight * scores[metric]
    return total


def best_path(intent_name, all_metrics):
    """The path with the highest reward (the 'perfect' answer)."""
    best_name = None
    best_value = -1.0
    for name, m in all_metrics.items():
        r = reward(intent_name, m)
        if r > best_value:
            best_name, best_value = name, r
    return best_name, best_value


# ---------------------------------------------------------------
# 4. TEST: run this file directly to see the rewards
# ---------------------------------------------------------------
if __name__ == "__main__":
    from network import get_base_links, all_path_metrics

    metrics = all_path_metrics(get_base_links())

    print("Reward of every path for every intent (original network)\n")
    header = f"{'Intent':<20}" + "".join(f"{p:>8}" for p in metrics) + "   Best"
    print(header)
    for intent in INTENTS:
        row = f"{intent:<20}"
        for p, m in metrics.items():
            row += f"{reward(intent, m):>8.3f}"
        best, _ = best_path(intent, metrics)
        print(row + f"   {best}")