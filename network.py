"""
network.py
The network topology: nodes, links, paths, and how a path's
quality is calculated from the links it uses.
"""

import copy

# ---------------------------------------------------------------
# 1. LINKS
# Each link connects two nodes and has 5 properties.
#   latency   : delay in milliseconds (ms)
#   jitter    : variation of the delay (ms)
#   loss      : packet loss in percent (%)
#   bandwidth : capacity in Mbps
#   cost      : money per GB
# 'up' says whether the link is working (False = failed).
# These numbers are our own design values for the simulation.
# ---------------------------------------------------------------
LINKS = {
    ("S", "R1"):  {"latency": 5,  "jitter": 1,  "loss": 0.1, "bandwidth": 150,  "cost": 8, "up": True},
    ("S", "R2"):  {"latency": 10, "jitter": 3,  "loss": 0.5, "bandwidth": 500,  "cost": 2, "up": True},
    ("R1", "R3"): {"latency": 5,  "jitter": 1,  "loss": 0.1, "bandwidth": 40,   "cost": 6, "up": True},
    ("R1", "R4"): {"latency": 15, "jitter": 4,  "loss": 0.3, "bandwidth": 200,  "cost": 4, "up": True},
    ("R2", "R4"): {"latency": 20, "jitter": 8,  "loss": 1.0, "bandwidth": 500,  "cost": 1, "up": True},
    ("R3", "D"):  {"latency": 5,  "jitter": 1,  "loss": 0.1, "bandwidth": 40,   "cost": 6, "up": True},
    ("R4", "D"):  {"latency": 15, "jitter": 5,  "loss": 0.5, "bandwidth": 300,  "cost": 3, "up": True},
    ("R4", "R5"): {"latency": 10, "jitter": 2,  "loss": 0.2, "bandwidth": 400,  "cost": 1, "up": True},
    ("R5", "D"):  {"latency": 20, "jitter": 10, "loss": 2.0, "bandwidth": 1000, "cost": 1, "up": True},
}

# ---------------------------------------------------------------
# 2. PATHS
# Each path is the list of nodes a packet travels through.
# ---------------------------------------------------------------
PATHS = {
    "P1": ["S", "R1", "R3", "D"],
    "P2": ["S", "R1", "R4", "D"],
    "P3": ["S", "R1", "R4", "R5", "D"],
    "P4": ["S", "R2", "R4", "D"],
    "P5": ["S", "R2", "R4", "R5", "D"],
}


def get_base_links():
    """Return a fresh copy of the original links (safe to modify)."""
    return copy.deepcopy(LINKS)


def get_link(links, a, b):
    """Find a link between two nodes, in either direction."""
    if (a, b) in links:
        return links[(a, b)]
    return links[(b, a)]


# ---------------------------------------------------------------
# 3. PATH METRICS
# How the quality of a whole path comes from its links:
#   latency   = SUM of link latencies
#   jitter    = SUM of link jitters
#   loss      = combined: 1 - (1-l1)(1-l2)...   (any link can drop a packet)
#   bandwidth = MINIMUM along the path (the bottleneck link)
#   cost      = SUM of link costs
# If any link on the path is down, the whole path is down (None).
# ---------------------------------------------------------------
def path_metrics(path_nodes, links):
    latency = 0.0
    jitter = 0.0
    survive = 1.0          # chance that a packet survives every link
    bandwidth = float("inf")
    cost = 0.0

    for i in range(len(path_nodes) - 1):
        link = get_link(links, path_nodes[i], path_nodes[i + 1])

        if not link["up"]:
            return None    # path is broken

        latency += link["latency"]
        jitter += link["jitter"]
        survive *= (1 - link["loss"] / 100)
        bandwidth = min(bandwidth, link["bandwidth"])
        cost += link["cost"]

    return {
        "latency": latency,
        "jitter": jitter,
        "loss": (1 - survive) * 100,
        "bandwidth": bandwidth,
        "cost": cost,
    }


def all_path_metrics(links):
    """Metrics of every path. A broken path gives None."""
    return {name: path_metrics(nodes, links) for name, nodes in PATHS.items()}


# ---------------------------------------------------------------
# 4. TEST: run this file directly to check the numbers
# ---------------------------------------------------------------
if __name__ == "__main__":
    links = get_base_links()
    print("Path metrics (original network)\n")
    print(f"{'Path':<5}{'Route':<20}{'Lat(ms)':>8}{'Jit(ms)':>9}{'Loss(%)':>9}{'BW(Mbps)':>10}{'Cost':>6}")
    for name, m in all_path_metrics(links).items():
        route = "-".join(PATHS[name])
        print(f"{name:<5}{route:<20}{m['latency']:>8.0f}{m['jitter']:>9.0f}"
              f"{m['loss']:>9.2f}{m['bandwidth']:>10.0f}{m['cost']:>6.0f}")

    print("\nNow we break link R1-R3 and check P1 again:")
    links[("R1", "R3")]["up"] = False
    print("P1 metrics:", path_metrics(PATHS["P1"], links))