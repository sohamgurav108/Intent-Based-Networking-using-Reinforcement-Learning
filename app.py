"""
app.py
The web page (Streamlit) for the demo.
Run it with:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from network import PATHS, get_base_links
from simulator import Simulator
from intents import INTENTS, reward, best_path
from agent import RandomStrategy, ShortestPathStrategy, LowestLatencyStrategy, QLearningAgent

st.set_page_config(page_title="Intent-Based Networking with RL", layout="wide")

STEPS = 600
FAILURE_START, FAILURE_END = 300, 450
AGENT = "RL agent (Q-learning)"
COLORS = {"Random": "#9e9e9e", "Shortest path": "#e69f00", "Lowest latency": "#009e73",
          AGENT: "#d62728", "Perfect (oracle)": "#000000"}

# where each router is drawn on the diagram
POS = {"S": (0, 1), "R1": (1, 1.8), "R2": (1, 0.2), "R3": (2, 2.0),
       "R4": (2, 1.0), "R5": (2, 0.0), "D": (3, 1)}

INTENT_TEXT = {
    "video_conferencing": "A smooth video call: low delay, low jitter, very little loss.",
    "gaming": "Online gaming: the lowest possible delay and steady jitter.",
    "file_transfer": "Moving a big file: as much bandwidth as possible, and cheap.",
    "video_streaming": "Watching a stream: good bandwidth and low loss.",
    "cheap_backup": "A nightly backup: the cheapest route, speed matters little.",
}


# ---------------------------------------------------------------
# Run the whole simulation once and record everything.
# st.cache_data remembers the answer, so moving a slider is instant.
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_simulation(intent, seed, with_failure):
    start = FAILURE_START if with_failure else 0
    end = FAILURE_END if with_failure else 0
    sim = Simulator(seed=seed, total_steps=STEPS, failure_start=start, failure_end=end)
    strategies = [RandomStrategy(seed), ShortestPathStrategy(),
                  LowestLatencyStrategy(), QLearningAgent([intent], seed=seed)]

    log = {"rewards": {s.name: [] for s in strategies},
           "paths": {s.name: [] for s in strategies},
           "oracle": [], "best": [], "util": [], "failed": [], "metrics": [], "q": []}
    log["rewards"]["Perfect (oracle)"] = log["oracle"]

    while not sim.done():
        sim.step()
        info = sim.true_metrics()
        best_name, best_value = best_path(intent, info)
        log["oracle"].append(best_value)
        log["best"].append(best_name)
        log["metrics"].append(info)
        log["util"].append({l: sim.utilisation(l) for l in sim.base})
        log["failed"].append([l for l in sim.base if sim.is_failed(l)])

        for s in strategies:
            path = s.choose(intent, info)
            s.update(intent, path, reward(intent, sim.measure(path)))   # learns from noisy data
            log["rewards"][s.name].append(reward(intent, info[path]))   # scored on the truth
            log["paths"][s.name].append(path)
            if s.name == AGENT:
                log["q"].append(dict(s.q[intent]))
    return log


def smooth(x, window=15):
    return np.convolve(x, np.ones(window) / window, mode="valid")


# ---------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------
def draw_reward_graph(log, with_failure):
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, values in log["rewards"].items():
        s = smooth(values)
        ax.plot(np.arange(len(s)) + 15, s, label=name, color=COLORS[name], linewidth=2,
                linestyle="--" if name.startswith("Perfect") else "-")
    if with_failure:
        ax.axvspan(FAILURE_START, FAILURE_END, color="red", alpha=0.08)
        ax.text((FAILURE_START + FAILURE_END) / 2, 0.02, "link R1-R3 down", ha="center",
                fontsize=9, color="red", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Time step")
    ax.set_ylabel("Reward (smoothed)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def draw_network(util, failed, agent_path, best_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.RdYlGn_r

    # the path chosen by the agent (thick blue line underneath)
    nodes = PATHS[agent_path]
    for a, b in zip(nodes, nodes[1:]):
        ax.plot([POS[a][0], POS[b][0]], [POS[a][1], POS[b][1]],
                color="#1f77b4", linewidth=11, alpha=0.35, solid_capstyle="round", zorder=1)

    # every link, coloured by how busy it is
    for (a, b), u in util.items():
        is_down = (a, b) in failed
        ax.plot([POS[a][0], POS[b][0]], [POS[a][1], POS[b][1]],
                color="#555555" if is_down else cmap(u), linewidth=3,
                linestyle=":" if is_down else "-", zorder=2)
        mx, my = (POS[a][0] + POS[b][0]) / 2, (POS[a][1] + POS[b][1]) / 2
        ax.text(mx, my + 0.08, "DOWN" if is_down else f"{u * 100:.0f}%",
                ha="center", fontsize=8, color="red" if is_down else "black", zorder=4)

    # routers
    for name, (x, y) in POS.items():
        ax.scatter([x], [y], s=900, color="white", edgecolor="black", zorder=3)
        ax.text(x, y, name, ha="center", va="center", fontsize=10, fontweight="bold", zorder=5)

    ax.set_title(f"Agent's path: {agent_path} (blue)   |   Perfect path: {best_name}", fontsize=10)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------
st.title("Intent-Based Networking using Reinforcement Learning")
st.caption("Tell the network WHAT you want. A learning agent picks the best path "
           "and adapts when the network changes. (Simulated network.)")

with st.sidebar:
    st.header("Choose the intent")
    intent = st.selectbox("What does the user want?", list(INTENTS),
                          format_func=lambda x: x.replace("_", " ").title())
    st.write(INTENT_TEXT[intent])
    st.divider()
    with_failure = st.checkbox("Break link R1-R3 (steps 300 to 450)", value=True)
    seed = st.number_input("Random seed", min_value=1, max_value=999, value=1, step=1)
    st.divider()
    st.subheader("How this intent is scored")
    weights = INTENTS[intent]["weights"]
    st.dataframe(pd.DataFrame({"weight": weights}))
    limits = INTENTS[intent]["limits"]
    st.write("Hard limits: " + (", ".join(f"{k} = {v}" for k, v in limits.items()) or "none"))

log = run_simulation(intent, int(seed), with_failure)

tab1, tab2, tab3 = st.tabs(["Results", "Network snapshot", "How it works"])

# ---------------- TAB 1: results ----------------
with tab1:
    st.subheader("Average reward over 600 steps (higher is better)")
    perfect = np.mean(log["oracle"])
    cols = st.columns(5)
    for col, (name, values) in zip(cols, log["rewards"].items()):
        avg = np.mean(values)
        col.metric(name, f"{avg:.3f}", f"{100 * avg / perfect:.0f}% of perfect" if perfect > 0 else None,
                   delta_color="off")

    st.pyplot(draw_reward_graph(log, with_failure))

    agent_avg = np.mean(log["rewards"][AGENT])
    st.write(f"For **{intent.replace('_', ' ')}**, the RL agent reached "
             f"**{100 * agent_avg / perfect:.0f}%** of the perfect score, "
             f"against {100 * np.mean(log['rewards']['Random']) / perfect:.0f}% for random choice "
             f"and {100 * np.mean(log['rewards']['Shortest path']) / perfect:.0f}% for the shortest-path rule.")
    if with_failure:
        st.info("During the red band, link R1-R3 is down. A fixed rule keeps using the dead path, "
                "while the agent notices the bad rewards and switches.")

# ---------------- TAB 2: snapshot ----------------
with tab2:
    t = st.slider("Time step", 1, STEPS, 350)
    i = t - 1
    agent_path = log["paths"][AGENT][i]
    best_name = log["best"][i]

    left, right = st.columns([1, 1])
    with left:
        st.pyplot(draw_network(log["util"][i], log["failed"][i], agent_path, best_name))
        st.caption("Link colour = how busy it is (green = quiet, red = busy). "
                   "The percentage is the link utilisation. Dotted grey = link is down.")

    with right:
        rows = []
        for p in PATHS:
            m = log["metrics"][i][p]
            if m is None:
                rows.append({"Path": p, "Route": "-".join(PATHS[p]), "Latency (ms)": None,
                             "Jitter (ms)": None, "Loss (%)": None, "Bandwidth (Mbps)": None,
                             "Cost": None, "Reward": 0.0})
            else:
                rows.append({"Path": p, "Route": "-".join(PATHS[p]),
                             "Latency (ms)": round(m["latency"], 1), "Jitter (ms)": round(m["jitter"], 1),
                             "Loss (%)": round(m["loss"], 2), "Bandwidth (Mbps)": round(m["bandwidth"]),
                             "Cost": m["cost"], "Reward": round(reward(intent, m), 3)})
        st.dataframe(pd.DataFrame(rows).set_index("Path"))
        st.caption("Empty row = the path is down (a link on it has failed). "
                   "Reward 0 = the path breaks a hard limit of this intent.")

        st.write("**What each strategy chose at this step**")
        picks = pd.DataFrame({"Chosen path": {n: p[i] for n, p in log["paths"].items()},
                              "Reward": {n: round(v[i], 3) for n, v in log["rewards"].items() if n in log["paths"]}})
        st.dataframe(picks)

    st.write("**What the agent believes (its Q-values) at this step**")
    st.bar_chart(pd.Series(log["q"][i], name="Q-value"))

# ---------------- TAB 3: how it works ----------------
with tab3:
    st.markdown("""
**The idea.** The user states an *intent* (for example "gaming"). The system turns it into
weights on five measurements, and a learning agent chooses the path with the best score.

**The flow**
1. The user picks an intent.
2. The agent chooses one of the 5 paths between S and D.
3. Traffic uses that path, and the system measures latency, jitter, loss, bandwidth and cost.
4. The measurements become a **reward** (weights of the intent x scores of the metrics).
5. The agent updates its Q-value for that path, so good paths are chosen more often.

**The learning rule:** `Q = Q + 0.2 x (reward - Q)`. With probability epsilon (it starts at 0.5 and
shrinks to 0.05) the agent tries a random path (exploration), otherwise it takes the path with the highest Q.

**The simulated network** changes all the time: a daily traffic wave, random congestion spikes,
and one link failure. A busier link has more delay, jitter and loss and less free bandwidth.

**Compared against**
- *Random*: any path.
- *Shortest path*: always the path with the fewest routers.
- *Lowest latency*: the working path with the lowest delay right now (it ignores the intent).
- *Perfect (oracle)*: the best path at every step. Nobody can reach it, but it shows how close we get.

**Limits.** This is a simplified simulation with link values chosen by us, not measurements from a real network.
""")