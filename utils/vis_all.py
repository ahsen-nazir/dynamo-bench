"""Results figures: round robin vs KV-aware routing (matplotlib + seaborn).

Usage:   python make_figures.py [DATA_DIR] [OUT_DIR]
Input:   DATA_DIR holds the eight files
         [con_]{priming,random}_{roundrobin,kvrouter}_prefix_cache_benchmark.csv
Output:  OUT_DIR/fig1..fig5 (+ fig3b, fig4b) as vector PDF + 200-dpi PNG.
"""

import glob, os, sys
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
import seaborn as sns

DATA = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/figs"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ style
RR, KV, AMBER = "#9aa5b1", "#0f8b8d", "#d9822b"
COLORS = {"roundrobin": RR, "kvrouter": KV}
NAMES = {"roundrobin": "Round robin", "kvrouter": "KV-aware"}
ROUTERS = ["roundrobin", "kvrouter"]
CONFIGS = [
    (False, "priming", "1. Priming, sequential"),
    (True, "priming", "2. Priming, concurrent"),
    (False, "random", "3. Random, sequential"),
    (True, "random", "4. Random, concurrent"),
]
sns.set_theme(
    style="ticks",
    context="paper",
    rc={
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.labelsize": 8,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e6e9ed",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    },
)


# ------------------------------------------------------------------ data
def load(path):
    frames = []
    for f in sorted(glob.glob(os.path.join(path, "*_prefix_cache_benchmark.csv"))):
        name = os.path.basename(f).replace("_prefix_cache_benchmark.csv", "")
        conc = name.startswith("con_")
        workload, router = name.replace("con_", "").split("_")
        d = pd.read_csv(f)
        d["rep"] = (
            d.index // 10
        )  # KV-aware files have run_number=1 throughout: use blocks of 10 rows
        d["workload"], d["router"], d["conc"] = workload, router, conc
        frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    # previous_workers = workers holding a matching cached prefix at request time ('-' = none)
    prev = D.previous_workers.fillna("-")
    D["opp"] = prev != "-"  # a cache hit was possible somewhere
    D["routed_ok"] = [w in p.split("/") for w, p in zip(D.actual_prefill_worker, prev)]
    D["ttft_s"] = D.ttft_ms / 1000
    return D


D = load(DATA)


def sel(conc, wl, router=None):
    m = (D.conc == conc) & (D.workload == wl)
    return D[m & (D.router == router)] if router else D[m]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)


def router_legend(fig, kind="dot"):
    mk = dict(marker="o", ls="", ms=4) if kind == "dot" else dict(lw=1.6)
    h = [plt.Line2D([], [], color=COLORS[r], label=NAMES[r], **mk) for r in ROUTERS]
    fig.legend(
        handles=h,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=7.5,
    )


def jain(x):
    x = np.asarray(x, float)
    return x.sum() ** 2 / (len(x) * (x**2).sum())


# ------------------------------------------------------------------ Fig 1: routing accuracy
fig, ax = plt.subplots(figsize=(7, 2.6))
bw = 0.36
for j, r in enumerate(ROUTERS):
    for i, (c, wl, _) in enumerate(CONFIGS):
        g = sel(c, wl, r)
        opp = int(g.opp.sum())
        ok = int((g.opp & g.routed_ok).sum())
        acc = 100 * ok / opp
        x = i + (j - 0.5) * bw
        ax.bar(x, acc, bw * 0.92, color=COLORS[r], label=NAMES[r] if i == 0 else None)
        ax.text(x, acc + 2, f"{ok}/{opp}", ha="center", fontsize=7, color="#333")
ax.set_xticks(range(4))
ax.set_xticklabels([c[2].replace(", ", "\n") for c in CONFIGS])
ax.set_ylim(0, 115)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Routing accuracy (%)")
ax.legend(
    frameon=False, loc="upper right", ncol=2, fontsize=7.5, bbox_to_anchor=(1, 1.1)
)
ax.grid(axis="x", visible=False)
save(fig, "fig1_routing_accuracy")

# ------------------------------------------------------------------ Fig 2: TTFT by routing outcome
fig, axs = plt.subplots(1, 4, figsize=(7, 2.7), sharey=True)
for ax, (c, wl, label) in zip(axs, CONFIGS):
    g = sel(c, wl)
    g = g[g.opp].assign(
        outcome=lambda t: np.where(t.routed_ok, "Routed to\ncache", "Missed")
    )
    sns.stripplot(
        data=g,
        x="outcome",
        y="ttft_ms",
        hue="router",
        hue_order=ROUTERS,
        order=["Routed to\ncache", "Missed"],
        palette=COLORS,
        dodge=True,
        size=3,
        alpha=0.55,
        jitter=0.12,
        legend=False,
        ax=ax,
    )
    for xi, oc in enumerate(
        ["Routed to\ncache", "Missed"]
    ):  # mean marks (dodge offsets are +/-0.2)
        for off, r in zip((-0.2, 0.2), ROUTERS):
            v = g[(g.outcome == oc) & (g.router == r)].ttft_ms
            if len(v):
                ax.hlines(
                    v.mean(),
                    xi + off - 0.17,
                    xi + off + 0.17,
                    color="#5b6672" if r == "roundrobin" else KV,
                    lw=1.8,
                )
    ax.set_yscale("log")
    ax.set_title(label.replace(", ", " "), fontsize=7.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("TTFT (ms, log)")
axs[0].set_yticks([200, 500, 1000, 2000, 5000])
axs[0].set_yticklabels(["200", "500", "1000", "2000", "5000"])
axs[0].minorticks_off()
router_legend(fig)
fig.tight_layout()
save(fig, "fig2_ttft_by_routing_outcome")

# ------------------------------------------------------------------ Fig 3: TTFT ECDF
fig, axs = plt.subplots(1, 4, figsize=(7, 2.3), sharey=True)
for ax, (c, wl, label) in zip(axs, CONFIGS):
    sns.ecdfplot(
        data=sel(c, wl),
        x="ttft_s",
        hue="router",
        hue_order=ROUTERS,
        palette=COLORS,
        lw=1.6,
        legend=False,
        ax=ax,
    )
    ax.set_title(label.replace(", ", " "), fontsize=7.5)
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("")
    ax.grid(axis="both")
axs[0].set_ylabel("Cumulative fraction")
router_legend(fig, "line")
fig.tight_layout()
save(fig, "fig3_ttft_ecdf")


# ------------------------------------------------------------------ Fig 3b: TTFT ECDF, opportunity requests only
fig, axs = plt.subplots(1, 4, figsize=(7, 2.3), sharey=True)
for ax, (c, wl, label) in zip(axs, CONFIGS):
    g = sel(c, wl)
    g = g[g.opp]
    sns.ecdfplot(
        data=g,
        x="ttft_s",
        hue="router",
        hue_order=ROUTERS,
        palette=COLORS,
        lw=1.6,
        legend=False,
        ax=ax,
    )
    n = g.groupby("router").size()
    ax.set_title(
        label.replace(", ", " ")
        + f"\n(n={n.get('roundrobin', 0)} / {n.get('kvrouter', 0)})",
        fontsize=7.5,
    )
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("")
    ax.grid(axis="both")
axs[0].set_ylabel("Cumulative fraction")
router_legend(fig, "line")
fig.tight_layout()
save(fig, "fig3b_ttft_ecdf_opportunity")

# ------------------------------------------------------------------ Fig 4: latency decomposition
fig, ax = plt.subplots(figsize=(7, 2.9))
y = 0
yt, yl = [], []
for gi, (c, wl, label) in enumerate(CONFIGS[::-1]):
    for r in ["kvrouter", "roundrobin"]:
        g = sel(c, wl, r)
        t, d = g.ttft_ms.mean() / 1000, g.response_ms.mean() / 1000
        ax.barh(y, t, 0.8, color=COLORS[r])
        ax.barh(y, d, 0.8, left=t, color=COLORS[r], alpha=0.35)
        ax.text(t + d + 0.15, y, f"{t + d:.1f} s", va="center", fontsize=7)
        yt.append(y)
        yl.append(NAMES[r])
        y += 1
    ax.text(
        0,
        y - 2 + 1.45,
        label.replace(", ", " \u00b7 "),
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#333",
    )
    y += 0.6
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=7)
ax.set_xlabel("Mean latency (s)")
ax.set_xlim(0, 12)
ax.set_ylim(-0.6, y - 0.1)
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.legend(
    handles=[
        mpl.patches.Patch(color="#555", label="TTFT (prefill)"),
        mpl.patches.Patch(color="#555", alpha=0.35, label="Decode"),
    ],
    frameon=False,
    loc="upper right",
    ncol=2,
    fontsize=7.5,
    bbox_to_anchor=(1, 1.12),
)
save(fig, "fig4_latency_decomposition")


# ------------------------------------------------------------------ Fig 4b: latency decomposition, opportunity requests only
fig, ax = plt.subplots(figsize=(7, 2.9))
y = 0
yt, yl = [], []
for c, wl, label in CONFIGS[::-1]:
    for r in ["kvrouter", "roundrobin"]:
        g = sel(c, wl, r)
        g = g[g.opp]
        t, d = g.ttft_ms.mean() / 1000, g.response_ms.mean() / 1000
        ax.barh(y, t, 0.8, color=COLORS[r])
        ax.barh(y, d, 0.8, left=t, color=COLORS[r], alpha=0.35)
        ax.text(
            t + d + 0.15, y, f"{t + d:.1f} s  (n={len(g)})", va="center", fontsize=7
        )
        yt.append(y)
        yl.append(NAMES[r])
        y += 1
    ax.text(
        0,
        y - 2 + 1.45,
        label.replace(", ", " \u00b7 "),
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#333",
    )
    y += 0.6
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=7)
ax.set_xlabel("Mean latency of cache-eligible requests (s)")
ax.set_xlim(0, 12.5)
ax.set_ylim(-0.6, y - 0.1)
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.legend(
    handles=[
        mpl.patches.Patch(color="#555", label="TTFT (prefill)"),
        mpl.patches.Patch(color="#555", alpha=0.35, label="Decode"),
    ],
    frameon=False,
    loc="upper right",
    ncol=2,
    fontsize=7.5,
    bbox_to_anchor=(1, 1.12),
)
save(fig, "fig4b_latency_decomposition_opportunity")

# ------------------------------------------------------------------ Fig 5: load distribution
fig, ax = plt.subplots(figsize=(7, 2.9))
y = 0
yt, yl = [], []
shades = ["#2f4858", "#4f7f95", "#8fb8c9", "#d3e4ec"]
for c, wl, label in CONFIGS[::-1]:
    for r in ["kvrouter", "roundrobin"]:
        cnt = (
            sel(c, wl, r)
            .actual_prefill_worker.value_counts()
            .reindex(["w0", "w1", "w2", "w3"])
            .fillna(0)
            .values
        )
        left = 0
        for k in range(4):
            ax.barh(
                y,
                cnt[k],
                0.8,
                left=left,
                color=shades[k],
                label=f"w{k}" if y == 0 else None,
            )
            if cnt[k] >= 6:
                ax.text(
                    left + cnt[k] / 2,
                    y,
                    int(cnt[k]),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if k < 2 else "#222",
                )
            left += cnt[k]
        if r == "kvrouter":
            ax.text(101.5, y, f"J = {jain(cnt):.2f}", va="center", fontsize=7)
        yt.append(y)
        yl.append(NAMES[r])
        y += 1
    ax.text(
        0,
        y - 2 + 1.45,
        label.replace(", ", " \u00b7 "),
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#333",
    )
    y += 0.6
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=7)
ax.set_xlim(0, 114)
ax.set_ylim(-0.6, y - 0.1)
ax.set_xlabel("Requests served (of 100)")
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.legend(
    frameon=False,
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.45, -0.38),
    fontsize=7.5,
)
save(fig, "fig5_load_distribution")
print("wrote 7 figures to", OUT)
