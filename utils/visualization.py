"""Make the five results figures for the routing study (round robin vs KV-aware).

Usage:  python make_figures.py [DATA_DIR] [OUT_DIR]
DATA_DIR must hold the eight files named
  [con_]{priming,random}_{roundrobin,kvrouter}_prefix_cache_benchmark.csv
"""

import sys, glob, os
import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/figs"
os.makedirs(OUT, exist_ok=True)

# ---- load: one row per request; 'rep' = repeat (10 blocks of 10 rows per file) ----
frames = []
for f in sorted(glob.glob(os.path.join(DATA, "*_prefix_cache_benchmark.csv"))):
    name = os.path.basename(f).replace("_prefix_cache_benchmark.csv", "")
    conc = name.startswith("con_")
    wl, router = name.replace("con_", "").split("_")
    d = pd.read_csv(f)
    d["rep"] = (
        d.index // 10
    )  # KV-aware files have run_number=1 throughout, so use row blocks
    d["wl"], d["router"], d["conc"] = wl, router, conc
    d["rn"] = d.request_id.str.extract(r"R(\d+)$").astype(int)
    frames.append(d)
D = pd.concat(frames, ignore_index=True)

# ---- routing ground truth ----
# previous_workers = workers holding a matching cached prefix when the request arrived ('-' = none)
D["prev"] = D.previous_workers.fillna("-")
D["opp"] = D.prev != "-"  # opportunity: some worker could have served a cache hit
D["ok"] = D.apply(
    lambda r: r.actual_prefill_worker in r.prev.split("/"), axis=1
)  # routed to a holder

RR, KV = "#9aa5b1", "#0f8b8d"
DK = {"roundrobin": RR, "kvrouter": KV}
NM = {"roundrobin": "Round robin", "kvrouter": "KV-aware"}
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e6e9ed",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)
CF = [
    (False, "priming", "1. Priming\nsequential"),
    (True, "priming", "2. Priming\nconcurrent"),
    (False, "random", "3. Random\nsequential"),
    (True, "random", "4. Random\nconcurrent"),
]


def sel(c, w, r):
    return D[(D.conc == c) & (D.wl == w) & (D.router == r)]


def save(f, n):
    f.savefig(f"{OUT}/{n}.pdf")
    f.savefig(f"{OUT}/{n}.png", dpi=200)
    plt.close(f)


# Fig1 routing accuracy
f, ax = plt.subplots(figsize=(7, 2.6))
w = 0.36
for j, r in enumerate(["roundrobin", "kvrouter"]):
    for i, (c, wl, l) in enumerate(CF):
        g = sel(c, wl, r)
        o = g.opp.sum()
        k = (g.opp & g.ok).sum()
        a = k / o * 100
        ax.bar(
            i + (j - 0.5) * w, a, w * 0.92, color=DK[r], label=NM[r] if i == 0 else None
        )
        ax.text(
            i + (j - 0.5) * w, a + 2, f"{k}/{o}", ha="center", fontsize=7, color="#333"
        )
ax.set_xticks(range(4))
ax.set_xticklabels([x[2] for x in CF])
ax.set_ylim(0, 115)
ax.set_ylabel("Routing accuracy (%)")
ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=7.5)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis="x", visible=False)
save(f, "fig1_routing_accuracy")
# Fig2 TTFT by routing outcome
f, axs = plt.subplots(1, 4, figsize=(7, 2.7), sharey=True)
rng = np.random.default_rng(0)
for ax, (c, wl, l) in zip(axs, CF):
    for r in ["roundrobin", "kvrouter"]:
        g = sel(c, wl, r)
        g = g[g.opp]
        for x, m in [(0, g.ok), (1, ~g.ok)]:
            v = g[m].ttft_ms.values
            if len(v) == 0:
                continue
            xs = (
                x
                + (-0.17 if r == "roundrobin" else 0.17)
                + rng.uniform(-0.07, 0.07, len(v))
            )
            ax.scatter(xs, v, s=7, color=DK[r], alpha=0.55, lw=0)
            ax.hlines(
                v.mean(),
                x + (-0.3 if r == "roundrobin" else 0.04),
                x + (-0.04 if r == "roundrobin" else 0.3),
                color=DK[r] if r == "kvrouter" else "#5b6672",
                lw=1.8,
            )
    ax.set_yscale("log")
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Routed to\ncache", "Missed"], fontsize=7)
    ax.set_title(l.replace("\n", " "), fontsize=7.5)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("TTFT (ms, log)")
axs[0].set_yticks([200, 500, 1000, 2000, 5000])
axs[0].set_yticklabels(["200", "500", "1000", "2000", "5000"])
axs[0].minorticks_off()
h = [
    plt.Line2D([], [], marker="o", ls="", color=DK[r], label=NM[r], ms=4)
    for r in ["roundrobin", "kvrouter"]
]
f.legend(
    handles=h,
    frameon=False,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, -0.08),
    fontsize=7.5,
)
f.tight_layout()
save(f, "fig2_ttft_by_routing_outcome")
# Fig3 ECDF
f, axs = plt.subplots(1, 4, figsize=(7, 2.3), sharey=True)
for ax, (c, wl, l) in zip(axs, CF):
    for r in ["roundrobin", "kvrouter"]:
        v = np.sort(sel(c, wl, r).ttft_ms.values)
        ax.step(
            v / 1000,
            np.arange(1, len(v) + 1) / len(v),
            where="post",
            color=DK[r],
            lw=1.6,
        )
    ax.set_title(l.replace("\n", " "), fontsize=7.5)
    ax.set_xlabel("TTFT (s)")
    ax.grid(axis="both")
axs[0].set_ylabel("Cumulative fraction")
h = [
    plt.Line2D([], [], color=DK[r], label=NM[r], lw=1.6)
    for r in ["roundrobin", "kvrouter"]
]
f.legend(
    handles=h,
    frameon=False,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, -0.12),
    fontsize=7.5,
)
f.tight_layout()
save(f, "fig3_ttft_ecdf")
# Fig4 decomposition
f, ax = plt.subplots(figsize=(7, 2.9))
y = 0
yt = []
yl = []
for c, wl, l in CF[::-1]:
    for r in ["kvrouter", "roundrobin"]:
        g = sel(c, wl, r)
        t = g.ttft_ms.mean() / 1000
        d = g.response_ms.mean() / 1000
        ax.barh(y, t, 0.8, color=DK[r])
        ax.barh(y, d, 0.8, left=t, color=DK[r], alpha=0.35)
        ax.text(t + d + 0.15, y, f"{t + d:.1f} s", va="center", fontsize=7)
        yt.append(y)
        yl.append(NM[r])
        y += 1
    y += 0.6
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=7)
ax.set_xlabel("Mean latency (s)")
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 12)
ys = []
for i, (c, wl, l) in enumerate(CF[::-1]):
    ax.text(
        0,
        i * 2.6 + 1.45,
        l.replace("\n", " "),
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#333",
        fontweight="bold",
    )
ax.set_ylim(-0.6, 4 * 2.6 - 0.1)
h = [
    mpl.patches.Patch(color="#555", label="TTFT (prefill)"),
    mpl.patches.Patch(color="#555", alpha=0.35, label="Decode"),
]
ax.legend(
    handles=h,
    frameon=False,
    loc="upper right",
    fontsize=7.5,
    ncol=2,
    bbox_to_anchor=(1, 1.12),
)
save(f, "fig4_latency_decomposition")
# Fig5 load
f, ax = plt.subplots(figsize=(7, 2.9))
cols = ["#2f4858", "#4f7f95", "#8fb8c9", "#d3e4ec"]
y = 0
yt = []
yl = []


def jain(x):
    x = np.array(x, float)
    return x.sum() ** 2 / (len(x) * (x**2).sum())


for c, wl, l in CF[::-1]:
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
                color=cols[k],
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
        yl.append(NM[r])
        y += 1
    y += 0.6
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=7)
ax.set_xlim(0, 114)
ax.set_xlabel("Requests served (of 100)")
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
for i, (c, wl, l) in enumerate(CF[::-1]):
    ax.text(-0.01, (i * 2.6 + 0.5) / (y), "", transform=ax.transAxes)
ax.legend(
    frameon=False,
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.45, -0.38),
    fontsize=7.5,
    title=None,
)
for i, (c, wl, l) in enumerate(CF[::-1]):
    ax.text(
        0,
        i * 2.6 + 1.45,
        l.replace("\n", " "),
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#333",
        fontweight="bold",
    )
ax.set_ylim(-0.6, 4 * 2.6 - 0.1)
save(f, "fig5_load_distribution")
