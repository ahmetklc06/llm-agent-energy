# Figures 1-5 and Tables 1-4 of the paper, generated from the result CSV files.
# Figures 4 and 5 require the outputs of notebooks/03_reanalysis.ipynb (results/analysis_A).
# Usage: python scripts/make_figures.py [results_dir]   (default: results)

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
MAIN = os.path.join(RESULTS, "main_run")
ANALYSIS = os.path.join(RESULTS, "analysis_A")
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"small": "#7FB3D5", "medium": "#F5B67A", "large": "#E08E8E"}
CAT = ["small", "medium", "large"]
TASKS = ["classification", "math", "summarization"]
TASK_NAME = {"classification": "Classification", "math": "Mathematical reasoning",
             "summarization": "Summarization"}
LEGEND = [Patch(color=COL[c], label=c.capitalize()) for c in CAT]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- data
runs = {}
for tau in (5, 7, 9):
    d = pd.read_csv(os.path.join(MAIN, f"agentic_energy_v4_threshold{tau}.csv"))
    d["first_conf"] = d["confidence_history"].apply(lambda s: json.loads(s)[0])
    d["metric"] = np.where(d["task"] == "summarization", d["rougeL"], d["correct"])
    runs[tau] = d
all_runs = pd.concat(runs.values(), ignore_index=True)
d7 = runs[7]

# ---------------------------------------------------------------- tables
cells = (all_runs.groupby(["threshold", "task", "model", "model_category"])
         .agg(accuracy=("metric", "mean"), energy_kwh=("energy_kwh", "mean"),
              turns=("n_turns", "mean"), truncated_share=("truncated_any", "mean"))
         .reset_index())
cells["efficiency"] = cells["accuracy"] / cells["energy_kwh"]
cells["kwh_per_accuracy_unit"] = cells["energy_kwh"] / cells["accuracy"]
cells.to_csv(os.path.join(OUT, "table2_model_task_threshold.csv"), index=False, float_format="%.4g")

rows = []
for (tau, task), g in cells.groupby(["threshold", "task"]):
    hi = g.loc[g["efficiency"].idxmax()]
    lo = g.loc[g["efficiency"].idxmin()]
    rows.append({"task": TASK_NAME[task], "tau": tau,
                 "highest": f"{hi.model} ({hi.model_category})",
                 "lowest": f"{lo.model} ({lo.model_category})",
                 "ratio": round(hi.efficiency / lo.efficiency, 1)})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table1_best_worst.csv"), index=False)

(all_runs.groupby(["threshold", "task", "model_category"])["n_turns"].mean()
 .unstack().reindex(columns=CAT).round(3)
 .to_csv(os.path.join(OUT, "table3_turns_task_category.csv")))

# ---------------------------------------------------------------- figure 1
fig, axs = plt.subplots(3, 3, figsize=(13, 10))
for i, task in enumerate(TASKS):
    for j, tau in enumerate((5, 7, 9)):
        g = cells[(cells.task == task) & (cells.threshold == tau)].sort_values("efficiency", ascending=False)
        ax = axs[i, j]
        ax.bar(g["model"], g["efficiency"], color=[COL[c] for c in g["model_category"]])
        ax.set_title(f"{TASK_NAME[task]}, τ = {tau}" + (" (primary)" if tau == 7 else ""), fontsize=9)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        if j == 0:
            ax.set_ylabel("Efficiency\n(accuracy or ROUGE-L per kWh)", fontsize=8)
fig.legend(handles=LEGEND, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "Fig1_efficiency_grid")

# ---------------------------------------------------------------- figure 2
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, task in zip(axs, TASKS):
    g = cells[(cells.task == task) & (cells.threshold == 7)]
    ax.scatter(g["energy_kwh"], g["accuracy"], c=[COL[c] for c in g["model_category"]], s=55, zorder=3)
    for _, r in g.iterrows():
        ax.annotate(r.model, (r.energy_kwh, r.accuracy), fontsize=7, xytext=(4, 3), textcoords="offset points")
    best, px, py = -np.inf, [], []
    for _, r in g.sort_values("energy_kwh").iterrows():
        if r.accuracy > best:
            best = r.accuracy
            px.append(r.energy_kwh)
            py.append(r.accuracy)
    ax.step(px, py, where="post", color="gray", ls="--", lw=1, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Average energy per sample (kWh, log scale)")
    ax.set_ylabel("Accuracy" if task != "summarization" else "ROUGE-L F1")
    ax.set_title(TASK_NAME[task])
    ax.grid(alpha=0.3)
fig.legend(handles=LEGEND + [plt.Line2D([], [], color="gray", ls="--", label="Pareto front")],
           loc="upper center", ncol=4, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, "Fig2_pareto_tau7")

# ---------------------------------------------------------------- figure 3
labels = [c.capitalize() for c in CAT]
cat_means = d7.groupby("model_category").agg(e=("energy_kwh", "mean"), t=("n_turns", "mean")).loc[CAT]
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6))
axs[0].bar(labels, cat_means["e"], color=[COL[k] for k in CAT])
axs[0].set_ylabel("kWh")
axs[0].set_title("Average energy per sample")
axs[1].bar(labels, cat_means["t"], color=[COL[k] for k in CAT])
axs[1].set_ylabel("Turns")
axs[1].set_title("Average number of turns")
conf = [d7.loc[d7.model_category == k, "first_conf"].values for k in CAT]
parts = axs[2].violinplot(conf, positions=[0, 1, 2], showmedians=True, widths=0.8)
for body, k in zip(parts["bodies"], CAT):
    body.set_facecolor(COL[k])
    body.set_edgecolor("gray")
    body.set_alpha(0.9)
for key in ("cmedians", "cbars", "cmins", "cmaxes"):
    parts[key].set_color("gray")
axs[2].axhline(7, color="gray", ls=":", lw=0.9)
axs[2].text(2.45, 7.15, "τ = 7", fontsize=7, color="gray", ha="right")
axs[2].set_xticks([0, 1, 2])
axs[2].set_xticklabels(labels)
axs[2].set_ylim(0.5, 10.5)
axs[2].set_ylabel("First-turn confidence score")
axs[2].set_title("Distribution of self-reported confidence")
fig.tight_layout()
save(fig, "Fig3_category_energy_turns_confidence")

# ---------------------------------------------------------------- figure 4
# Energy ratios are loop energy divided by the energy of a single generation call
# (Table 4 of the paper; derived from the component-level run as described in Section 4.8).
RATIO = {
    "classification": {"llama-3.2-3b": 4.3, "qwen2.5-3b": 5.9, "gemma-2-2b": 4.7, "llama-3-8b": 2.8,
                       "qwen2.5-7b": 3.5, "mistral-7b": 4.4, "llama-3-70b": 2.7, "qwen2.5-72b": 2.5,
                       "mistral-large": 2.3},
    "math": {"llama-3.2-3b": 1.02, "qwen2.5-3b": 1.12, "gemma-2-2b": 1.17, "llama-3-8b": 1.02,
             "qwen2.5-7b": 1.05, "mistral-7b": 1.02, "llama-3-70b": 1.06, "qwen2.5-72b": 1.08,
             "mistral-large": 1.06},
    "summarization": {"llama-3.2-3b": 1.06, "qwen2.5-3b": 2.79, "gemma-2-2b": 1.07, "llama-3-8b": 1.23,
                      "qwen2.5-7b": 1.11, "mistral-7b": 1.05, "llama-3-70b": 1.11, "qwen2.5-72b": 1.17,
                      "mistral-large": 1.14},
}
LABEL = {"classification": None,
         "math": {"qwen2.5-3b", "gemma-2-2b", "llama-3-70b"},
         "summarization": {"qwen2.5-3b", "llama-3-8b", "qwen2.5-72b"}}
OFFSET = {"llama-3.2-3b": (4, 12), "qwen2.5-7b": (-62, 8), "mistral-7b": (14, 6)}

paired = pd.read_csv(os.path.join(ANALYSIS, "paired_first_vs_loop.csv"))
paired["ratio"] = [RATIO[t][m] for t, m in zip(paired.task, paired.model)]
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, task in zip(axs, TASKS):
    g = paired[paired.task == task]
    ax.errorbar(g.ratio, g.gain, yerr=[g.gain - g.ci_lo, g.ci_hi - g.gain],
                fmt="none", ecolor="lightgray", zorder=2)
    ax.scatter(g.ratio, g.gain, c=[COL[c] for c in g.category], s=55, zorder=3)
    for _, r in g.iterrows():
        if LABEL[task] is not None and r.model not in LABEL[task]:
            continue
        moved = task == "classification" and r.model in OFFSET
        dx, dy = OFFSET[r.model] if moved else (4, 3)
        ax.annotate(r.model, (r.ratio, r.gain), fontsize=7, xytext=(dx, dy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, shrinkA=0, shrinkB=3) if moved else None)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(1, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Loop energy / single generation call energy")
    ax.set_ylabel("Paired accuracy difference" if task != "summarization" else "Paired ROUGE-L difference")
    ax.set_title(TASK_NAME[task])
    ax.grid(alpha=0.3)
    if task != "summarization":
        ax.set_ylim(-0.05, 0.05)
fig.legend(handles=LEGEND, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.92])
save(fig, "Fig4_loop_vs_single_shot")

# ---------------------------------------------------------------- figure 5
rel = pd.read_csv(os.path.join(ANALYSIS, "reliability_table.csv"))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].bar(rel.first_conf, rel.acc, color="#9FA8DA")
for x, m, n in zip(rel.first_conf, rel.acc, rel.n):
    axs[0].text(x, m + 0.01, f"n={int(n)}", ha="center", fontsize=7)
axs[0].set_xlabel("First-turn self-reported confidence (1–10)")
axs[0].set_ylabel("Accuracy")
axs[0].set_ylim(0, 1.05)
axs[0].set_xticks(range(1, 11))
axs[0].set_title("Accuracy by confidence score")
axs[1].plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Perfect calibration")
axs[1].scatter(rel.conf_p, rel.acc, s=rel.n / 8, color="#9FA8DA", edgecolor="gray", zorder=3,
               label="Observed (size ∝ n)")
axs[1].set_xlabel("Self-reported confidence / 10")
axs[1].set_ylabel("Observed accuracy")
axs[1].set_xlim(0, 1.05)
axs[1].set_ylim(0, 1.05)
axs[1].set_title("Reliability diagram (pooled, ECE = 0.16)")
axs[1].legend(frameon=False, fontsize=8, loc="lower right")
axs[1].grid(alpha=0.3)
fig.tight_layout()
save(fig, "Fig5_confidence_calibration")
