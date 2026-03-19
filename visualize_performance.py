"""
Visualisation des performances : output_v3, output_optimise, greedy, random
Graphique 1 : Comparaison des métriques finales (barres groupées)
Graphique 2 : Courbes d'apprentissage (win rate et reward par épisode)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ─── Chargement des données ───────────────────────────────────────────────────

BASE = Path(__file__).parent

with open(BASE / "outputs_v3/logs/training_history.json") as f:
    v3 = json.load(f)

with open(BASE / "outputs_optimise/logs/training_history.json") as f:
    opt = json.load(f)

with open(BASE / "benchmark_results.json") as f:
    bench = {b["policy"]: b for b in json.load(f)}

# Meilleur checkpoint par win_rate pour chaque modèle DQN
def best_eval(history):
    wr = history["eval_win_rates"]
    idx = int(np.argmax(wr))
    return {
        "win_rate":  wr[idx],
        "reward":    history["eval_rewards"][idx],
        "score":     history["eval_scores"][idx],
        "avg_rank":  history["eval_avg_ranks"][idx],
        "episode":   history["eval_checkpoints"][idx],
    }

v3_best  = best_eval(v3)
opt_best = best_eval(opt)

# ─── Palette ──────────────────────────────────────────────────────────────────

COLORS = {
    "random":         "#e74c3c",
    "greedy":         "#f39c12",
    "output_v3":      "#2980b9",
    "output_optimise":"#27ae60",
}

LABELS = {
    "random":          f"Random",
    "greedy":          f"Greedy",
    "output_v3":       f"DQN v3\n(best ep {v3_best['episode']})",
    "output_optimise": f"DQN optimisé\n(best ep {opt_best['episode']})",
}

# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 1 — Comparaison finale (4 métriques en barres groupées)
# ══════════════════════════════════════════════════════════════════════════════

fig1, axes = plt.subplots(1, 3, figsize=(16, 6))
fig1.suptitle("Comparaison des agents – meilleur checkpoint DQN vs baselines",
              fontsize=14, fontweight="bold", y=1.01)

models   = ["random", "greedy", "output_v3", "output_optimise"]
colors   = [COLORS[m] for m in models]
x        = np.arange(len(models))
bar_w    = 0.6

# --- Taux de victoire (win_rate) ---
ax = axes[0]
values = [
    bench["random"]["win_rate"],
    bench["greedy"]["win_rate"],
    v3_best["win_rate"],
    opt_best["win_rate"],
]
bars = ax.bar(x, values, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
ax.set_title("Taux de victoire", fontsize=12, fontweight="bold")
ax.set_ylabel("Win rate")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
ax.set_ylim(0, 1.0)
ax.axhline(0.33, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="Chance (1/3)")
ax.legend(fontsize=8)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{val:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# --- Reward moyen ---
ax = axes[1]
values = [
    bench["random"]["reward"],
    bench["greedy"]["reward"],
    v3_best["reward"],
    opt_best["reward"],
]
bars = ax.bar(x, values, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
ax.set_title("Reward moyen", fontsize=12, fontweight="bold")
ax.set_ylabel("Reward")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
for bar, val in zip(bars, values):
    ypos = bar.get_height() + (1.5 if val >= 0 else -4)
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# --- Rang moyen (lower = better) ---
ax = axes[2]
values = [
    bench["random"]["avg_rank"],
    bench["greedy"]["avg_rank"],
    v3_best["avg_rank"],
    opt_best["avg_rank"],
]
bars = ax.bar(x, values, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
ax.set_title("Rang moyen\n(1 = meilleur)", fontsize=12, fontweight="bold")
ax.set_ylabel("Rang moyen")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in models], fontsize=9)
ax.set_ylim(0, 3.2)
ax.axhline(1.0, color="gold", linestyle="--", linewidth=1, alpha=0.8, label="Rang 1 (1er)")
ax.legend(fontsize=8)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f8f9fa")

fig1.tight_layout()
out1 = BASE / "graph1_comparaison_finale.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Graphique 1 sauvegardé : {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 2 — Courbes d'apprentissage
# ══════════════════════════════════════════════════════════════════════════════

fig2, (ax_wr, ax_rw) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
fig2.suptitle("Courbes d'apprentissage – DQN v3 vs DQN optimisé",
              fontsize=14, fontweight="bold")

# ── Win rate (éval périodique + MA entraînement) ──────────────────────────────

# v3 — évals
ax_wr.plot(v3["eval_checkpoints"], v3["eval_win_rates"],
           "o-", color=COLORS["output_v3"], linewidth=1.8, markersize=5,
           label="DQN v3 – éval (20 ep)", zorder=3)
# v3 — MA entraînement (fenêtre 50)
ep_v3 = np.arange(1, len(v3["win_rate_ma"]) + 1)
ax_wr.plot(ep_v3, v3["win_rate_ma"],
           color=COLORS["output_v3"], linewidth=0.8, alpha=0.35,
           linestyle="--", label="DQN v3 – MA train (50 ep)")

# optimise — évals
ax_wr.plot(opt["eval_checkpoints"], opt["eval_win_rates"],
           "s-", color=COLORS["output_optimise"], linewidth=1.8, markersize=5,
           label="DQN optimisé – éval (20 ep)", zorder=3)
# optimise — MA entraînement
ep_opt = np.arange(1, len(opt["win_rate_ma"]) + 1)
ax_wr.plot(ep_opt, opt["win_rate_ma"],
           color=COLORS["output_optimise"], linewidth=0.8, alpha=0.35,
           linestyle="--", label="DQN optimisé – MA train (50 ep)")

# baselines horizontales
ax_wr.axhline(bench["greedy"]["win_rate"], color=COLORS["greedy"],
              linestyle=":", linewidth=1.5, alpha=0.8, label=f"Greedy ({bench['greedy']['win_rate']:.0%})")
ax_wr.axhline(bench["random"]["win_rate"], color=COLORS["random"],
              linestyle=":", linewidth=1.5, alpha=0.8, label=f"Random ({bench['random']['win_rate']:.0%})")
ax_wr.axhline(0.333, color="grey", linestyle="--", linewidth=0.8, alpha=0.4)

ax_wr.set_ylabel("Taux de victoire", fontsize=11)
ax_wr.set_xlabel("Épisode", fontsize=11)
ax_wr.set_title("Évolution du taux de victoire", fontsize=12, fontweight="bold")
ax_wr.set_ylim(-0.02, 1.02)
ax_wr.legend(fontsize=9, loc="upper left", ncol=2)
ax_wr.spines["top"].set_visible(False)
ax_wr.spines["right"].set_visible(False)
ax_wr.set_facecolor("#f8f9fa")

# ── Reward moyen (éval périodique) ────────────────────────────────────────────

ax_rw.plot(v3["eval_checkpoints"], v3["eval_rewards"],
           "o-", color=COLORS["output_v3"], linewidth=1.8, markersize=5,
           label="DQN v3 – éval")
ax_rw.fill_between(v3["eval_checkpoints"], v3["eval_rewards"],
                   alpha=0.12, color=COLORS["output_v3"])

ax_rw.plot(opt["eval_checkpoints"], opt["eval_rewards"],
           "s-", color=COLORS["output_optimise"], linewidth=1.8, markersize=5,
           label="DQN optimisé – éval")
ax_rw.fill_between(opt["eval_checkpoints"], opt["eval_rewards"],
                   alpha=0.12, color=COLORS["output_optimise"])

ax_rw.axhline(bench["greedy"]["reward"], color=COLORS["greedy"],
              linestyle=":", linewidth=1.5, alpha=0.8, label=f"Greedy ({bench['greedy']['reward']:.0f})")
ax_rw.axhline(bench["random"]["reward"], color=COLORS["random"],
              linestyle=":", linewidth=1.5, alpha=0.8, label=f"Random ({bench['random']['reward']:.1f})")
ax_rw.axhline(0, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)

ax_rw.set_ylabel("Reward moyen", fontsize=11)
ax_rw.set_xlabel("Épisode", fontsize=11)
ax_rw.set_title("Évolution du reward moyen (évaluation périodique)", fontsize=12, fontweight="bold")
ax_rw.legend(fontsize=9, loc="upper left")
ax_rw.spines["top"].set_visible(False)
ax_rw.spines["right"].set_visible(False)
ax_rw.set_facecolor("#f8f9fa")

fig2.tight_layout()
out2 = BASE / "graph2_courbes_apprentissage.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Graphique 2 sauvegardé : {out2}")

plt.show()
print("\nTerminé.")
