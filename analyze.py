"""Aggregate runs/*.json into LaTeX tables and figures for the paper (English and Turkish)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "runs"
OUT = "paper"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
KEYS = ["add_easy", "rev_easy", "sort_easy", "add_hard", "rev_hard", "sort_hard"]
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1a1a19", "#6b6a63", "#e6e5df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "font.family": "serif"})

NAMES = [None, "rl", "sft", "rl_nobase", "rl_dense", "rl_nokl", "rl_sortonly", "rl_hilr_nokl", "rl_hilr_kl1"]
T = {
    "en": dict(sfx="", dec=".",
               rows=["Pretrained base", "RL (default)", "SFT, gold labels (reference)", "RL, no baseline",
                     "RL, dense reward", "RL, no KL anchor", "RL, sort only",
                     "RL, lr $3{\\times}10^{-4}$, no KL", "RL, lr $3{\\times}10^{-4}$, KL 1.0"],
               titles=["Addition (hard)", "Reverse (hard)", "Sort (hard)"], rl="RL (reward only)",
               sft="SFT (gold labels)", x1="fine-tuning steps", x2="RL steps", y="greedy exact match (%)",
               st=[r"lr $3\times10^{-4}$, no KL", r"lr $3\times10^{-4}$, KL 1.0", r"lr $10^{-4}$, KL 0.1"],
               series=["sort (hard)", "reverse (hard)", "addition (easy)"]),
    "tr": dict(sfx="_tr", dec=",",
               rows=["Ön-eğitimli temel model", "PÖ (varsayılan)", "Dİİ, altın etiketli (referans)",
                     "PÖ, taban değersiz", "PÖ, yoğun ödül", "PÖ, KL çapasız", "PÖ, yalnızca sıralama",
                     "PÖ, öo $3{\\times}10^{-4}$, KL yok", "PÖ, öo $3{\\times}10^{-4}$, KL 1{,}0"],
               titles=["Toplama (zor)", "Ters çevirme (zor)", "Sıralama (zor)"], rl="PÖ (yalnızca ödül)",
               sft="Dİİ (altın etiketler)", x1="ince ayar adımı", x2="PÖ adımı", y="açgözlü tam eşleşme (%)",
               st=[r"öo $3\times10^{-4}$, KL yok", r"öo $3\times10^{-4}$, KL 1,0", r"öo $10^{-4}$, KL 0,1"],
               series=["sıralama (zor)", "ters çevirme (zor)", "toplama (kolay)"]),
}


def load(name):
    return [json.load(open(f"{R}/{name}_s{s}.json")) for s in SEEDS]


def final(name):
    """seeds x keys matrix of final accuracies (in %)."""
    return np.array([[100 * d["log"][-1][k] for k in KEYS] for d in load(name)])


def curve(name, key):
    runs = load(name)
    steps = [e["step"] for e in runs[0]["log"]]
    return np.array(steps), np.array([[100 * e[key] for e in d["log"]] for d in runs])


def base():
    return np.array([[100 * d["log"][0][k] for k in KEYS] for d in load("rl")])


for lang, t in T.items():
    def cell(m):
        s = f"{m.mean():.1f} $\\pm$ {m.std():.1f}"
        return s.replace(".", "{,}") if t["dec"] == "," else s

    lines, summary = [], {}
    for label, name in zip(t["rows"], NAMES):
        m = base() if name is None else final(name)
        summary[label] = {k: [float(m[:, i].mean()), float(m[:, i].std())] for i, k in enumerate(KEYS)}
        summary[label]["per_seed"] = m.tolist()
        lines.append(label + " & " + " & ".join(cell(m[:, i]) for i in range(6)) + " \\\\")
        if name in (None, "sft", "rl_sortonly"):
            lines.append("\\midrule")
    open(f"{OUT}/table_main{t['sfx']}.tex", "w").write("\n".join(lines) + "\n")
    if lang == "en":
        json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
        for k, v in summary.items():
            print(f"{k:45s}", " ".join(f"{v[q][0]:5.1f}±{v[q][1]:4.1f}" for q in KEYS))

    # ---- Figure 1: RL vs SFT learning curves on the hard regime -------------
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    for ax, key, title in zip(axs, ["add_hard", "rev_hard", "sort_hard"], t["titles"]):
        for name, col, lab in [("rl", BLUE, t["rl"]), ("sft", ORANGE, t["sft"])]:
            st, ys = curve(name, key)
            ax.fill_between(st, ys.min(0), ys.max(0), color=col, alpha=0.15, lw=0)
            ax.plot(st, ys.mean(0), color=col, lw=2, label=lab)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(t["x1"])
        ax.set_ylim(-3, 103)
    axs[0].set_ylabel(t["y"])
    axs[0].legend(frameon=False, loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_curves{t['sfx']}.pdf"); fig.savefig(f"{OUT}/fig_curves{t['sfx']}.png", dpi=200)

    # ---- Figure 2: stability -- per-seed curves -----------------------------
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    for ax, name, title in zip(axs, ["rl_hilr_nokl", "rl_hilr_kl1", "rl"], t["st"]):
        for key, col, lab in zip(["sort_hard", "rev_hard", "add_easy"], [BLUE, ORANGE, AQUA], t["series"]):
            st, ys = curve(name, key)
            for i, y in enumerate(ys):
                ax.plot(st, y, color=col, lw=1.4, alpha=0.85, label=lab if i == 0 else None)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(t["x2"])
        ax.set_ylim(-3, 103)
    axs[0].set_ylabel(t["y"])
    axs[2].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_stability{t['sfx']}.pdf"); fig.savefig(f"{OUT}/fig_stability{t['sfx']}.png", dpi=200)
