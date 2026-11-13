"""Figures for the Results section, drawn from the result files, plus the derived numbers the text uses.

    python paper/figures/scripts/r_results_figures.py

Writes four 300-dpi PNGs to paper/figures/:
  r_proxy.png          proxy vs medium val mAP@50 of the 8 proxy-check settings (Gen 1)
  r_convergence.png    best-so-far proxy val mAP@50 vs evaluation, Gen 1 and Gen 2 side by side
  r_gen1_per_class.png per-class test AP@50 of the three Gen 1 models
  r_domains.png        Gen 2 test mAP@50 with 95% CIs per test domain and arm

and prints the derived statistics quoted in the text: proxy ranks, search summaries (per-round mean and SD,
evaluations below the defaults' proxy score), the cross-generation Spearman correlation on the
configurations that both generations evaluated, and the recorded GPU time per run (Table 8).

Inputs (read only): results/proxy_check.json, results/search_{pso,random}.json, results/gen2/search_{pso,random}.json,
results/{baseline,arm_b,arm_c}.json, results/gen2/arm_{a,b,c}.json,
results/test_results.json, results/gen2/test_{all,studio,real_world,india}_results.json.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results"
OUT = REPO / "paper" / "figures"

# Line and bar styles of the three arms, kept the same in all figures.
ARM = {
    "A": {"label": "Defaults", "color": "k", "marker": "o", "ls": "-.", "face": "white", "hatch": ""},
    "B": {"label": "Random search", "color": "#0072BD", "marker": "^", "ls": "--", "face": "0.80", "hatch": "///"},
    "C": {"label": "PSO", "color": "#D95319", "marker": "s", "ls": "-", "face": "0.45", "hatch": ""},
}
COL_W = 3.5  # IEEE column width, inches
PAGE_W = 7.16  # IEEE text width, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "hatch.linewidth": 0.5,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "k",
    "legend.framealpha": 1.0,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def style_axes(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, color="0.75", linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    for lg in [ax.get_legend()]:
        if lg is not None:
            lg.get_frame().set_linewidth(0.6)


def legend(ax, **kw):
    lg = ax.legend(**kw)
    lg.get_frame().set_linewidth(0.6)
    return lg


def ranks(values):
    """1 = lowest, as scipy's rankdata (no ties occur in these data)."""
    order = np.argsort(values)
    r = np.empty(len(values), dtype=int)
    r[order] = np.arange(1, len(values) + 1)
    return r


# --------------------------------------------------------------------------------------------------------------
# Fig. 3: proxy validity
# --------------------------------------------------------------------------------------------------------------
def fig_proxy():
    check = load(RES / "proxy_check.json")["checks"]["proxy_e12_s416 vs medium_e40_s416"]
    ids, prox, med = check["configs"], np.array(check["proxy_map50"]), np.array(check["medium_map50"])
    rho = spearmanr(prox, med).statistic
    print("== Proxy check (Gen 1)")
    print(f"stored rho {check['rho']}, recomputed {rho:.4f}, n {check['n']}")
    for i, p, m, rp, rm in zip(ids, prox, med, ranks(prox), ranks(med)):
        print(f"  {i:<9} proxy {p:.5f} (rank {rp})  medium {m:.5f} (rank {rm})")

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    style_axes(ax)
    lhs = [k for k, i in enumerate(ids) if i != "defaults"]
    d = ids.index("defaults")
    ax.plot(prox[lhs], med[lhs], "ko", markersize=4.5, label="LHS settings")
    ax.plot(prox[d], med[d], linestyle="none", marker="^", markersize=6, markerfacecolor="white",
            markeredgecolor="#D95319", markeredgewidth=1.1, label="Defaults")
    offsets = {"lhs1": (5, -3), "lhs2": (5, -8), "lhs3": (-22, 3), "lhs5": (5, -8), "lhs7": (-7, -11),
               "lhs6": (-16, 5), "lhs4": (2, 5), "defaults": (2, -11)}
    for k, i in enumerate(ids):
        if i == "defaults":
            continue
        ax.annotate(i, (prox[k], med[k]), xytext=offsets.get(i, (5, 3)), textcoords="offset points", fontsize=7)
    ax.set_xlim(0.04, 0.54)
    ax.set_ylim(0.25, 0.68)
    ax.set_xlabel("Proxy validation mAP@50 (12 epochs, 40% subset)")
    ax.set_ylabel("Medium validation mAP@50 (40 epochs)")
    ax.text(0.96, 0.07, rf"$\rho$ = {check['rho']:.3f} ($n$ = {check['n']})", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    legend(ax, loc="upper left", handletextpad=0.3)
    fig.savefig(OUT / "r_proxy.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
# Fig. 4: search convergence (and the derived search statistics)
# --------------------------------------------------------------------------------------------------------------
def search_stats(tag, log, defaults=None):
    ev = log["evaluations"]
    s = np.array([e["map50"] for e in ev])
    best = int(np.argmax(s))
    line = (f"  {tag:<11} best {s[best]:.5f} ({ev[best]['label']}, evaluation {best + 1}) "
            f"mean {s.mean():.5f} SD {s.std(ddof=1):.5f} min {s.min():.5f} "
            f"time {sum(e['seconds'] for e in ev):.1f} s")
    if defaults is not None:
        line += f"  below defaults {(s < defaults).sum()} of {len(s)}"
    print(line)
    if ev[0].get("round"):
        for r in sorted({e["round"] for e in ev}):
            rs = np.array([e["map50"] for e in ev if e["round"] == r])
            print(f"    round {r}: mean {rs.mean():.5f} SD {rs.std(ddof=1):.5f} max {rs.max():.5f}")
    return s


def same_params(a, b):
    return all(np.isclose(a["params"][k], b["params"][k], rtol=1e-9, atol=0) for k in a["params"])


def fig_convergence():
    g1 = {"B": load(RES / "search_random.json"), "C": load(RES / "search_pso.json")}
    g2 = {"B": load(RES / "gen2" / "search_random.json"), "C": load(RES / "gen2" / "search_pso.json")}
    defaults = load(RES / "proxy_check.json")["runs"]["proxy_e12_s416_defaults"]["map50"]
    n_particles = g1["C"]["settings"]["n_particles"]

    print("== Search (proxy val mAP@50)")
    print(f"  Gen 1 defaults on the same proxy: {defaults}")
    scores = {}
    for gen, logs, ref in (("Gen 1", g1, defaults), ("Gen 2", g2, None)):
        for arm, log in logs.items():
            scores[(gen, arm)] = search_stats(f"{gen} {arm}", log, ref)

    print("== Cross-generation rank agreement on identical configurations")
    for arm in ("B", "C"):
        e1, e2 = g1[arm]["evaluations"], g2[arm]["evaluations"]
        idx = [i for i in range(min(len(e1), len(e2))) if same_params(e1[i], e2[i])]
        r = spearmanr([e1[i]["map50"] for i in idx], [e2[i]["map50"] for i in idx])
        rounds = sorted({e1[i]["round"] for i in idx}) if e1[0].get("round") else None
        print(f"  {arm}: {len(idx)} identical configurations (indices {idx[0]}-{idx[-1]}"
              f"{', rounds ' + str(rounds) if rounds else ''}); Spearman {r.statistic:.4f}, p {r.pvalue:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.5))
    panels = (("(a)", "Gen 1", g1, (0.05, 0.56)), ("(b)", "Gen 2", g2, (0.03, 0.38)))
    for ax, (tag, gen, logs, ylim) in zip(axes, panels):
        style_axes(ax, "y")
        n = len(logs["C"]["evaluations"])
        for b in range(n_particles, n, n_particles):
            ax.axvline(b + 0.5, color="0.4", linestyle=":", linewidth=0.7)
        for arm in ("B", "C"):
            st = ARM[arm]
            s_ = scores[(gen, arm)]
            x = np.arange(1, len(s_) + 1)
            ax.plot(x, s_, linestyle="none", marker=st["marker"], markersize=3.2, markerfacecolor="none",
                    markeredgecolor=st["color"], markeredgewidth=0.6, label=f"{st['label']} (evaluations)")
            ax.step(x, np.maximum.accumulate(s_), where="post", color=st["color"], linestyle=st["ls"],
                    linewidth=1.3, label=f"{st['label']} (best so far)")
        if gen == "Gen 1":
            ax.axhline(defaults, color="k", linestyle="-.", linewidth=0.9, label="Defaults")
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(*ylim)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
        ax.set_xlabel(f"Number of evaluations\n{tag}")
        ax.set_ylabel("Proxy validation mAP@50")
    h, l = axes[0].get_legend_handles_labels()
    lg = fig.legend(h, l, loc="upper center", ncol=5, handlelength=2.4, columnspacing=1.2, borderpad=0.4)
    lg.get_frame().set_linewidth(0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)
    fig.savefig(OUT / "r_convergence.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
# Fig. 6: Gen 1 per-class test AP@50
# --------------------------------------------------------------------------------------------------------------
def fig_gen1_per_class():
    test = load(RES / "test_results.json")
    classes = list(test["models"]["A"]["per_class"].keys())
    boxes = test["boxes_per_class"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    style_axes(ax, "y")
    x = np.arange(len(classes))
    width = 0.26
    print("== Gen 1 per-class test AP@50")
    for c in classes:
        vals = [test["models"][a]["per_class"][c]["ap50"] for a in "ABC"]
        print(f"  {c:<14} ({boxes[c]:,} boxes) " + "  ".join(f"{a} {v:.5f}" for a, v in zip("ABC", vals)))
    for j, a in enumerate("ABC"):
        vals = [test["models"][a]["per_class"][c]["ap50"] for c in classes]
        ax.bar(x + (j - 1) * width, vals, width, color=ARM[a]["face"], edgecolor="k", linewidth=0.6,
               hatch=ARM[a]["hatch"], label=ARM[a]["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], rotation=25, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", top=False, length=0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Test AP@50")
    legend(ax, loc="upper center", ncol=3, handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(OUT / "r_gen1_per_class.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
# Fig. 5: Gen 2 test mAP@50 per domain
# --------------------------------------------------------------------------------------------------------------
def fig_domains():
    tags = [("all", "All"), ("studio", "Studio"), ("real_world", "Real-world"), ("india", "India (DWSD)")]
    runs = {t: load(RES / "gen2" / f"test_{t}_results.json") for t, _ in tags}
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    style_axes(ax, "y")
    width = 0.26
    x = np.arange(len(tags))
    print("== Gen 2 test mAP@50 [95% CI] per domain")
    for j, a in enumerate("ABC"):
        vals = np.array([runs[t]["models"][a]["metrics"]["map50"] for t, _ in tags])
        ci = np.array([runs[t]["models"][a]["bootstrap"]["map50_ci95"] for t, _ in tags])
        err = np.vstack([vals - ci[:, 0], ci[:, 1] - vals])
        xs = x + (j - 1) * width
        ax.bar(xs, vals, width, color=ARM[a]["face"], edgecolor="k", linewidth=0.6, hatch=ARM[a]["hatch"],
               label=ARM[a]["label"])
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor="k", elinewidth=0.7, capsize=2, capthick=0.7)
        print(f"  {a}: " + "  ".join(f"{t} {v:.3f} [{lo:.3f}, {hi:.3f}]"
                                     for (t, _), v, (lo, hi) in zip(tags, vals, ci)))
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in tags])
    ax.tick_params(axis="x", top=False, length=0)
    ax.set_ylim(0, 0.8)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_xlabel("Test set")
    ax.set_ylabel("Test mAP@50")
    legend(ax, loc="upper right", handlelength=1.4, handletextpad=0.4)
    fig.savefig(OUT / "r_domains.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
# Table 8: recorded GPU training time (no figure; printed for the text)
# --------------------------------------------------------------------------------------------------------------
def compute_cost():
    """Sums of the recorded `seconds` fields: training wall-clock only, idle and setup time excluded."""
    check = load(RES / "proxy_check.json")["runs"]
    rows = {
        "Gen 1": {
            "Proxy-validity check": sum(v["seconds"] for v in check.values()),
            "PSO search": sum(e["seconds"] for e in load(RES / "search_pso.json")["evaluations"]),
            "Random search": sum(e["seconds"] for e in load(RES / "search_random.json")["evaluations"]),
            "Arm A training": load(RES / "baseline.json")["seconds"],
            "Arm B training": load(RES / "arm_b.json")["seconds"],
            "Arm C training": load(RES / "arm_c.json")["seconds"],
        },
        "Gen 2": {
            "PSO search": sum(e["seconds"] for e in load(RES / "gen2" / "search_pso.json")["evaluations"]),
            "Random search": sum(e["seconds"] for e in load(RES / "gen2" / "search_random.json")["evaluations"]),
            "Arm A training": load(RES / "gen2" / "arm_a.json")["seconds"],
            "Arm B training": load(RES / "gen2" / "arm_b.json")["seconds"],
            "Arm C training": load(RES / "gen2" / "arm_c.json")["seconds"],
        },
    }
    print("== Recorded GPU time (Tesla T4)")
    for gen, r in rows.items():
        for name, sec in r.items():
            print(f"  {gen} {name:<22} {sec:9.1f} s  {sec / 3600:5.2f} h")
        total = sum(r.values())
        print(f"  {gen} {'Total':<22} {total:9.1f} s  {total / 3600:5.2f} h")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    compute_cost()
    fig_proxy()
    fig_convergence()
    fig_gen1_per_class()
    fig_domains()
    for name in ("r_proxy.png", "r_convergence.png", "r_gen1_per_class.png", "r_domains.png"):
        print("wrote", (OUT / name).relative_to(REPO), (OUT / name).stat().st_size, "bytes")
