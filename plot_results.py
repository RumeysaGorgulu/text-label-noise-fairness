"""Turn results/<model>/results.csv files into figures and a Markdown summary.

    python plot_results.py --inp results/tfidf/results.csv results/transformer/results.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "group_flip": "Group's non-toxic comments relabeled toxic",
    "matched_random": "Same number of labels flipped at random",
    "matched_other": "Same number relabeled toxic, other comments",
}
COLORS = {"group_flip": "#C44E52", "matched_random": "#4C72B0", "matched_other": "#55A868"}
MODEL_NAMES = {"tfidf": "TF-IDF + logistic regression", "transformer": "DistilBERT (fine-tuned)"}
METRICS = [
    ("auc", "Overall AUC", None),
    ("fpr_gap", "FPR gap (group − other)", 0),
    ("bpsn_auc", "BPSN AUC (lower = group over-flagged)", None),
]


def with_clean(res):
    """Copy the clean (severity 0) runs into every corruption so lines start at 0."""
    clean = res[res["corruption"] == "clean"]
    parts = [res[res["corruption"] != "clean"]]
    for kind in res.loc[res["corruption"] != "clean", "corruption"].unique():
        parts.append(clean.assign(corruption=kind))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", nargs="+", default=["results/tfidf/results.csv"])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    res = with_clean(pd.concat([pd.read_csv(p) for p in args.inp], ignore_index=True))
    models = [m for m in MODEL_NAMES if m in res["model"].unique()]

    # Figure: rows = models, columns = metrics
    fig, axes = plt.subplots(len(models), len(METRICS), figsize=(15, 3.6 * len(models)),
                             squeeze=False)
    agg = (res.groupby(["model", "corruption", "severity"])[[m for m, _, _ in METRICS]]
              .agg(["mean", "std"]).reset_index())
    for r, model in enumerate(models):
        for c, (metric, label, hline) in enumerate(METRICS):
            ax = axes[r, c]
            for kind in LABELS:
                d = agg[(agg["model"] == model) & (agg["corruption"] == kind)].sort_values("severity")
                if d.empty:
                    continue
                m, s = d[(metric, "mean")], d[(metric, "std")].fillna(0)
                ax.plot(d["severity"], m, marker="o", ms=3, color=COLORS[kind], label=LABELS[kind])
                ax.fill_between(d["severity"], m - s, m + s, color=COLORS[kind], alpha=0.15)
            if hline is not None:
                ax.axhline(hline, color="grey", lw=0.8, ls="--")
            ax.set_title(f"{MODEL_NAMES[model]}\n{label}", fontsize=9)
            ax.set_xlabel("Severity (share of group's non-toxic comments)")
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary table: clean vs. worst severity
    m = res.groupby(["model", "corruption", "severity"]).mean(numeric_only=True).reset_index()
    rows = []
    for model in models:
        for kind in LABELS:
            d = m[(m["model"] == model) & (m["corruption"] == kind)].sort_values("severity")
            if d.empty:
                continue
            c, w = d.iloc[0], d.iloc[-1]
            rows.append({
                "Model": MODEL_NAMES[model], "Corruption": LABELS[kind],
                "Severity": w["severity"],
                "Labels changed": f"{100 * w['share_labels_changed']:.1f}%",
                "AUC": f"{c['auc']:.3f} → {w['auc']:.3f}",
                "Accuracy": f"{c['accuracy']:.3f} → {w['accuracy']:.3f}",
                "FPR gap": f"{c['fpr_gap']:+.3f} → {w['fpr_gap']:+.3f}",
                "BPSN AUC": f"{c['bpsn_auc']:.3f} → {w['bpsn_auc']:.3f}",
            })
    table = pd.DataFrame(rows)
    md = ["# Results: clean training data → highest severity (mean over seeds)", "",
          table.to_markdown(index=False), "",
          f"Seeds: {res['seed'].nunique()}. FPR gap > 0 means the group's non-toxic comments "
          "are flagged as toxic more often than other non-toxic comments.", "",
          "![Results](fig_results.png)"]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:-1]))


if __name__ == "__main__":
    main()
