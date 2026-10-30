"""
Prepare a CivilComments (WILDS) sample for the label-noise experiments.

Source: Hugging Face mirror of CivilComments-WILDS (Koh et al., 2021; Borkan et al., 2019)
  pietrolesci/civilcomments-wilds  (columns: comment_text, toxicity, identity flags)

Group definition (default): comments that mention a Black, Muslim or LGBTQ identity.
Identity-mentioning comments are known to be over-flagged as toxic by classifiers
(Dixon et al., 2018; Borkan et al., 2019), so false-positive gaps are the main harm.

The sample is enriched so the group is large enough for stable group metrics:
  train: --n-train comments, --group-share-train of them from the group
  test : --n-test comments,  --group-share-test of them from the group

Usage
-----
    python prepare_data.py                      # downloads from Hugging Face
    python prepare_data.py --group-cols muslim   # a single identity group
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_split(split: str) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("pietrolesci/civilcomments-wilds", "default", split=split)
    return ds.to_pandas()


def make_sample(df, group_cols, n, group_share, rng):
    df = df.copy()
    df["text"] = df["comment_text"].astype(str)
    df["label"] = df["toxicity"].astype(int)          # 1 = toxic
    df["group"] = (df[group_cols].sum(axis=1) > 0).astype(int)
    df = df[df["text"].str.strip().str.len() > 0]

    n_group = int(round(n * group_share))
    g = df[df["group"] == 1]
    o = df[df["group"] == 0]
    if len(g) < n_group:
        raise ValueError(f"Only {len(g)} group comments available, need {n_group}")
    sample = pd.concat([
        g.sample(n_group, random_state=int(rng.integers(1e9))),
        o.sample(n - n_group, random_state=int(rng.integers(1e9))),
    ]).sample(frac=1, random_state=int(rng.integers(1e9)))
    return sample[["text", "label", "group"]].reset_index(drop=True)


def describe(name, d):
    t = d.groupby("group")["label"].agg(["count", "mean"]).rename(
        columns={"count": "n", "mean": "toxic_rate"})
    t.index = t.index.map({0: "other", 1: "group"})
    print(f"\n{name}: n = {len(d)}\n{t.round(3).to_string()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-cols", nargs="+", default=["black", "muslim", "LGBTQ"])
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=8000)
    ap.add_argument("--group-share-train", type=float, default=0.3)
    ap.add_argument("--group-share-test", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    train = make_sample(load_split("train"), args.group_cols, args.n_train,
                        args.group_share_train, rng)
    test = make_sample(load_split("test"), args.group_cols, args.n_test,
                       args.group_share_test, rng)

    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    describe("Train", train)
    describe("Test", test)
    print(f"\nGroup = mentions of: {', '.join(args.group_cols)}. Saved to {out}/")


if __name__ == "__main__":
    main()
