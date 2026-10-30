"""
Does group-targeted label noise hide fairness harm in text classifiers?

Training labels are corrupted in controlled ways; every model is evaluated on the
same CLEAN test set.

Corruptions (training data only)
  clean           no corruption (severity 0)
  group_flip      non-toxic comments that mention the group are relabeled toxic with
                  probability = severity (models annotator bias against identity mentions)
  matched_random  the SAME NUMBER of labels as group_flip, flipped at random rows
                  (control 1: separates "which labels are wrong" from "how many")
  matched_other   the SAME NUMBER of non-toxic comments relabeled toxic, but chosen from
                  comments that do NOT mention the group (control 2: same number and
                  direction of errors, different target)
  random          a random share (= severity) of all labels is flipped (optional)

Models
  tfidf           TF-IDF (word 1-2 grams) + logistic regression        (CPU, seconds)
  transformer     fine-tuned DistilBERT, plain PyTorch training loop   (GPU, minutes)

Usage
-----
    python run_experiment.py --model tfidf
    python run_experiment.py --model transformer          # needs a GPU (e.g., Colab T4)

Results are appended to results/<model>/results.csv after every run, so an
interrupted session can be resumed by running the same command again.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CORRUPTIONS = ["group_flip", "matched_random", "matched_other"]
KIND_OFFSET = {"clean": 0, "group_flip": 1, "matched_random": 2, "matched_other": 3, "random": 4}


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------
def corrupt(y, g, kind, severity, rng):
    y = y.copy()
    if kind == "clean" or severity == 0:
        return y
    target = (g == 1) & (y == 0)
    if kind == "group_flip":
        flip = target & (rng.random(len(y)) < severity)
        y[flip] = 1
    elif kind == "matched_random":
        k = int(round(severity * target.sum()))
        idx = rng.choice(len(y), size=k, replace=False)
        y[idx] = 1 - y[idx]
    elif kind == "matched_other":
        k = int(round(severity * target.sum()))
        pool = np.flatnonzero((g == 0) & (y == 0))
        idx = rng.choice(pool, size=min(k, len(pool)), replace=False)
        y[idx] = 1
    elif kind == "random":
        flip = rng.random(len(y)) < severity
        y[flip] = 1 - y[flip]
    else:
        raise ValueError(kind)
    return y


# ---------------------------------------------------------------------------
# Metrics (on the clean test set)
# ---------------------------------------------------------------------------
def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def evaluate(y, p, g, threshold=0.5):
    pred = (p >= threshold).astype(int)

    def rate(mask, cond):
        m = mask & cond
        return pred[m].mean() if m.any() else np.nan

    grp, oth = g == 1, g == 0
    fpr_g, fpr_o = rate(grp, y == 0), rate(oth, y == 0)
    tpr_g, tpr_o = rate(grp, y == 1), rate(oth, y == 1)
    bpsn = (oth & (y == 1)) | (grp & (y == 0))   # background positive, subgroup negative
    bnsp = (oth & (y == 0)) | (grp & (y == 1))   # background negative, subgroup positive
    return {
        "accuracy": (pred == y).mean(),
        "auc": safe_auc(y, p),
        "fpr_group": fpr_g, "fpr_other": fpr_o, "fpr_gap": fpr_g - fpr_o,
        "tpr_group": tpr_g, "tpr_other": tpr_o, "tpr_gap": tpr_g - tpr_o,
        "subgroup_auc": safe_auc(y[grp], p[grp]),
        "bpsn_auc": safe_auc(y[bpsn], p[bpsn]),
        "bnsp_auc": safe_auc(y[bnsp], p[bnsp]),
    }


# ---------------------------------------------------------------------------
# Model 1: TF-IDF + logistic regression
# ---------------------------------------------------------------------------
def fit_predict_tfidf(train_text, y_train, test_text, seed, cache):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    if "tfidf" not in cache:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                              sublinear_tf=True)
        cache["tfidf"] = (vec.fit_transform(train_text), vec.transform(test_text))
    X_tr, X_te = cache["tfidf"]
    clf = LogisticRegression(C=4.0, max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_train)
    return clf.predict_proba(X_te)[:, 1]


# ---------------------------------------------------------------------------
# Model 2: fine-tuned transformer (plain PyTorch loop)
# ---------------------------------------------------------------------------
def fit_predict_transformer(train_text, y_train, test_text, seed, cache, args):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "tok" not in cache:
        tok = AutoTokenizer.from_pretrained(args.hf_model)
        enc = lambda texts: tok(list(texts), truncation=True, max_length=args.max_len,
                                padding="max_length", return_tensors="pt")
        cache["tok"] = (enc(train_text), enc(test_text))
    enc_tr, enc_te = cache["tok"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.hf_model, num_labels=2).to(device)

    labels = torch.tensor(y_train, dtype=torch.long)
    train_dl = DataLoader(TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"], labels),
                          batch_size=args.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    test_dl = DataLoader(TensorDataset(enc_te["input_ids"], enc_te["attention_mask"]),
                         batch_size=args.batch_size * 2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    for _ in range(args.epochs):
        for ids, mask, lab in train_dl:
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for ids, mask in test_dl:
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(probs)


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tfidf", "transformer"], default="tfidf")
    ap.add_argument("--hf-model", default="distilbert-base-uncased")
    ap.add_argument("--data", default="data")
    ap.add_argument("--severities", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--corruptions", nargs="+", default=CORRUPTIONS,
                    choices=["group_flip", "matched_random", "matched_other", "random"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    train = pd.read_parquet(Path(args.data) / "train.parquet")
    test = pd.read_parquet(Path(args.data) / "test.parquet")
    y_tr, g_tr = train["label"].to_numpy(), train["group"].to_numpy()
    y_te, g_te = test["label"].to_numpy(), test["group"].to_numpy()

    out = Path(args.outdir or f"results/{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.csv"
    done = set()
    if res_path.exists():
        prev = pd.read_csv(res_path)
        done = set(zip(prev["seed"], prev["corruption"], prev["severity"].round(3)))

    conditions = [("clean", 0.0)] + [(k, s) for k in args.corruptions for s in args.severities]
    todo = [(seed, k, s) for seed in range(args.seeds) for k, s in conditions
            if (seed, k, round(s, 3)) not in done]
    print(f"Model: {args.model} | train n = {len(y_tr)} | test n = {len(y_te)} | "
          f"{len(todo)} runs to do ({len(done)} already done)")

    cache = {}
    start = time.time()
    for i, (seed, kind, sev) in enumerate(todo, 1):
        rng = np.random.default_rng([seed, int(round(sev * 1000)), KIND_OFFSET[kind]])
        y_c = corrupt(y_tr, g_tr, kind, sev, rng)
        if args.model == "tfidf":
            p = fit_predict_tfidf(train["text"], y_c, test["text"], seed, cache)
        else:
            p = fit_predict_transformer(train["text"], y_c, test["text"], seed, cache, args)
        row = {"model": args.model, "seed": seed, "corruption": kind, "severity": sev,
               "share_labels_changed": float((y_c != y_tr).mean()),
               **evaluate(y_te, p, g_te)}
        pd.DataFrame([row]).to_csv(res_path, mode="a", header=not res_path.exists(), index=False)
        elapsed = time.time() - start
        eta = elapsed / i * (len(todo) - i)
        print(f"  [{i}/{len(todo)}] seed {seed} {kind:<15} sev {sev:.1f} | "
              f"AUC {row['auc']:.3f} | FPR gap {row['fpr_gap']:+.3f} | "
              f"{elapsed / 60:.1f} min, ~{math.ceil(eta / 60)} min left")

    print(f"Done. Results in {res_path}. Next: python plot_results.py --inp {res_path}")


if __name__ == "__main__":
    main()
