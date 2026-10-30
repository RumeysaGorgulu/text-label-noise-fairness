# Which Labels Are Wrong Matters: Group-Targeted Label Noise in Toxicity Classifiers

**Question:** Human annotators sometimes label harmless comments that mention an identity group (e.g., "As a Muslim, I…") as toxic. If a small share of training labels carries this kind of bias, does the classifier inherit it, and would standard accuracy metrics notice?

This project extends my [data-quality stress test](https://github.com/RumeysaGorgulu/data-quality-stress-test) from tabular data to text and fine-tuned transformers. It tests whether the harm comes from *which* labels are wrong, using two control conditions that change the same number of labels.

## Design

- **Data:** [CivilComments-WILDS](https://huggingface.co/datasets/pietrolesci/civilcomments-wilds) (Borkan et al., 2019; Koh et al., 2021). The task is to predict whether a comment is toxic.
- **Group:** comments that mention a Black, Muslim or LGBTQ identity
- **Sample:**
  - 20,000 training comments (30% from the group)
  - 8,000 test comments (50% from the group)
  - The group is over-sampled so that group-level metrics are stable
- **Training labels are corrupted; every model is tested on the same clean test set.**

| Condition | What happens to the training labels |
|---|---|
| **Group-targeted** | A share (= severity) of the group's non-toxic comments are relabeled toxic |
| Control 1: matched random | The **same number** of labels flipped at random rows, in either direction |
| Control 2: matched other | The **same number** of non-toxic comments relabeled toxic, chosen from comments that do **not** mention the group |

Control 2 keeps the number and direction of errors identical, so any difference from the group-targeted condition comes only from *who* the errors fall on.

- **Severity:** 0, 0.1, 0.2, 0.3, 0.4
- **Models:**
  - TF-IDF + logistic regression (baseline)
  - **DistilBERT**, fine-tuned for 1 epoch with a plain PyTorch training loop: AdamW, linear warmup, mixed precision
- **Seeds:** 3
- **Metrics:**
  - Overall AUC and accuracy
  - **FPR gap:** false-positive rate on the group's non-toxic comments minus the rate on other non-toxic comments
  - **BPSN AUC** (Borkan et al., 2019): a fairness metric for this dataset; low values mean the group's non-toxic comments score higher than other toxic comments

## Results

At the highest severity, 40% of the group's non-toxic training comments were relabeled toxic, which is **8.9% of all training labels**. Each control changes the same 8.9%. Means over 3 seeds; full numbers in [`results/summary.md`](results/summary.md).

| Model | Training labels | AUC | FPR gap | BPSN AUC |
|---|---|---|---|---|
| TF-IDF + LR | Clean | 0.818 | 0.039 | 0.625 |
| | **Group-targeted** | 0.784 | **0.298** | **0.345** |
| | Matched random | 0.776 | 0.051 | 0.609 |
| | Matched other | 0.763 | 0.024 | 0.736 |
| DistilBERT | Clean | 0.882 | 0.051 | 0.783 |
| | **Group-targeted** | 0.857 | **0.439** | **0.532** |
| | Matched random | 0.872 | 0.051 | 0.761 |
| | Matched other | 0.859 | 0.027 | 0.861 |

![Results](results/fig_results.png)

*Red: group-targeted errors. Blue and green: the same number of errors placed at random or on other comments. Shaded bands show ± 1 SD across seeds.*

**Key findings**
- **Which labels are wrong matters more than how many.** Relabeling 40% of the group's non-toxic comments (8.9% of all training labels) raised DistilBERT's FPR gap from 0.05 to 0.44: the model flagged the group's harmless comments as toxic far more often than other harmless comments. The same number of random flips left the gap unchanged (0.05), and the same number of errors placed on other comments slightly lowered it (0.03).
- **Overall AUC does not reveal the harm.** DistilBERT's AUC fell by about the same amount under group-targeted errors (0.882 → 0.857) as under errors on other comments (→ 0.859), although only the first created a fairness gap. For TF-IDF, the group-targeted condition even had the *highest* AUC of the three corrupted conditions. The subgroup metrics (FPR gap, BPSN AUC) are what detect it.
- **The stronger model inherits the bias more.** DistilBERT starts fairer than TF-IDF (BPSN AUC 0.78 vs. 0.63) but reaches a larger gap under group-targeted errors (0.44 vs. 0.30), with the sharpest rise between 30% and 40% severity and the most variation across seeds there (0.35–0.54).
- **Note on accuracy:** accuracy did drop under group-targeted errors (DistilBERT 0.866 → 0.774). This is partly because the test set is enriched to 50% group comments; in data with the natural group share, the drop would be smaller.

## Run it

**Colab (recommended, free GPU):** open `colab_run.ipynb` in Google Colab, choose a T4 GPU and run all cells. Results are saved to Google Drive, so the run can resume after a disconnect.

**Locally:**
```bash
pip install -r requirements.txt
python prepare_data.py
python run_experiment.py --model tfidf
python run_experiment.py --model transformer     # GPU recommended
python plot_results.py --inp results/tfidf/results.csv results/transformer/results.csv
```

## Limitations

- Synthetic corruption: real annotator bias is not random within the group, and it may depend on dialect or wording (Sap et al., 2019).
- The group is a combination of three identities. Effects may differ for each one; use `--group-cols` to run a single identity.
- One dataset, two model families and a fixed 0.5 decision threshold
- Enriched sample: group shares differ from the natural data, so absolute rates are not population estimates
- 3 seeds give a rough sense of variability, not formal tests.

## Related work

- Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced metrics for measuring unintended bias with real data for text classification. *WWW Companion.*
- Dixon, L., Li, J., Sorensen, J., Thain, N., & Vasserman, L. (2018). Measuring and mitigating unintended bias in text classification. *AIES.*
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The risk of racial bias in hate speech detection. *ACL.*
- Koh, P. W., et al. (2021). WILDS: A benchmark of in-the-wild distribution shifts. *ICML.*

## License

MIT, see `LICENSE`.
