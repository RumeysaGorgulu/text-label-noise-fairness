# Results: clean training data → highest severity (mean over seeds)

| Model                        | Corruption                                  |   Severity | Labels changed   | AUC           | Accuracy      | FPR gap         | BPSN AUC      |
|:-----------------------------|:--------------------------------------------|-----------:|:-----------------|:--------------|:--------------|:----------------|:--------------|
| TF-IDF + logistic regression | Group's non-toxic comments relabeled toxic  |        0.4 | 8.9%             | 0.818 → 0.784 | 0.842 → 0.792 | +0.039 → +0.298 | 0.625 → 0.345 |
| TF-IDF + logistic regression | Same number of labels flipped at random     |        0.4 | 8.9%             | 0.818 → 0.776 | 0.842 → 0.835 | +0.039 → +0.051 | 0.625 → 0.609 |
| TF-IDF + logistic regression | Same number relabeled toxic, other comments |        0.4 | 8.9%             | 0.818 → 0.763 | 0.842 → 0.837 | +0.039 → +0.024 | 0.625 → 0.736 |
| DistilBERT (fine-tuned)      | Group's non-toxic comments relabeled toxic  |        0.4 | 8.9%             | 0.882 → 0.857 | 0.866 → 0.774 | +0.051 → +0.439 | 0.783 → 0.532 |
| DistilBERT (fine-tuned)      | Same number of labels flipped at random     |        0.4 | 8.9%             | 0.882 → 0.872 | 0.866 → 0.861 | +0.051 → +0.051 | 0.783 → 0.761 |
| DistilBERT (fine-tuned)      | Same number relabeled toxic, other comments |        0.4 | 8.9%             | 0.882 → 0.859 | 0.866 → 0.860 | +0.051 → +0.027 | 0.783 → 0.861 |

Seeds: 3. FPR gap > 0 means the group's non-toxic comments are flagged as toxic more often than other non-toxic comments.

![Results](fig_results.png)