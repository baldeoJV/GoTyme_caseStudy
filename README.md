# GoTyme Case Study: Cold-Call Response Model

An end-to-end, independently built model to predict whether a customer will take up a sales offer on a cold call, plus a campaign profit overlay. The goal is to give a client's new data science team a second opinion on their own model and approach.

Everything lives in one executed notebook, `GoTyme_case_study.ipynb`. Each stage explains what was done and why, and records the decision made.

## Business problem

| | |
|---|---|
| **Primary aim** | Classify whether a customer will take up the offer when cold-called, so the client calls the right people. |
| **Secondary aim** | Overlay revenue and cost on the classification outcomes to estimate campaign profit. |
| **Bonus** | Apply the model to a 10,000-record prospect list and compute (a) expected net profit and (b) profit lost to misclassification. |

## Data

| File | Description |
|---|---|
| `dataset.csv` | 35,000 rows, pipe-delimited (`\|`). 20 features (10 numeric, 10 categorical) and a binary `Response`. Category labels are masked (e.g. `Cat_3_h***l`). |
| `GoTyme_interview_case_study.xlsx` | The brief, the data dictionary, and the Finance risk-band profit table. |

Response rate is 11.2%, so the classes are imbalanced.

## Repository contents

| File | Purpose |
|---|---|
| `GoTyme_case_study.ipynb` | The full project, executed top to bottom with all outputs and charts. |
| `README.md` | This file. |

## Notebook map

| Stage | Topic | Case questions it answers |
|---|---|---|
| 1 | Data understanding & cleaning | What the data looks like, outliers, corrections, relevance of inputs |
| 2 | Feature screening | Train/test split, Information Value (IV), correlation, VIF, leakage test |
| 3 | Preparation | Encoding, rare-level grouping, binning, scaling, class imbalance, macro-variable choice |
| 4 | Modelling & validation | Model comparison, over/under-fit checks, ablations |
| 5 | Tuning, selection, cut-off | Tuning, one-standard-error rule, importance, cut-off, test evaluation, confusion matrix |
| 6 | Bonus | Profit and lost opportunity on the 10,000 list, with bootstrap uncertainty |
| 6B | Extension | Reusable functions linking precision, recall and accuracy to profit |
| 7 | Summary | All 17 case questions answered, limitations, mentor answer key |

## Key results

| Item | Result |
|---|---|
| Final model | Regularised logistic regression (L1, C = 0.1) on 14 inputs known before the call |
| Cross-validated AUC | 0.785 |
| Test AUC / Gini / KS | 0.80 / 0.60 / 0.50 |
| Cut-off | 0.2306 (break-even probability, 300 / (1,001 + 300)) |
| At the cut-off | Precision 0.46, recall 0.52, calls about 13% of the list, lift about 4x |
| Expected net profit per 10,000 | about 379,000 (95% bootstrap interval roughly 331k to 428k) |
| Profit lost to misclassification | about 745,000 (about 537k missed buyers plus about 208k wasted calls) |
| Benchmarks | Call everyone: about -1.54 million. Perfect model: about 1.12 million |

## Main decisions and why

1. **Call duration (`Feature_dn_1`) is excluded.** It is only known after the call, so it is data leakage. It lifts test AUC from 0.80 to 0.94 but cannot be used in production.
2. **Split before selecting features.** A stratified 70/30 split is made first. IV, correlation and every tuning step use the training data only. The test set is scored once.
3. **The 999 sentinel in `Feature_ps_3` becomes a "previously contacted" flag.** 96% of values are 999, meaning "never contacted".
4. **One macro variable out of three.** `ee_5`, `em_8` and `nd_9` are almost identical (VIF 32 to 64). Cross-validation could not tell them apart, so `ee_5` was chosen on interpretability. This is a judgement call and is flagged as one.
5. **No binning, no resampling.** Binning added 0.0004 AUC. Re-weighting or resampling would distort the probabilities that the profit cut-off depends on.
6. **Simplest model within noise wins.** Tuned logistic, boosting and random forest all score about 0.785 AUC. The one-standard-error rule picks logistic regression.
7. **The cut-off comes from the economics, not from 0.5.** It is confirmed on out-of-fold data.
8. **Accuracy is the wrong metric here.** The model's accuracy (87.7%) is below "never call" (88.8%), and maximising accuracy earns about a third less profit than maximising profit.

## Bonus assumptions

- Confusion-matrix rates come from the held-out test set at the chosen cut-off.
- Customers the model flags are called. Others are not, at no cost and no profit.
- The risk bands (High 10%, Medium 25%, Low 65%) are independent of the model's predictions.
- A sale earns 285 / 705 / 1,225 by band (expected 1,001). A call that does not convert costs 300 in every band.
- 100% approval.
- "Lost opportunity" can be read as missed buyers only (about 537k) or missed buyers plus wasted calls (about 745k). The notebook shows both.

## Stage 6B functions

| Function | Question it answers |
|---|---|
| `campaign_economics(precision, recall)` | What profit and accuracy does this precision and recall imply? |
| `breakeven_precision(V, c)` | Below what precision should we stop calling? (23.06% here) |
| `required_precision(recall, target)` | What precision is needed at this recall to hit a profit target? |
| `required_recall(precision, target)` | What recall is needed at this precision to hit a profit target? |
| `balance_table(y, p)` | Precision, recall, accuracy, F1 and profit at every cut-off |

Profit per customer = `pi * R * n * [V - c * (1/P - 1)]`, where `pi` is the share who would buy, `R` recall, `P` precision, `V` the expected profit of a sale and `c` the cost of a wasted call.

## How to run

1. Install Python 3.10 or later and the packages below.
2. Put `dataset.csv` somewhere accessible.
3. Open the notebook and change `DATA_PATH` in the second code cell to the location of `dataset.csv`.
4. Run all cells top to bottom. The seed is fixed (`SEED = 42`), so results are reproducible. A full run takes about 8 minutes, mostly from the boosting and random-forest searches in Stage 5.

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn jupyter
jupyter notebook GoTyme_case_study.ipynb
```

The stored outputs were produced with Python 3.13 and a recent scikit-learn. Version differences may shift the last decimal place of some figures.

## Limitations

- **The macro variables act as a calendar.** The strongest driver is the economic cycle (employment variation rate). It works in-sample but may not transfer if conditions change. An out-of-time validation and ongoing score monitoring are recommended.
- **The split is random, not time-based.** A time-based hold-out would be a more honest test for deployment.
- **The new list may convert differently.** A purchased prospect list could have a lower response rate than the development sample. A small pilot would tighten the profit estimate.
- **Masked categories.** Levels are anonymised, so their effects can be reported but not interpreted in plain language.
- **Profit inputs are given, not estimated.** If the sale value or call cost changes, recompute the break-even precision and the cut-off.
- **Call attempts** includes the current call, which is slightly leak-prone. The model loses almost nothing without it.
- **Final deployment** should refit on all labelled data with the same hyper-parameters. The test metrics here are for the model fitted on the 70% training split.
