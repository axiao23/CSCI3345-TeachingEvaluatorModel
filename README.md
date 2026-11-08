# Predicting teaching interaction quality

Course project, CSCI3345, Fall 2026. Team: 4 (Audrey Xiao, Sydney Luciano, Charles Ross).

We predict a five-level teaching quality label (Poor, Needs Improvement, Average, Good, Excellent) from classroom interaction measurements, and compare a few model families on a held-out set of teachers.

## Data

15,000 classroom interaction records from `teaching_interaction_quality_dataset.csv`, 27 measured columns plus the label (32 model features after the five engineered ones are added). There are no missing values or duplicates. The label is ordinal and moderately imbalanced (Average is the largest class at 28%, Poor the smallest at 11%).

Preprocessing is documented in `data/PREPROCESSING.md`. In short:

- identifier columns are dropped (interaction, session, class, teacher)
- 26 rows where a time exceeded session length were capped at session length
- five per-minute / share features were added because session length ranges from 30 to 113 minutes
- the split is by teacher, so no teacher appears in more than one split (143 / 28 / 29 teachers in train / val / test)

| split | rows |
|-------|------|
| train | 10,699 |
| val   | 2,150 |
| test  | 2,151 |

Class counts (Poor, Needs Improvement, Average, Good, Excellent):

| split | Poor | Needs Impr. | Average | Good | Excellent |
|-------|------|-------------|---------|------|-----------|
| train | 1,159 | 1,918 | 3,009 | 2,783 | 1,830 |
| val   | 256 | 385 | 587 | 571 | 351 |
| test  | 235 | 397 | 604 | 546 | 369 |

The CSVs in `data/` are the cleaned, unscaled versions. Scaling happens inside each model pipeline and is fit on train only. `scaler.json` is kept for reference.

## Metric

Macro-F1 is the main metric, since classes are unequal and we care about the rare ones. We also report accuracy, quadratic weighted kappa and mean absolute error in label levels, because the classes are ordered and predicting "Good" for an "Excellent" teacher is a smaller miss than predicting "Poor".

Models are chosen on validation. Test numbers are reported alongside but were not used to make any decisions.

## Reproducing the baseline

```
pip install pandas scikit-learn matplotlib jupyter
jupyter notebook baseline.ipynb
```

Open the notebook from the repo root and run all cells.

Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, seed 42. Running all cells takes under a minute on a laptop CPU and writes everything in `results/`.

## Baseline results

Validation (test in parentheses), all 32 features:

| model | macro-F1 | accuracy | QW kappa |
|-------|----------|----------|----------|
| Majority class | 0.086 (0.088) | 0.273 (0.281) | 0.000 (0.000) |
| Logistic regression | 0.715 (0.724) | 0.700 (0.708) | 0.906 (0.904) |
| Random forest, 300 trees | 0.714 (0.724) | 0.697 (0.706) | 0.902 (0.903) |
| Hist. gradient boosting | 0.709 (0.715) | 0.691 (0.698) | 0.898 (0.896) |

Logistic regression and random forest use balanced class weights. Everything else is scikit-learn defaults. Full table, including a run without `overall_interaction_score`, is in `results/baseline_results.csv`.

## Where the baseline fails

About 70% of validation predictions are exact and almost all of the rest (30.0%) are off by one level. Only 3 of 2,150 are off by two or more. The confusion matrix (`results/confusion_val.png`) shows the three middle classes (Needs Improvement, Average, Good) are the weakest (F1 around 0.66) while Poor and Excellent are easier (0.81 and 0.79). The classes look like cut points on a continuous quality score, so records near a boundary are close to a coin flip for every model we have tried.

Three very different model families land within 0.01 macro-F1 of each other. That points to the ceiling coming from the data rather than the model, so we expect tuning to give small gains and plan to say so plainly in the final report.

## Layout

```
processed/       cleaned splits, scaler.json, meta.json, PREPROCESSING.md
baseline.ipynb
baselinemodel/    baseline_results.csv, baseline.ipynb
report/     ML Project Proposal PDF, ML Baseline Checkpoint PDF, (coming soon) final PDF
```

## Planned experiments

1. Tune logistic regression, random forest and gradient boosting with grouped cross-validation on train. This needs teacher ids, which the cleaned CSVs no longer carry, so we will add them back from the raw file.
2. Add an ordinal approach (regression on the label level with tuned thresholds) and compare against the classifiers.
3. Small MLP as a neural baseline.
4. Feature ablation, mainly with and without `overall_interaction_score`, which is a composite of other columns and the strongest single predictor.
5. Final comparison on test, once, with the tuned configurations.
