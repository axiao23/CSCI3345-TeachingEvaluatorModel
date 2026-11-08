# Preprocessing summary

Source: `teaching_interaction_quality_dataset.csv` — 15,000 rows, label `teaching_quality_class`.

## Steps
1. **Quality checks:** no missing values, no duplicate rows, no values outside plausible ranges, no extreme outliers (none beyond 3×IQR). Labels are clean (5 spellings, no stray whitespace).
2. **Removed as features:** interaction_id, session_id, class_id, teacher_id (identifiers; `session_id` maps to up to 27 teachers, so it is not a real session key).
3. **Fixed impossible values:** teacher explanation time longer than the session in 21 rows and discussion time longer than the session in 5 rows; capped at session length.
4. **Derived features** (session length varies 30–113 min, so raw counts are not comparable): `explanation_share`, `discussion_share`, `questions_per_min`, `student_responses_per_min`, `feedback_per_question`.
5. **Split by teacher** (StratifiedGroupKFold, seed 42): train 10,699 rows / 143 teachers, val 2,150 / 28, test 2,151 / 29.
6. **Standardization:** z-score using train mean/std only (`scaler.json`), applied to val and test.
7. **Label encoding:** Poor=0, Needs Improvement=1, Average=2, Good=3, Excellent=4.

Kept as-is: many 0–100 scores pile up at exactly 100 (e.g. attendance in 943 rows), consistent with capped measurements; no transformation needed for the current models.

## Class counts per split (Poor → Excellent)
- train: [1159, 1918, 3009, 2783, 1830]
- val: [256, 385, 587, 571, 351]
- test: [235, 397, 604, 546, 369]

## Files
- `../train.csv`, `../val.csv`, `../test.csv`: cleaned, unscaled, readable labels
- `train.csv`, `val.csv`, `test.csv` (here): 32 standardized features + `label` (0–4) + `label_name`
- `scaler.json`: means/stds to transform new data
