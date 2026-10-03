# Causal Inference: What If - executable core

This companion script implements the main computational ideas in Hernan and Robins, *Causal Inference: What If* (19 Aug 2026 PDF supplied in this chat).

It is an educational implementation. It is not a replacement for the book and it does not reproduce every technical point, proof, SWIG construction, front-door/proximal method, structural AFT variant, or path-specific mediation estimand.

## Run

```bash
python -m pip install -r requirements_what_if_core.txt
python what_if_core.py
```

The script is sequential and REPL-friendly. Linear and logistic regressions are implemented directly. The linear systems inside those estimators use a small Gaussian-elimination solver with partial pivoting rather than `np.linalg` regression routines. Scikit-learn is used only for the high-dimensional cross-fitting example.

## Sections in the script

0. Hand-checkable causal estimands: counterfactual risks, risk difference, risk ratio, odds ratio, NNH.
1. Potential outcomes, randomization, association versus causation, effect modification, standardization/g-formula, IPW, AIPW, positivity diagnostics.
2. Propensity-score matching for ATT.
3. Joint interventions and additive interaction.
4. Confounding, collider/selection bias, measurement error, and a small DAG/backdoor checker.
5. G-estimation of a simple structural nested mean model.
6. Instrumental-variable estimation with the Wald estimator.
7. Causal survival analysis with outcome g-formula and treatment/censoring IP weighting.
8. Doubly robust machine learning with 5-fold cross-fitting.
9. Time-varying treatment, treatment-confounder feedback, longitudinal g-formula, sequential IPW, and simple sequential g-estimation.
10. Target-trial protocol and a time-zero/immortal-time bias demonstration.
11. Interventionist mediation with modeled stochastic mediator interventions.
12. Optional NHEFS smoking-cessation standardization example.

## Optional NHEFS data

Put `nhefs.csv` in the directory from which the script is run. The script requires the book-data schema, including `education`, `exercise`, `active`, `qsmk`, and `wt82_71`. It does not guess or reconstruct missing variables.

Official book page and data:
https://miguelhernan.org/whatifbook

Direct NHEFS CSV link advertised by the book page:
https://miguelhernan.org/s/nhefs.csv

The official book page also links Python notebooks by James Fiedler. That repository currently states that its Python code covers Part II and was checked against the 30 March 2021 book version, so it is useful as a reference but is not synchronized to every change in the supplied 19 Aug 2026 PDF:
https://github.com/jrfiedler/causal_inference_python_code

## Files produced

`what_if_results.csv` contains the principal estimates from the simulations.

`target_trial_protocol.csv` contains the target-trial components used in the longitudinal example.

## Verification in this build

This packaged build was executed end-to-end in the supplied environment with NumPy 2.3.5, pandas 2.2.3, and scikit-learn 1.8.0. The high-dimensional example uses `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`.
