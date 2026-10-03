# Causal Inference: What If - complete executable companion

This package is based on the user-supplied 19 August 2026 version of Miguel A. Hernan and James M. Robins, *Causal Inference: What If*.

The package is split into two sequential Python files so that the basic ideas remain readable and the advanced material does not turn the first script into one large monolith.

## Run

```bash
python -m pip install -r requirements.txt
python what_if_core.py
python what_if_advanced.py
```

Both files are top-level, sequential, and REPL-friendly. There is no `main()` wrapper. The numerical regression code uses explicit Gaussian elimination with partial pivoting instead of `np.linalg` regression routines. Part 2 also uses explicit finite-difference Newton solving for nonlinear estimating equations.

The supplied build was executed end-to-end with:

- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0

Only Part 1 section 8 needs scikit-learn. Part 2 needs only NumPy and pandas.

## Part 1: `what_if_core.py`

Part 1 covers the central workflow of the book.

1. Potential outcomes and hand-calculated causal risk difference, risk ratio, odds ratio, and NNT/NNH.
2. Randomization, association versus causation, exchangeability, standardization, IP weighting, AIPW, and positivity diagnostics.
3. Propensity-score matching.
4. Joint interventions and interaction.
5. Confounding, collider/selection bias, measurement error, and a small backdoor checker.
6. One-parameter additive g-estimation.
7. Instrumental-variable estimation.
8. Causal survival analysis with treatment and censoring weights.
9. Doubly robust machine learning with cross-fitting.
10. Time-varying treatment, treatment-confounder feedback, longitudinal g-formula, sequential IPW, and simple sequential g-estimation.
11. Target-trial protocol and a time-zero/immortal-time demonstration.
12. Interventionist mediation.
13. Optional NHEFS standardization example if `nhefs.csv` is placed beside the script.

The detailed Part 1 notes are retained in `README_part1.md`.

## Part 2: `what_if_advanced.py`

Part 2 implements the advanced topics that were intentionally omitted from Part 1.

### 1. General SWIG construction

Book basis: Chapter 7.5 and Chapter 19.

`make_swig()` transforms a DAG by splitting each intervened treatment node into its natural/random half and intervention half. Incoming arrows go to the natural half. Outgoing arrows originate from the intervention half. Descendants receive counterfactual labels.

The implementation handles:

- one static intervention;
- multiple static interventions;
- time-ordered interventions;
- deterministic dynamic strategies by adding history-to-strategy arrows.

Generated examples:

- `swig_A0_1.dot`
- `swig_A0_1_A1_0.dot`
- `swig_dynamic_A1_g1L1.dot`
- `swig_examples.csv`

The DOT files can be rendered with Graphviz if Graphviz is installed, but Graphviz is not required to run the Python code.

### 2. Front-door identification

Book basis: Technical Point 7.4 and Technical Point 23.3.

The code simulates unmeasured confounding of treatment and outcome, with a mediator that satisfies the front-door conditions. It computes

```text
sum_m P(M=m | A=a) * sum_a' E[Y | M=m,A=a'] P(A=a')
```

and compares the result with the known interventional truth from the data-generating mechanism.

### 3. Proximal causal identification

Book basis: Technical Point 7.3. The book states that two suitable negative-control/proxy variables can identify a causal effect in the presence of unmeasured confounding under additional rank/completeness conditions.

The finite binary implementation goes one step beyond the formula printed in the book. It uses the standard outcome-confounding bridge representation from the proximal-causal-inference literature. For each treatment level `a`, it solves

```text
E[Y | A=a,Z=z] = sum_c h_a(c) P(C=c | A=a,Z=z)
```

for the bridge values `h_a(c)`, then evaluates

```text
E[Y^a] = sum_c h_a(c) P(C=c).
```

Here `Z` is a treatment-inducing proxy/negative treatment control and `C` is an outcome-inducing proxy/negative outcome control. The determinant of the finite bridge matrix is reported as a simple rank diagnostic.

Relevant references named by the 2026 book are Miao, Geng, and Tchetgen Tchetgen (2018) and Cui et al. (2024).

### 4. Quantitative bias analysis

Book basis: Fine Point 10.2 and the sensitivity-analysis discussion in Chapter 13.

The code implements a transparent unmeasured-confounding bias factor for a risk ratio:

```text
B = [p1 * RR_UY + (1-p1)] / [p0 * RR_UY + (1-p0)]
corrected RR = observed RR / B
```

where `p1=P(U=1|A=1)` and `p0=P(U=1|A=0)` are sensitivity parameters. It then performs probabilistic bias analysis by sampling plausible sensitivity parameters and random error jointly. The draws are written to `qba_samples.csv`.

This is a worked quantitative-bias-analysis construction. Fine Point 10.2 introduces the framework but does not prescribe this specific bias-factor parameterization.

### 5-8. Structural nested mean models

Book basis: Chapter 14 and Chapter 21.

Implemented model families are:

- additive SNMM with two parameters and effect modification;
- multiplicative SNMM with nonlinear g-estimation;
- structural nested logistic model, with a numerical demonstration of noncollapsibility;
- longitudinal saturated additive SNMM with the Chapter 21-style blip functions at two treatment times.

For additive and multiplicative SNMMs, the estimating equations use treatment residuals after modeling treatment conditional on history. For the longitudinal example, the later treatment blip is removed first, then the earlier blip is estimated from the resulting candidate counterfactual.

The logistic example is intentionally not presented as a general longitudinal g-estimator. Technical Point 14.1 states that structural nested logistic models are noncollapsible and do not generalize easily to time-varying treatment.

### 9-12. CFT, CST, and structural AFT models

Book basis: Chapter 17.6 and Technical Points 17.2-17.3.

Implemented model families are:

- structural cumulative failure-time (CFT) model parameterization;
- structural cumulative survival-time (CST) model parameterization;
- one-parameter structural accelerated failure-time model;
- structural AFT model with covariate-dependent treatment effect;
- administrative censoring plus artificial censoring for AFT g-estimation.

For the AFT model

```text
T^a / T^0 = exp(-psi * a)
```

the candidate untreated survival time is

```text
H(psi) = T * exp(psi * A).
```

The artificial-censoring section follows Technical Point 17.3. It constructs `Delta(psi)` so that comparison of treatment groups does not condition on treatment-dependent observability of the transformed failure time.

CFT and CST are included as explicit model-family calculations. The book notes that these models have practical restrictions and gives less implementation detail than it gives for AFT models.

### 13-15. Chapter 23 mediation and path-specific effects

Book basis: Chapter 23.

The code implements the complete set of computational ideas developed in that chapter's simplified examples:

- controlled direct effects;
- the mediation formula;
- `E[Y(0,M(0))]`, `E[Y(1,M(0))]`, and `E[Y(1,M(1))]`;
- pure/natural direct effect;
- total/natural indirect effect;
- exact decomposition of the total effect in the simulated NPSEM-IE setting;
- separable treatment components `N` and `O`;
- the future component-intervention interpretation of `E[Y(n=0,o=1)]`;
- an explicit violation in which `O` also affects the mediator, making the future component-trial mean disagree with the original mediation formula;
- the Technical Point 23.3 path-specific front-door estimand, which isolates the `L -> A -> Y` pathway while leaving the direct `L -> Y` pathway at its natural value.

The script keeps the conceptual distinction made by the book. Cross-world pure/natural effects require stronger assumptions than the FFRCISTG model used in most of the book. Separable effects instead refer to interventions on substantively meaningful treatment components and can, in principle, be checked in future randomized experiments.

## Verification

Both scripts were run after packaging.

`advanced_verification.csv` contains explicit truth-versus-estimate checks. All included checks passed in this build. Selected examples are:

| Method | Estimate | Simulation truth | Absolute error |
|---|---:|---:|---:|
| Front-door risk difference | 0.154806 | 0.155036 | 0.000230 |
| Proximal risk difference | 0.174616 | 0.171348 | 0.003267 |
| QBA-corrected risk ratio | 1.544612 | 1.550000 | 0.005388 |
| Additive SNMM beta1 | 1.991721 | 2.000000 | 0.008279 |
| Multiplicative SNMM beta1 | 0.442809 | 0.450000 | 0.007191 |
| One-parameter AFT psi | -0.410000 | -0.400000 | 0.010000 |
| AFT with artificial censoring psi | -0.346000 | -0.350000 | 0.004000 |
| Path-specific front-door effect | 0.632051 | 0.631550 | 0.000501 |

The full console traces are in:

- `what_if_core_run.txt`
- `what_if_advanced_run.txt`

## Output files

Running Part 1 writes:

- `what_if_results.csv`
- `target_trial_protocol.csv`

Running Part 2 writes:

- `what_if_advanced_results.csv`
- `advanced_verification.csv`
- `qba_samples.csv`
- `swig_examples.csv`
- `swig_A0_1.dot`
- `swig_A0_1_A1_0.dot`
- `swig_dynamic_A1_g1L1.dot`

The copies already in this ZIP are the outputs from the verified packaged run.

## Scope

This is an executable teaching implementation, not a replacement for specialist causal-inference software. It implements the formulas and simplified settings needed to expose the logic of the 2026 book.

It is not a general symbolic do-calculus/ID engine for arbitrary hidden-variable ADMGs. The front-door and proximal sections implement the identification structures discussed in the book. The SWIG generator handles static deterministic interventions and deterministic dynamic strategies supplied with their history parents, but it does not infer a scientific treatment strategy from raw data.

The original book PDF is not included in this ZIP. It is not needed to run the code.
