import numpy as np
import pandas as pd
from pathlib import Path

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(7)
outdir = Path.cwd()


def expit(x):
    x = np.clip(np.asarray(x, dtype=float), -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def add_intercept(*cols):
    arrays = [np.asarray(c, dtype=float).reshape(len(np.asarray(c)), -1) for c in cols]
    return np.column_stack([np.ones(len(arrays[0])), *arrays])



def solve_linear(A, b):
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    n = len(b)
    for k in range(n):
        pivot = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[pivot, k]) < 1e-12:
            raise ValueError("Singular linear system")
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
        for i in range(k + 1, n):
            f = A[i, k] / A[k, k]
            A[i, k:] -= f * A[k, k:]
            b[i] -= f * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def ols_fit(X, y, w=None, ridge=1e-8):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        XtX = X.T @ X
        Xty = X.T @ y
    else:
        w = np.asarray(w, dtype=float)
        XtX = X.T @ (w[:, None] * X)
        Xty = X.T @ (w * y)
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return solve_linear(XtX + penalty, Xty)


def logistic_fit(X, y, max_iter=100, ridge=1e-6):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(max_iter):
        p = expit(X @ beta)
        W = np.clip(p * (1.0 - p), 1e-6, None)
        z = X @ beta + (y - p) / W
        XtWX = X.T @ (W[:, None] * X) + penalty
        XtWz = X.T @ (W * z)
        beta_new = solve_linear(XtWX, XtWz)
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return beta


def hajek_mean(y, w):
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    return np.sum(w * y) / np.sum(w)



def descendants(node, edges):
    out = set()
    frontier = [node]
    while frontier:
        u = frontier.pop()
        for a, b in edges:
            if a == u and b not in out:
                out.add(b)
                frontier.append(b)
    return out


def ancestors(nodes0, edges):
    out = set(nodes0)
    frontier = list(nodes0)
    while frontier:
        v = frontier.pop()
        for a, b in edges:
            if b == v and a not in out:
                out.add(a)
                frontier.append(a)
    return out


def d_separated(nodes, edges, X, Y, Z):
    X, Y, Z = set(X), set(Y), set(Z)
    anc = ancestors(X | Y | Z, edges)
    undirected = {u: set() for u in anc}
    parents = {u: [] for u in anc}
    for a, b in edges:
        if a in anc and b in anc:
            undirected[a].add(b)
            undirected[b].add(a)
            parents[b].append(a)
    for child, pars in parents.items():
        for i in range(len(pars)):
            for j in range(i + 1, len(pars)):
                undirected[pars[i]].add(pars[j])
                undirected[pars[j]].add(pars[i])
    seen = set(Z)
    frontier = [x for x in X if x not in seen]
    seen.update(frontier)
    while frontier:
        u = frontier.pop()
        if u in Y:
            return False
        for v in undirected.get(u, set()):
            if v not in seen:
                seen.add(v)
                frontier.append(v)
    return True


def valid_backdoor(nodes, edges, treatment, outcome, adjust):
    adjust = set(adjust)
    if any(z in descendants(treatment, edges) for z in adjust):
        return False
    backdoor_graph = [(a, b) for a, b in edges if a != treatment]
    return d_separated(nodes, backdoor_graph, {treatment}, {outcome}, adjust)


def report(title, rows):
    print("\n" + title)
    print("-" * len(title))
    for name, value in rows:
        if isinstance(value, float):
            print(f"{name:42s} {value: .4f}")
        else:
            print(f"{name:42s} {value}")


# 0. Small hand-checkable causal estimands
pL = np.array([0.5, 0.5])
risk1 = np.array([0.10, 0.30])
risk0 = np.array([0.05, 0.20])
mu1_small = np.sum(pL * risk1)
mu0_small = np.sum(pL * risk0)
rd_small = mu1_small - mu0_small
rr_small = mu1_small / mu0_small
odds1_small = mu1_small / (1 - mu1_small)
odds0_small = mu0_small / (1 - mu0_small)
or_small = odds1_small / odds0_small
nnh_small = 1 / rd_small
report("0. Hand-checkable causal effect measures", [
    ("E[Y^1] = .5*.10 + .5*.30", mu1_small),
    ("E[Y^0] = .5*.05 + .5*.20", mu0_small),
    ("Causal risk difference", rd_small),
    ("Causal risk ratio", rr_small),
    ("Causal odds ratio", or_small),
    ("Number needed to harm", nnh_small),
])

# 1. Potential outcomes, causal effect, association, effect modification
n = 12000
L1 = rng.normal(size=n)
L2 = rng.binomial(1, 0.40, size=n)
e_true = expit(-0.4 + 0.9 * L1 + 0.7 * L2)
A = rng.binomial(1, e_true)
epsilon = rng.normal(0, 1.0, size=n)
Y0 = 2.0 + 1.2 * L1 - 0.8 * L2 + epsilon
Y1 = Y0 + 2.0 + 1.5 * L2
Y = np.where(A == 1, Y1, Y0)

true_ate = np.mean(Y1 - Y0)
A_random = rng.binomial(1, 0.5, size=n)
Y_random = np.where(A_random == 1, Y1, Y0)
randomized_difference = Y_random[A_random == 1].mean() - Y_random[A_random == 0].mean()
naive = Y[A == 1].mean() - Y[A == 0].mean()
true_ate_l20 = np.mean((Y1 - Y0)[L2 == 0])
true_ate_l21 = np.mean((Y1 - Y0)[L2 == 1])

X_ps = add_intercept(L1, L2)
beta_ps = logistic_fit(X_ps, A)
e_hat = np.clip(expit(X_ps @ beta_ps), 0.01, 0.99)

X_y = add_intercept(A, L1, L2, A * L2)
beta_y = ols_fit(X_y, Y)
X_y1 = add_intercept(np.ones(n), L1, L2, L2)
X_y0 = add_intercept(np.zeros(n), L1, L2, np.zeros(n))
m1 = X_y1 @ beta_y
m0 = X_y0 @ beta_y
std_ate = np.mean(m1 - m0)

w1 = A / e_hat
w0 = (1 - A) / (1 - e_hat)
ipw_ate = hajek_mean(Y, w1) - hajek_mean(Y, w0)

aipw_i = m1 - m0 + A * (Y - m1) / e_hat - (1 - A) * (Y - m0) / (1 - e_hat)
aipw_ate = np.mean(aipw_i)
aipw_se = np.std(aipw_i - aipw_ate, ddof=1) / np.sqrt(n)
weights = A / e_hat + (1 - A) / (1 - e_hat)
ess = weights.sum() ** 2 / np.sum(weights ** 2)

report("1. Counterfactuals and point-treatment adjustment", [
    ("True ATE E[Y1-Y0]", true_ate),
    ("Randomized difference in means", randomized_difference),
    ("Confounded observed association", naive),
    ("Standardization / parametric g-formula", std_ate),
    ("Inverse-probability weighting", ipw_ate),
    ("Augmented IPW / doubly robust", aipw_ate),
    ("AIPW 95% CI lower", aipw_ate - 1.96 * aipw_se),
    ("AIPW 95% CI upper", aipw_ate + 1.96 * aipw_se),
    ("True ATE when L2=0", true_ate_l20),
    ("True ATE when L2=1", true_ate_l21),
    ("Estimated propensity minimum", float(e_hat.min())),
    ("Estimated propensity maximum", float(e_hat.max())),
    ("Largest inverse-probability weight", float(weights.max())),
    ("IPW effective sample size", float(ess)),
])

# 2. Propensity-score matching for the ATT
control_idx = np.where(A == 0)[0]
treated_idx = np.where(A == 1)[0]
control_ps = e_hat[control_idx]
matched_controls = []
for i in treated_idx:
    matched_controls.append(control_idx[np.argmin(np.abs(control_ps - e_hat[i]))])
matched_controls = np.asarray(matched_controls)
att_match = np.mean(Y[treated_idx] - Y[matched_controls])
true_att = np.mean((Y1 - Y0)[A == 1])
report("2. Propensity-score matching", [
    ("True ATT in treated", true_att),
    ("1:1 nearest-neighbor propensity ATT", att_match),
])

# 3. Joint intervention and additive interaction
n2 = 20000
C = rng.normal(size=n2)
A2 = rng.binomial(1, expit(-0.2 + 0.5 * C))
B2 = rng.binomial(1, expit(0.1 - 0.3 * C + 0.3 * A2))
noise2 = rng.normal(size=n2)
Y2 = 1 + 1.0 * A2 + 1.5 * B2 + 2.0 * A2 * B2 + 0.7 * C + noise2
X2 = add_intercept(A2, B2, A2 * B2, C)
b2 = ols_fit(X2, Y2)
mu = {}
for a in [0, 1]:
    for b in [0, 1]:
        Xab = add_intercept(np.full(n2, a), np.full(n2, b), np.full(n2, a * b), C)
        mu[(a, b)] = np.mean(Xab @ b2)
additive_interaction = mu[(1, 1)] - mu[(1, 0)] - mu[(0, 1)] + mu[(0, 0)]
report("3. Interaction requires a joint intervention", [
    ("mu(0,0)", mu[(0, 0)]),
    ("mu(1,0)", mu[(1, 0)]),
    ("mu(0,1)", mu[(0, 1)]),
    ("mu(1,1)", mu[(1, 1)]),
    ("Additive interaction contrast", additive_interaction),
])

# 4. DAG logic in data: confounding, collider conditioning, measurement error
n3 = 30000
U = rng.normal(size=n3)
A3 = rng.binomial(1, expit(0.8 * U))
Y3 = 2.0 * A3 + 1.5 * U + rng.normal(size=n3)
naive_b = ols_fit(add_intercept(A3), Y3)[1]
adj_b = ols_fit(add_intercept(A3, U), Y3)[1]

Uc = rng.normal(size=n3)
Ac = rng.binomial(1, 0.5, size=n3)
Cc = 1.2 * Ac + 1.2 * Uc + rng.normal(size=n3)
Yc = 2.0 * Ac + 1.5 * Uc + rng.normal(size=n3)
collider_unadjusted = ols_fit(add_intercept(Ac), Yc)[1]
collider_adjusted = ols_fit(add_intercept(Ac, Cc), Yc)[1]

Lm = rng.normal(size=n3)
Am = rng.binomial(1, expit(0.9 * Lm))
Ym = 2.0 * Am + 1.3 * Lm + rng.normal(size=n3)
Lm_star = Lm + rng.normal(0, 1.2, size=n3)
trueL_b = ols_fit(add_intercept(Am, Lm), Ym)[1]
measuredL_b = ols_fit(add_intercept(Am, Lm_star), Ym)[1]

report("4. Bias structures represented by causal diagrams", [
    ("Confounded crude treatment coefficient", naive_b),
    ("Adjusted for common cause U", adj_b),
    ("Collider example, do not condition on C", collider_unadjusted),
    ("Collider example, after conditioning on C", collider_adjusted),
    ("Adjustment with true confounder", trueL_b),
    ("Adjustment with mismeasured confounder", measuredL_b),
])

conf_nodes = {"L", "A", "Y"}
conf_edges = [("L", "A"), ("L", "Y"), ("A", "Y")]
coll_nodes = {"A", "U", "C", "Y"}
coll_edges = [("A", "C"), ("U", "C"), ("U", "Y"), ("A", "Y")]
report("4b. Backdoor checks from graph structure", [
    ("Confounding DAG, adjust for nothing", valid_backdoor(conf_nodes, conf_edges, "A", "Y", set())),
    ("Confounding DAG, adjust for L", valid_backdoor(conf_nodes, conf_edges, "A", "Y", {"L"})),
    ("Collider DAG, adjust for nothing", valid_backdoor(coll_nodes, coll_edges, "A", "Y", set())),
    ("Collider DAG, adjust for collider C", valid_backdoor(coll_nodes, coll_edges, "A", "Y", {"C"})),
])

# 5. G-estimation of a simple structural nested mean model
ng = 20000
Lg = rng.normal(size=ng)
pg = expit(-0.3 + 0.9 * Lg)
Ag = rng.binomial(1, pg)
Yg = 1.0 + 2.5 * Ag + 1.4 * Lg + rng.normal(size=ng)
Xg = add_intercept(Lg)
pg_hat = np.clip(expit(Xg @ logistic_fit(Xg, Ag)), 0.01, 0.99)
rAg = Ag - pg_hat
psi_g = np.sum(rAg * Yg) / np.sum(rAg * Ag)
report("5. G-estimation, one-parameter structural nested mean model", [
    ("True additive treatment effect", 2.5),
    ("G-estimate psi", psi_g),
])

# 6. Instrumental-variable estimation
niv = 50000
Uiv = rng.normal(size=niv)
Z = rng.binomial(1, 0.5, size=niv)
pAiv = expit(-0.3 + 1.2 * Z + 0.9 * Uiv)
Aiv = rng.binomial(1, pAiv)
Yiv = 3.0 * Aiv + 1.6 * Uiv + rng.normal(size=niv)
naive_iv = Yiv[Aiv == 1].mean() - Yiv[Aiv == 0].mean()
first_stage = Aiv[Z == 1].mean() - Aiv[Z == 0].mean()
reduced_form = Yiv[Z == 1].mean() - Yiv[Z == 0].mean()
wald_iv = reduced_form / first_stage
report("6. Instrumental variable", [
    ("True constant treatment effect", 3.0),
    ("Confounded treatment association", naive_iv),
    ("Instrument first stage", first_stage),
    ("Wald IV estimate", wald_iv),
])

# 7. Causal survival analysis with dependent censoring
ns = 9000
K = 8
Ls = rng.normal(size=ns)
ps = expit(-0.3 + 0.8 * Ls)
As = rng.binomial(1, ps)
rows = []
for i in range(ns):
    alive = True
    uncensored = True
    for t in range(K):
        if not (alive and uncensored):
            break
        pD = expit(-3.3 + 0.25 * t - 0.65 * As[i] + 0.65 * Ls[i])
        D = rng.binomial(1, pD)
        Cens = 0
        if D == 0:
            pC = expit(-4.0 + 0.18 * t + 0.75 * Ls[i] + 0.15 * As[i])
            Cens = rng.binomial(1, pC)
        rows.append((i, t, Ls[i], As[i], D, Cens))
        if D == 1:
            alive = False
        elif Cens == 1:
            uncensored = False
pp = pd.DataFrame(rows, columns=["id", "t", "L", "A", "D", "C"])

Xtr = add_intercept(Ls)
btr = logistic_fit(Xtr, As)
eA = np.clip(expit(Xtr @ btr), 0.01, 0.99)

Xc = add_intercept(pp["A"], pp["L"], pp["t"], pp["t"] ** 2)
bc = logistic_fit(Xc, pp["C"].values)
pp["p_unc"] = np.clip(1 - expit(Xc @ bc), 0.05, 1.0)
pp["cum_unc"] = pp.groupby("id")["p_unc"].cumprod()
base_e = np.where(As == 1, eA, 1 - eA)
pp["w"] = 1.0 / base_e[pp["id"].to_numpy()] / pp["cum_unc"].to_numpy()

risk_ipcw = {}
for a in [0, 1]:
    surv = 1.0
    for t in range(K):
        r = pp[(pp.A == a) & (pp.t == t)]
        hazard = np.sum(r.w * r.D) / np.sum(r.w)
        surv *= 1 - hazard
    risk_ipcw[a] = 1 - surv

Xd = add_intercept(pp["A"], pp["L"], pp["t"], pp["t"] ** 2)
bd = logistic_fit(Xd, pp["D"].values)
risk_g = {}
risk_true = {}
for a in [0, 1]:
    S_i = np.ones(ns)
    S_true = np.ones(ns)
    for t in range(K):
        Xdt = add_intercept(np.full(ns, a), Ls, np.full(ns, t), np.full(ns, t * t))
        h = expit(Xdt @ bd)
        h_true = expit(-3.3 + 0.25 * t - 0.65 * a + 0.65 * Ls)
        S_i *= 1 - h
        S_true *= 1 - h_true
    risk_g[a] = np.mean(1 - S_i)
    risk_true[a] = np.mean(1 - S_true)

report("7. Causal survival analysis", [
    ("True risk under A=0", risk_true[0]),
    ("True risk under A=1", risk_true[1]),
    ("True risk difference A=1 minus A=0", risk_true[1] - risk_true[0]),
    ("G-formula risk difference", risk_g[1] - risk_g[0]),
    ("IP treatment + IPC censoring weighted RD", risk_ipcw[1] - risk_ipcw[0]),
])

# 8. Doubly robust machine learning with cross-fitting
try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    ndml = 7000
    Xdml = rng.normal(size=(ndml, 6))
    e_dml = expit(0.5 * Xdml[:, 0] - 0.45 * Xdml[:, 1] ** 2 + 0.35 * np.sin(Xdml[:, 2]))
    Adml = rng.binomial(1, e_dml)
    fX = np.sin(Xdml[:, 0]) + Xdml[:, 1] ** 2 + 0.5 * Xdml[:, 2] * Xdml[:, 3]
    Ydml = 1.5 * Adml + fX + rng.normal(size=ndml)

    e_cf = np.zeros(ndml)
    m1_cf = np.zeros(ndml)
    m0_cf = np.zeros(ndml)
    kf = KFold(n_splits=5, shuffle=True, random_state=17)
    for train, test in kf.split(Xdml):
        clf = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
                                             l2_regularization=1.0, random_state=19)
        clf.fit(Xdml[train], Adml[train])
        e_cf[test] = clf.predict_proba(Xdml[test])[:, 1]

        reg1 = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
                                             l2_regularization=1.0, random_state=23)
        reg0 = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
                                             l2_regularization=1.0, random_state=29)
        tr1 = train[Adml[train] == 1]
        tr0 = train[Adml[train] == 0]
        reg1.fit(Xdml[tr1], Ydml[tr1])
        reg0.fit(Xdml[tr0], Ydml[tr0])
        m1_cf[test] = reg1.predict(Xdml[test])
        m0_cf[test] = reg0.predict(Xdml[test])

    e_cf = np.clip(e_cf, 0.02, 0.98)
    phi = m1_cf - m0_cf + Adml * (Ydml - m1_cf) / e_cf - (1 - Adml) * (Ydml - m0_cf) / (1 - e_cf)
    dml_ate = np.mean(phi)
    dml_se = np.std(phi - dml_ate, ddof=1) / np.sqrt(ndml)
    report("8. Doubly robust machine learning with 5-fold cross-fitting", [
        ("True ATE", 1.5),
        ("Cross-fit AIPW estimate", dml_ate),
        ("95% CI lower", dml_ate - 1.96 * dml_se),
        ("95% CI upper", dml_ate + 1.96 * dml_se),
    ])
except ImportError:
    print("\n8. DML example skipped because scikit-learn is not installed.")

# 9. Time-varying treatment, treatment-confounder feedback, g-methods
nl = 30000
L0 = rng.normal(size=nl)
p0 = expit(-0.2 + 0.7 * L0)
A0 = rng.binomial(1, p0)
L1_long = 0.6 * L0 + 1.0 * A0 + rng.normal(size=nl)
p1 = expit(-0.3 + 0.4 * A0 + 0.8 * L1_long + 0.2 * L0)
A1 = rng.binomial(1, p1)
Ylong = 1 + 1.5 * A0 + 2.0 * A1 + 1.2 * L1_long + 0.5 * L0 + rng.normal(size=nl)

true_never = 1.0
true_always = 1.0 + 1.5 + 2.0 + 1.2 * 1.0
true_long_ate = true_always - true_never

bL1 = ols_fit(add_intercept(A0, L0), L1_long)
bYlong = ols_fit(add_intercept(A0, A1, L0, L1_long), Ylong)

def longitudinal_gmean(a0, a1):
    L1a = add_intercept(np.full(nl, a0), L0) @ bL1
    Xa = add_intercept(np.full(nl, a0), np.full(nl, a1), L0, L1a)
    return np.mean(Xa @ bYlong)

g_never = longitudinal_gmean(0, 0)
g_always = longitudinal_gmean(1, 1)

bp0 = logistic_fit(add_intercept(L0), A0)
p0h = np.clip(expit(add_intercept(L0) @ bp0), 0.01, 0.99)
bp1 = logistic_fit(add_intercept(A0, L0, L1_long), A1)
p1h = np.clip(expit(add_intercept(A0, L0, L1_long) @ bp1), 0.01, 0.99)

def longitudinal_ipw_mean(a0, a1):
    keep = (A0 == a0) & (A1 == a1)
    q0 = np.where(a0 == 1, p0h, 1 - p0h)
    q1 = np.where(a1 == 1, p1h, 1 - p1h)
    w = 1.0 / (q0[keep] * q1[keep])
    return hajek_mean(Ylong[keep], w)

ipw_never = longitudinal_ipw_mean(0, 0)
ipw_always = longitudinal_ipw_mean(1, 1)

b_naive_long = ols_fit(add_intercept(A0, A1, L0, L1_long), Ylong)
naive_always_never = b_naive_long[1] + b_naive_long[2]

r1 = A1 - p1h
psi1 = np.sum(r1 * Ylong) / np.sum(r1 * A1)
H1 = Ylong - psi1 * A1
r0 = A0 - p0h
psi0 = np.sum(r0 * H1) / np.sum(r0 * A0)
gest_long = psi0 + psi1

report("9. Time-varying treatment and treatment-confounder feedback", [
    ("True always-treat minus never-treat", true_long_ate),
    ("Naive regression adjusting for post-A0 L1", naive_always_never),
    ("Longitudinal g-formula", g_always - g_never),
    ("Sequential IP weighting", ipw_always - ipw_never),
    ("Sequential g-estimation, simple additive SNMM", gest_long),
    ("G-estimated first-time blip", psi0),
    ("G-estimated second-time blip", psi1),
])

# 10. Target-trial emulation and time-zero error
protocol = pd.DataFrame({
    "component": ["Eligibility", "Strategies", "Assignment", "Time zero", "Follow-up", "Outcome", "Causal contrast", "Analysis"],
    "specification": [
        "Eligible at baseline before strategy assignment",
        "Always treat versus never treat",
        "Emulate assignment using measured history",
        "Same instant eligibility, strategy assignment, and follow-up start",
        "Baseline through final scheduled visit",
        "Continuous final outcome Y",
        "Per-protocol mean difference",
        "Sequential exchangeability + positivity + consistency; use g-methods",
    ],
})
protocol.to_csv(outdir / "target_trial_protocol.csv", index=False)

ni = 120000
baseline_hazard = 0.06
Tevent = rng.exponential(1 / baseline_hazard, size=ni)
Tstart = rng.exponential(6.0, size=ni)
end = 10.0
ever_treated = (Tstart < Tevent) & (Tstart < end)
dead_by_end = Tevent < end
immortal_naive = dead_by_end[ever_treated].mean() - dead_by_end[~ever_treated].mean()
report("10. Target trial and time zero", [
    ("Naive risk difference using future-defined 'ever treated'", immortal_naive),
    ("True treatment effect in this simulation", 0.0),
    ("Protocol written to", "target_trial_protocol.csv"),
])

# 11. Interventionist mediation
nm = 12000
Cm = rng.normal(size=nm)
At = rng.binomial(1, 0.5, size=nm)
M = 0.8 * At + 0.5 * Cm + rng.normal(size=nm)
Ym = 1.0 * At + 1.2 * M + 0.4 * Cm + 0.5 * At * M + rng.normal(size=nm)

bM = ols_fit(add_intercept(At, Cm), M)
Mhat = add_intercept(At, Cm) @ bM
Mresid = M - Mhat
bYm = ols_fit(add_intercept(At, M, Cm, At * M), Ym)

def mediation_mu(a_y, a_m, draws=30):
    out = []
    mean_M = add_intercept(np.full(nm, a_m), Cm) @ bM
    for _ in range(draws):
        eps_m = rng.choice(Mresid, size=nm, replace=True)
        Mdraw = mean_M + eps_m
        XY = add_intercept(np.full(nm, a_y), Mdraw, Cm, np.full(nm, a_y) * Mdraw)
        out.append(np.mean(XY @ bYm))
    return float(np.mean(out))

mu00 = mediation_mu(0, 0)
mu10 = mediation_mu(1, 0)
mu11 = mediation_mu(1, 1)
ide = mu10 - mu00
iie = mu11 - mu10
total_med = mu11 - mu00
report("11. Interventionist mediation", [
    ("mu(A=0, mediator distribution under A=0)", mu00),
    ("mu(A=1, mediator distribution under A=0)", mu10),
    ("mu(A=1, mediator distribution under A=1)", mu11),
    ("Interventional direct effect", ide),
    ("Interventional indirect effect", iie),
    ("Total effect", total_med),
    ("Direct + indirect", ide + iie),
])

# 12. Optional NHEFS reproduction from the book
nhefs_path = outdir / "nhefs.csv"
if nhefs_path.exists():
    d = pd.read_csv(nhefs_path)
    needed = ["qsmk", "wt82_71", "sex", "race", "age", "wt71", "smokeintensity", "smokeyrs", "exercise", "active", "education"]
    missing = [x for x in needed if x not in d.columns]
    if missing:
        raise ValueError("NHEFS file does not have the book-code schema. Missing columns: " + ", ".join(missing))
    d = d.dropna(subset=needed).copy()
    for col in ["age", "wt71", "smokeintensity", "smokeyrs"]:
        d[col + "2"] = d[col] ** 2
    edu = pd.get_dummies(d["education"], prefix="edu", dtype=float)
    ex = pd.get_dummies(d["exercise"], prefix="exercise", dtype=float)
    act = pd.get_dummies(d["active"], prefix="active", dtype=float)
    d = pd.concat([d, edu, ex, act], axis=1)
    for name in ["edu_2", "edu_3", "edu_4", "edu_5", "exercise_1", "exercise_2", "active_1", "active_2"]:
        if name not in d:
            d[name] = 0.0
    base = ["sex", "race", "edu_2", "edu_3", "edu_4", "edu_5", "exercise_1", "exercise_2", "active_1", "active_2",
            "age", "age2", "wt71", "wt712", "smokeintensity", "smokeintensity2", "smokeyrs", "smokeyrs2"]
    A_n = d["qsmk"].to_numpy(float)
    Y_n = d["wt82_71"].to_numpy(float)
    Xbase = d[base].to_numpy(float)
    Xn = np.column_stack([np.ones(len(d)), A_n, Xbase, A_n * d["smokeintensity"].to_numpy(float)])
    bn = ols_fit(Xn, Y_n)
    Xn0 = np.column_stack([np.ones(len(d)), np.zeros(len(d)), Xbase, np.zeros(len(d))])
    Xn1 = np.column_stack([np.ones(len(d)), np.ones(len(d)), Xbase, d["smokeintensity"].to_numpy(float)])
    nhefs_std = np.mean(Xn1 @ bn - Xn0 @ bn)
    report("12. NHEFS smoking-cessation example", [
        ("Complete-case n", len(d)),
        ("Standardized mean weight-gain difference", nhefs_std),
    ])
else:
    print("\n12. Optional NHEFS example not run. Put nhefs.csv beside this script and rerun.")

summary = pd.DataFrame({
    "method": [
        "true_point_ATE", "naive_association", "standardization", "IPW", "AIPW", "g_estimation_point",
        "IV_Wald", "survival_g_RD", "survival_IPCW_RD", "true_longitudinal_ATE", "longitudinal_g_formula",
        "longitudinal_IPW", "longitudinal_g_estimation", "interventional_direct", "interventional_indirect"
    ],
    "estimate": [
        true_ate, naive, std_ate, ipw_ate, aipw_ate, psi_g, wald_iv,
        risk_g[1] - risk_g[0], risk_ipcw[1] - risk_ipcw[0], true_long_ate,
        g_always - g_never, ipw_always - ipw_never, gest_long, ide, iie
    ]
})
summary.to_csv(outdir / "what_if_results.csv", index=False)
print("\nSaved what_if_results.csv and target_trial_protocol.csv")
