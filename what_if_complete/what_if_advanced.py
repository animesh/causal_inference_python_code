import numpy as np
import pandas as pd
from pathlib import Path

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(20261003)
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


def ols_fit(X, y, ridge=1e-8):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return solve_linear(X.T @ X + penalty, X.T @ y)


def logistic_fit(X, y, max_iter=100, ridge=1e-6):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(max_iter):
        p = expit(X @ beta)
        W = np.clip(p * (1 - p), 1e-6, None)
        z = X @ beta + (y - p) / W
        beta_new = solve_linear(X.T @ (W[:, None] * X) + penalty, X.T @ (W * z))
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return beta


def numerical_jacobian(fun, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(fun(x), dtype=float)
    J = np.zeros((len(f0), len(x)))
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += h
        J[:, j] = (np.asarray(fun(xp), dtype=float) - f0) / h
    return J


def solve_nonlinear(fun, x0, max_iter=60, tol=1e-9):
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(max_iter):
        f = np.asarray(fun(x), dtype=float)
        if np.max(np.abs(f)) < tol:
            break
        J = numerical_jacobian(fun, x)
        step = solve_linear(J, -f)
        scale = 1.0
        old = np.sum(f * f)
        while scale > 1e-4:
            candidate = x + scale * step
            new = np.sum(np.asarray(fun(candidate), dtype=float) ** 2)
            if new <= old:
                x = candidate
                break
            scale *= 0.5
        if np.max(np.abs(scale * step)) < tol:
            break
    return x


def report(title, rows):
    print("\n" + title)
    print("-" * len(title))
    for name, value in rows:
        if isinstance(value, (float, np.floating)):
            print(f"{name:48s} {float(value): .6f}")
        else:
            print(f"{name:48s} {value}")


def ancestors_of(node, edges):
    out = set()
    frontier = [node]
    while frontier:
        v = frontier.pop()
        for a, b in edges:
            if b == v and a not in out:
                out.add(a)
                frontier.append(a)
    return out


def topological_order(nodes, edges):
    nodes = list(nodes)
    incoming = {n: 0 for n in nodes}
    children = {n: [] for n in nodes}
    for a, b in edges:
        incoming[b] += 1
        children[a].append(b)
    queue = [n for n in nodes if incoming[n] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in children[u]:
            incoming[v] -= 1
            if incoming[v] == 0:
                queue.append(v)
    if len(order) != len(nodes):
        raise ValueError("Graph must be acyclic")
    return order


def make_swig(nodes, edges, interventions, strategy_parents=None):
    strategy_parents = {} if strategy_parents is None else strategy_parents
    order = topological_order(nodes, edges)
    intervention_order = [n for n in order if n in interventions]
    labels = {}
    swig_nodes = []
    for node in order:
        prior = [a for a in intervention_order if a in ancestors_of(node, edges)]
        suffix = ""
        if prior:
            suffix = "(" + ",".join(f"{a}={interventions[a]}" for a in prior) + ")"
        rid = f"r::{node}"
        labels[rid] = node + suffix
        swig_nodes.append(rid)
        if node in interventions:
            fid = f"f::{node}"
            labels[fid] = f"{node}={interventions[node]}"
            swig_nodes.append(fid)
    swig_edges = []
    for a, b in edges:
        src = f"f::{a}" if a in interventions else f"r::{a}"
        dst = f"r::{b}"
        swig_edges.append((src, dst))
    for treatment, parents in strategy_parents.items():
        if treatment not in interventions:
            raise ValueError(f"Dynamic strategy parent supplied for non-intervened node {treatment}")
        for parent in parents:
            swig_edges.append((f"r::{parent}", f"f::{treatment}"))
    return swig_nodes, swig_edges, labels


def write_dot(path, nodes, edges, labels):
    with open(path, "w", encoding="utf-8") as f:
        f.write("digraph SWIG {\n")
        f.write("  rankdir=LR;\n")
        for node in nodes:
            shape = "box" if node.startswith("f::") else "ellipse"
            f.write(f'  "{node}" [label="{labels[node]}", shape={shape}];\n')
        for a, b in edges:
            f.write(f'  "{a}" -> "{b}";\n')
        f.write("}\n")


def empirical_prob(mask):
    return float(np.mean(mask))


def empirical_mean(y, mask):
    return float(np.mean(np.asarray(y)[mask]))


# 1. General SWIG construction by node splitting
nodes = ["U", "A0", "L1", "A1", "Y"]
edges = [("U", "A0"), ("U", "L1"), ("U", "Y"), ("A0", "L1"), ("A0", "A1"), ("L1", "A1"), ("L1", "Y"), ("A1", "Y")]

swig1_nodes, swig1_edges, swig1_labels = make_swig(nodes, edges, {"A0": 1})
write_dot(outdir / "swig_A0_1.dot", swig1_nodes, swig1_edges, swig1_labels)

swig2_nodes, swig2_edges, swig2_labels = make_swig(nodes, edges, {"A0": 1, "A1": 0})
write_dot(outdir / "swig_A0_1_A1_0.dot", swig2_nodes, swig2_edges, swig2_labels)

swig3_nodes, swig3_edges, swig3_labels = make_swig(
    nodes, edges, {"A0": 0, "A1": "g1(L1)"}, strategy_parents={"A1": ["L1"]}
)
write_dot(outdir / "swig_dynamic_A1_g1L1.dot", swig3_nodes, swig3_edges, swig3_labels)

swig_rows = []
for example, ns, es, ls in [
    ("do(A0=1)", swig1_nodes, swig1_edges, swig1_labels),
    ("do(A0=1,A1=0)", swig2_nodes, swig2_edges, swig2_labels),
    ("g: A0=0, A1=g1(L1)", swig3_nodes, swig3_edges, swig3_labels),
]:
    for node in ns:
        swig_rows.append((example, "node", node, ls[node]))
    for a, b in es:
        swig_rows.append((example, "edge", a, b))
pd.DataFrame(swig_rows, columns=["example", "kind", "from_or_id", "to_or_label"]).to_csv(outdir / "swig_examples.csv", index=False)

report("1. SWIG generation", [
    ("Single intervention DOT", "swig_A0_1.dot"),
    ("Joint intervention DOT", "swig_A0_1_A1_0.dot"),
    ("Dynamic strategy DOT", "swig_dynamic_A1_g1L1.dot"),
    ("Joint-intervention outcome label", swig2_labels["r::Y"]),
    ("Natural A1 under prior A0 intervention", swig2_labels["r::A1"]),
    ("Dynamic-strategy intervention edge", "L1(A0=0) -> A1=g1(L1)"),
])

# 2. Front-door identification
n_fd = 220000
U = rng.binomial(1, 0.45, size=n_fd)
A = rng.binomial(1, expit(-0.9 + 1.8 * U))
M = rng.binomial(1, expit(-1.1 + 2.2 * A))
pY = expit(-2.2 + 1.6 * M + 1.2 * U)
Y = rng.binomial(1, pY)

pA = {a: empirical_prob(A == a) for a in [0, 1]}
pM_given_A = {(m, a): empirical_prob(M[A == a] == m) for a in [0, 1] for m in [0, 1]}
pY_given_MA = {(m, a): empirical_mean(Y, (M == m) & (A == a)) for a in [0, 1] for m in [0, 1]}

frontdoor_risk = {}
for a in [0, 1]:
    total = 0.0
    for m in [0, 1]:
        inner = sum(pY_given_MA[(m, ap)] * pA[ap] for ap in [0, 1])
        total += pM_given_A[(m, a)] * inner
    frontdoor_risk[a] = total

true_fd_risk = {}
for a in [0, 1]:
    pm1 = expit(-1.1 + 2.2 * a)
    py1 = expit(-2.2 + 1.6 + 1.2 * U)
    py0 = expit(-2.2 + 1.2 * U)
    true_fd_risk[a] = float(np.mean(pm1 * py1 + (1 - pm1) * py0))

naive_fd = empirical_mean(Y, A == 1) - empirical_mean(Y, A == 0)
fd_rd = frontdoor_risk[1] - frontdoor_risk[0]
true_fd_rd = true_fd_risk[1] - true_fd_risk[0]

report("2. Front-door identification", [
    ("True do(A=1) risk", true_fd_risk[1]),
    ("True do(A=0) risk", true_fd_risk[0]),
    ("True causal risk difference", true_fd_rd),
    ("Naive observed risk difference", naive_fd),
    ("Front-door risk difference", fd_rd),
])

# 3. Proximal identification with two binary proxy variables
n_px = 300000
U = rng.binomial(1, 0.45, size=n_px)
Z = rng.binomial(1, expit(-1.0 + 2.0 * U))
C = rng.binomial(1, expit(-1.2 + 2.4 * U))
A = rng.binomial(1, expit(-1.2 + 1.4 * U + 0.9 * Z))
pY = expit(-2.0 + 0.9 * A + 1.5 * U)
Y = rng.binomial(1, pY)

proximal_risk = {}
proximal_det = {}
proximal_bridge = {}
for a in [0, 1]:
    P = np.zeros((2, 2))
    mu = np.zeros(2)
    for z in [0, 1]:
        idx = (A == a) & (Z == z)
        P[z, 0] = np.mean(C[idx] == 0)
        P[z, 1] = np.mean(C[idx] == 1)
        mu[z] = np.mean(Y[idx])
    h = solve_linear(P, mu)
    proximal_bridge[a] = h
    proximal_det[a] = P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]
    proximal_risk[a] = float(np.mean(C == 0) * h[0] + np.mean(C == 1) * h[1])

true_px_risk = {a: float(np.mean(expit(-2.0 + 0.9 * a + 1.5 * U))) for a in [0, 1]}
naive_px = empirical_mean(Y, A == 1) - empirical_mean(Y, A == 0)
px_rd = proximal_risk[1] - proximal_risk[0]
true_px_rd = true_px_risk[1] - true_px_risk[0]

report("3. Proximal causal identification, finite binary bridge", [
    ("True causal risk difference", true_px_rd),
    ("Naive observed risk difference", naive_px),
    ("Proximal bridge risk difference", px_rd),
    ("Bridge h_0(C=0)", proximal_bridge[0][0]),
    ("Bridge h_0(C=1)", proximal_bridge[0][1]),
    ("Bridge h_1(C=0)", proximal_bridge[1][0]),
    ("Bridge h_1(C=1)", proximal_bridge[1][1]),
    ("Proxy relevance determinant at A=0", proximal_det[0]),
    ("Proxy relevance determinant at A=1", proximal_det[1]),
])

# 4. Quantitative bias analysis for unmeasured confounding
n_qba = 450000
U = rng.binomial(1, 0.35, size=n_qba)
A = rng.binomial(1, expit(-1.0 + 1.7 * U))
base = 0.04
rr_a_true = 1.55
rr_u_true = 3.20
risk = base * (rr_a_true ** A) * (rr_u_true ** U)
Y = rng.binomial(1, risk)

risk1 = empirical_mean(Y, A == 1)
risk0 = empirical_mean(Y, A == 0)
rr_observed = risk1 / risk0
p_u1 = empirical_mean(U, A == 1)
p_u0 = empirical_mean(U, A == 0)
bias_factor = (p_u1 * rr_u_true + 1 - p_u1) / (p_u0 * rr_u_true + 1 - p_u0)
rr_corrected = rr_observed / bias_factor

n1 = np.sum(A == 1)
n0 = np.sum(A == 0)
y1 = np.sum(Y[A == 1])
y0 = np.sum(Y[A == 0])
se_log_rr = np.sqrt(1 / y1 - 1 / n1 + 1 / y0 - 1 / n0)

mc = 20000
p1_draw = np.clip(rng.normal(p_u1, 0.035, size=mc), 0.001, 0.999)
p0_draw = np.clip(rng.normal(p_u0, 0.035, size=mc), 0.001, 0.999)
rru_draw = np.exp(rng.normal(np.log(rr_u_true), 0.18, size=mc))
log_rr_obs_draw = rng.normal(np.log(rr_observed), se_log_rr, size=mc)
bf_draw = (p1_draw * rru_draw + 1 - p1_draw) / (p0_draw * rru_draw + 1 - p0_draw)
rr_corrected_draw = np.exp(log_rr_obs_draw) / bf_draw
qba_q = np.quantile(rr_corrected_draw, [0.025, 0.5, 0.975])

pd.DataFrame({
    "p_U_exposed": p1_draw,
    "p_U_unexposed": p0_draw,
    "RR_UY": rru_draw,
    "observed_RR_draw": np.exp(log_rr_obs_draw),
    "bias_factor": bf_draw,
    "corrected_RR": rr_corrected_draw,
}).to_csv(outdir / "qba_samples.csv", index=False)

report("4. Quantitative bias analysis", [
    ("True treatment risk ratio", rr_a_true),
    ("Observed confounded risk ratio", rr_observed),
    ("P(U=1|A=1)", p_u1),
    ("P(U=1|A=0)", p_u0),
    ("Deterministic confounding bias factor", bias_factor),
    ("Bias-corrected risk ratio", rr_corrected),
    ("Probabilistic QBA median", qba_q[1]),
    ("Probabilistic QBA 2.5%", qba_q[0]),
    ("Probabilistic QBA 97.5%", qba_q[2]),
])

# 5. Additive structural nested mean model with effect modification
n_sn = 120000
L = rng.normal(size=n_sn)
V = (L > 0).astype(float)
e = expit(-0.4 + 0.8 * L)
A = rng.binomial(1, e)
Y0 = 1.0 + 0.8 * L + rng.normal(size=n_sn)
beta_add_true = np.array([2.0, 1.2])
Y = Y0 + A * (beta_add_true[0] + beta_add_true[1] * V)

e_hat = expit(add_intercept(L) @ logistic_fit(add_intercept(L), A))
rA = A - e_hat
q = np.column_stack([np.ones(n_sn), V])
x = q.copy()
Mmat = (q * rA[:, None]).T @ (A[:, None] * x)
bvec = (q * rA[:, None]).T @ Y
beta_add_hat = solve_linear(Mmat, bvec)

report("5. Additive SNMM with two parameters", [
    ("True beta1", beta_add_true[0]),
    ("Estimated beta1", beta_add_hat[0]),
    ("True beta2 for A*V", beta_add_true[1]),
    ("Estimated beta2 for A*V", beta_add_hat[1]),
])

# 6. Multiplicative structural nested mean model
n_mult = 140000
L = rng.normal(size=n_mult)
V = (L > 0).astype(float)
e = expit(-0.3 + 0.7 * L)
A = rng.binomial(1, e)
beta_mult_true = np.array([0.45, -0.25])
mu0 = np.exp(0.2 + 0.2 * L)
mu = mu0 * np.exp(A * (beta_mult_true[0] + beta_mult_true[1] * V))
Y = rng.poisson(mu)

e_hat = expit(add_intercept(L) @ logistic_fit(add_intercept(L), A))
rA = A - e_hat
q = np.column_stack([np.ones(n_mult), V])

def mult_moments(beta):
    H = Y * np.exp(-A * (beta[0] + beta[1] * V))
    return np.mean(q * (rA * H)[:, None], axis=0)

beta_mult_hat = solve_nonlinear(mult_moments, [0.0, 0.0])
report("6. Multiplicative SNMM", [
    ("True beta1", beta_mult_true[0]),
    ("Estimated beta1", beta_mult_hat[0]),
    ("True beta2", beta_mult_true[1]),
    ("Estimated beta2", beta_mult_hat[1]),
    ("Maximum final estimating moment", float(np.max(np.abs(mult_moments(beta_mult_hat))))),
])

# 7. Structural nested logistic model: noncollapsibility demonstration
n_log = 350000
L = rng.normal(size=n_log)
A = rng.binomial(1, 0.5, size=n_log)
conditional_log_or = 0.8
p = expit(-1.2 + 1.2 * L + conditional_log_or * A)
Y = rng.binomial(1, p)

b_log = logistic_fit(add_intercept(A, L), Y)
p1 = float(np.mean(expit(-1.2 + 1.2 * L + conditional_log_or)))
p0 = float(np.mean(expit(-1.2 + 1.2 * L)))
marginal_or = (p1 / (1 - p1)) / (p0 / (1 - p0))
conditional_or = np.exp(conditional_log_or)

report("7. Structural nested logistic family and noncollapsibility", [
    ("True conditional causal odds ratio", conditional_or),
    ("Fitted conditional treatment odds ratio", np.exp(b_log[1])),
    ("Marginal causal odds ratio", marginal_or),
])

# 8. Longitudinal saturated structural nested mean model
n_long = 170000
L0 = rng.normal(size=n_long)
p0 = expit(-0.2 + 0.7 * L0)
A0 = rng.binomial(1, p0)
L1 = 0.5 * L0 + 0.8 * A0 + rng.normal(size=n_long)
p1 = expit(-0.3 + 0.4 * A0 + 0.8 * L1 + 0.2 * L0)
A1 = rng.binomial(1, p1)
psi1_true = np.array([2.2, 0.5, -0.4, 0.3])
psi0_direct = 1.4
baseline = 1.0 + 0.6 * L0 + 0.4 * L1 + rng.normal(size=n_long)
x1 = np.column_stack([np.ones(n_long), L1, A0, A0 * L1])
Y = baseline + psi0_direct * A0 + A1 * (x1 @ psi1_true)
psi0_blip_true = psi0_direct + 0.4 * 0.8

p0h = expit(add_intercept(L0) @ logistic_fit(add_intercept(L0), A0))
p1h = expit(add_intercept(A0, L0, L1) @ logistic_fit(add_intercept(A0, L0, L1), A1))
r1 = A1 - p1h
q1 = x1.copy()
M1 = (q1 * r1[:, None]).T @ (A1[:, None] * x1)
b1 = (q1 * r1[:, None]).T @ Y
psi1_hat = solve_linear(M1, b1)
H1 = Y - A1 * (x1 @ psi1_hat)
r0 = A0 - p0h
psi0_hat = np.sum(r0 * H1) / np.sum(r0 * A0)

report("8. Longitudinal saturated SNMM", [
    ("True psi0 blip through future covariate", psi0_blip_true),
    ("Estimated psi0", psi0_hat),
    ("True psi11", psi1_true[0]),
    ("Estimated psi11", psi1_hat[0]),
    ("True psi12 for A1*L1", psi1_true[1]),
    ("Estimated psi12", psi1_hat[1]),
    ("True psi13 for A1*A0", psi1_true[2]),
    ("Estimated psi13", psi1_hat[2]),
    ("True psi14 for A1*A0*L1", psi1_true[3]),
    ("Estimated psi14", psi1_hat[3]),
])

# 9. Structural nested cumulative failure/survival models
risk0 = 0.08
risk1 = 0.05
surv0 = 1 - risk0
surv1 = 1 - risk1
psi_cft = np.log(risk1 / risk0)
psi_cst = np.log(surv1 / surv0)
report("9. CFT and CST structural nested model parameters", [
    ("Risk under A=0", risk0),
    ("Risk under A=1", risk1),
    ("CFT psi = log risk ratio", psi_cft),
    ("Survival under A=0", surv0),
    ("Survival under A=1", surv1),
    ("CST psi = log survival ratio", psi_cst),
])

# 10. Structural AFT model, one parameter without censoring
n_aft = 120000
L = rng.normal(size=n_aft)
e = expit(-0.3 + 0.8 * L)
A = rng.binomial(1, e)
psi_aft_true = -0.40
T0 = np.exp(1.0 + 0.4 * L + rng.normal(0, 0.5, size=n_aft))
T = T0 * np.exp(-psi_aft_true * A)

e_hat = expit(add_intercept(L) @ logistic_fit(add_intercept(L), A))
rA = A - e_hat

def aft_moment(psi):
    H = T * np.exp(float(psi) * A)
    return float(np.mean(rA * H))

grid = np.linspace(-1.2, 0.4, 3201)
psi_aft_hat = float(grid[np.argmin(np.abs([aft_moment(x) for x in grid]))])

report("10. Structural AFT, one-parameter g-estimation", [
    ("True psi", psi_aft_true),
    ("Estimated psi", psi_aft_hat),
    ("True survival-time ratio exp(-psi)", np.exp(-psi_aft_true)),
    ("Estimated survival-time ratio", np.exp(-psi_aft_hat)),
])

# 11. Structural AFT with covariate effect modification
n_aft2 = 150000
L = rng.normal(size=n_aft2)
e = expit(-0.2 + 0.7 * L)
A = rng.binomial(1, e)
psi_aft2_true = np.array([-0.32, 0.16])
T0 = np.exp(1.0 + 0.3 * L + rng.normal(0, 0.45, size=n_aft2))
T = T0 * np.exp(-A * (psi_aft2_true[0] + psi_aft2_true[1] * L))

e_hat = expit(add_intercept(L) @ logistic_fit(add_intercept(L), A))
rA = A - e_hat
q = np.column_stack([np.ones(n_aft2), L])

def aft2_moments(psi):
    H = T * np.exp(A * (psi[0] + psi[1] * L))
    return np.mean(q * (rA * H)[:, None], axis=0)

psi_aft2_hat = solve_nonlinear(aft2_moments, [-0.1, 0.0])
report("11. Structural AFT with effect modification", [
    ("True psi1", psi_aft2_true[0]),
    ("Estimated psi1", psi_aft2_hat[0]),
    ("True psi2", psi_aft2_true[1]),
    ("Estimated psi2", psi_aft2_hat[1]),
])

# 12. Structural AFT with administrative and artificial censoring
n_ac = 260000
L = rng.normal(size=n_ac)
e = expit(-0.3 + 0.8 * L)
A = rng.binomial(1, e)
psi_ac_true = -0.35
T0 = np.exp(1.2 + 0.3 * L + rng.normal(0, 0.7, size=n_ac))
T = T0 * np.exp(-psi_ac_true * A)
K = 5.0
X = np.minimum(T, K)
event = (T <= K).astype(int)

e_hat = expit(add_intercept(L) @ logistic_fit(add_intercept(L), A))
rA = A - e_hat

def artificial_censoring_moment(psi):
    psi = float(psi)
    H = X * np.exp(psi * A)
    Kpsi = K * min(1.0, np.exp(psi))
    Delta = (H < Kpsi).astype(float)
    return float(np.mean(rA * Delta)), float(np.mean(Delta))

grid = np.linspace(-1.0, 0.5, 3001)
obj = [abs(artificial_censoring_moment(x)[0]) for x in grid]
psi_ac_hat = float(grid[np.argmin(obj)])
kept_fraction = artificial_censoring_moment(psi_ac_hat)[1]

report("12. AFT g-estimation with artificial censoring", [
    ("True psi", psi_ac_true),
    ("Estimated psi", psi_ac_hat),
    ("Observed event fraction", float(np.mean(event))),
    ("Fraction retained after artificial censoring", kept_fraction),
])

# 13. Cross-world mediation formula: pure direct and total indirect effects
n_med = 230000
A = rng.binomial(1, 0.5, size=n_med)
M = rng.binomial(1, expit(-0.8 + 1.6 * A))
Y = 1.0 + 0.8 * A + 1.2 * M + 0.5 * A * M + rng.normal(size=n_med)

pM = {a: empirical_mean(M, A == a) for a in [0, 1]}
EY = {(a, m): empirical_mean(Y, (A == a) & (M == m)) for a in [0, 1] for m in [0, 1]}

def mediation_formula(a_y, a_m):
    pm1 = pM[a_m]
    return EY[(a_y, 1)] * pm1 + EY[(a_y, 0)] * (1 - pm1)

mu00 = mediation_formula(0, 0)
mu10 = mediation_formula(1, 0)
mu11 = mediation_formula(1, 1)
pure_direct = mu10 - mu00
total_indirect = mu11 - mu10
total_effect_med = mu11 - mu00
cde0 = EY[(1, 0)] - EY[(0, 0)]
cde1 = EY[(1, 1)] - EY[(0, 1)]

report("13. Chapter 23 cross-world mediation formula", [
    ("E[Y(0,M(0))]", mu00),
    ("E[Y(1,M(0))]", mu10),
    ("E[Y(1,M(1))]", mu11),
    ("Pure direct effect", pure_direct),
    ("Total indirect effect", total_indirect),
    ("Pure direct + total indirect", pure_direct + total_indirect),
    ("Total effect", total_effect_med),
    ("Controlled direct effect at M=0", cde0),
    ("Controlled direct effect at M=1", cde1),
])

# 14. Separable effects and empirical verification by a future component trial
n_sep = 260000
A = rng.binomial(1, 0.5, size=n_sep)
N = A.copy()
O = A.copy()
M = rng.binomial(1, expit(-1.0 + 1.6 * N))
Y = 1.0 + 1.2 * M + 0.9 * O + 0.3 * M * O + rng.normal(size=n_sep)

pm0 = empirical_mean(M, A == 0)
mediation_cross = empirical_mean(Y, (A == 1) & (M == 1)) * pm0 + empirical_mean(Y, (A == 1) & (M == 0)) * (1 - pm0)

p_m_n0 = expit(-1.0)
mu_n0_o1 = 1.0 + 1.2 * p_m_n0 + 0.9 + 0.3 * p_m_n0
p_m_n1 = expit(-1.0 + 1.6)
mu_n1_o1 = 1.0 + 1.2 * p_m_n1 + 0.9 + 0.3 * p_m_n1
mu_n1_o0 = 1.0 + 1.2 * p_m_n1
sep_n_effect = mu_n1_o1 - mu_n0_o1
sep_o_effect = mu_n1_o1 - mu_n1_o0

p_m_invalid = expit(-1.0 + 0.8)
mu_n0_o1_invalid = 1.0 + 1.2 * p_m_invalid + 0.9 + 0.3 * p_m_invalid

report("14. Separable effects and empirical falsification", [
    ("Mediation formula from original two-arm trial", mediation_cross),
    ("Future trial E[Y(n=0,o=1)] if assumptions hold", mu_n0_o1),
    ("Difference under valid separability", mediation_cross - mu_n0_o1),
    ("Separable N effect at O=1", sep_n_effect),
    ("Separable O effect at N=1", sep_o_effect),
    ("Future E[Y(n=0,o=1)] if O also affects M", mu_n0_o1_invalid),
    ("Mismatch that would refute separability story", mediation_cross - mu_n0_o1_invalid),
])

# 15. Path-specific effect through the front-door pathway in Technical Point 23.3
n_ps = 420000
H = rng.binomial(1, 0.5, size=n_ps)
L = rng.binomial(1, expit(-0.4 + 1.3 * H))
A = rng.binomial(1, expit(-0.8 + 1.8 * L))
Y = 1.0 + 1.5 * A + 0.7 * L + 1.2 * H + rng.normal(size=n_ps)

inner_y = {}
for a in [0, 1]:
    inner_y[a] = sum(empirical_mean(Y, (A == a) & (L == l)) * empirical_prob(L == l) for l in [0, 1])

path_mean = {}
for n in [0, 1]:
    p_a1 = empirical_mean(A, L == n)
    path_mean[n] = inner_y[1] * p_a1 + inner_y[0] * (1 - p_a1)

path_specific_fd = path_mean[1] - path_mean[0]
p_a1_l1 = expit(-0.8 + 1.8)
p_a1_l0 = expit(-0.8)
true_path_specific = 1.5 * (p_a1_l1 - p_a1_l0)
true_total_L = 0.7 + true_path_specific

report("15. Path-specific front-door effect", [
    ("True effect of L through L->A->Y only", true_path_specific),
    ("Front-door path-specific estimate", path_specific_fd),
    ("True total effect of setting L in this simulation", true_total_L),
    ("Direct-path contribution excluded by path estimand", 0.7),
])

# 16. Basic numerical verification checks
checks = [
    ("frontdoor", fd_rd, true_fd_rd, 0.02),
    ("proximal", px_rd, true_px_rd, 0.025),
    ("QBA", rr_corrected, rr_a_true, 0.06),
    ("additive_SNMM_beta1", beta_add_hat[0], beta_add_true[0], 0.08),
    ("additive_SNMM_beta2", beta_add_hat[1], beta_add_true[1], 0.10),
    ("multiplicative_SNMM_beta1", beta_mult_hat[0], beta_mult_true[0], 0.06),
    ("multiplicative_SNMM_beta2", beta_mult_hat[1], beta_mult_true[1], 0.08),
    ("AFT_one_parameter", psi_aft_hat, psi_aft_true, 0.03),
    ("AFT_artificial_censoring", psi_ac_hat, psi_ac_true, 0.05),
    ("path_specific_frontdoor", path_specific_fd, true_path_specific, 0.03),
]
verification = []
for name, estimate, truth, tolerance in checks:
    error = abs(float(estimate) - float(truth))
    passed = error <= tolerance
    verification.append((name, estimate, truth, error, tolerance, passed))
    if not passed:
        raise AssertionError(f"Verification failed for {name}: estimate={estimate}, truth={truth}, tolerance={tolerance}")

pd.DataFrame(verification, columns=["check", "estimate", "truth", "absolute_error", "tolerance", "passed"]).to_csv(outdir / "advanced_verification.csv", index=False)

advanced_results = pd.DataFrame({
    "method": [
        "frontdoor_RD",
        "proximal_RD",
        "QBA_corrected_RR",
        "additive_SNMM_beta1",
        "additive_SNMM_beta2",
        "multiplicative_SNMM_beta1",
        "multiplicative_SNMM_beta2",
        "longitudinal_SNMM_psi0",
        "AFT_psi",
        "AFT_artificial_censoring_psi",
        "pure_direct_effect",
        "total_indirect_effect",
        "separable_cross_mean",
        "path_specific_frontdoor_effect",
    ],
    "estimate": [
        fd_rd,
        px_rd,
        rr_corrected,
        beta_add_hat[0],
        beta_add_hat[1],
        beta_mult_hat[0],
        beta_mult_hat[1],
        psi0_hat,
        psi_aft_hat,
        psi_ac_hat,
        pure_direct,
        total_indirect,
        mediation_cross,
        path_specific_fd,
    ],
})
advanced_results.to_csv(outdir / "what_if_advanced_results.csv", index=False)

print("\nSaved what_if_advanced_results.csv, advanced_verification.csv, qba_samples.csv, swig_examples.csv, and SWIG DOT files")
