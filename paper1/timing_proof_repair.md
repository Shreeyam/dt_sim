# Repairing the latest-feasible timing proposition

The timing result can be retained as a conditional theorem. The current assumptions do not establish unconditional optimality for a moving rectangular detector. No simulation or manuscript source is changed by this note.

## A sufficient condition weaker than nondecreasing density

Fix pitch, physical spacing `g > 0`, the anchor interval, the baseline burden, and the information available before the proposed lookahead. Let `t` denote maneuver initiation and evaluate detector support at acquisition, `t + t_man/2`, under the paper's symmetric maneuver model. Define

\[
U(t)=\max\{0,W(t)-c_L(t,g)-c_R(t,g)\},\qquad
m(t)=\lambda_c(t)U(t)
=pN_{\mathrm{obs}}(t)\frac{U(t)}{W(t)}.
\]

Here `m` is the expected number of clear candidates in the **usable** support under the paper's uniform/Poissonized model; it is not the total expected clear count in the detector. Empty support has `m = 0` and chain value zero.

**Proposition.** Suppose every admissible maneuver fits within the same schedule gap under the stated maneuver model. If both `U(t)` and `m(t)` are nondecreasing with initiation time, then the two-anchor estimated net advantage is nondecreasing. Consequently, the latest feasible initiation,

\[
t^\star=t_{\mathrm{next}}-t_{\mathrm{man}},
\]

is a maximizer. For this endpoint conclusion alone, it suffices that `U(t*) >= U(t)` and `m(t*) >= m(t)` for every admissible `t`; monotonicity at all intermediate times is unnecessary.

**Proof.** For positive usable support, rewrite the argument of the regularized gamma CDF as

\[
z_k(t)=\lambda_c(t)[U(t)-(k-1)g]_+
=m(t)\left[1-\frac{(k-1)g}{U(t)}\right]_+.
\]

For every integer `k >= 1`, this expression is nondecreasing in both `m` and `U`. The regularized lower gamma CDF is nondecreasing in its argument, since

\[
\frac{\partial}{\partial z}\frac{\gamma(k,z)}{\Gamma(k)}
=\frac{z^{k-1}e^{-z}}{\Gamma(k)}\geq0.
\]

Therefore every term of the finite chain expectation is nondecreasing, and so is their sum. Empty support contributes zero and causes no exception. Gap containment gives zero displaced-task cost within the assumed maneuver model; subtracting the fixed baseline and multiplying by fixed nonnegative mean utility preserve the ordering. This proves the claim.

The same conditions establish monotonicity of the renewal-rate approximation **at fixed physical spacing**:

\[
\widehat K_b=\frac{mU}{U+mg},\qquad
\frac{\partial\widehat K_b}{\partial m}
=\frac{U^2}{(U+mg)^2}\geq0,\qquad
\frac{\partial\widehat K_b}{\partial U}
=\frac{m^2g}{(U+mg)^2}\geq0.
\]

For a claim about maximizing the break-even spacing itself, require this dominance for **every tested spacing** `g`, since the threshold varies `g` and hence changes anchor clipping. A simple stronger sufficient condition is nested detector-support intervals and nondecreasing density: these imply usable-length and usable-count dominance for every spacing. Constant homogeneous density is a special case. The fixed-count uniform assumption alone does not establish any of these orderings over time.

## What the repair establishes

The theorem does not require candidate density to increase. A longer usable interval can spread the same expected count farther apart and improve packing even while density falls. For example, with `g = 10` and `m = 5`, increasing `U` from 40 to 80 halves density but increases the finite estimate from 2.37601 to 3.15089.

The original counterexample is excluded for the correct reason: extending support from `[20,90]` to `[20,95]` between anchors 0 and 100, with `g = 10`, `p = 0.5`, and `N_obs = 10`, keeps `U = 70` but reduces `m` from 5 to 4.66667. The finite estimate falls from 3.00347 to 2.88000.

The instantaneous rectangular footprint can lose accesses as it moves, and the policy recomputes `N_obs/W`. The current implementation therefore does not automatically satisfy the proposed conditions. The honest manuscript framing is:

> We use the latest feasible initiation to preserve scheduled imaging and maximize forward reach. Under nondecreasing usable-support length and expected usable clear-candidate count, this timing also maximizes the two-anchor estimated advantage. For general moving detector footprints, it is a timing heuristic.

The phrase “preserve scheduled imaging” is conditional on the maneuver model accounting for the required transitions; this theorem does not repair the separate roll/pitch accounting concern. Also replace the existing claim that later observations necessarily enlarge the revealed set, and distinguish initiation from acquisition in the actionable-window equation. Retaining this conditional result does not require changing the evaluated policy or rerunning the reported trials.

## Checks

The rewritten finite formula was compared against `chain_anchor_support` on 500 reproducible random pairs ordered in both `U` and `m` (seed 20260909). It matched the implementation, and both the finite and fixed-spacing rate estimates preserved the asserted ordering. The counterexample and falling-density example above were recomputed using the implementation. These numerical checks supplement the analytic proof; they do not verify the new assumptions on orbit-derived windows.
