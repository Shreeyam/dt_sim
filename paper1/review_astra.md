# Review of “Look Before You Slew”

**Follow-up status after discussion and the FoV edit:** The FoV contact/containment issue below has been corrected, and the original six-panel detector figure has been restored in the main text. The author confirmed that greedy's early acquisition and ability to displace scheduled images are intentional; Section 3 now treats that as a policy definition and reporting issue. A conditional repair of the timing proposition, with a complete proof, is in [timing_proof_repair.md](/Users/shreeyam/Projects/phd_extensions/dt_sim/paper1/timing_proof_repair.md). Its additional conditions have not been verified for the moving detector, and the manuscript timing proposition is still unchanged. The initial assessment below records the review before these follow-ups.

Reviewed 8 September 2026. I reviewed the current LaTeX sources and the 20-page `template.pdf` dated 24 August, including the figures and appendices. The PDF in `output/pdf/` is an older July version. I also inspected the relevant simulator and figure-generation code, recomputed the headline aggregates from the saved IID and BCM trials, and checked numerical counterexamples. I did not rerun the full experimental sweep or edit the manuscript.

My assessment is **major revision before submission**. The paper has a useful central contribution: a cheap approximation to the schedule value of a body-fixed lookahead, with an interpretable dead-time threshold. The progression from synthetic calibration to paired IID and structured-cloud trials is sensible. The reported aggregate numbers are internally consistent with the saved results. The main concerns are the physical feasibility model, the timing guarantee, and what the experimental comparison establishes.

## 1. The maneuver model does not establish the claimed physical feasibility

**Locations:** [method, lines 431 and 438 onward](/Users/shreeyam/Projects/phd_extensions/dt_sim/paper1/sections/method.tex:431); [simulator, pitch timing](/Users/shreeyam/Projects/phd_extensions/dt_sim/src/dt_sim/simulator.py:179); [camera projection call](/Users/shreeyam/Projects/phd_extensions/dt_sim/src/dt_sim/simulator.py:198).

The paper schedules primary images at potentially different roll angles, but its lookahead round trip is timed only from the pitch displacement:

\[
t_{\rm man}=2\{t_{\rm slew}(\alpha-\alpha_0)-t_s\}.
\]

In the simulator, lookahead projection uses roll zero, and this maneuver duration does not depend on either neighboring image's roll. Thus a lookahead at the rest pitch has zero maneuver duration and can occur at `t_next`, even when the next primary image requires nonzero roll. There is no explicit transition from the lookahead attitude to that image's attitude. Subtracting one settling interval from the gap does not, by itself, account for these attitude transitions.

This matters because physical feasibility is one of the paper's central contributions. A conservative spacing used to *score* insertions cannot certify the feasibility of the actual sensing maneuver.

**Revision:** Define the spacecraft attitude at the previous image, lookahead acquisition, and next image. Require the two attitude transitions and appropriate settling/acquisition time to fit in the gap. If roll and pitch maneuvers overlap, state and justify the model for that overlap. Distinguish observation time from maneuver initiation time throughout. Audit accepted actions against this model before interpreting the scheduling results as physically feasible.

There is also a separate angle inconsistency. The text calls `g_phys = t_slew(theta_FoR)` the worst-case cross-track slew across a ±45° FoR. That excursion is 90°, whereas the implementation evaluates 45°. With the default settling time of 10 s, the three nominal 45° slew settings of 25/35/45 s imply 90° times of **31.2/45.4/59.5 s**, respectively. Either use `t_slew(2 theta_FoR)` for the stated bound or describe the current quantity as an effective approximation and quantify its adequacy. This also affects the connection to Appendix A's 90° platform comparisons.

## 2. Proposition 1 is not established by the current assumptions

**Locations:** [timing proposition](/Users/shreeyam/Projects/phd_extensions/dt_sim/paper1/sections/method.tex:438); [Appendix B.4 proof](/Users/shreeyam/Projects/phd_extensions/dt_sim/paper1/sections/appendix.tex:151).

The proof relies on two steps that need correction:

1. An expanding horizon interval does not imply an expanding instantaneous rectangular camera footprint. At fixed pitch, delaying the observation moves both footprint edges; an access can leave through the near or lateral edge. Therefore the visible candidate count need not be nondecreasing.
2. Even granting nondecreasing count and support end, the two-anchor estimate need not increase. Its density is recomputed as `p N_obs / W`, and the right-anchor clearance can prevent additional support from contributing usable length.

A concrete counterexample to the second step uses the paper's own estimator. Fix anchors at 0 and 100 s, `g = 10 s`, `p = 0.5`, and `N_obs = 10`. Compare support `[20,90]` with `[20,95]`. The supports are nested and the count is unchanged. In both cases usable length is 70 s, because the extension lies inside the right-anchor exclusion zone. Density falls from `5/70` to `5/75`, and the finite-window estimate falls from **3.00347 to 2.88000**. A fixed burden of 3 even changes the acceptance decision. These values were checked independently from the gamma/Poisson formula and against the implementation.

This counterexample addresses the asserted monotonic implication; it is not a claim that every orbit produces those exact supports. The moving-footprint issue is an additional reason the theorem needs stronger hypotheses for the actual policy.

**Revision:** Keep “latest feasible” as an explicit timing heuristic unless a correct guarantee is proved. A restricted theorem would need sufficient conditions on nested usable visibility and the point-process intensity, not merely count and horizon extent. Add a modest time–pitch search diagnostic on actual gaps to measure the loss from fixing time analytically. Restrict the zero-opportunity-cost statement to the maneuver model for which it is verified.

## 3. Describe greedy's intended timing and displacement behavior

**Locations:** [baseline description](/Users/shreeyam/Projects/phd_extensions/dt_sim/paper1/sections/experiments.tex:9); [policy-specific timing](/Users/shreeyam/Projects/phd_extensions/dt_sim/src/dt_sim/simulator.py:110); [greedy missed-image removal](/Users/shreeyam/Projects/phd_extensions/dt_sim/src/dt_sim/simulator.py:348).

The manuscript presents greedy primarily as choosing the view with the most unobserved accesses without a value gate. The implementation also gives it different timing and feasibility rules:

- Value-based policies reject a pitch whose maneuver does not fit in the gap and acquire at the latest feasible time.
- Greedy starts from the previous image, can overrun the next image, and explicitly deletes images displaced by that overrun.

This is consequential in the saved experiments. Averaged over the 72 trial/cell combinations per cloud model, greedy records **18.57 missed tasks per IID replay and 18.38 per BCM replay**. The value-based policies record zero. The respective missed-clear counts are 13.83 and 13.47. These describe the policy differences, not a causal estimate of how much utility a revised baseline would recover. The author has confirmed that these differences are intentional.

The reported policy-package comparison remains valid within the simulator, but it does not isolate the benefit of the renewal/value gate. The striking deterioration of greedy with slower slews could partly reflect its different overrun behavior.

**Revision:** Keep the intended greedy policy and describe its early acquisition and possible displacement of scheduled images explicitly. Frame the reported improvements as a comparison of complete policies. A matched-timing/feasibility baseline would be an optional ablation if the paper seeks to isolate the value gate's causal contribution; it is not required merely to report the intended policy comparison.

## 4. The horizontal-FoV requirement switches between containment and contact

**Locations:** [Equations 8–9](/Users/shreeyam/Projects/phd_extensions/dt_sim/paper1/sections/method.tex:159); [rolled contact implementation](/Users/shreeyam/Projects/phd_extensions/dt_sim/paper1/scripts/regenerate_lookahead_figures.py:266).

Equation 8 takes a maximum over every admissible point on each FoR edge. This describes containing the entire vertically admissible edge. Equation 9 instead evaluates an upper-edge/limb contact, and the rolled implementation explicitly takes the **minimum** horizontal angle along each edge before maximizing across the two edges. This describes touching each edge somewhere. These are different design requirements.

For the nadir attitude, altitude 400 km, vertical FoV 60°, and primary FoR ±45°, the valid edge point at `x = 0` requires a full horizontal FoV of about **98.2°**. The upper-edge contact used by Equation 9 requires about **54.4°**. Thus the nadir component of Equation 9 cannot equal the maximum stated in Equation 8. This changes what the 60° × 60° design can be claimed to cover.

**Revision:** Choose the intended quantifier. For “touch both edges,” the inner operation should be a minimum over admissible contacts, followed by a maximum over edges and design attitudes. For “contain every admissible edge point,” retain the maximum and revise the sizing equations and figures. State whether contact provides sufficient lead time to be actionable; geometric contact alone does not guarantee that. Align the captions and the stationary/agile design discussion with the chosen requirement.

## 5. Validate the approximation where the policy actually uses it

**Locations:** [calibration setup](/Users/shreeyam/Projects/phd_extensions/dt_sim/paper1/sections/experiments.tex:24); [fixed replacement burden](/Users/shreeyam/Projects/phd_extensions/dt_sim/paper1/sections/method.tex:379).

Experiment 1 usefully tests an idealized fixed-count, IID, constant-spacing model. It does not directly calibrate actual orbit-derived, partially observed repair windows with angle-dependent slews and previous revelations. The CCC of 0.992 should therefore be presented as validation of the stochastic approximation under its sampling model, rather than evidence that onboard action values are equally accurate.

The policy also estimates utility from *unobserved* candidates and subtracts `p N_sched` across the common anchor interval. After earlier lookaheads, some scheduled tasks have known clear utility rather than prior utility `p`; repair may retain them and other tasks outside the scored support. This makes the estimator's relation to actual incremental repair value more complicated than the initial IID picture. In particular, “conservative” needs a stated domain or empirical qualification.

**Revision:** Sample real decision windows before the acceptance decision, including rejected actions. For each, freeze geometry and current beliefs, sample only unrevealed cloud states, and compute paired local-MILP improvement with and without the proposed observation. Report sign errors and regret near zero as well as overall agreement. Include support coverage, candidate clustering, and previously observed scheduled utility as diagnostics. This would connect the derivation to the operational claim much more directly.

## 6. Clarify the scope of the BCM evidence and uncertainty estimates

The BCM trials demonstrate performance under structured masks with perfect revelation of access-time truth. A sensor at lookahead time does not automatically know the state at the later imaging time, even with a perfect classifier. The current text mentions perfect classification, but should explicitly state the additional cloud persistence/perfect future-state assumption. A useful sensitivity check would reveal the mask at lookahead time and evaluate utility at acquisition time, or vary a persistence/error model with lead time.

The twelve dates share the phase-matched ground track. This is a sensible controlled comparison, but generalization across request geography, inclination, and orbital phase remains untested. Report paired confidence intervals for policy differences, preserving date pairing and dependence across FoV/agility cells. The displayed SDs characterize trial spread, not confidence intervals on the improvement. In particular, the small two-anchor versus break-even differences should not be given a causal explanation based solely on Experiment 1's bias result.

## Numerical checks that passed

I recomputed the following from the saved `iid_prior_weighted_20260718` and `bcm_prior_weighted_20260718` trial CSVs:

| Cloud truth | Policy | Mean gap closed | Mean lookaheads / initial scheduled access |
|---|---|---:|---:|
| IID | Greedy | 58.09% | 0.26893 |
| IID | Break-even | 78.20% | 0.16228 |
| IID | Two-anchor | 79.01% | 0.16576 |
| BCM | Greedy | 43.60% | 0.26427 |
| BCM | Break-even | 77.59% | 0.16134 |
| BCM | Two-anchor | 78.96% | 0.16478 |

These support the rounded headline values and “about 40% fewer” statement for the evaluated policies. The paper also correctly distinguishes normalized gap closure from absolute clear-image utility. The main concern is interpretation and model validity, not the arithmetic of those summaries.

## Presentation, definitions, and reproducibility

- **Make the advantage definition consistent.** Equations 13 and 22 give net advantage as `c_bar K - J_minus`, whereas Equation 24 additionally subtracts `p c_bar N_sched`. Define gross repaired utility and incremental advantage separately, or include the baseline subtraction in the first definition. [Source](/Users/shreeyam/Projects/phd_extensions/dt_sim/paper1/sections/method.tex:275).
- **Separate initiation and acquisition epochs.** The algorithm initiates at `t_next - t_man`; the simulator acquires at `t_next - t_man/2`. Equation 31's horizon endpoint uses `t` without this offset and without the pitch-specific far edge. Use distinct symbols and derive support at acquisition time.
- **Add an experiment parameter table.** Include inclination, FoR half-angle, settling time, pitch bounds and limb cap, acquisition/inference assumptions, city selection, seed, solver/version, and optimality status. For calibration, state the sample ranges and Monte Carlo draws per window. Relevant defaults exist in code but are not fully specified in the paper.
- **Provide figure-to-run provenance.** The current headline data can be found locally, but the paper's public repository points to `dt_sim`, while the paper experiment drivers I inspected are in the parent workspace's `experiments/` directory. Verify that readers can access these drivers and the configuration/data needed for every published result; add an archived version or commit and a reproduction command.
- **Tighten the related-work characterization.** Kangaslahti et al. (2024) explicitly searches primary-observation paths using lookahead information and slewing constraints. The manuscript's phrase “searches lookahead choices” needs a precise definition and supporting passage. Describe the actual decision-variable differences instead of relying on “precluding a direct numerical comparison” to dismiss an adapted comparator. [Primary paper](https://ai.jpl.nasa.gov/public/documents/papers/Kangaslahti_DT_ICRA_2024.pdf).
- **Show scheduling behavior in the case study.** The Brazil maps show footprints, but not which observations were removed, inserted, or preserved. Add a compact timeline with selected pitches, before/after schedules, and realized clear counts. Explain the virtual anchor's attitude and whether its initialization is applied consistently to every policy.
- **Shorten the route to the main idea.** The estimator starts on page 7, after several pages of geometry. A small worked gap example immediately after the ConOps would make the value model easier to follow. Consider moving more of the detector derivation to the appendix.
- **Clean up minor notation and claims.** `beta` is reused for the slew coefficient and half vertical FoV. Define all timing support symbols once. Avoid treating conventional execution as a guaranteed realized lower bound; the saved greedy BCM results include a trial below it. Qualify the Appendix B.3 concavity claim or prove it, and avoid implying that exact finite-count evaluation would inherently lose monotonicity in spacing.

The rendered paper is generally legible, with no obvious broken references or clipped content in the inspected pages. Following discussion, the FoV quantifiers are resolved and the greedy comparison can remain with an explicit policy description. The principal remaining corrections are attitude-feasibility accounting and the scope of the timing theorem. Calibration on actual repair windows and a matched greedy ablation would strengthen the empirical interpretation, especially any claim isolating the value gate's contribution.
