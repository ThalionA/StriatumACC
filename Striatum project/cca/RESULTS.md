# Spatial CCA of striato-cortical communication across learning

**Methods and results.** Draft, 2026-05-23. Destined for ResearchVault
(`Methods/`); drafted here because that vault was not mounted in the session
that produced it. Pipeline: `StriatumACC/Striatum project/cca/`. Design
record: `UNDERSTANDING.md`; chronological log: `NOTES.md`.

> **Status (2026-05-23, round 7): this writeup describes rounds 1–2 and is now
> out of date.** The pipeline has since changed substantially — significance
> switched to a held-out-CC permutation test (round 6); the intermediate epoch
> was dropped, leaving a direct naive-vs-expert contrast; the cohort gate
> relaxed to `lp >= 10`; z-scoring moved to the whole engaged period; and
> Stage 2 now runs a residual/signal × FS-in/out robustness grid with paired,
> unpaired and vs-0 tests (round 7). The §3 Methods and §4 Results below
> predate all of that. A full rewrite against the round-7 figures is pending
> Theo's review of them — see `NOTES.md` (rounds 6–7) for the current state.

---

## 1. Question

How does communication between striatal and cortical/hippocampal areas change
as mice learn the head-fixed VR-corridor task? Four sub-questions: do
inter-areal connections strengthen or weaken; does the direction of
information flow change; which units take part in the communication subspaces,
and are they shared across area pairs; and how do those subspaces reorient
across learning. The analysis adapts — rather than copies — canonical
correlation analysis (CCA) as used by Han & Helmchen (2024, "H&H") and the
hippocampal–retrosplenial subspace-communication study (Gonzalez et al. 2026).

## 2. Data and cohort

Sixteen mice were recorded with Neuropixels probes during within-session
learning of the corridor task. The analysis uses the pre-computed
spatially-binned firing rates (`spatial_binned_fr_all`, 100 bins of 2.5 cm
over the 250 cm corridor; the pipeline is bin-count-agnostic and also runs on
the earlier 50-bin/5 cm preprocessing). Five areas are in scope — DMS, DLS,
ACC, V1, CA1 — giving ten area pairs; DG was excluded for want of units.

Each mouse is assigned a learning point: the first trial that is itself below
threshold ($z \le -2$) and from which performance is sustained — the ten-trial
window starting there contains at least seven sub-threshold trials. (An earlier
version reproduced the MATLAB `movsum` rule, which returns the window's *start*
trial; that trial can be above threshold, so the learning point could land on
a non-learning trial — corrected.) Trials are truncated at the disengagement
point
(`change_point_mean`). Three ten-trial epochs are then defined per mouse:
*naive* (trials 1–10), *intermediate* (the ten trials ending at the learning
point) and *expert* (the ten trials after it). Twelve mice have a genuine
learning point; three (animals 8, 12, 16) are non-learners given the
cohort-mean "yoked" learning point ($\approx 42$) so their early/mid/late
windows serve as a learning-specificity control. Animal 8's detected learning
point (trial 16) is implausibly early and was treated as yoked. Animal 15 is
dropped — only 27 usable trials.

Fast-spiking units (`final_neurontypes` type 2) are excluded in the primary
analysis and retained in a comparison variant. The per-epoch sample budget is
ten trials $\times$ 100 bins $= 1000$ nominal samples; missing bins (see §3)
reduce this somewhat.

The binding constraint is **co-recording**. Counting mice with a usable
learning point, $\geq 30$ usable trials and $\geq 4$ units in *both* areas of a
pair, the well-powered pairs are the striatal–cingulate triangle (DMS–ACC
$n=10$–12, DMS–DLS $n=8$–9, DLS–ACC $n=7$–9). V1 pairs reach only $n=3$–4 and
CA1 pairs $n=1$–2. Group inference is therefore restricted to the striatal
triangle; V1 and CA1 results are reported as per-animal exploratory
observations and should not be over-interpreted.

## 3. Methods

**Residualisation.** "Communication" is defined as trial-to-trial co-fluctuation
rather than shared spatial tuning. For each epoch the per-(bin, unit) trial
mean is subtracted, leaving residuals; CCA on the raw activity would be
dominated by both areas trivially encoding corridor position. Missing spatial
bins (unvisited on a given trial) are **dropped** from every fit rather than
imputed — at 2.5 cm resolution they reach 23% of within-epoch samples for the
fastest animal (0–3% for most), too many to impute without corrupting the
canonical correlations. The missing pattern is per-(trial, bin) and shared
across areas, so the dropped-sample set is common to both members of a pair.

**Dimensionality reduction.** Each area's residuals are reduced by PCA to $k$
components, with $k$ chosen from the observed (non-missing) sample budget
($k = \lfloor n_{\text{valid}}/25 \rfloor$, typically the cap of 30),
capped by the smaller area's unit count and by the per-epoch numerical rank.
PCA is fitted *per epoch*; cross-epoch comparisons are made in neuron space.

**Canonical correlation.** CCA is computed by an SVD-based, rank-robust
routine: each population is reduced to an orthonormal basis of its column
space and the canonical correlations are the singular values of the
cross-product of those bases. The estimator is the plain PCA-then-CCA of H&H.

**Cross-validation.** Canonical correlations are cross-validated with five
folds over *whole trials* (train on eight, test on two) so that within-trial
spatial autocorrelation cannot leak between train and test. The held-out CC1
is reported as the effect size; the in-sample CC1 is biased high by roughly
two- to four-fold (see §4.1) and is used only as the permutation-test
statistic.

**Lagged CCA and directionality.** CCA is refit at every spatial-bin lag from
$-5$ to $+5$ bins ($\pm25$ cm); a positive lag pairs area X's residual at bin
$b$ with area Y's at bin $b+\ell$, i.e. X leading Y. The Information Flow Index
$\mathrm{IFI}=(\overline{CC}_{\ell>0}-\overline{CC}_{\ell<0})/(\overline{CC}_{\ell>0}+\overline{CC}_{\ell<0})$,
on held-out CC1 clipped at zero, summarises direction in $[-1,1]$.

**Surrogates.** Significance of CC1 uses an in-sample permutation test: the
trial correspondence between the two areas is permuted 200 times (H&H scheme),
and the real in-sample CC1 is compared with the shuffled distribution by a
non-parametric $p$-value. Because the real and shuffled statistics carry the
same overfitting bias, the comparison is valid. A circular position-bin shift
provides a second, robustness null. (The in-sample test is a compute-driven
choice; the held-out CC1 remains the reported effect size.)

**Membership and subspaces.** Each canonical dimension is back-projected to
neuron space two ways — as a structure coefficient (the correlation of a
neuron's in-subspace residual activity with the canonical variate) and as a
raw weight (the PCA loadings times the canonical coefficient). A neuron is a
"member" of the communication subspace if its contribution is in the top
quartile. Weight sparsity is summarised by the Gini coefficient. The
communication subspace is taken to be the *dominant* canonical direction only:
at three dimensions the within-epoch split-half principal angle was already
near-orthogonal, so the higher dimensions are not estimable from ten-trial
epochs. Subspace reorientation across epochs is measured by the principal
angle between the dominant directions in neuron space, calibrated against a
within-epoch split-half angle (the sampling-noise floor).

**Partial CCA.** For the DMS/DLS/ACC triplet (the only three areas recorded
together in enough mice), partial CCA regresses the third area's activity out
of both members of a pair before CCA, testing whether the pair's coupling
survives removal of a shared striatal input.

**Implementation.** A tested Python package (`striatum_cca`): 67 unit tests on
synthetic data with known ground truth, `ruff`-clean, NumPy/SciPy with CPU
multiprocessing. Stages run in seconds to tens of seconds.

## 4. Results

### 4.1 Communication strength is modest and heterogeneous

Held-out CC1 for the striatal triangle (FS-excluded, learner-group means;
naive / intermediate / expert):

| pair | $n$ | naive | intermediate | expert |
|---|---|---|---|---|
| DMS–DLS | 8 | 0.27 | 0.34 | 0.26 |
| DMS–ACC | 10 | 0.18 | 0.11 | 0.17 |
| DLS–ACC | 7 | 0.13 | 0.10 | 0.14 |

About half the learner mice show a CC1 significantly above the trial-shuffle
null in any given epoch (DMS–DLS 4–5 of 8; DMS–ACC 4–5 of 10; DLS–ACC 2–3 of
7). Residual communication is therefore real but modest and heterogeneous
across animals — a expected outcome for trial-to-trial noise correlations, and
much weaker than the in-sample CC1 (0.4–0.9) that an un-cross-validated
analysis would have reported. DMS–DLS is consistently the strongest pair.

The epoch profile is suggestive but not decisive: DMS–DLS bulges at the
intermediate epoch while DMS–ACC and DLS–ACC dip there. With $n\leq10$ this
should be read as a hint, not a result; a formal group epoch test is the
natural next step. There is no clean monotonic strengthening or weakening of
any connection across learning.

### 4.2 No directional asymmetry in the communication

The held-out lagged-CCA curves peak sharply at zero spatial lag and fall off
roughly symmetrically for every well-powered pair. The Information Flow Index
is close to zero at the group level in every epoch, with no epoch showing a
significant deviation from zero. The hypothesis that the lead–lag relationship
flips across learning (for example DMS leading early, DLS later) is **not
supported**: residual communication is spatially symmetric and shows no
detectable directionality. (A caveat: spatial lag maps only loosely onto
temporal lead/lag because running speed varies.)

### 4.3 Communication subspaces reorient modestly across learning

The dominant communication direction rotates between epochs. The principal
angle between the naive and expert dominant directions is $\approx1.3$–1.4 rad
for the striatal pairs, consistently above the within-epoch split-half noise
floor of $\approx1.0$–1.2 rad. For DMS–ACC and DLS–ACC the naive$\to$expert
angle exceeds the floor significantly (paired $t$-test across animals). So the
communication subspace **does reorient with learning** — but on a noisy
baseline: even within a single epoch the dominant direction is only loosely
constrained by ten trials, so the reorientation is modest relative to the
estimation noise.

Weight sparsity (Gini of the dominant-dimension weights) is $\approx0.6$–0.8
and increases mildly from naive to expert for the striatal pairs (for example
DMS–DLS $0.67\to0.76$). Tentatively, communication becomes carried by slightly
fewer units as the animal learns.

### 4.4 Membership is pair-specific and turns over with learning

Which units carry the communication? The member sets are **largely
pair-specific**: for each area, the Jaccard overlap of its member set between
the different pairs it participates in is $\approx0.22$–0.31, essentially at
the chance level of 0.25 for top-quartile sets. There is no shared "hub"
population that couples an area to all of its partners. Membership is also
**unstable across learning**: the Jaccard overlap between the naive and expert
member sets is $\approx0.22$ (again near chance). The units carrying a pair's
communication in the naive epoch are largely not those carrying it at expert.

### 4.5 Striatal–cingulate communication is direct

Partial CCA on the DMS/DLS/ACC triplet ($n=7$ mice with all three areas)
barely changes the canonical correlation: regressing the third striatal area
out of a pair leaves CC1 essentially unchanged (DMS–DLS$\,|\,$ACC
$0.30\to0.30$; DMS–ACC$\,|\,$DLS $0.22\to0.25$; DLS–ACC$\,|\,$DMS
$0.14\to0.12$, naive epoch). Each pair's residual coupling is therefore
**direct and pair-specific**, not an artefact of a shared third-area input —
consistent with the pair-specific membership in §4.4.

### 4.6 The conclusions are robust to fast-spiking cells

Re-running the whole pipeline with fast-spiking units retained gives held-out
CC1 values that correlate with the FS-excluded values at $r=0.89$ across
matched cells; the per-pair pattern and the significance counts are unchanged.
FS inclusion adds a little power (more units, a few more mice clear the
inclusion floor) but changes no conclusion.

### 4.7 Directionality and robustness checks (round 2)

Three further checks, requested after the first pass, all support the §4.2
conclusion. First, taking the statistical unit to be the **significant
subspace dimension** rather than the animal (the Gonzalez/Buzsáki convention):
across the striatal learner pairs there are 357 significant canonical
dimensions, and their IFI distribution is centred at +0.023 (one-sample
$t$-test $p=0.18$) — no net directionality. Subspace dimensionality is
bimodal (mean 4.76 significant dimensions per epoch; many epochs with 0–1, a
tail toward 20). Second, sweeping the lag-integration window from $\pm1$ to
$\pm5$ bins leaves IFI in the range 0.005–0.03 with every standard error
overlapping zero — the directionality null does not depend on the window.
Third, the full factorial over FS-cell exclusion and residual-vs-signal CCA:
signal CCA (which retains shared spatial tuning) gives modestly higher
canonical correlations than residual CCA (≈0.33 vs 0.25 for DMS–DLS at
expert) and a non-zero IFI in places — but that reflects shared position
coding, not communication, and is exactly why the residual definition is the
primary one. FS exclusion barely moves any number. The held-out vs in-sample
contrast remains the largest single effect (≈0.25 vs ≈0.55), underlining that
cross-validation is essential.

## 5. Caveats and limitations

The per-epoch sample budget is modest (~1000 nominal samples at 2.5 cm bins,
fewer after dropping missing bins; $k$ up to 30), and adjacent spatial bins
are autocorrelated so the effective sample count is lower still — canonical
correlations are modest and individual-animal estimates noisy; the held-out
cross-validation is essential and is what keeps the reported magnitudes
honest. (Finer 2.5 cm bins did not raise the canonical correlations relative
to 5 cm — the extra bins are more autocorrelated — but the conclusions are
unchanged between the two resolutions.) The V1 and CA1 arms are underpowered ($n=1$–4) and cannot support
group inference — the CA1 arm in particular rests on one or two animals. Only
the dominant canonical dimension is reliably estimable from ten-trial epochs.
The significance test is an in-sample permutation test (valid, but a
compute-driven choice over CV'd surrogates). Spatial lag is only a loose proxy
for temporal lead/lag. "Communication" here means trial-to-trial residual
co-fluctuation; shared spatial tuning is deliberately removed and is not
quantified by this analysis.

## 6. Figures

All in `cca/figures/`. Stage 1: `stage1_dms_acc_detail`,
`stage1_all_pairs_grid`. Stage 2: `stage2_cc1_significance_*`, `stage2_ifi_*`,
`stage2_lag_curves_*`, `stage2_fs_comparison`. Stage 3: `stage3_principal_angles`,
`stage3_gini`, `stage3_membership_overlap`, `partial_cca_{z0,z1}`. Factorial
and directionality: `stage2_factorial_{z0,z1}` (FS × mean-subtraction ×
holdout, without / with unit z-scoring), `stage2_ifi_per_pair_*` (IFI over
each pair's significant subspace dimensions, by epoch), `stage2_ifi_window_*`
(IFI vs lag-integration window, per pair).

## 7. Suggested next steps

A formal group-level test of the epoch profile of CC1 (a repeated-measures or
Friedman test per pair) would settle whether the DMS–DLS intermediate bump and
the DMS–ACC/DLS–ACC dip are real. More CA1/V1 co-recordings are the only way
to rescue the hippocampal and visual arms — no analysis choice can manufacture
that power. If directionality matters, a genuinely temporal (within-bin) lag
analysis would test lead/lag more directly than spatial lag. Finally, the
non-subtracted ("signal") CCA variant is wired in and would quantify how much
of the inter-areal structure is shared position coding versus communication.
