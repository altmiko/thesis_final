# The Domain Validator — Design, Implementation, and Rationale

This document explains the rule-based domain validator used throughout the thesis to
distinguish *successful* adversarial perturbations from *plausible* ones. It is written
to be lifted more or less directly into the thesis report: it covers what the validator
is, how it is built, how every rule works at the code level, **why** each rule exists and
where its constraint comes from, and how the validator's output is consumed downstream to
produce the central `ASR_raw` vs `ASR_valid` finding.

Source files:

- `src/attack/validator.py` — the validator (rule definitions + result container).
- `src/preprocessing/feature_groups.py` — the 39-feature schema the rules are written against.
- `src/evaluation/validity_analysis.py` — inverse-transform bridge, `ASR_valid` computation, and impossible-traffic exhibits.

---

## 1. Purpose and Role in the Thesis

The core thesis claim is that the *attack success rate* reported by a standard adversarial
attack is misleading for tabular network-intrusion data, because gradient-based attacks
(FGSM, PGD, CW) freely move features into regions that **cannot correspond to any real
network flow**. A classifier can be "fooled" by a feature vector that no packet capture
could ever produce — a negative packet count, a TTL of 4,000, a flow simultaneously
flagged as both TCP and ICMP, or a variance that is not the square of its own standard
deviation.

The validator exists to make that distinction measurable. For each feature vector it
answers one question:

> *Does this sample satisfy all domain constraints expected of a genuine network-traffic
> feature vector?*

It returns both a per-rule breakdown of which constraints failed and a single overall
valid/invalid verdict per sample. Feeding adversarial examples through it lets us split
attack success into:

- **`ASR_raw`** — the classifier was fooled, validity ignored (the optimistic, standard number).
- **`ASR_valid`** — the classifier was fooled *and* the adversarial sample is still a
  domain-valid flow (the honest, defensible number).

The gap between these two is the empirical contribution of the thesis. The validator is
therefore not a peripheral utility; it is the instrument that operationalises the entire
argument.

---

## 2. Design Philosophy

Four deliberate design decisions shape the validator. Each is defensible and each is worth
stating explicitly in the report.

### 2.1 Rule-based, not learned

Validity is enforced with a fixed set of transparent, auditable rules rather than a learned
"is-this-real-traffic" model (e.g. a density estimator or one-class classifier). This is
intentional:

- **No circularity.** A learned validity model would itself be attackable and would share
  blind spots with the NIDS under study, contaminating the measurement. Hard rules cannot
  be adversarially flattered.
- **Interpretability.** Every rejection has a named cause (`R_pkts_positive`,
  `R_proto_tcp`, …). The thesis can point at *which* physical or protocol invariant an
  attack broke, not merely that "a model scored it low."
- **Soundness over completeness.** The rule set is intentionally conservative: it encodes
  only invariants that *must* hold for any real flow. It does not attempt to capture the
  full manifold of realistic traffic. Consequently a sample the validator accepts is not
  guaranteed realistic, but a sample it rejects is *guaranteed impossible*. This one-sided
  guarantee is exactly what is needed to report a *lower bound* on the over-statement of
  raw ASR. (See §11 for the formal implication.)

### 2.2 Operates in raw feature space only

Models are trained on scaled features (a fitted `StandardScaler`, persisted as
`scaler.pkl`), and the attacks perturb samples in that scaled space. Every domain
constraint, however, is a statement about *physical* quantities: bytes, packet counts,
protocol numbers, TTL hops. A "non-negative packet count" or "TTL ≤ 255" rule is
meaningless against z-scored values.

The validator therefore assumes its input `X` is already in **raw (inverse-transformed)
feature space**. Callers are responsible for inverting the scaler first; this is exactly
what `validity_analysis.inverse_transform_results()` does before validation (§8). Keeping
the validator purely raw-space makes the rules readable as plain network semantics and
keeps it decoupled from whichever scaler the preprocessing happened to fit.

### 2.3 Tolerant of float round-trip noise

`StandardScaler.inverse_transform` is `x * scale + mean` in float32. Round-tripping a value
that was exactly `0`, `1`, or `6` rarely returns exactly that integer — it returns
`6.0000001` or `-0.0000003`. A naïve `value < 0` or `protocol == 6` test would flag huge
numbers of perfectly valid samples purely as numerical artifacts.

Every rule is therefore written with an explicit tolerance band (§4) so that
inverse-transform noise never counts as a domain violation. The tolerances are *policy*,
not physics: they are chosen large enough to absorb float error and small enough that they
cannot launder a genuinely impossible value into a valid one.

### 2.4 Schema-agnostic via column gating

The validator never assumes a fixed column layout. Each rule is guarded by a `has(column)`
check and only fires if the columns it needs are present (§5). This lets the same code
validate the full 39-feature schema, an ablated subset, or a future schema revision without
edits, and it means a missing column silently disables only the rules that depend on it
rather than crashing the run.

---

## 3. Core API and Data Structures

The single public entry point is:

```python
validate_batch(X: np.ndarray, feature_names: List[str]) -> ValidationResult
```

**Inputs**

- `X` — 2-D NumPy array of shape `(n_samples, n_features)` in **raw** feature space.
- `feature_names` — ordered list of column names aligned to the columns of `X`
  (canonically `feature_groups.FEATURE_NAMES`).

**Output — `ValidationResult`** (a `@dataclass`):

| Member | Type | Meaning |
|---|---|---|
| `n_samples` | `int` | Number of rows validated. |
| `violations_per_rule` | `Dict[str, np.ndarray]` | Maps each rule name to a boolean mask of shape `(n_samples,)`; `True` = that rule was **violated** for that row. |
| `overall_valid` *(property)* | `np.ndarray[bool]` | `True` where **no** rule was violated for the row. |
| `validity_rate` *(property)* | `float` | Mean of `overall_valid` — the fraction of valid rows. |
| `per_rule_violation_rate()` | `Dict[str, float]` | Fraction of rows violating each rule. |
| `summary()` | `str` | Human-readable validity rate plus per-rule violation rates (non-zero only, sorted descending). |

The result object stores **only boolean masks**, not the data. This keeps it cheap to hold
for millions of rows and makes every downstream metric (overall validity, per-rule rates,
per-sample violation lists) a pure function of those masks. `overall_valid` is computed by
stacking all rule masks into an `(n_samples, n_rules)` array and taking
`~V.any(axis=1)` — i.e. a row is valid iff it triggers zero rules (see §7).

---

## 4. Constants and Tolerance Policy

Four module-level constants govern the validator (`src/attack/validator.py:7-10`):

```python
VALID_PROTOCOLS = {0, 1, 2, 6, 17, 47}
FLOAT_TOL       = 0.01
VAR_REL_TOL     = 0.05
VAR_ABS_TOL     = FLOAT_TOL   # == 0.01
```

| Constant | Value | Used by | Purpose |
|---|---|---|---|
| `VALID_PROTOCOLS` | `{0,1,2,6,17,47}` | G2 | Allowed IP protocol numbers for this dataset (§6.2). |
| `FLOAT_TOL` | `0.01` | G1–G5, G7, G8 | Universal slack for "integer-like", "non-negative", and range/ordering checks. |
| `VAR_REL_TOL` | `0.05` | G6 | Relative-error ceiling for the `Variance = Std²` identity. |
| `VAR_ABS_TOL` | `0.01` | G6 | Absolute-error floor so the relative test is not applied near zero. |

How the slack is applied in practice:

- **Integer-likeness** (protocol number, binary flags, packet count): a value is accepted
  as an integer if it lies within `FLOAT_TOL` of its nearest integer
  (`|x − round(x)| ≤ 0.01`).
- **Non-negativity**: a value violates only if it drops below `−FLOAT_TOL`, so `−0.003`
  is tolerated but `−0.5` is not.
- **Range / ordering checks**: bounds are widened by `±FLOAT_TOL` (e.g. TTL ≤ `255 + 0.01`,
  `Min ≤ Max + 0.01`).
- **Variance identity**: handled with a *combined* absolute-and-relative gate (§6.6), the
  only place where `FLOAT_TOL` alone is insufficient.

The choice of `0.01` is comfortably larger than float32 inverse-transform error (typically
`1e-5`–`1e-6`) yet far smaller than the granularity of any real violation (the smallest
meaningful integer step is `1`). It is a tunable policy knob: loosening it inflates
reported validity, tightening it risks counting numerical noise as violations. The reported
statistics are therefore stated relative to this fixed tolerance.

---

## 5. Validation Pipeline Internals

`validate_batch` is a single linear pass (`src/attack/validator.py:43-129`):

1. **Wrap in a DataFrame.** `X` is converted to a `pandas.DataFrame` with the supplied
   `feature_names`, so rules can address features by name rather than by positional index.
   This is what makes the rules readable (`df['Protocol Type']`) and order-independent.
2. **Gate every rule.** A nested helper `has(c) = c in df.columns` guards each rule. A rule
   contributes a mask to the result dictionary `V` only if its required columns exist
   (§2.4).
3. **Compute masks under stable names.** Each rule writes a boolean array into `V` under a
   fixed, documented key (e.g. `V['R_protocol_valid']`). The key naming convention is the
   contract the downstream report code relies on (§8, §9).
4. **Wrap and return.** `V` and `n` are packaged into a `ValidationResult`.

If no rules fire at all (e.g. an empty or fully unrecognised schema), `violations_per_rule`
is empty and `overall_valid` defaults to all-`True` — i.e. *absence of any applicable rule
is treated as "valid"*, not "invalid". This is a deliberate fail-open choice consistent
with the soundness stance in §2.1: the validator only ever asserts impossibility when it has
a concrete rule to point to.

All rule logic is fully vectorised over NumPy arrays — there is no per-row Python loop — so
the validator scales to the multi-million-row full-dataset validation (§9) and to large
adversarial batches without becoming a bottleneck.

---

## 6. The Rule Catalogue (G1–G8)

The rules are organised into eight groups. For each group below: the rule name(s), the
features it targets, the exact violation condition as coded, and — most importantly for the
report — the **provenance** of the constraint (where in network/protocol reality it comes
from, and why it is safe to enforce).

The feature partition the rules rely on is defined in `feature_groups.py`: 15 binary
protocol/service indicators (`BINARY_FEATURES`), 12 integer-valued counters/flags
(`INTEGER_FEATURES`), and the continuous statistical aggregates (`Min`, `Max`, `AVG`,
`Std`, `Variance`, `Tot sum`, `Tot size`, `IAT`, `Rate`, `Header_Length`). The rule groups
map onto exactly these partitions.

### G1 — Non-negativity (`R_nonneg_<feature>`)

**Targets.** Two families:
- Continuous / statistical fields: `Header_Length`, `Rate`, `Time_To_Live`, `Tot sum`,
  `Min`, `Max`, `AVG`, `Std`, `Tot size`, `IAT`, `Number`, `Variance`.
- Count / flag-count fields: `fin_flag_number`, `syn_flag_number`, `rst_flag_number`,
  `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`,
  `ack_count`, `syn_count`, `fin_count`, `rst_count`.

**Condition.** Violation iff `value < −FLOAT_TOL`.

**Provenance.** Every one of these quantities is a count, a size in bytes, a rate, a
duration-like inter-arrival time, or a non-negative dispersion statistic (a standard
deviation or variance is non-negative by definition). None can be physically negative. This
is the single most frequently broken rule under unconstrained gradient attacks, because the
optimiser has no reason to respect a sign it was never told about — which is precisely the
phenomenon the thesis highlights.

### G2 — Protocol validity (`R_protocol_valid`)

**Target.** `Protocol Type`.

**Condition.** With `p_round = round(p_raw)`, violation iff **either**
`|p_raw − p_round| > FLOAT_TOL` (not integer-like) **or** `int(p_round) ∉ VALID_PROTOCOLS`.

**Provenance.** `Protocol Type` records the IANA IP protocol number, which is a *discrete
code*, not a continuous quantity — a value of `9.5` is meaningless. The allowed set
`{0, 1, 2, 6, 17, 47}` is exactly the set of protocol numbers that occur in the CICIoT2023
traffic used here:

| Number | Protocol | Why it appears |
|---|---|---|
| `1` | ICMP | Ping sweeps, ICMP floods, ICMP fragmentation attacks. |
| `2` | IGMP | Multicast group management. |
| `6` | TCP | The bulk of flows: SYN/ACK/RST floods, web attacks, brute force. |
| `17` | UDP | UDP floods, DNS spoofing, Mirai UDP-plain. |
| `47` | GRE | Mirai GRE-Ethernet / GRE-IP floods (`MIRAI-GREETH_FLOOD`, `MIRAI-GREIP_FLOOD`). |
| `0` | — | Sentinel for link-layer frames with no IP protocol number (e.g. ARP, LLC). |

The presence of `47` (GRE) is a concrete example of why the set is dataset-specific rather
than "the usual TCP/UDP/ICMP": it is there *because* the dataset contains Mirai GRE flood
classes. The combined integer-likeness + membership test rejects both "fractional
protocol" artifacts and out-of-vocabulary protocol numbers an attacker might drift into.

### G3 — Binary indicators (`R_binary_<feature>`)

**Targets.** The 15 protocol/service indicator columns: `HTTP`, `HTTPS`, `DNS`, `Telnet`,
`SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IGMP`, `IPv`, `LLC`
(= `feature_groups.BINARY_FEATURES`).

**Condition.** With `b_round = round(b_raw)`, violation iff **either**
`|b_raw − b_round| > FLOAT_TOL` (fractional) **or** `int(b_round) ∉ {0, 1}`.

**Provenance.** These columns are one-hot-style presence indicators: a flow either used a
protocol/service or it did not. Valid values are exactly `0` and `1`; anything else (a
fractional `0.6`, or a `2`) is not a state the indicator can represent. Gradient attacks
routinely push these toward fractional or out-of-range values because, in scaled space,
they are just another continuous axis to exploit.

### G4 — Protocol-indicator consistency (`R_proto_tcp/udp/icmp/igmp`)

**Targets.** `Protocol Type` against the `TCP`, `UDP`, `ICMP`, `IGMP` indicators.

**Condition (one-directional implication, on rounded values).**
- `R_proto_tcp`: violation iff `TCP == 1` **and** `Protocol Type ≠ 6`.
- `R_proto_udp`: violation iff `UDP == 1` **and** `Protocol Type ≠ 17`.
- `R_proto_icmp`: violation iff `ICMP == 1` **and** `Protocol Type ≠ 1`.
- `R_proto_igmp`: violation iff `IGMP == 1` **and** `Protocol Type ≠ 2`.

**Provenance & the one-way design choice.** Physically, the protocol number and the
protocol indicator must agree. But the validator enforces the implication in *one direction
only*: "if the indicator is firmly set (= 1), the protocol number must match." It does **not**
enforce the converse ("if `Protocol Type == 6` then `TCP` must be 1").

The reason is the nature of the source columns. In this CICIoT2023 release the indicator
columns can carry noisy/soft values before rounding (they behave like aggregated
proportions over a flow window), so demanding the reverse implication would reject many
genuine flows whose indicator rounded to 0 despite a defined protocol number. The forward
implication is the robust, low-false-positive constraint: a flow that *asserts* it is TCP
while declaring a non-TCP protocol number is unambiguously inconsistent and impossible. This
keeps G4 sound (it never rejects a real flow) at the cost of not catching every conceivable
inconsistency — again the deliberate soundness-over-completeness stance of §2.1.

These four indicators are exactly the IMMUTABLE protocol features in
`feature_groups.IMMUTABLE_FEATURES` (`Protocol Type`, `TCP`, `UDP`, `ICMP`), reinforcing
that they are network-stack-determined and not attacker-chosen at will.

### G5 — Statistical ordering (`R_min_leq_max`, `R_avg_in_range`)

**Targets.** `Min`, `Max`, `AVG` (per-flow packet-size statistics).

**Condition.**
- `R_min_leq_max`: violation iff `Min > Max + FLOAT_TOL`.
- `R_avg_in_range`: violation iff `AVG < Min − FLOAT_TOL` **or** `AVG > Max + FLOAT_TOL`.

**Provenance.** `Min`, `AVG`, `Max` are the minimum, mean, and maximum of the same
underlying packet-size sample. By the definition of min/mean/max over any non-empty set,
`Min ≤ AVG ≤ Max` always holds. An adversarial example with `Min = 900, Max = 40` or
`AVG = 5000` while `Max = 1500` describes a statistically impossible flow, even though every
individual value might be "in range". This rule catches violations of the *relationship
between* features that per-feature bounds (G1, G7) cannot.

### G6 — Variance / standard-deviation consistency (`R_var_eq_std_sq`)

**Targets.** `Std`, `Variance`.

**Condition (corrected — combined absolute + relative gate).** Let `exp = Std²`,
`abs_err = |Variance − exp|`, and `rel_err = abs_err / (|exp| + 1e-8)`. Then:

```python
R_var_eq_std_sq = (abs_err > VAR_ABS_TOL) & (rel_err > VAR_REL_TOL)
```

i.e. a sample is flagged **only if both** the absolute error exceeds `0.01` **and** the
relative error exceeds `5%`.

**Why both terms (and why this differs from a naïve relative-only check).** `Variance` and
`Std` are not independent features: by definition `Variance = Std²`. This is an *algebraic
identity* the dataset carries redundantly, so it is a powerful validity probe — a free
constraint that real data always satisfies and attacks rarely preserve.

The subtlety is the tolerance near zero. A pure relative test `rel_err > 0.05` would be
hyper-sensitive when the true variance is `0`: a tiny inverse-transform wobble like
`Variance = 0.0003` against `Std² = 0` yields an enormous relative error and would be
falsely flagged. The combined gate fixes this: the absolute term `abs_err > 0.01` first
requires a *meaningful* discrepancy before the relative term is even consulted, so small
round-trip noise around zero is tolerated, while genuine identity violations away from zero
(where both terms trip) are still caught. (The earlier version of this document described
only the relative term; the implemented rule is the stricter-to-pass / safer-to-trust
conjunction above.)

### G7 — TTL range (`R_ttl_range`)

**Target.** `Time_To_Live`.

**Condition.** Violation iff `Time_To_Live < −FLOAT_TOL` **or** `Time_To_Live > 255 + FLOAT_TOL`.

**Provenance.** TTL is carried in a single 8-bit IP header field and is therefore bounded to
`[0, 255]` by the protocol itself — no real packet can carry a TTL of `−3` or `4000`. (Note
`Time_To_Live` is also listed in `QUASI_IMMUTABLE_FEATURES`, because it is dominated by the
OS/network stack and should not be freely perturbed; G7 enforces the hard byte-range part of
that constraint.)

### G8 — Packet count (`R_pkts_positive`, `R_pkts_integer`)

**Target.** `Number` (packet count of the flow).

**Condition.**
- `R_pkts_positive`: violation iff `Number < 1 − FLOAT_TOL`.
- `R_pkts_integer`: violation iff `|Number − round(Number)| > FLOAT_TOL`.

**Provenance.** A flow that exists in the dataset contains at least one packet, and a packet
count is a whole number — you cannot observe `2.7` packets, nor a flow of `0` packets. The
two-part rule enforces both the lower bound (≥ 1) and integrality, catching attacks that
push the count fractional or below one.

---

## 7. How Overall Validity Is Computed

Given the rule masks in `violations_per_rule`, the per-sample verdict is the logical NOR of
all rules:

```
overall_valid[i] = NOT ( any rule mask is True at row i )
```

implemented as `~np.stack(list(V.values()), axis=1).any(axis=1)`. **A single failed rule is
sufficient to mark a sample invalid** — validity is conjunctive across all applicable rules.
`validity_rate` is then simply `overall_valid.mean()`. This strictness is intentional: the
sample must be self-consistent on *every* checked invariant to count as a plausible flow.

---

## 8. Integration: `validity_analysis.py`

`src/evaluation/validity_analysis.py` is the bridge between scaled attack artifacts and the
raw-space validator, and it is where the headline metrics are computed.

**Inverse-transform.** `inverse_transform_results(npz_path, scaler_path)` loads an attack
`.npz` (containing `X_clean`, `X_adv`, `y_true`, `y_pred_clean`, `y_pred_adv`, …), applies
`scaler.inverse_transform` to both clean and adversarial matrices, and returns them as
`X_clean_raw` / `X_adv_raw`. This is the step that satisfies the validator's raw-space
contract (§2.2).

**Per-sample validation.** `validate_adversarial_examples(X_adv_raw, feature_names)` calls
`validate_batch` and repackages the result into: the overall `validity_rate`, per-rule
violation counts, a per-sample boolean `per_sample_valid`, the per-rule violation rates, and
a `violation_details` list naming exactly which rules each row broke.

**`ASR_valid` computation.** `compute_asr_valid(result, per_sample_valid)` defines the
thesis metrics precisely. Let `clean_correct = (y_pred_clean == y_true)` and
`successful_attack = clean_correct & (y_pred_adv != y_true)`. Then:

```
ASR_raw   = |successful_attack|                         / |clean_correct|
ASR_valid = |successful_attack & per_sample_valid|      / |clean_correct|
validity_among_successful = |successful_attack & valid| / |successful_attack|
```

Note both rates use the **same denominator** (`clean_correct`), so they are directly
comparable and `ASR_valid ≤ ASR_raw` always. The third quantity,
`validity_rate_among_successful`, is the fraction of *winning* adversarial examples that are
still domain-valid — the most direct expression of the gap.

**Impossible-traffic exhibits.** `generate_impossible_traffic_exhibit(...)` selects samples
that are *both* successful flips *and* invalid, and builds a feature-level table showing,
for each, the clean value, adversarial value, delta, and — via `_rules_to_features` and
`_describe_violation` — a plain-English statement of which invariant was broken (e.g.
"`Number` (packet count) must be ≥ 1; got −4.20", or "TCP indicator implies Protocol Type
should be 6"). These tables are the qualitative evidence behind the quantitative gap: they
let the report *show* an attack producing a flow with a negative packet count rather than
merely asserting it.

---

## 9. Downstream Consumers

**Full-dataset validation (`src/validate_full_dataset.py`).** Inverse-transforms the
train/val/test splits, runs `validate_batch` chunk-by-chunk, and aggregates split-level
validity, per-rule fail counts, per-class and per-category validity, plus a synthetic
corruption sanity test (deliberately corrupted rows *should* be flagged). Establishes that
the validator passes the *real* data at very high rates — a precondition for trusting it as
a measuring instrument — and writes results under `results/validation/`.

**Compact adversarial exhibits (`src/compact_exhibits.py`).** For each attack artifact,
inverse-transforms the adversarial rows, applies `validate_batch`, and uses `overall_valid`
plus the rule masks to drive the `Validity` and `ASR_valid` columns, per-sample violation
counts, and the violation tags shown in the exhibit files.

---

## 10. Practical Interpretation of the Key Metrics

- **`ASR_raw`** — attack success rate ignoring validity. The optimistic number an attacker
  would quote.
- **`Validity` (among successful)** — of the flips that fooled the model, the fraction still
  passing *all* validator rules.
- **`ASR_valid`** — attack success rate counting only valid successful flips; the honest
  number.

If `Validity` is near zero, almost every "successful" attack relied on an out-of-domain,
physically impossible perturbation — the model was fooled only by traffic that could never
occur on a real network. The size of `ASR_raw − ASR_valid` is the headline result.

---

## 11. Caveats and Scope

- **Soundness, not completeness.** Formally, the validator guarantees
  *invalid ⇒ impossible*, but **not** *valid ⇒ realistic*. It checks feature-level domain
  consistency, not full packet-level protocol semantics or joint distributional realism. A
  sample can pass every rule and still be unusual traffic. Consequently `ASR_valid` is an
  **upper bound** on the genuinely realistic attack success rate, and `ASR_raw − ASR_valid`
  is a **lower bound** on the over-statement — which is the conservative, defensible way to
  state the finding.
- **Dataset-specific protocol set.** `VALID_PROTOCOLS` is fixed to the protocols present in
  this workflow (including GRE for the Mirai classes). If the protocol coverage changes, the
  constant must be updated.
- **Tolerances are policy.** `FLOAT_TOL`, `VAR_REL_TOL`, and `VAR_ABS_TOL` are chosen to
  absorb float32 round-trip error without laundering real violations. They are tunable, but
  changing them changes every reported validity statistic, so they are held fixed across all
  experiments.
- **Raw-space precondition.** The validator is meaningless on scaled input; callers must
  inverse-transform first. This is enforced by convention (always go through
  `validity_analysis`), not by the validator itself.

---

## 12. Quick Example Usage

```python
from src.attack.validator import validate_batch
from src.preprocessing.feature_groups import FEATURE_NAMES

vr = validate_batch(X_raw, FEATURE_NAMES)      # X_raw must be inverse-transformed
print(vr.validity_rate)                        # overall fraction valid
print(vr.summary())                            # validity + non-zero per-rule rates
print(vr.per_rule_violation_rate())            # full per-rule breakdown
mask_valid = vr.overall_valid                  # boolean (n_samples,) for filtering
```

---

## 13. Rule-Name Reference

| Group | Rule name(s) | Constraint enforced |
|---|---|---|
| G1 | `R_nonneg_<feature>` | Counts, sizes, rates, dispersions ≥ 0. |
| G2 | `R_protocol_valid` | `Protocol Type` is integer-like and ∈ {0,1,2,6,17,47}. |
| G3 | `R_binary_<feature>` | Protocol/service indicators ∈ {0,1}. |
| G4 | `R_proto_tcp`, `R_proto_udp`, `R_proto_icmp`, `R_proto_igmp` | Set indicator ⇒ matching protocol number (one-way). |
| G5 | `R_min_leq_max`, `R_avg_in_range` | `Min ≤ AVG ≤ Max`. |
| G6 | `R_var_eq_std_sq` | `Variance = Std²` (combined absolute + relative tolerance). |
| G7 | `R_ttl_range` | `Time_To_Live ∈ [0, 255]`. |
| G8 | `R_pkts_positive`, `R_pkts_integer` | `Number ≥ 1` and integer-like. |
