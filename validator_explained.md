# Validator Detailed Explanation

## Purpose
The validator in `src/validator.py` performs rule-based domain checks on traffic feature vectors after inverse scaling (raw feature space). It is used to answer one question for each sample:

- Does this sample satisfy all domain constraints expected for valid network traffic features?

It returns both per-rule failures and an overall valid/invalid decision per sample.

## Core API
The main entry point is:

- `validate_batch(X, feature_names) -> ValidationResult`

Inputs:

- `X`: 2D NumPy array of shape `(n_samples, n_features)` in raw feature space.
- `feature_names`: ordered feature-name list aligned with columns in `X`.

Output (`ValidationResult`):

- `n_samples`: number of rows validated.
- `violations_per_rule`: dictionary mapping each rule name to a boolean array of shape `(n_samples,)` where `True` means the rule is violated.
- `overall_valid` (property): boolean array where `True` means no rules were violated for that sample.
- `validity_rate` (property): mean of `overall_valid`.
- `per_rule_violation_rate()`: fraction of rows violating each rule.
- `summary()`: text summary of validity and non-zero rule violation rates.

## Constants and Tolerances
The validator uses two key constants:

- `VALID_PROTOCOLS = {0, 1, 2, 6, 17, 47}`
- `FLOAT_TOL = 0.01`

How tolerance is applied:

- Integer-like checks allow small float noise up to `0.01` from nearest integer.
- Non-negativity checks allow tiny negative noise down to `-0.01`.
- Range checks include a tolerance margin of `0.01`.
- Variance consistency uses relative error threshold `5%`.

This design intentionally avoids false alarms from inverse-transform float artifacts.

## Validation Pipeline Internals
Inside `validate_batch`:

1. The array is converted to a pandas DataFrame with provided feature names.
2. A helper `has(c)` gates each rule so checks only run if required columns exist.
3. Rule masks are computed and stored in `V` under stable rule names.
4. `ValidationResult` wraps `V`.

If no rules are generated (unlikely), all samples are considered valid.

## Rule Groups (G1 to G8)

### G1: Non-negativity rules
Rule names:

- `R_nonneg_<feature>`

Applied to:

- Continuous/statistical fields: `Header_Length`, `Rate`, `Time_To_Live`, `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `IAT`, `Number`, `Variance`
- Count/flag-count fields: `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`, `ack_count`, `syn_count`, `fin_count`, `rst_count`

Condition:

- Violation if value `< -FLOAT_TOL`.

Interpretation:

- Enforces physically meaningful non-negative quantities while tolerating tiny numerical drift.

### G2: Protocol validity rule
Rule name:

- `R_protocol_valid`

Inputs:

- `Protocol Type`

Logic:

- Let `p_raw` be protocol value.
- Let `p_round = round(p_raw)`.
- Violation if either:
  - `abs(p_raw - p_round) > FLOAT_TOL` (not integer-like), or
  - `int(p_round)` is not in `VALID_PROTOCOLS`.

Interpretation:

- Protocol must be integer-like and in allowed IANA protocol set used by this dataset workflow.

### G3: Binary indicator rules
Rule names:

- `R_binary_HTTP`, `R_binary_HTTPS`, `R_binary_DNS`, `R_binary_Telnet`, `R_binary_SMTP`, `R_binary_SSH`, `R_binary_IRC`, `R_binary_TCP`, `R_binary_UDP`, `R_binary_DHCP`, `R_binary_ARP`, `R_binary_ICMP`, `R_binary_IGMP`, `R_binary_IPv`, `R_binary_LLC`

Logic per feature `b`:

- `b_round = round(b_raw)`
- Violation if either:
  - `abs(b_raw - b_round) > FLOAT_TOL` (fractional/non-integer-like), or
  - `int(b_round)` is not in `{0, 1}`.

Interpretation:

- Protocol/service indicator columns must be binary, with tolerance for tiny float noise only.

### G4: Protocol-indicator consistency (implication direction)
Rule names:

- `R_proto_tcp`, `R_proto_udp`, `R_proto_icmp`, `R_proto_igmp`

Important design note:

- The validator enforces only one-way implication:
  - If an indicator is 1, protocol must match.
- It does not enforce the reverse direction.

Per-rule logic (using rounded values):

- `R_proto_tcp`: violation when `TCP == 1` and `Protocol Type != 6`
- `R_proto_udp`: violation when `UDP == 1` and `Protocol Type != 17`
- `R_proto_icmp`: violation when `ICMP == 1` and `Protocol Type != 1`
- `R_proto_igmp`: violation when `IGMP == 1` and `Protocol Type != 2`

Why one-way:

- This is robust to noisy/soft indicator columns in this CICIoT2023 processing context.

### G5: Statistical ordering rules
Rule names:

- `R_min_leq_max`
- `R_avg_in_range`

Logic:

- `R_min_leq_max`: violation if `Min > Max + FLOAT_TOL`
- `R_avg_in_range`: violation if `AVG < Min - FLOAT_TOL` or `AVG > Max + FLOAT_TOL`

Interpretation:

- Enforces expected ordering relation: `Min <= AVG <= Max` with tolerance.

### G6: Variance and standard deviation consistency
Rule name:

- `R_var_eq_std_sq`

Logic:

- Expected variance: `exp = Std^2`
- Relative error: `err = abs(Variance - exp) / (exp + 1e-8)`
- Violation if `err > 0.05`

Interpretation:

- Accepts up to 5% relative mismatch between `Variance` and `Std^2`.

### G7: TTL range
Rule name:

- `R_ttl_range`

Logic:

- Violation if `Time_To_Live < -FLOAT_TOL` or `Time_To_Live > 255 + FLOAT_TOL`

Interpretation:

- TTL is constrained to the valid byte-like range `[0, 255]` with small tolerance.

### G8: Packet count rules
Rule names:

- `R_pkts_positive`
- `R_pkts_integer`

Applied to:

- `Number`

Logic:

- `R_pkts_positive`: violation if `Number < 1 - FLOAT_TOL`
- `R_pkts_integer`: violation if `abs(Number - round(Number)) > FLOAT_TOL`

Interpretation:

- Packet count must be at least 1 and integer-like.

## How Overall Validity Is Computed
Given all rule masks in `violations_per_rule`:

- A sample is valid if none of the rule masks are `True` for that row.
- Formally: `overall_valid = NOT(any rule violated for row)`.

So one failed rule is enough to mark a sample invalid.

## How This Feeds Your Reports

### Full dataset validation (`src/validate_full_dataset.py`)
The script:

- inverse-transforms train/val/test splits,
- calls `validate_batch` chunk-by-chunk,
- aggregates:
  - split-level validity,
  - per-rule fail counts,
  - per-class/per-category validity,
  - synthetic corruption test pass/fail,
- writes outputs under `results/validation/`.

### Compact adversarial exhibits (`src/compact_exhibits.py`)
For each attack artifact:

- adversarial rows are inverse-transformed,
- `validate_batch` is applied on adversarial rows,
- `overall_valid` and rule masks drive:
  - `Validity` and `ASR_valid` metrics,
  - per-sample violation counts,
  - displayed violation tags in the exhibit file.

## Practical Interpretation of Key Metrics

- `ASR_raw`: attack success rate ignoring validity.
- `Validity`: among flipped samples, fraction still passing all validator rules.
- `ASR_valid`: attack success rate constrained to valid adversarial samples.

If `Validity` is near zero, most successful flips are out-of-domain perturbations.

## Caveats and Scope

- This validator checks feature-level domain consistency, not packet-level protocol semantics beyond encoded constraints.
- `VALID_PROTOCOLS` is intentionally restricted to the workflow's accepted set; if protocol coverage changes, update this constant.
- Tolerance choices (`FLOAT_TOL`, 5% var-std tolerance) are policy decisions and can be tuned, but changing them changes reported validity statistics.

## Quick Example Usage

```python
from validator import validate_batch
from feature_groups import FEATURE_NAMES

vr = validate_batch(X_raw, FEATURE_NAMES)
print(vr.validity_rate)
print(vr.per_rule_violation_rate())
mask_valid = vr.overall_valid
```

## Rule Name Reference

- Non-negativity: `R_nonneg_*`
- Protocol validity: `R_protocol_valid`
- Binary fields: `R_binary_*`
- Protocol-indicator consistency: `R_proto_tcp`, `R_proto_udp`, `R_proto_icmp`, `R_proto_igmp`
- Statistical ordering: `R_min_leq_max`, `R_avg_in_range`
- Variance consistency: `R_var_eq_std_sq`
- TTL range: `R_ttl_range`
- Packet count constraints: `R_pkts_positive`, `R_pkts_integer`
