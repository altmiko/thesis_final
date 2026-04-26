# Adversarial Validity One-Pager (Supervisor Brief)

## Thesis Claim in One Sentence
Unconstrained adversarial attacks on scaled tabular features can show very high raw attack success, but almost all successful examples are physically invalid in raw network feature space.

## Credibility Checks
- Clean inverse-transformed samples pass domain validation at 100.0% (sanity check on binary + PGD eps=0.30).
- Adversarial samples from the same run have 0.0% validity.
- Therefore, the validator is not over-rejecting clean data; it is specifically detecting impossible adversarial traffic.

## Key Result: Raw ASR vs Domain-Valid ASR

| Framing | Attack | eps | ASR_raw | Validity rate (adv) | ASR_valid | Gap (raw-valid) |
|---|---|---:|---:|---:|---:|---:|
| binary | PGD | 0.30 | 37.8% | 0.0% | 0.0% | 37.8% |
| 8class | PGD | 0.30 | 74.6% | 0.0% | 0.0% | 74.6% |
| 34class | PGD | 0.30 | 88.3% | 0.0% | 0.0% | 88.3% |

Interpretation: the apparent threat is strongly inflated if domain validity is ignored.

## Most Frequently Broken Constraints (All Attacks Combined)

| Rule | Violation rate |
|---|---:|
| Protocol Type allowlist | 85.6% |
| Variance approximately equals Std^2 | 84.8% |
| rst_flag_number non-negative | 61.3% |
| fin_flag_number non-negative | 58.6% |
| psh_flag_number non-negative | 58.2% |
| syn_flag_number non-negative | 52.6% |
| ece_flag_number non-negative | 50.1% |
| rst_count non-negative | 48.9% |
| TCP indicator implies Protocol Type=6 | 46.9% |
| fin_count non-negative | 42.3% |

## Impossible Traffic Exhibit (Concrete Examples)

### Example A (binary + PGD eps=0.30)
- Tot sum: 6000.0000 -> -8681.7002 (violates non-negativity)
- Max: 66.0000 -> -238.2000 (violates non-negativity and Min <= Max)
- Variance: 0.0000 -> -13294.6006 (violates non-negativity and Std/Variance consistency)

### Example B (8class + PGD eps=0.30)
- Rate: 4137.3330 -> -3288.2339 (violates non-negativity)
- Tot sum: 6000.0000 -> -5011.2754 (violates non-negativity)
- Max: 112.0000 -> -192.2000 (violates non-negativity and Min <= Max)

### Example C (34class + PGD eps=0.30)
- Protocol Type: 6.0000 -> 6.8250 (no longer in allowed set {0,1,2,6,17,47})
- Variance: 0.0000 -> -13294.6006 (violates non-negativity)
- Std: 0.0000 -> -63.1536 (violates non-negativity)

## Bottom Line for Supervisor Discussion
- Raw adversarial success alone overstates practical risk.
- Once realistic network constraints are enforced, effective adversarial success collapses.
- This supports the thesis argument that unconstrained tabular attacks can produce impossible traffic and inflated threat estimates.

## Artifacts
- Shock table: results/attacks/shock_table.csv
- Rule breakdown: results/attacks/violation_breakdown.csv
- Main figure: figures/asr_raw_vs_valid.png
- Slide exhibit pack: tables/impossible_traffic_slide_pack.csv
