import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List


VALID_PROTOCOLS = {0, 1, 2, 6, 17, 47}
FLOAT_TOL = 0.01


@dataclass
class ValidationResult:
    n_samples: int
    violations_per_rule: Dict[str, np.ndarray]

    @property
    def overall_valid(self) -> np.ndarray:
        if not self.violations_per_rule:
            return np.ones(self.n_samples, dtype=bool)
        V = np.stack(list(self.violations_per_rule.values()), axis=1)
        return ~V.any(axis=1)

    @property
    def validity_rate(self) -> float:
        return float(self.overall_valid.mean())

    def per_rule_violation_rate(self) -> Dict[str, float]:
        return {k: float(v.mean()) for k, v in self.violations_per_rule.items()}

    def summary(self) -> str:
        rates = self.per_rule_violation_rate()
        out = [f"Validity rate: {self.validity_rate:.4%} "
               f"({int(self.overall_valid.sum())}/{self.n_samples})"]
        out.append("Per-rule violation rates (non-zero only):")
        for k, r in sorted(rates.items(), key=lambda x: -x[1]):
            if r > 0:
                out.append(f"  {k}: {r:.4%}")
        return "\n".join(out)


def validate_batch(X: np.ndarray, feature_names: List[str]) -> ValidationResult:
    df = pd.DataFrame(X, columns=feature_names)
    n = len(df)
    V = {}

    def has(c): return c in df.columns

    # G1 — Non-negativity
    for c in ['Header_Length', 'Rate', 'Time_To_Live',
              'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT',
              'Number', 'Variance']:
        if has(c):
            V[f'R_nonneg_{c}'] = df[c].values < -FLOAT_TOL
    for c in ['fin_flag_number', 'syn_flag_number', 'rst_flag_number',
              'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
              'cwr_flag_number',
              'ack_count', 'syn_count', 'fin_count', 'rst_count']:
        if has(c):
            V[f'R_nonneg_{c}'] = df[c].values < -FLOAT_TOL

    # G2 — Protocol Type in valid IP protocol numbers
    if has('Protocol Type'):
        proto_raw = df['Protocol Type'].values
        proto_rounded = np.round(proto_raw)
        proto_int = proto_rounded.astype(int)
        non_integer = np.abs(proto_raw - proto_rounded) > FLOAT_TOL
        not_allowed = ~np.isin(proto_int, list(VALID_PROTOCOLS))
        V['R_protocol_valid'] = non_integer | not_allowed

    # G3 — Binary features in {0, 1}
    for c in ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
              'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC']:
        if has(c):
            raw = df[c].values
            rounded = np.round(raw)
            not_integer_like = np.abs(raw - rounded) > FLOAT_TOL
            not_binary = ~np.isin(rounded.astype(int), [0, 1])
            V[f'R_binary_{c}'] = not_integer_like | not_binary

    # G4 — Protocol-indicator consistency (indicator implies protocol)
    # Note: for this CICIoT2023 release, protocol-indicator columns may carry
    # noisy/soft values before rounding. We enforce the robust implication
    # direction only: if indicator=1, Protocol Type must match.
    if has('Protocol Type') and has('TCP'):
        p = np.round(df['Protocol Type'].values).astype(int)
        t = np.round(df['TCP'].values).astype(int)
        V['R_proto_tcp'] = (t == 1) & (p != 6)
    if has('Protocol Type') and has('UDP'):
        p = np.round(df['Protocol Type'].values).astype(int)
        u = np.round(df['UDP'].values).astype(int)
        V['R_proto_udp'] = (u == 1) & (p != 17)
    if has('Protocol Type') and has('ICMP'):
        p = np.round(df['Protocol Type'].values).astype(int)
        i = np.round(df['ICMP'].values).astype(int)
        V['R_proto_icmp'] = (i == 1) & (p != 1)
    if has('Protocol Type') and has('IGMP'):
        p = np.round(df['Protocol Type'].values).astype(int)
        g = np.round(df['IGMP'].values).astype(int)
        V['R_proto_igmp'] = (g == 1) & (p != 2)

    # G5 — Statistical ordering Min <= AVG <= Max
    if has('Min') and has('Max'):
        V['R_min_leq_max'] = df['Min'].values > (df['Max'].values + FLOAT_TOL)
    if has('Min') and has('AVG') and has('Max'):
        V['R_avg_in_range'] = (df['AVG'].values < (df['Min'].values - FLOAT_TOL)) | \
                              (df['AVG'].values > (df['Max'].values + FLOAT_TOL))

    # G6 — Variance = Std^2 (5% tolerance)
    if has('Std') and has('Variance'):
        exp = df['Std'].values ** 2
        err = np.abs(df['Variance'].values - exp) / (exp + 1e-8)
        V['R_var_eq_std_sq'] = err > 0.05

    # G7 — TTL / Time_To_Live range [0, 255]
    if has('Time_To_Live'):
        V['R_ttl_range'] = (df['Time_To_Live'].values < -FLOAT_TOL) | \
                           (df['Time_To_Live'].values > (255 + FLOAT_TOL))

    # G8 — Packet count positive integer
    if has('Number'):
        V['R_pkts_positive'] = df['Number'].values < (1 - FLOAT_TOL)
        V['R_pkts_integer'] = np.abs(df['Number'].values -
                                      np.round(df['Number'].values)) > FLOAT_TOL

    return ValidationResult(n_samples=n, violations_per_rule=V)
