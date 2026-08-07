"""
CICIoT2023 feature definitions — adapted to the actual 39-feature schema
found in ciciot2023_base.csv.

Schema variant: Modified Schema A (paper-derived).
  - Has Schema A exclusives: ack_count, syn_count, fin_count, rst_count, Number
  - Missing vs full Schema A: flow_duration, Duration, Srate, Drate,
    urg_count, Magnitude, Radius, Covariance, Weight
  - Extra vs Schema A: Time_To_Live, IGMP
  - Label column: "Label" (capital L), values ALL UPPERCASE
"""

import numpy as np

LABEL_COLUMN = 'Label'

# ── Feature name list (39 features, order matches CSV minus Label) ──────────

FEATURE_NAMES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Variance',
]

assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES), "Duplicate feature names"
assert len(FEATURE_NAMES) == 39

# Expected dataset columns (features + label) in raw_loaded parquet/csv.
EXPECTED_COLUMNS = FEATURE_NAMES + [LABEL_COLUMN]

# ── Immutable: defined by the network stack; attacker cannot change ─────────

IMMUTABLE_FEATURES = ['Protocol Type', 'TCP', 'UDP', 'ICMP']

# ── Quasi-immutable: application/protocol-layer indicators ──────────────────

QUASI_IMMUTABLE_FEATURES = [
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'DHCP', 'ARP', 'IPv', 'LLC', 'IGMP',
    # TTL is OS/network-stack dominated and should not be freely perturbed.
    'Time_To_Live',
]

# ── Binary features (protocol indicators, values in {0, 1}) ────────────────

BINARY_FEATURES = [
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC',
]

# ── Integer-valued features ─────────────────────────────────────────────────

INTEGER_FEATURES = [
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'Number',
]

# ── Mutable features (attacker can influence via packet crafting) ───────────

BASE_MUTABLE = [
    'Rate', 'Header_Length', 'Variance',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT',
]

MUTABLE_FEATURES = BASE_MUTABLE + [
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'Number',
]

# Explicit full-perturbation override for high-IQR, attacker-controllable
# continuous/aggregate features used in baseline attacks.
FULL_PERTURBABLE_OVERRIDE_FEATURES = [
    'Header_Length',
    'Rate',
    'Tot sum',
    'Min',
    'Max',
    'AVG',
    'Std',
    'Tot size',
    'IAT',
    'Number',
    'Variance',
]

# ── Near-zero-IQR governance for perturbation policy ───────────────────────

NEAR_ZERO_IQR_THRESHOLD = 1e-6
RARE_SIGNAL_NONZERO_THRESHOLD = 1e-3
NEAR_ZERO_FREEZE_POLICY = {
    'constant': 'auto_freeze',
    'rare_signal': 'allow',
    'concentrated': 'manual',
}

# Explicit manual decisions for near-zero-IQR concentrated features.
# Allowed values: 'allow_mutable' | 'force_freeze'
MANUAL_CONCENTRATED_DECISIONS = {
    'fin_flag_number': 'allow_mutable',
    'syn_flag_number': 'allow_mutable',
    'rst_flag_number': 'allow_mutable',
    'psh_flag_number': 'allow_mutable',
    'ack_flag_number': 'allow_mutable',
    'fin_count': 'allow_mutable',
    'rst_count': 'allow_mutable',
    # Min (min packet length) and Number (packet count) fell below the near-zero
    # IQR threshold once preprocessing moved to the FULL labelled parquet, whose
    # DDoS-dominated (72.65%) distribution concentrates both around a single
    # value (Q25==Q75) even though they genuinely vary (nunique 1360 / 99, always
    # non-zero). They are attacker-influenceable aggregates already listed in
    # FULL_PERTURBABLE_OVERRIDE_FEATURES, so they stay mutable — consistent with
    # every other concentrated mutable feature above.
    'Min': 'allow_mutable',
    'Number': 'allow_mutable',
}

# ── Category mapping (uppercase labels as found in the CSV) ─────────────────

CATEGORY_MAP = {
    # DDoS (12)
    'DDOS-ICMP_FLOOD': 'DDoS', 'DDOS-UDP_FLOOD': 'DDoS',
    'DDOS-TCP_FLOOD': 'DDoS', 'DDOS-PSHACK_FLOOD': 'DDoS',
    'DDOS-SYN_FLOOD': 'DDoS', 'DDOS-RSTFINFLOOD': 'DDoS',
    'DDOS-SYNONYMOUSIP_FLOOD': 'DDoS', 'DDOS-UDP_FRAGMENTATION': 'DDoS',
    'DDOS-ACK_FRAGMENTATION': 'DDoS', 'DDOS-ICMP_FRAGMENTATION': 'DDoS',
    'DDOS-HTTP_FLOOD': 'DDoS', 'DDOS-SLOWLORIS': 'DDoS',
    # DoS (4)
    'DOS-UDP_FLOOD': 'DoS', 'DOS-TCP_FLOOD': 'DoS',
    'DOS-SYN_FLOOD': 'DoS', 'DOS-HTTP_FLOOD': 'DoS',
    # Mirai (3)
    'MIRAI-GREETH_FLOOD': 'Mirai', 'MIRAI-UDPPLAIN': 'Mirai',
    'MIRAI-GREIP_FLOOD': 'Mirai',
    # Benign
    'BENIGN': 'Benign',
    # Spoofing (2)
    'MITM-ARPSPOOFING': 'Spoofing', 'DNS_SPOOFING': 'Spoofing',
    # Recon (5)
    'RECON-PINGSWEEP': 'Recon', 'RECON-OSSCAN': 'Recon',
    'RECON-PORTSCAN': 'Recon', 'RECON-HOSTDISCOVERY': 'Recon',
    'VULNERABILITYSCAN': 'Recon',
    # Web (6)
    'BROWSERHIJACKING': 'Web', 'BACKDOOR_MALWARE': 'Web',
    'XSS': 'Web', 'SQLINJECTION': 'Web',
    'COMMANDINJECTION': 'Web', 'UPLOADING_ATTACK': 'Web',
    # BruteForce (1)
    'DICTIONARYBRUTEFORCE': 'BruteForce',
}

# ── Perturbation mask ───────────────────────────────────────────────────────

def get_feature_indices(feats):
    return [FEATURE_NAMES.index(f) for f in feats if f in FEATURE_NAMES]

PERTURBATION_MASK = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
for _i in get_feature_indices(MUTABLE_FEATURES):
    PERTURBATION_MASK[_i] = 1.0
