import pandas as pd
import numpy as np
import json
import sys
import os

np.random.seed(42)
sys.path.insert(0, 'D:/thesis_final/src')
from feature_groups import FEATURE_NAMES, BINARY_FEATURES, INTEGER_FEATURES, CATEGORY_MAP

df = pd.read_parquet('D:/thesis_final/data/processed/raw_loaded.parquet')
print(f"Loaded: {df.shape}")

# ── T1: Feature schema table ─────────────────────────────────────────
DESCRIPTIONS = {
    'Header_Length': 'Total header length of all packets in the flow',
    'Protocol Type': 'IP protocol number (6=TCP, 17=UDP, 1=ICMP, 2=IGMP)',
    'Time_To_Live': 'Time-to-live value (hop limit) of packets in the flow',
    'Rate': 'Packet rate of the flow (packets per second)',
    'fin_flag_number': 'Number of packets with FIN flag set',
    'syn_flag_number': 'Number of packets with SYN flag set',
    'rst_flag_number': 'Number of packets with RST flag set',
    'psh_flag_number': 'Number of packets with PSH flag set',
    'ack_flag_number': 'Number of packets with ACK flag set',
    'ece_flag_number': 'Number of packets with ECE flag set',
    'cwr_flag_number': 'Number of packets with CWR flag set',
    'ack_count': 'Number of packets with ACK flag set in the same flow',
    'syn_count': 'Number of packets with SYN flag set in the same flow',
    'fin_count': 'Number of packets with FIN flag set in the same flow',
    'rst_count': 'Number of packets with RST flag set in the same flow',
    'HTTP': 'Binary: flow uses HTTP protocol',
    'HTTPS': 'Binary: flow uses HTTPS protocol',
    'DNS': 'Binary: flow uses DNS protocol',
    'Telnet': 'Binary: flow uses Telnet protocol',
    'SMTP': 'Binary: flow uses SMTP protocol',
    'SSH': 'Binary: flow uses SSH protocol',
    'IRC': 'Binary: flow uses IRC protocol',
    'TCP': 'Binary: transport-layer protocol is TCP',
    'UDP': 'Binary: transport-layer protocol is UDP',
    'DHCP': 'Binary: flow uses DHCP protocol',
    'ARP': 'Binary: flow uses ARP protocol',
    'ICMP': 'Binary: transport-layer protocol is ICMP',
    'IGMP': 'Binary: transport-layer protocol is IGMP',
    'IPv': 'Binary: flow uses IPv4/IPv6',
    'LLC': 'Binary: flow uses LLC sub-layer protocol',
    'Tot sum': 'Sum of packet sizes in the flow (bytes)',
    'Min': 'Minimum packet size in the flow (bytes)',
    'Max': 'Maximum packet size in the flow (bytes)',
    'AVG': 'Average packet size in the flow (bytes)',
    'Std': 'Standard deviation of packet sizes in the flow',
    'Tot size': 'Total size of all packets in the flow (bytes)',
    'IAT': 'Inter-arrival time between consecutive packets',
    'Number': 'Total packet count in the flow',
    'Variance': 'Variance of packet sizes in the flow',
}

t1_rows = []
for feat in FEATURE_NAMES:
    if feat in BINARY_FEATURES:
        ftype = 'binary'
    elif feat in INTEGER_FEATURES:
        ftype = 'integer'
    elif feat == 'Protocol Type':
        ftype = 'categorical'
    else:
        ftype = 'float'

    col = df[feat]
    stats = col.describe()
    t1_rows.append({
        'Name': feat,
        'Type': ftype,
        'Description': DESCRIPTIONS.get(feat, ''),
        'Domain_Range': f"[{col.min():.2f}, {col.max():.2f}]",
        'Mean': f"{col.mean():.4f}",
        'Std': f"{col.std():.4f}",
        'Min': f"{col.min():.4f}",
        '25%': f"{stats['25%']:.4f}",
        '50%': f"{stats['50%']:.4f}",
        '75%': f"{stats['75%']:.4f}",
        'Max': f"{col.max():.4f}",
    })

t1_df = pd.DataFrame(t1_rows)
t1_df.to_csv('D:/thesis_final/tables/T1_feature_schema.csv', index=False)
with open('D:/thesis_final/tables/T1_feature_schema.md', 'w') as f:
    f.write("# T1 - Feature Schema\n\n")
    f.write(t1_df.to_markdown(index=False))
print("T1 saved.")

# ── T2a/T2b: Class counts ────────────────────────────────────────────
vc = df['Label'].value_counts().sort_values(ascending=False)
total = len(df)

t2a_rows = []
for label, count in vc.items():
    t2a_rows.append({
        'class': label,
        'category': CATEGORY_MAP.get(label, 'UNKNOWN'),
        'count': count,
        'percentage': f"{count/total*100:.4f}%",
    })
t2a_df = pd.DataFrame(t2a_rows)
t2a_df.to_csv('D:/thesis_final/tables/T2a_class_counts.csv', index=False)
with open('D:/thesis_final/tables/T2a_class_counts.md', 'w') as f:
    f.write("# T2a - Per-Class Counts (sorted desc)\n\n")
    f.write(t2a_df.to_markdown(index=False))

df['category'] = df['Label'].map(CATEGORY_MAP)
cat_vc = df['category'].value_counts().sort_values(ascending=False)
t2b_rows = []
for cat, count in cat_vc.items():
    n_attacks = sum(1 for v in CATEGORY_MAP.values() if v == cat)
    t2b_rows.append({
        'category': cat,
        'count': count,
        'percentage': f"{count/total*100:.4f}%",
        'num_sub_attacks': n_attacks,
    })
t2b_df = pd.DataFrame(t2b_rows)
t2b_df.to_csv('D:/thesis_final/tables/T2b_category_counts.csv', index=False)
with open('D:/thesis_final/tables/T2b_category_counts.md', 'w') as f:
    f.write("# T2b - Per-Category Counts (sorted desc)\n\n")
    f.write(t2b_df.to_markdown(index=False))

with open('D:/thesis_final/data/processed/class_to_category.json', 'w') as f:
    json.dump(CATEGORY_MAP, f, indent=2)
print("T2a, T2b saved.")

# ── T3: Summary statistics ───────────────────────────────────────────
feat_df = df[FEATURE_NAMES]
desc = feat_df.describe().T
desc['skewness'] = feat_df.skew()
desc['kurtosis'] = feat_df.kurtosis()
desc['pct_zeros'] = (feat_df == 0).mean() * 100
desc['pct_unique'] = feat_df.nunique() / len(feat_df) * 100
desc.index.name = 'feature'
desc.to_csv('D:/thesis_final/tables/T3_summary_stats.csv')
with open('D:/thesis_final/tables/T3_summary_stats.md', 'w') as f:
    f.write("# T3 - Summary Statistics\n\n")
    f.write(desc.to_markdown())
print("T3 saved.")

# ── T4: Clean data validity audit ────────────────────────────────────
from validator import validate_batch

X = df[FEATURE_NAMES].values
result = validate_batch(X, FEATURE_NAMES)
print(f"\n{result.summary()}")

rates = result.per_rule_violation_rate()
t4_rows = []
for rule, rate in sorted(rates.items(), key=lambda x: -x[1]):
    n_viol = int(result.violations_per_rule[rule].sum())
    if rate > 0.01:
        interp = "WARNING: >1% violation rate - possible dataset quirk"
    elif rate > 0:
        interp = "Minor violations; likely float precision"
    else:
        interp = "No violations"
    t4_rows.append({
        'rule_name': rule,
        'violation_rate': f"{rate:.6f}",
        'n_violations': n_viol,
        'interpretation': interp,
    })
t4_df = pd.DataFrame(t4_rows)
t4_df.to_csv('D:/thesis_final/tables/T4_clean_validity.csv', index=False)
with open('D:/thesis_final/tables/T4_clean_validity.md', 'w') as f:
    f.write("# T4 - Clean Data Validity Audit\n\n")
    f.write(f"Overall validity rate: {result.validity_rate:.4%}\n\n")
    f.write(t4_df.to_markdown(index=False))
    high_viol = [(r, rates[r]) for r in rates if rates[r] > 0.01]
    if high_viol:
        f.write("\n\n## Known Dataset Quirks\n\n")
        for rule, rate in high_viol:
            f.write(f"- **{rule}**: {rate:.4%} violation rate. ")
            if 'var_eq_std_sq' in rule:
                f.write("Float precision in original feature extraction.\n")
            elif 'proto' in rule:
                f.write("Protocol/indicator mismatch in original data.\n")
            elif 'ttl' in rule:
                f.write("Time_To_Live values outside [0,255] range.\n")
            else:
                f.write("See sample violations in logs.\n")

validity_report = {
    'validity_rate': result.validity_rate,
    'n_samples': result.n_samples,
    'n_valid': int(result.overall_valid.sum()),
    'per_rule_violation_rates': {k: float(v) for k, v in rates.items()},
}
with open('D:/thesis_final/data/processed/clean_data_validity_report.json', 'w') as f:
    json.dump(validity_report, f, indent=2)
print("T4 saved.")

# ── T5: Sample rows ──────────────────────────────────────────────────
sample_classes = ['BENIGN', 'DDOS-ICMP_FLOOD', 'RECON-PORTSCAN', 'MIRAI-GREETH_FLOOD']
sample_rows_list = []
for cls in sample_classes:
    cls_df = df[df['Label'] == cls]
    sampled = cls_df.sample(n=5, random_state=42)
    sampled = sampled[FEATURE_NAMES].copy()
    sampled.insert(0, 'Label', cls)
    sample_rows_list.append(sampled)
t5_wide = pd.concat(sample_rows_list, ignore_index=True)
t5_wide.to_csv('D:/thesis_final/tables/T5_sample_rows_wide.csv', index=False)

t5_t = t5_wide.set_index('Label').T
with open('D:/thesis_final/tables/T5_sample_rows_transposed.md', 'w') as f:
    f.write("# T5 - Sample Rows (transposed: features x samples)\n\n")
    f.write(t5_t.to_markdown())
print("T5 saved.")

print("\nAll Phase 3 tables (T1-T5) complete.")
