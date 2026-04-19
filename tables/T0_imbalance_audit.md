# T0 — Imbalance Audit (sorted by count, rarest first)

| class | category | count | pct_total | imbalance_ratio_vs_max | tier | thesis_strategy |
|---|---|---:|---:|---:|---|---|
| UPLOADING_ATTACK | Web | 1,196 | 0.0027% | 0.000174 | rare | keep full; class-weighted loss; VAE at CATEGORY level (not per-class); flag reconstruction quality |
| RECON-PINGSWEEP | Recon | 2,161 | 0.0048% | 0.000313 | rare | keep full; class-weighted loss; VAE at CATEGORY level (not per-class); flag reconstruction quality |
| BACKDOOR_MALWARE | Web | 3,078 | 0.0068% | 0.000447 | rare | keep full; class-weighted loss; VAE at CATEGORY level (not per-class); flag reconstruction quality |
| XSS | Web | 3,705 | 0.0082% | 0.000537 | rare | keep full; class-weighted loss; VAE at CATEGORY level (not per-class); flag reconstruction quality |
| SQLINJECTION | Web | 5,022 | 0.0112% | 0.000729 | rare | keep full; class-weighted loss; VAE at CATEGORY level (not per-class); flag reconstruction quality |
| COMMANDINJECTION | Web | 5,168 | 0.0115% | 0.000750 | rare | keep full; class-weighted loss; VAE at CATEGORY level (not per-class); flag reconstruction quality |
| BROWSERHIJACKING | Web | 5,630 | 0.0125% | 0.000817 | rare | keep full; class-weighted loss; VAE at CATEGORY level (not per-class); flag reconstruction quality |
| DICTIONARYBRUTEFORCE | BruteForce | 12,522 | 0.0278% | 0.001817 | minority | keep full; class-weighted loss; VAE trainable |
| DDOS-SLOWLORIS | DDoS | 22,400 | 0.0498% | 0.003250 | minority | keep full; class-weighted loss; VAE trainable |
| DDOS-HTTP_FLOOD | DDoS | 27,597 | 0.0613% | 0.004003 | minority | keep full; class-weighted loss; VAE trainable |
| DOS-HTTP_FLOOD | DoS | 68,799 | 0.1528% | 0.009981 | minority | keep full; class-weighted loss; VAE trainable |
| RECON-PORTSCAN | Recon | 78,730 | 0.1749% | 0.011421 | minority | keep full; class-weighted loss; VAE trainable |
| RECON-OSSCAN | Recon | 93,970 | 0.2087% | 0.013632 | minority | keep full; class-weighted loss; VAE trainable |
| RECON-HOSTDISCOVERY | Recon | 128,677 | 0.2858% | 0.018667 | medium | keep full; use class-weighted loss in baseline |
| DNS_SPOOFING | Spoofing | 171,468 | 0.3809% | 0.024875 | medium | keep full; use class-weighted loss in baseline |
| DDOS-ACK_FRAGMENTATION | DDoS | 272,793 | 0.6059% | 0.039574 | medium | keep full; use class-weighted loss in baseline |
| DDOS-UDP_FRAGMENTATION | DDoS | 274,909 | 0.6106% | 0.039881 | medium | keep full; use class-weighted loss in baseline |
| MITM-ARPSPOOFING | Spoofing | 294,469 | 0.6541% | 0.042718 | medium | keep full; use class-weighted loss in baseline |
| VULNERABILITYSCAN | Recon | 357,583 | 0.7943% | 0.051874 | medium | keep full; use class-weighted loss in baseline |
| DDOS-ICMP_FRAGMENTATION | DDoS | 433,157 | 0.9622% | 0.062838 | medium | keep full; use class-weighted loss in baseline |
| MIRAI-GREIP_FLOOD | Mirai | 719,655 | 1.5986% | 0.104400 | medium | keep full; use class-weighted loss in baseline |
| MIRAI-UDPPLAIN | Mirai | 852,695 | 1.8941% | 0.123700 | medium | keep full; use class-weighted loss in baseline |
| MIRAI-GREETH_FLOOD | Mirai | 949,381 | 2.1088% | 0.137726 | medium | keep full; use class-weighted loss in baseline |
| BENIGN | Benign | 1,051,373 | 2.3354% | 0.152522 | majority | cap at 200K for baseline; cap at 200K for VAE training |
| DOS-SYN_FLOOD | DoS | 1,942,176 | 4.3141% | 0.281750 | majority | cap at 200K for baseline; cap at 200K for VAE training |
| DOS-TCP_FLOOD | DoS | 2,558,256 | 5.6826% | 0.371124 | majority | cap at 200K for baseline; cap at 200K for VAE training |
| DOS-UDP_FLOOD | DoS | 3,177,323 | 7.0577% | 0.460932 | majority | cap at 200K for baseline; cap at 200K for VAE training |
| DDOS-SYNONYMOUSIP_FLOOD | DDoS | 3,445,659 | 7.6537% | 0.499859 | majority | cap at 200K for baseline; cap at 200K for VAE training |
| DDOS-RSTFINFLOOD | DDoS | 3,872,808 | 8.6026% | 0.561825 | majority | cap at 200K for baseline; cap at 200K for VAE training |
| DDOS-SYN_FLOOD | DDoS | 3,886,130 | 8.6322% | 0.563758 | majority | cap at 200K for baseline; cap at 200K for VAE training |
| DDOS-PSHACK_FLOOD | DDoS | 3,920,372 | 8.7082% | 0.568725 | majority | cap at 200K for baseline; cap at 200K for VAE training |
| DDOS-TCP_FLOOD | DDoS | 4,306,086 | 9.5650% | 0.624681 | majority | cap at 200K for baseline; cap at 200K for VAE training |
| DDOS-UDP_FLOOD | DDoS | 5,181,027 | 11.5085% | 0.751608 | majority | cap at 200K for baseline; cap at 200K for VAE training |
| DDOS-ICMP_FLOOD | DDoS | 6,893,259 | 15.3118% | 1.000000 | majority | cap at 200K for baseline; cap at 200K for VAE training |
