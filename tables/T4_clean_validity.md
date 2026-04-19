# T4 - Clean Data Validity Audit

Overall validity rate: 81.3724%

| rule_name                |   violation_rate |   n_violations | interpretation                                       |
|:-------------------------|-----------------:|---------------:|:-----------------------------------------------------|
| R_protocol_valid         |         0.08948  |         396389 | WARNING: >1% violation rate - possible dataset quirk |
| R_proto_tcp              |         0.050967 |         225779 | WARNING: >1% violation rate - possible dataset quirk |
| R_proto_udp              |         0.045462 |         201393 | WARNING: >1% violation rate - possible dataset quirk |
| R_proto_icmp             |         0.000368 |           1629 | Minor violations; likely float precision             |
| R_nonneg_Header_Length   |         0        |              0 | No violations                                        |
| R_nonneg_Rate            |         0        |              0 | No violations                                        |
| R_nonneg_Time_To_Live    |         0        |              0 | No violations                                        |
| R_nonneg_Tot sum         |         0        |              0 | No violations                                        |
| R_nonneg_Min             |         0        |              0 | No violations                                        |
| R_nonneg_Max             |         0        |              0 | No violations                                        |
| R_nonneg_AVG             |         0        |              0 | No violations                                        |
| R_nonneg_Std             |         0        |              0 | No violations                                        |
| R_nonneg_Tot size        |         0        |              0 | No violations                                        |
| R_nonneg_IAT             |         0        |              0 | No violations                                        |
| R_nonneg_Number          |         0        |              0 | No violations                                        |
| R_nonneg_Variance        |         0        |              0 | No violations                                        |
| R_nonneg_fin_flag_number |         0        |              0 | No violations                                        |
| R_nonneg_syn_flag_number |         0        |              0 | No violations                                        |
| R_nonneg_rst_flag_number |         0        |              0 | No violations                                        |
| R_nonneg_psh_flag_number |         0        |              0 | No violations                                        |
| R_nonneg_ack_flag_number |         0        |              0 | No violations                                        |
| R_nonneg_ece_flag_number |         0        |              0 | No violations                                        |
| R_nonneg_cwr_flag_number |         0        |              0 | No violations                                        |
| R_nonneg_ack_count       |         0        |              0 | No violations                                        |
| R_nonneg_syn_count       |         0        |              0 | No violations                                        |
| R_nonneg_fin_count       |         0        |              0 | No violations                                        |
| R_nonneg_rst_count       |         0        |              0 | No violations                                        |
| R_binary_HTTP            |         0        |              0 | No violations                                        |
| R_binary_HTTPS           |         0        |              0 | No violations                                        |
| R_binary_DNS             |         0        |              0 | No violations                                        |
| R_binary_Telnet          |         0        |              0 | No violations                                        |
| R_binary_SMTP            |         0        |              0 | No violations                                        |
| R_binary_SSH             |         0        |              0 | No violations                                        |
| R_binary_IRC             |         0        |              0 | No violations                                        |
| R_binary_TCP             |         0        |              0 | No violations                                        |
| R_binary_UDP             |         0        |              0 | No violations                                        |
| R_binary_DHCP            |         0        |              0 | No violations                                        |
| R_binary_ARP             |         0        |              0 | No violations                                        |
| R_binary_ICMP            |         0        |              0 | No violations                                        |
| R_binary_IGMP            |         0        |              0 | No violations                                        |
| R_binary_IPv             |         0        |              0 | No violations                                        |
| R_binary_LLC             |         0        |              0 | No violations                                        |
| R_min_leq_max            |         0        |              0 | No violations                                        |
| R_avg_in_range           |         0        |              0 | No violations                                        |
| R_var_eq_std_sq          |         0        |              0 | No violations                                        |
| R_ttl_range              |         0        |              0 | No violations                                        |
| R_pkts_positive          |         0        |              0 | No violations                                        |
| R_pkts_integer           |         0        |              0 | No violations                                        |

## Known Dataset Quirks

- **R_protocol_valid**: 8.9480% violation rate. Protocol/indicator mismatch in original data.
- **R_proto_tcp**: 5.0967% violation rate. Protocol/indicator mismatch in original data.
- **R_proto_udp**: 4.5462% violation rate. Protocol/indicator mismatch in original data.
