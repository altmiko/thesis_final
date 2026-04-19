# T1 - Feature Schema

| Name            | Type        | Description                                           | Domain_Range         |        Mean |         Std |   Min |     25% |       50% |       75% |              Max |
|:----------------|:------------|:------------------------------------------------------|:---------------------|------------:|------------:|------:|--------:|----------:|----------:|-----------------:|
| Header_Length   | float       | Total header length of all packets in the flow        | [0.00, 60.00]        |     15.0499 |     10.3224 |     0 |    7.92 |   19.88   |    20     |     60           |
| Protocol Type   | categorical | IP protocol number (6=TCP, 17=UDP, 1=ICMP, 2=IGMP)    | [0.00, 47.00]        |     11.7165 |     12.1506 |     0 |    6    |    6      |    17     |     47           |
| Time_To_Live    | float       | Time-to-live value (hop limit) of packets in the flow | [0.00, 255.00]       |     73.6326 |     27.8015 |     0 |   64    |   64      |    65.91  |    255           |
| Rate            | float       | Packet rate of the flow (packets per second)          | [0.00, 10485760.00]  |  18820.4    |  44116.6    |     0 | 1088.23 | 6511.28   | 25833.4   |      1.04858e+07 |
| fin_flag_number | integer     | Number of packets with FIN flag set                   | [0.00, 1.00]         |      0.0515 |      0.2079 |     0 |    0    |    0      |     0     |      1           |
| syn_flag_number | integer     | Number of packets with SYN flag set                   | [0.00, 1.00]         |      0.1615 |      0.3459 |     0 |    0    |    0      |     0.02  |      1           |
| rst_flag_number | integer     | Number of packets with RST flag set                   | [0.00, 1.00]         |      0.0673 |      0.2293 |     0 |    0    |    0      |     0     |      1           |
| psh_flag_number | integer     | Number of packets with PSH flag set                   | [0.00, 1.00]         |      0.0924 |      0.2272 |     0 |    0    |    0      |     0.02  |      1           |
| ack_flag_number | integer     | Number of packets with ACK flag set                   | [0.00, 1.00]         |      0.253  |      0.3733 |     0 |    0    |    0      |     0.5   |      1           |
| ece_flag_number | integer     | Number of packets with ECE flag set                   | [0.00, 1.00]         |      0.0001 |      0.005  |     0 |    0    |    0      |     0     |      1           |
| cwr_flag_number | integer     | Number of packets with CWR flag set                   | [0.00, 1.00]         |      0.0001 |      0.0036 |     0 |    0    |    0      |     0     |      1           |
| ack_count       | integer     | Number of packets with ACK flag set in the same flow  | [0.00, 100.00]       |     10.7846 |     24.7401 |     0 |    0    |    0      |     7     |    100           |
| syn_count       | integer     | Number of packets with SYN flag set in the same flow  | [0.00, 100.00]       |     14.2477 |     33.7888 |     0 |    0    |    0      |     1     |    100           |
| fin_count       | integer     | Number of packets with FIN flag set in the same flow  | [0.00, 100.00]       |      4.6855 |     20.6843 |     0 |    0    |    0      |     0     |    100           |
| rst_count       | integer     | Number of packets with RST flag set in the same flow  | [0.00, 100.00]       |      5.6444 |     21.5824 |     0 |    0    |    0      |     0     |    100           |
| HTTP            | binary      | Binary: flow uses HTTP protocol                       | [0.00, 1.00]         |      0.065  |      0.2246 |     0 |    0    |    0      |     0     |      1           |
| HTTPS           | binary      | Binary: flow uses HTTPS protocol                      | [0.00, 1.00]         |      0.1376 |      0.2992 |     0 |    0    |    0      |     0.02  |      1           |
| DNS             | binary      | Binary: flow uses DNS protocol                        | [0.00, 1.00]         |      0.0111 |      0.0484 |     0 |    0    |    0      |     0     |      1           |
| Telnet          | binary      | Binary: flow uses Telnet protocol                     | [0.00, 0.60]         |      0      |      0.002  |     0 |    0    |    0      |     0     |      0.6         |
| SMTP            | binary      | Binary: flow uses SMTP protocol                       | [0.00, 0.90]         |      0      |      0.0022 |     0 |    0    |    0      |     0     |      0.9         |
| SSH             | binary      | Binary: flow uses SSH protocol                        | [0.00, 1.00]         |      0.0011 |      0.021  |     0 |    0    |    0      |     0     |      1           |
| IRC             | binary      | Binary: flow uses IRC protocol                        | [0.00, 0.90]         |      0.0001 |      0.0026 |     0 |    0    |    0      |     0     |      0.9         |
| TCP             | binary      | Binary: transport-layer protocol is TCP               | [0.00, 1.00]         |      0.5487 |      0.4542 |     0 |    0    |    0.7    |     1     |      1           |
| UDP             | binary      | Binary: transport-layer protocol is UDP               | [0.00, 1.00]         |      0.2191 |      0.364  |     0 |    0    |    0      |     0.3   |      1           |
| DHCP            | binary      | Binary: flow uses DHCP protocol                       | [0.00, 0.80]         |      0.0008 |      0.0115 |     0 |    0    |    0      |     0     |      0.8         |
| ARP             | binary      | Binary: flow uses ARP protocol                        | [0.00, 1.00]         |      0.0112 |      0.0434 |     0 |    0    |    0      |     0     |      1           |
| ICMP            | binary      | Binary: transport-layer protocol is ICMP              | [0.00, 1.00]         |      0.0913 |      0.2774 |     0 |    0    |    0      |     0     |      1           |
| IGMP            | binary      | Binary: transport-layer protocol is IGMP              | [0.00, 0.40]         |      0.0001 |      0.003  |     0 |    0    |    0      |     0     |      0.4         |
| IPv             | binary      | Binary: flow uses IPv4/IPv6                           | [0.00, 1.00]         |      0.9888 |      0.0434 |     0 |    1    |    1      |     1     |      1           |
| LLC             | binary      | Binary: flow uses LLC sub-layer protocol              | [0.00, 1.00]         |      0.9888 |      0.0434 |     0 |    1    |    1      |     1     |      1           |
| Tot sum         | float       | Sum of packet sizes in the flow (bytes)               | [120.00, 289481.00]  |  24509      |  31815.9    |   120 | 6000    | 6000      | 54939     | 289481           |
| Min             | float       | Minimum packet size in the flow (bytes)               | [42.00, 4410.00]     |    118.558  |    190.687  |    42 |   60    |   60      |    66     |   4410           |
| Max             | float       | Maximum packet size in the flow (bytes)               | [46.00, 39162.00]    |    637.052  |    959.335  |    46 |   60    |  214      |  1074     |  39162           |
| AVG             | float       | Average packet size in the flow (bytes)               | [46.00, 7885.20]     |    338.195  |    399.201  |    46 |   60    |   82.8    |   578     |   7885.2         |
| Std             | float       | Standard deviation of packet sizes in the flow        | [0.00, 10625.82]     |    180.71   |    322.205  |     0 |    0    |    3.3    |   210.675 |  10625.8         |
| Tot size        | float       | Total size of all packets in the flow (bytes)         | [46.00, 7885.20]     |    338.195  |    399.201  |    46 |   60    |   82.8    |   578     |   7885.2         |
| IAT             | float       | Inter-arrival time between consecutive packets        | [0.00, 65724.01]     |      0.0396 |     38.8917 |     0 |    0    |    0.0002 |     0.001 |  65724           |
| Number          | integer     | Total packet count in the flow                        | [2.00, 100.00]       |     77.3026 |     39.0476 |     2 |   10    |  100      |   100     |    100           |
| Variance        | float       | Variance of packet sizes in the flow                  | [0.00, 112907971.56] | 136472      | 599648      |     0 |    0    |   10.89   | 44384     |      1.12908e+08 |