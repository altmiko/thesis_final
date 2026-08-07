import json, collections
m = json.load(open("data/processed/ciciot2023_labeled_full_manifest.json"))
per_folder = collections.defaultdict(list)
for path, n in m["per_file_rows"].items():
    per_folder[path.split("/")[0]].append((path, n))
for folder in ["DDoS-PSHACK_FLOOD", "DDoS-SYN_Flood", "Mirai-udpplain"]:
    for p, n in sorted(per_folder[folder]):
        print(f"{n:>9,}  {p}")
    print()