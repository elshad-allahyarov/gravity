import pickle
import os

# 1. Load Q=1 names
q1_names = set()
with open('0-0-0-0-0-sparc-data.txt', 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) < 5: continue
        if parts[-2] == '1':
            q1_names.add(parts[0])

print(f"Identified {len(q1_names)} Q=1 galaxies.")

# 2. Load Master Table
pkl_path = 'external/gravity/active/scripts/sparc_master.pkl'
with open(pkl_path, 'rb') as f:
    master_table = pickle.load(f)

# 3. Filter
q1_table = {k: v for k, v in master_table.items() if k in q1_names}
print(f"Filtered master table has {len(q1_table)} entries.")

# 4. Save
out_path = 'external/gravity/active/scripts/sparc_q1.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(q1_table, f)

print(f"Saved Q=1 subset to {out_path}")

