import pickle
import numpy as np

# Definitions
def get_galaxy_type(v_obs):
    if len(v_obs) == 0: return 'Unknown'
    v_max = np.nanmax(v_obs)
    if v_max < 80.0: return 'Dwarf'
    if v_max > 200.0: return 'Massive'
    return 'Spiral'

# Load Q flags
q_dict = {}
try:
    with open('0-0-0-0-0-sparc-data.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                try:
                    q_dict[parts[0]] = int(parts[-2])
                except ValueError:
                    pass
except FileNotFoundError:
    print("Q file not found")

# Load Data
with open('external/gravity/active/scripts/sparc_master.pkl', 'rb') as f:
    data = pickle.load(f)

# Counters
counts = {
    'Total': {'Dwarf': 0, 'Spiral': 0, 'Massive': 0},
    'Q1': {'Dwarf': 0, 'Spiral': 0, 'Massive': 0},
    'Q2': {'Dwarf': 0, 'Spiral': 0, 'Massive': 0},
    'Q3': {'Dwarf': 0, 'Spiral': 0, 'Massive': 0},
    'Unknown': {'Dwarf': 0, 'Spiral': 0, 'Massive': 0}
}

for name, g in data.items():
    df = g['data']
    v_obs = df['vobs'].values
    if len(v_obs) == 0: continue
    
    gtype = get_galaxy_type(v_obs)
    q = q_dict.get(name, 0)
    
    # Total
    counts['Total'][gtype] += 1
    
    # By Quality
    if q == 1: counts['Q1'][gtype] += 1
    elif q == 2: counts['Q2'][gtype] += 1
    elif q == 3: counts['Q3'][gtype] += 1
    else: counts['Unknown'][gtype] += 1

# Format Table
print("-" * 40)
print(f"{'':<10} | {'Dwarf':<8} | {'Spiral':<8} | {'Massive':<8}")
print("-" * 40)
row_labels = ['Total', 'Q1', 'Q2', 'Q3']
for label in row_labels:
    d = counts[label]['Dwarf']
    s = counts[label]['Spiral']
    m = counts[label]['Massive']
    print(f"{label:<10} | {d:<8} | {s:<8} | {m:<8}")
print("-" * 40)

