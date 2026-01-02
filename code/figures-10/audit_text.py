import re

patterns = [
    r"0\.21", r"2\.75", r"2\.21", r"2\.43", r"126", r"175"
]

with open('0-0-0-gravity-submission-aaa.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("--- Audit Report ---")
for i, line in enumerate(lines):
    for pat in patterns:
        if re.search(pat, line):
            print(f"Line {i+1} [{pat}]: {line.strip()}")

