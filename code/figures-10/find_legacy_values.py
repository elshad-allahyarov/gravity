with open('0-0-0-gravity-submission-aaa.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if '0.21' in line:
        print(f"Line {i+1}: {line.strip()}")

