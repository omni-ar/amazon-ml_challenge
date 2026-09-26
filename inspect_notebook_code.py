import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

nb_path = r"c:\Users\arjit\Desktop\ml_challenge\colab_file.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [7, 8, 9, 10, 11, 13, 16]:
    if idx < len(nb['cells']):
        print(f"\n{'='*25} CELL {idx} {'='*25}")
        print("".join(nb['cells'][idx]['source']))
