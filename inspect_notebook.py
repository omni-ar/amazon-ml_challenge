import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

nb_path = r"c:\Users\arjit\Desktop\ml_challenge\colab_file.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb.get('cells', []))}")

for i, cell in enumerate(nb.get('cells', [])):
    cell_type = cell.get('cell_type', '')
    source = "".join(cell.get('source', []))
    print(f"\n{'='*20} CELL {i} ({cell_type}) {'='*20}")
    print(source[:500])
    if len(source) > 500:
        print(f"... [truncated, total {len(source)} chars]")
