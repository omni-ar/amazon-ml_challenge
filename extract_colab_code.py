import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

nb_path = r"c:\Users\arjit\Desktop\ml_challenge\colab_file.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(r"c:\Users\arjit\Desktop\ml_challenge\colab_extracted_code.py", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            out.write(f"\n# {'='*30} CELL {i} {'='*30}\n")
            out.write("".join(cell.get('source', [])))
            out.write("\n")

print("Extracted all code cells to colab_extracted_code.py")
