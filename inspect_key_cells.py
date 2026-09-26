import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

nb_path = r"c:\Users\arjit\Desktop\ml_challenge\colab_file.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [6, 12, 14, 16, 17, 18, 19, 20, 22]:
    if idx < len(nb['cells']):
        print(f"\n{'='*25} CELL {idx} {'='*25}")
        print("".join(nb['cells'][idx]['source']))
        outputs = nb['cells'][idx].get('outputs', [])
        if outputs:
            print("--- CELL OUTPUTS ---")
            for out in outputs:
                if 'text' in out:
                    print("".join(out['text']))
                elif 'data' in out and 'text/plain' in out['data']:
                    print("".join(out['data']['text/plain']))
