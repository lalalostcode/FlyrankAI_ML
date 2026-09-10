# Scratch script to test w06_validation_audit.ipynb code cells
import json, sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../..')
with open('work/notebooks/w06_validation_audit.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Executing {len(code_cells)} code cells...")
all_code = []
for i, cell in enumerate(code_cells):
    src = cell['source']
    code = ''.join(src) if isinstance(src, list) else src
    all_code.append(f"# === CELL {i} ===\n{code}")
exec(compile('\n\n'.join(all_code), 'w06_validation_audit.ipynb', 'exec'))
print("\n=== ALL CELLS EXECUTED SUCCESSFULLY ===")
