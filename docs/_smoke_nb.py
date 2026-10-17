"""Smoke-run a notebook: execute every code cell in order with matplotlib Agg.

Usage: python docs/_smoke_nb.py <path-to-ipynb>
"""
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

nb_path = sys.argv[1]
with io.open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

ns = {}
n_code = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    try:
        exec(compile(src, f"{os.path.basename(nb_path)}[{i}]", "exec"), ns)
        n_code += 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAIL {nb_path} cell [{i}]: {type(e).__name__}: {e}")
        sys.exit(1)

print(f"OK   {nb_path} ({n_code} code cells)")
