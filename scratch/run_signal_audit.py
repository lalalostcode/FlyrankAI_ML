import json
import io
import sys
import os

# Fix Unicode encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Change to project root so relative paths in notebook cells work
os.chdir(r"c:\Local D\Galeri Belajar\Project Code\FlyRank Internship\work\notebooks")

import pandas as pd
import numpy as np

def execute_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    global_namespace = {"__name__": "__main__"}
    cell_count = 0
    
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            if not code.strip():
                continue
            cell_count += 1
            print(f"\n{'='*60}")
            print(f"Executing code cell {cell_count} (cell index {idx})...")
            print(f"{'='*60}")
            
            # Capture stdout
            old_stdout = sys.stdout
            captured_stdout = io.StringIO()
            sys.stdout = captured_stdout
            
            try:
                exec(code, global_namespace)
                output_text = captured_stdout.getvalue()
                cell["execution_count"] = cell_count
                cell["outputs"] = [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": output_text.splitlines(keepends=True)
                    }
                ]
            except Exception as e:
                output_text = captured_stdout.getvalue()
                sys.stdout = old_stdout
                print(f"Output so far: {output_text[:500]}")
                print(f"ERROR in cell {cell_count} (index {idx}): {e}")
                import traceback
                traceback.print_exc()
                cell["execution_count"] = cell_count
                cell["outputs"] = [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": output_text.splitlines(keepends=True) + [f"\nEXECUTION ERROR: {e}\n"]
                    }
                ]
                raise e
            finally:
                sys.stdout = old_stdout
            
            if output_text.strip():
                # Print first 500 chars of output
                preview = output_text[:500]
                print(preview)
                if len(output_text) > 500:
                    print(f"  ... ({len(output_text)} total chars)")
            print(f"Cell {cell_count} completed successfully.")

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Notebook executed successfully: {cell_count} code cells run.")
    print(f"{'='*60}")

if __name__ == "__main__":
    execute_notebook(r"c:\Local D\Galeri Belajar\Project Code\FlyRank Internship\work\notebooks\w04_signal_audit.ipynb")
