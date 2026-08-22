import json
import io
import sys
import pandas as pd
import numpy as np
import os

def execute_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    global_namespace = {}
    
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            print(f"Executing code cell {idx}...")
            
            # Capture stdout
            old_stdout = sys.stdout
            captured_stdout = io.StringIO()
            sys.stdout = captured_stdout
            
            try:
                exec(code, global_namespace)
                output_text = captured_stdout.getvalue()
                cell["execution_count"] = idx
                cell["outputs"] = [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": output_text.splitlines(keepends=True)
                    }
                ]
            except Exception as e:
                output_text = captured_stdout.getvalue()
                print(f"Error in cell {idx}: {e}")
                cell["execution_count"] = idx
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
            
            print(f"Cell {idx} completed successfully.")

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Notebook {nb_path} executed top-to-bottom and updated successfully.")

if __name__ == "__main__":
    execute_notebook(r"c:\Local D\Galeri Belajar\Project Code\FlyRank Internship\work\notebooks\w04_baseline_score.ipynb")
