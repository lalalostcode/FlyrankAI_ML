import json

nb_path = r"c:\Local D\Galeri Belajar\Project Code\FlyRank Internship\work\notebooks\w05_model.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and len(cell["source"]) > 0 and "Cell 2:" in cell["source"][0]:
        new_source = []
        for line in cell["source"]:
            new_source.append(line)
            if "X = pd.concat" in line:
                new_source.append("    # Clean feature names for XGBoost (cannot contain [, ], or <)\n")
                new_source.append("    import re\n")
                new_source.append("    X.columns = [re.sub(r'[\[\]<>]', '_', c) for c in X.columns]\n")
        cell["source"] = new_source
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
