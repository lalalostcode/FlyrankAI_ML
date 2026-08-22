import json

nb_path = r"c:\Local D\Galeri Belajar\Project Code\FlyRank Internship\work\notebooks\w05_model.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Insert the advanced models cell after Cell 6
new_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# Cell 6b: Advanced Ensemble Methods (XGBoost & LightGBM)\n",
  "import xgboost as xgb\n",
  "import lightgbm as lgb\n",
  "\n",
  "print('Training Advanced Ensembles...')\n",
  "advanced_models = {\n",
  "    'XGBoost': xgb.XGBClassifier(\n",
  "        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),\n",
  "        max_depth=6,\n",
  "        n_estimators=200,\n",
  "        n_jobs=-1,\n",
  "        random_state=RANDOM_STATE,\n",
  "        eval_metric='logloss'\n",
  "    ),\n",
  "    'LightGBM': lgb.LGBMClassifier(\n",
  "        class_weight='balanced',\n",
  "        max_depth=6,\n",
  "        n_estimators=200,\n",
  "        n_jobs=-1,\n",
  "        random_state=RANDOM_STATE,\n",
  "        verbose=-1\n",
  "    ),\n",
  "}\n",
  "\n",
  "for name, model in advanced_models.items():\n",
  "    print(f'Training {name}...')\n",
  "    model.fit(X_train, y_train)\n",
  "    proba = model.predict_proba(X_test)[:, 1]\n",
  "    results = evaluate_model(y_test, proba, name)\n",
  "    \n",
  "    models[name] = model\n",
  "    client_holdout_results[name] = results\n",
  "    model_probas[name] = proba\n",
  "    print(f'  P@50={results[\"p_at_50\"]*100:.1f}%, ROC-AUC={results[\"roc_auc\"]:.4f}')\n",
  "\n",
  "print('\\nAdvanced models added to comparison pool.')\n"
 ]
}

idx_to_insert = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and len(cell["source"]) > 0 and "Cell 6:" in cell["source"][0]:
        idx_to_insert = i + 1
        break

if idx_to_insert != -1:
    nb["cells"].insert(idx_to_insert, new_cell)
    print("Inserted new cell.")
else:
    print("Could not find Cell 6.")

# Update the gap analysis cell (Cell 10)
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and len(cell["source"]) > 0 and "Cell 10:" in cell["source"][0]:
        new_source = []
        for line in cell["source"]:
            new_source.append(line)
            if "        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE" in line:
                # Add xgboost and lightgbm to the list
                new_source.append("    )),\n")
                new_source.append("    ('XGBoost', xgb.XGBClassifier(scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(), max_depth=6, n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, eval_metric='logloss')),\n")
                new_source.append("    ('LightGBM', lgb.LGBMClassifier(class_weight='balanced', max_depth=6, n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1\n")
        cell["source"] = new_source
        print("Updated Cell 10.")
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
