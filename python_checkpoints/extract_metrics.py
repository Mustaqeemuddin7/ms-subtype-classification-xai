"""Extract metrics from all executed notebooks for updated comparison."""
import json

notebooks = ['Logistic_Regression.ipynb', 'Random_Forest.ipynb', 'Extra_Trees.ipynb', 'XGBoost.ipynb', 'CatBoost.ipynb']

for nb_name in notebooks:
    print(f"\n{'='*60}")
    print(f"  {nb_name}")
    print(f"{'='*60}")
    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                text = ''
                if output.get('output_type') == 'stream' and 'text' in output:
                    text = ''.join(output['text'])
                if any(kw in text for kw in ['Test Set Performance', 'Cross-Validation Results', 'ROC-AUC (macro)']):
                    print(text[:400])
                    print('---')
