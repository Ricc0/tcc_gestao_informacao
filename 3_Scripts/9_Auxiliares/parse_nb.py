import json
import sys

def parse_notebook(filepath, out_f):
    out_f.write(f"=== {filepath} ===\n")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        out_f.write(f"Error reading {filepath}: {e}\n")
        return

    for i, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] == 'markdown':
            out_f.write(f"\n--- Markdown Cell {i} ---\n")
            out_f.write(''.join(cell.get('source', [])))
            out_f.write('\n')
        elif cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            text_outputs = []
            for o in outputs:
                if 'text' in o:
                    val = o['text']
                    if isinstance(val, list):
                        text_outputs.append(''.join(val))
                    else:
                        text_outputs.append(val)
                elif 'text/plain' in o.get('data', {}):
                    val = o['data']['text/plain']
                    if isinstance(val, list):
                        text_outputs.append(''.join(val))
                    else:
                        text_outputs.append(val)
            if text_outputs:
                out_f.write(f"\n--- Code Output Cell {i} ---\n")
                out_f.write('\n'.join(text_outputs))
                out_f.write('\n')

with open('nb_summary.txt', 'w', encoding='utf-8') as out_f:
    parse_notebook('TCC_google_drive/TCC_classificação.ipynb', out_f)
    parse_notebook('TCC_google_drive/TCC_SINTETICOS.ipynb', out_f)
