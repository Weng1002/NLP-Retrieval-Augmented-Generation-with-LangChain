import json

notebook_path = '/mnt/sda/chihhung/NYCU_homework/114-1/NLP/HW4/main.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Iterate through cells to find and replace the token
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'hf_token = "hf_' in line:
                new_source.append('hf_token = "YOUR_HF_TOKEN_HERE"\\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Successfully removed secrets from main.ipynb")
