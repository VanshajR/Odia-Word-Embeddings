# save as scripts/flatten_monolingual.py
input_path = 'data/OdiEnCorp_1.0/monolingual.final'
output_path = 'data/raw/odia_literature.txt'

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        for sent in line.strip().split('\t'):
            if sent.strip():
                outfile.write(sent.strip() + '\n')