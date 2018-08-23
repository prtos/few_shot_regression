import os
import numpy as np
import pandas as pd
from Bio import SeqIO
SOURCE_FOLDER = "../../datasets/originals"
DEST_FOLDER = "../../datasets/mhc_all"

if not os.path.exists(DEST_FOLDER):
    os.makedirs(DEST_FOLDER, exist_ok=True)

# mhc_ligand_full was downloaded at http://www.iedb.org/assay/3352106


def load_fasta(input_file):
    fasta_sequences = SeqIO.parse(open(input_file), 'fasta')
    return {fasta.description.upper(): fasta.seq.tostring() for fasta in fasta_sequences}


def create_meta_dataset():
    sequences = load_fasta(os.path.join(SOURCE_FOLDER, 'mhc_prot.fasta'))
    class1 = pd.read_csv(os.path.join(SOURCE_FOLDER, 'class_I.txt'), delim_whitespace=True)
    class2 = pd.read_csv(os.path.join(SOURCE_FOLDER, 'class_II.txt'), delim_whitespace=True)
    class1 = class1[class1.inequality == '=']
    class2 = class2[class2.inequality == '=']

    for data in [class1, class2]:
        for mhc_name, mhc_data in data.groupby('mhc'):
            mhc_name = mhc_name.replace('/', '_').upper()
            if mhc_data.size > 20:
                specie, allele = mhc_name.split('-')[0], '-'.join(mhc_name.split('-')[1:])
                seq_keys = [el for el in sequences.keys() if (allele in el) and (specie in el)]
                print(mhc_name, seq_keys)
                mhc_data = mhc_data[['sequence', 'meas']]
                # mhc_data.to_csv(os.path.join(DEST_FOLDER, mhc_name), sep='\t', index=False, header=False)


if __name__ == '__main__':
    from collections import Counter
    data = pd.read_csv(os.path.join(SOURCE_FOLDER, 'mhc_ligand_full.csv'), skiprows=1,
                       usecols=['Assay Group', 'Units', 'Quantitative measurement', 'Allele Name', 'MHC allele class'])
    data = data.dropna(subset=['Quantitative measurement'])
    print(data.shape)
    for col in data.columns:
        temp = Counter(data[col])
        temp = [el for el, v in temp.items() if v >100]
        print(col, len(temp))
    # create_meta_dataset()