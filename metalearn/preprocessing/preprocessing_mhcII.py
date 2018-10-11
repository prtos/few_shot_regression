import os
import numpy as np
from glob import glob
#FOLDER = "../datasets/mhcII_similarity_reduced/"


def group_folds_to_dataset(folder):

    file_pattern = folder + "*_test_*.txt"
    all_files = glob(file_pattern)
    allele_names = [(fname.split('/')[-1]).split('_')[0] for fname in all_files]

    for allele in allele_names:
        dataset_path = folder + allele + ".txt"
        data = [np.loadtxt(fname, dtype=str, delimiter='\t') for fname in all_files if allele in fname]
        data = np.concatenate(data)
        data = data[:, [4, 6]]
        np.savetxt(dataset_path, data, delimiter='\t', fmt='%s\t%s')
        print(allele, len(data))

    for f in all_files:
        os.remove(f)

if __name__ == '__main__':
    folder = "../datasets/mhcII_DRB_all/"
    group_folds_to_dataset(folder)
    # file_pattern = FOLDER + "*_nooverlap_?.txt"
    # for f in glob(file_pattern):
    #     os.remove(f)