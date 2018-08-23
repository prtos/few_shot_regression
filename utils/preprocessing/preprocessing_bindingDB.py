import os
import pandas as pd
from rdkit import Chem
import numpy as np
from scipy import stats
from joblib import Parallel, delayed


BINDINGDB_FOLDER = "../../datasets/bindingDB/"

FILENAME = os.path.join("../../datasets/originals/", "BindingDB_All.tsv")

LIGAND_SMILES_COLUMN = 'Ligand SMILES'
TARGET_NAME_COLUMN = 'Target Name Assigned by Curator or DataSource'
TARGET_SEQUENCE_COLUMN = 'BindingDB Target Chain  Sequence'  # The two spaces between "Chain" and "Sequence" are normal.
TARGET_SOURCE_COLUMN = "Target Source Organism According to Curator or DataSource"

BIOACTIVITIES_COLUMNS = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
cols_of_interest = [LIGAND_SMILES_COLUMN, TARGET_NAME_COLUMN, TARGET_SOURCE_COLUMN, TARGET_SEQUENCE_COLUMN] + BIOACTIVITIES_COLUMNS


# def get_latest_bindingDB_file():
#     download('https://www.bindingdb.org/bind/downloads/BindingDB_All_2018m0.tsv.zip')
#     and put it at the FILENAME location


def get_filename(target_name,  bioactivity_col):
    prefix1 = bioactivity_col.split(' ')[0].lower()
    core = ''.join(e for e in target_name if e.isalnum()).lower()
    filename = "{}_{}".format(prefix1, core)
    current_files = os.listdir(BINDINGDB_FOLDER)
    n = sum([f.startswith(filename) for f in current_files])
    if n > 0:
        filename += str(n)
    return os.path.join(BINDINGDB_FOLDER, filename)+'.tsv'


def is_convertible(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def create_few_shot_dataset():
    data = pd.read_csv(FILENAME, delimiter='\t', usecols=cols_of_interest, skiprows=[], dtype=str)
    data = data.dropna(how='any', subset=[LIGAND_SMILES_COLUMN, TARGET_NAME_COLUMN, TARGET_SEQUENCE_COLUMN])
    for bioactivty_column in BIOACTIVITIES_COLUMNS:
        data[bioactivty_column] = data[bioactivty_column].str.replace(">", '')
        data[bioactivty_column] = data[bioactivty_column].str.replace("<", '')
        data[bioactivty_column] = data[bioactivty_column].astype('float')
    data = data[data[LIGAND_SMILES_COLUMN].str.len() <= 300]
    print(data.shape)
    filter_unconvertible = Parallel(n_jobs=-1)(delayed(is_convertible)(el) for el in data[LIGAND_SMILES_COLUMN])
    # filter_unconvertible = [Chem.MolFromSmiles(el) is not None for el in data[LIGAND_SMILES_COLUMN]]
    data = data.loc[filter_unconvertible]
    print(data.shape)

    data_sizes = []
    for target in set(data[TARGET_NAME_COLUMN].values):
        target_data = data.loc[data[TARGET_NAME_COLUMN] == target]
        for bioactivty_column in BIOACTIVITIES_COLUMNS:
            target_task_data = target_data.dropna(how='any', subset=[bioactivty_column])
            target_sequences =set(target_task_data[TARGET_SEQUENCE_COLUMN])
            for sequence in target_sequences:
                target_task_src_data = target_task_data.loc[target_task_data[TARGET_SEQUENCE_COLUMN] == sequence]
                if target_task_src_data.shape[0] >= 20 and target_task_src_data[bioactivty_column].std() > 0:
                    fname = get_filename(target, bioactivty_column)
                    with open(fname, 'w') as metaf:
                        metaf.write(sequence+'\n')
                        metaf.write(bioactivty_column.split(' ')[0]+'\n')
                    target_task_src_data[[LIGAND_SMILES_COLUMN, bioactivty_column]].to_csv(path_or_buf=fname,
                                                sep='\t', header=False, index=False, mode='a')
                    data_sizes += [target_task_src_data.shape[0]]
                    print(target, bioactivty_column, target_task_src_data.shape[0])
    print(stats.describe(data_sizes))
    print("10-percent = {}, 25-percent = {}, 50-percent = {}, 75-percent = {}, 90-percent = {}".format(
        np.percentile(data_sizes, 10), np.percentile(data_sizes, 25), np.percentile(data_sizes, 50),
        np.percentile(data_sizes, 75), np.percentile(data_sizes, 90)
    ))
    print('sum = ', sum(data_sizes), 'n = ', len(data_sizes))
    print(data.size)


if __name__ == '__main__':
    create_few_shot_dataset()