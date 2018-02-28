import os
import pandas as pd
import numpy as np
from scipy import stats


BINDINGDB_FOLDER = "../datasets/bindingDB/"
FILENAME = os.path.join(BINDINGDB_FOLDER, "BindingDB_All.tsv")

LIGAND_SMILES_COLUMN = 'Ligand SMILES'
TARGET_NAME_COLUMN = 'Target Name Assigned by Curator or DataSource'
TARGET_SEQUENCE_COLUMN = 'BindingDB Target Chain  Sequence'  # The two spaces between "Chain" and "Sequence" are normal.
TARGET_SOURCE_COLUMN = "Target Source Organism According to Curator or DataSource"
BIOACTIVITIES_COLUMNS = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
cols_of_interest = [LIGAND_SMILES_COLUMN, TARGET_NAME_COLUMN, TARGET_SOURCE_COLUMN, TARGET_SEQUENCE_COLUMN] + BIOACTIVITIES_COLUMNS


# def get_latest_bindingDB_file():
#     download('https://www.bindingdb.org/bind/downloads/BindingDB_All_2018m0.tsv.zip')


def get_filename(target_name, target_source, bioactivity_col):
    prefix1 = bioactivity_col.split(' ')[0].lower()
    core = ''.join(e for e in target_name if e.isalnum()).lower()
    prefix2 = ''.join(e for e in target_source if e.isalnum()).lower() if not pd.isnull(target_source) else "unknownsource"
    filename = "{}_{}_{}.tsv".format(prefix1, prefix2, core)
    return os.path.join(BINDINGDB_FOLDER, filename)


def create_few_shot_dataset():
    data = pd.read_csv(FILENAME, delimiter='\t', usecols=cols_of_interest, skiprows=[], dtype=str)
    data = data.dropna(how='any', subset=[LIGAND_SMILES_COLUMN, TARGET_NAME_COLUMN])
    data_sizes = []
    for target in set(data[TARGET_NAME_COLUMN].values):
        target_data = data.loc[data[TARGET_NAME_COLUMN] == target]
        for bioactivty_column in BIOACTIVITIES_COLUMNS:
            target_task_data = target_data.dropna(how='any', subset=[bioactivty_column])
            sources =set(target_task_data[TARGET_SOURCE_COLUMN])
            for source in sources:
                target_task_src_data = target_task_data.loc[target_task_data[TARGET_SOURCE_COLUMN] == source]
                # suppress uncertain values
                if target_task_src_data.shape[0] > 0:
                    target_task_src_data = target_task_src_data[~target_task_src_data[bioactivty_column].str.contains(">")]
                if target_task_src_data.shape[0] > 0:
                    target_task_src_data = target_task_src_data[~target_task_src_data[bioactivty_column].str.contains("<")]
                if target_task_src_data.shape[0] > 0:
                    target_task_src_data = target_task_src_data[target_task_src_data[LIGAND_SMILES_COLUMN].str.len() <= 300]
                if target_task_src_data.shape[0] >= 20:
                    target_task_src_data = target_task_src_data[[LIGAND_SMILES_COLUMN, bioactivty_column]]
                    target_task_src_data.to_csv(path_or_buf=get_filename(target, source, bioactivty_column),
                                                sep='\t', header=False, index=False, mode='a')
                    data_sizes += [target_task_src_data.shape[0]]
                    print(target, source, target_task_src_data.shape[0])
    print(stats.describe(data_sizes))
    print("10-percent = {}, 25-percent = {}, 50-percent = {}, 75-percent = {}, 90-percent = {}".format(
        np.percentile(data_sizes, 10), np.percentile(data_sizes, 25), np.percentile(data_sizes, 50),
        np.percentile(data_sizes, 75), np.percentile(data_sizes, 90)
    ))
    print('sum = ', sum(data_sizes), 'n = ', len(data_sizes))
    print(data.size)


if __name__ == '__main__':
    create_few_shot_dataset()
    os.remove(FILENAME)