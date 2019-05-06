import numpy as np
import pandas as pd
from joblib import Parallel, delayed
# from ivbase.utils.datasets.pubchem import PubChem

# pubchem = PubChem('/home/prtos/workspace/code/invivoai/data_repository/')


# def rewrite_with_smiles(sourcename, destname):
#     data = pd.read_csv(sourcename, header=None, skiprows=1).values
#     x, y = data[:, 0].astype('int64'), data[:, 1]
#     res = Parallel(n_jobs=1, verbose=1)(delayed(get_smiles)(cid) for cid in x)
#     res = pd.DataFrame(dict(x=res, y=y))
#     res = res.replace('NA', '')
#     res = res[res.x != '']
#     res.dropna(subset=['x', 'y']).to_csv(destname, index=False)


# def get_smiles(cid):
#     try:
#         res = pubchem.get_smiles(cid)
#     except:
#         # print(f'{cid} not found')
#         res = ''
#     return res


def load_data(filename, max_samples=None, y_dtype='int32'):
    data = pd.read_csv(filename, header=None, skiprows=1).values
    np.random.shuffle(data)
    data = data[:max_samples]
    x, y = data[:max_samples, 0], data[:max_samples, 1:].astype(y_dtype)
    return x, y


# if __name__ == '__main__':
#     import os
#     data_folder = 'Bioassay'
#     for filename in os.listdir(data_folder):
#         if 'AID' in filename:
#             src = os.path.join(data_folder, filename)
#             dest = os.path.join(data_folder, filename.replace('AID', 'R3'))
#             print(src, dest)
#             rewrite_with_smiles(src, dest)
