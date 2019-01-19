import numpy as np
from rdkit import Chem

AMINO_ACID_ALPHABET = list('ARNDCQEGHILKMFPSTWYVBZX*')
ATOM_ALPHABET = [Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, 121)]
BOND_ALPHABET = list(Chem.BondType().names.values())  # if you want the names: use temp_bond_name.names.keys()
MOL_ALPHABET = ATOM_ALPHABET + BOND_ALPHABET
SMILES_ALPHABET = list('#%)(+*-/.1032547698:=@[]\\cons') + ATOM_ALPHABET + ['se']
MAPPING_ATOM_TO_INT = dict(zip(ATOM_ALPHABET, np.arange(len(ATOM_ALPHABET))))
MAPPING_BOND_TO_INT = dict(zip(BOND_ALPHABET,  np.arange(len(BOND_ALPHABET))))

ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
             'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
             'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
ATOM_NUM_H = [0, 1, 2, 3, 4]
IMPLICIT_VALENCE = [0, 1, 2, 3, 4, 5, 6]
HYBRIDATION_LIST = [Chem.rdchem.HybridizationType.names[k] for k in sorted(
    Chem.rdchem.HybridizationType.names.keys(), reverse=True) if k != "OTHER"]
ATOM_DEGREE_LIST = range(5)
CHIRALITY_LIST = ['R'] # alternative is just S
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_STEREO = [0, 1, 2, 3, 4, 5]
