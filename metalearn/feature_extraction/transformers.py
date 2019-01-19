import torch
import warnings
import numpy as np
from itertools import zip_longest, product
from sklearn.base import TransformerMixin
from rdkit import Chem
from joblib import delayed, Parallel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from rdkit.Chem.rdmolops import RDKFingerprint, RenumberAtoms
from dgl import DGLGraph
from .constants import *


def one_of_k_encoding(val, allowed_choices):
    """Converts a single value to a one-hot vector.

    Parameters
    ----------
        val: class to be converted into a one hot vector
            (integers from 0 to num_classes).
        allowed_choices: a list of allowed choices for val to take

    Returns
    -------
        A list of size len(allowed_choices) + 1
    """
    encoding = np.zeros(len(allowed_choices)+1, dtype=int)
    # not using index of, in case, someone fuck up
    # and there are duplicates in the allowed choices
    for i, v in enumerate(allowed_choices):
        if v == val:
            encoding[i] = 1
    if np.sum(encoding) == 0:  # aka not found
        encoding[-1] = 1
    return encoding


def get_atom_features(atom, use_chirality=True, explicit_H=False):
    feats = []
    # Set type symbol
    feats.extend(one_of_k_encoding(atom.GetSymbol(), ATOM_LIST))
    # add the degree of the atom now
    feats.extend(one_of_k_encoding(atom.GetDegree(), ATOM_DEGREE_LIST))
    # mplicit valence
    feats.extend(one_of_k_encoding(atom.GetImplicitValence(), IMPLICIT_VALENCE))
    # add hybridization type of atom
    feats.extend(one_of_k_encoding(atom.GetHybridization(), HYBRIDATION_LIST))
    # whether the atom is aromatic or not
    feats.append(int(atom.GetIsAromatic()))
    # atom formal charge
    feats.append(atom.GetFormalCharge())
    # add number of radical electrons
    feats.append(atom.GetNumRadicalElectrons())
    # atom is in ring
    feats.append(int(atom.IsInRing()))

    if not explicit_H:
        # number of hydrogene, is usually 0 after Chem.AddHs(mol) is called
        feats.extend(one_of_k_encoding(atom.GetTotalNumHs(), ATOM_NUM_H))

    if use_chirality:
        try:
            feats.extend(one_of_k_encoding(
                atom.GetProp('_CIPCode'), CHIRALITY_LIST))
            feats.append(int(atom.HasProp('_ChiralityPossible')))

        except:
            feats.extend([0, 0, int(atom.HasProp('_ChiralityPossible'))])

    return np.asarray(feats, dtype=np.float32)


def get_edge_features(bond):
    # Initialise bond feature vector as an empty list
    edge_features = []
    # Encode bond type as a feature vector
    bond_type = bond.GetBondType()
    edge_features.extend(one_of_k_encoding(bond_type, BOND_TYPES))
    # Encode whether the bond is conjugated or not
    edge_features.append(int(bond.GetIsConjugated()))
    # Encode whether the bond is in a ring or not
    edge_features.append(int(bond.IsInRing()))
    edge_features.append(int(bond.GetStereo()))
    return np.array(edge_features, dtype=np.float32)


def totensor(x, gpu=True, dtype=torch.float):
    """convert a np array to tensor"""
    x = torch.from_numpy(x)
    x = x.type(dtype)
    if torch.cuda.is_available() and gpu:
        x = x.cuda()
    return x


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class MoleculeTransformer(TransformerMixin):
    """
    This class is abstract all children should implement fit and transform
    """
    def __init__(self):
        super(MoleculeTransformer, self).__init__()

    def fit(self, X):
        return self

    @classmethod
    def to_mol(clc, mol, addHs=False, ordered=True, explicitOnly=False):
        """Convert the input into a Chem.Mol with implicit hydrogens

        Parameters
        ----------
        mol: str or rdkit.Chem.Mol
            SMILES of a molecule or a molecule
        addHs: bool, optional, default=False
            Whether the implicit hydrogens should be added the molecule.
        ordered: bool, optional, default=False
            Whether the atom should be ordered for graph conv.
        explicitOnly: bool, optional, default=False
            Whether the explicit hydrogen are included in the output molecule

        Returns
        -------
        mol: rdkit.Chem.Molecule
            the molecule if some conversion have been made.
            If the conversion fails None is returned so make sure that you handle this case on your own.

        Raises
        ------
        ValueError
            if the input is neither a CHem.Mol neither a string
        """
        if not isinstance(mol, (str, Chem.Mol)):
            raise ValueError("Input should be a CHem.Mol or a string")
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        if ordered:
            new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
            mol = RenumberAtoms(mol, new_order)
        if addHs and (mol is not None):
            mol = Chem.AddHs(mol, explicitOnly=explicitOnly)
        return mol

    def _transform(self, mol):
        """
        Compute features for a single molecule.
        """
        raise NotImplementedError('Missing implementation of _transform.')

    def transform(self, mols, as_numpy=False, **kwargs):
        """
        Compute the features of a molecule

        Args:
        -----
            mols: a list containing smiles
            addH: a bool to specify if hydrogen atom should be added

        Returns:
        --------
            features: the list of features

        """
        features = []
        for i, mol in enumerate(mols):
            feat = []
            try:
                mol = self.to_mol(mol, **kwargs)
                if mol:
                    feat = self._transform(mol)
            except:
                pass
            features.append(feat)
        if as_numpy:
            return np.array(features)
        return features

    def __call__(self, mols, **kwargs):
        """
        Calculate features for molecules.
        Parameters
        ----------
        mols : iterable
                RDKit Mol objects.
        """
        return self.transform(mols, **kwargs)


class SequenceTransformer:
    padding_value = 0

    def __init__(self, alphabet, returnTensor=True):
        """
        This is a transformer the sequence (string)
        :param alphabet: list,
        :param returnTensor: bool, if true return a tensor for pytorch/cuda compatibility
        """
        self.alphabet = alphabet
        self.alphabet2int = {el: i + 1 for i, el in enumerate(alphabet)}
        self.returnTensor = returnTensor

    def __transform_seq2intlist(self, seq):
        i, l, result = 0, 0, [self.padding_value for _ in range(len(seq))]
        while i < len(seq):
            # Is two-letter symbol?
            if self.alphabet2int.get(seq[i:i + 2]):
                result[l] = self.alphabet2int[seq[i:i + 2]]
                i += 2
            else:
                result[l] = self.alphabet2int[seq[i]]
                i += 1
            l += 1
        return result[:l]

    def transform(self, sequences):
        vectorized_sequences = [self.__transform_seq2intlist(seq) for seq in sequences]
        vectorized_sequences = np.array(list(zip_longest(*vectorized_sequences, fillvalue=self.padding_value))).T
        final_res = np.array([to_categorical(row, len(self.alphabet2int) + 1)
                              for row in vectorized_sequences], dtype=np.int)
        final_res[:, :, 0] = 0

        if self.returnTensor:
            final_res = torch.from_numpy(final_res)
            if torch.cuda.is_available():
                final_res = final_res.cuda()
        return final_res


class AdjGraphTransformer(MoleculeTransformer):
    """Transform a molecule into an adjacency matrix of atom, and a tensor od feature
    for each atom.

    Parameters
    ----------
    max_n_atoms: Maximum number of atom, to set the size of the graph.
        Use default value None, to allow graph with different size that 
        will be packed together later
    with_bond: bool, optional, default=False
        whether to return bond feature too
    explicit_H: bool, optional, default=False
        Whethet to consider hydrogen atoms explicitely
    chirality: bool, optional, default=True
        Use chirality as a feature.
    max_valence: int, optional, default=4
        Maximum number of neighbor for each atom

    """

    def __init__(self, max_n_atoms=None, with_bond=False, explicit_H=False, chirality=True, max_valence=4):
        self.max_valence = max_valence
        self.max_n_atoms = max_n_atoms # if this is not set, packing of graph would be expected later
        self.n_atom_feat = 0
        self.n_bond_feat = 0
        self.explicit_H = explicit_H
        self.use_chirality = chirality
        self.with_bond = with_bond
        self._set_num_features()

    def _set_num_features(self):
        """Compute the number of features for each atom and bond
        """
        self.n_atom_feat = 0
        # add atom type required
        self.n_atom_feat += len(ATOM_LIST) + 1
        # add atom degree
        self.n_atom_feat += len(ATOM_DEGREE_LIST) + 1
        # add valence implicit
        self.n_atom_feat += len(IMPLICIT_VALENCE) + 1
        # aromatic, formal charge, radical electrons, in_ring
        self.n_atom_feat += 4
        # hybridation_list
        self.n_atom_feat += len(HYBRIDATION_LIST) + 1
        # number of hydrogen
        if not self.explicit_H:
            self.n_atom_feat += len(ATOM_NUM_H) + 1
        # chirality
        if self.use_chirality:
            self.n_atom_feat += 3

        # do the same thing but with bond now
        # start with bond types
        self.n_bond_feat += len(BOND_TYPES) + 1
        # bond is conjugated, in rings
        self.n_bond_feat += 2

    def transform(self, mols):
        """Transform a batch of N molecules or smiles into a graph and 

        Parameters
        ----------
        mols: (str or rdkit.Chem.Mol) iterable
            The molecules to be converted

        Returns
        -------
        features: tuple list
            A list of tuple (G, x), where G is an adjacency matrix and x is the atom features

        """
        features = []
        mol_list = []
        for i, mol in enumerate(mols):
            feat = np.array([])
            try:
                mol = self.to_mol(mol, addHs=self.explicit_H)
                if mol:
                    if self.max_n_atoms and self.max_n_atoms < mol.GetNumAtoms():
                        warnings.warn("Max number of atoms is not enough, Updating to {}".format(mol.GetNumAtoms()))
                        self.max_n_atoms = mol.GetNumAtoms()
                    mol_list.append(mol)
            except:
                pass

        for mol in mol_list:
            feat = self._transform(mol)
            features.append(feat)

        return features

    def _transform(self, mol):
        # then for each atom, we would have one neighbor at each of its valence state
        if self.with_bond:
            bond_matrix = np.zeros(
                (self.max_n_atoms, self.n_bond_feat * self.max_valence)).astype(np.uint8)
            # type of bond for each of its neighbor respecting max valence

        n_atoms = self.max_n_atoms or mol.GetNumAtoms()
        adj_matrix = np.zeros((n_atoms, n_atoms), dtype=np.uint8)
        atom_arrays = []
        for a_idx in range(0, mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(a_idx)
            atom_arrays.append(get_atom_features(atom))
            adj_matrix[a_idx, a_idx] = 1 # add self loop
            for n_idx, neighbor in enumerate(atom.GetNeighbors()):
                adj_matrix[neighbor.GetIdx(), a_idx] = 1
                adj_matrix[a_idx, neighbor.GetIdx()] = 1
                if self.with_bond:
                    bond = mol.GetBondBetweenAtoms(a_idx, neighbor.GetIdx())
                    bond_feat = get_edge_features(bond)
                    bond_matrix[a_idx][(self.n_bond_feat * n_idx):(self.n_bond_feat) * (n_idx + 1)] = bond_feat

        atom_matrix = np.zeros(
            (n_atoms, self.n_atom_feat)).astype(np.uint8)
        for idx, atom_array in enumerate(atom_arrays):
            atom_matrix[idx, :] = atom_array
        
        if self.with_bond:
            atom_matrix = np.concatenate(
                [atom_matrix, bond_matrix], axis=1).astype(np.uint8)
        return (adj_matrix.astype(np.uint8), atom_matrix.astype(np.uint8))


class DGLGraphTransformer(AdjGraphTransformer):
    """
    Graph feature extraction using DGL

    """
    def __init__(self, explicit_H=False, chirality=True):
        super(DGLGraphTransformer, self).__init__(max_n_atoms=None, with_bond=False, explicit_H=explicit_H, chirality=chirality)


    def _transform(self, mol):
        atom_feats = []
        bond_feats = []
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()
        graph = DGLGraph()

        for a_idx in range(0, n_atoms):
            atom = mol.GetAtomWithIdx(a_idx)
            atom_feats.append(get_atom_features(atom))
        graph.add_nodes(n_atoms)

        bond_src = []
        bond_dst = []
        for i, bond in enumerate(mol.GetBonds()):
            begin_idx = bond.GetBeginAtom().GetIdx()
            end_idx = bond.GetEndAtom().GetIdx()
            features = get_edge_features(bond)
            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_feats.append(features)
            # set up the reverse direction
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
            bond_feats.append(features)
        graph.add_edges(bond_src, bond_dst)

        graph.ndata["hv"] = totensor(np.asarray(atom_feats), gpu=torch.cuda.is_available())
        graph.edata["he"] = totensor(np.asarray(bond_feats), gpu=torch.cuda.is_available())
        return graph
