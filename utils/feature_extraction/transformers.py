import torch
import numpy as np
from itertools import zip_longest, product
from sklearn.base import TransformerMixin
from rdkit import Chem
from joblib import delayed, Parallel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


AMINO_ACID_ALPHABET = list('ARNDCQEGHILKMFPSTWYVBZX*')
periodic_elements = [Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, 121)]
ATOM_ALPHABET = periodic_elements
BOND_ALPHABET = list(Chem.BondType().names.values())  # if you want the names: use temp_bond_name.names.keys()
MOL_ALPHABET = ATOM_ALPHABET + BOND_ALPHABET
SMILES_ALPHABET = list('#%)(+*-/.1032547698:=@[]\\cons') + ATOM_ALPHABET + ['se']

MAPPING_ATOM_TO_INT = dict(zip(ATOM_ALPHABET, np.arange(len(ATOM_ALPHABET))))
MAPPING_BOND_TO_INT = dict(zip(BOND_ALPHABET,  np.arange(len(BOND_ALPHABET))))


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


def decomposition_into_tree(mol):
    """
    Given a molecule, the function outputs the maximum spanning tree over all its clusters.
    :param mol : (rdkit.Chem.Molecule) The molecule of interest
    :return: (int list list, int list)
    - nodes: list of all the cluster formed by the decomposition
    Each cluster is a list of integers which represent the ids of the atoms in the molecule.
    - edges, list of the edges. Each edges is a tuple of source node and destination node
    represented by their position in the list of clusters
    """
    n_atoms = mol.GetNumAtoms()
    # rings clusters
    rings = [list(x) for x in Chem.GetSymmSSSR(mol)]
    i, temp = 0, []
    while i < len(rings):
        j = i + 1
        while j < len(rings):
            atoms_ring_i = set(rings[i])
            atoms_ring_j = set(rings[j])
            if len(atoms_ring_i.intersection(atoms_ring_j)) > 2:
                rings[i] = list(atoms_ring_i.union(atoms_ring_j))
                rings[j:j+1] = []
            j += 1
        i += 1
    # non rings clusters aka bonds
    non_rings = [[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
                 for bond in mol.GetBonds() if not bond.IsInRing()]
    clusters = rings + non_rings
    # singleton clusters
    nb_appearence_per_atoms = dict(zip(range(n_atoms), [0]*n_atoms))
    for cluster in clusters:
        for atom_id in cluster:
            nb_appearence_per_atoms[atom_id] += 1
    for atom_id, nb_appearence in nb_appearence_per_atoms.items():
        if nb_appearence >= 3:
            clusters.append([atom_id])

    # clusters are the nodes of the cluster graph
    # here we construct the edges of that graph
    clusters_sets = [set(el) for el in clusters]
    clusters_edges = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            has_intersection = len(clusters_sets[i].intersection(clusters_sets[j])) > 0
            has_singleton = len(clusters_sets[i]) > 1 or len(clusters_sets[j]) > 1
            if has_singleton and has_intersection:
                clusters_edges.append((i, j, 1))
                clusters_edges.append((j, i, 1))
            elif has_intersection:
                clusters_edges.append((i, j, 0.5))
                clusters_edges.append((j, i, 0.5))

    # Compute Maximum Spanning Tree
    if len(clusters_edges) == 0:
        return clusters, []
    else:
        row, col, weights = zip(*clusters_edges)
        n_clusters = len(clusters)
        graph = csr_matrix((weights, (row, col)), shape=(n_clusters, n_clusters))
        junc_tree = minimum_spanning_tree(graph)
        row, col = junc_tree.nonzero()
        edges = [(row[i], col[i]) for i in range(len(row))]
        return clusters, edges


def get_mol_fragment(mol, fragment_atoms):
    """
    Returns a fragment of a molecule given ids of atoms in that fragment
    :param mol: (rdkit.Chem.Molecule) The molecule of interest
    :param fragment_atoms: (int list) ids of atoms of interest
    :return:
    """
    smiles = Chem.MolFragmentToSmiles(mol, fragment_atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles)
    return new_mol


def __get_mol_fragment_smiles(mol):
    if mol is None:
        return []
    nodes, _ = decomposition_into_tree(mol)
    return [Chem.MolFragmentToSmiles(mol, f) for f in nodes]


def generate_fragments_vocab(molecule_list, output_file=None):
    """
    Generate the list of all possible fragments given a set of molecules.
    This can be useful to build the vocabulary of fragments before doing some ML.

    :param molecule_list: (rdkit.Chem.Molecule list) the list of molecules of interest
    :param output_file:
    :return:
    """
    res = set()
    frags = Parallel(n_jobs=-1, verbose=True)(delayed(__get_mol_fragment_smiles)(mol) for mol in molecule_list)
    for el in frags:
        res.update(el)
    print('A vocab of {} elements has been generated'.format(len(res)))
    res = sorted(res, key=lambda x: len(x))
    if output_file:
        with open(output_file, 'w') as fd:
            fd.write('\n'.join(res))
    else:
        print('\n'.join(res))
    return res


class MoleculeTransformer(TransformerMixin):
    """
    This class is abstract all children should implement fit and transform
    """

    def __init__(self):
        super(TransformerMixin, self).__init__()

    @staticmethod
    def to_mol(s, addHs=False, explicitOnly=False):
        """
        Convert the input into a Chem.Mol with implicit or explicit hydrogens
        :param s: the input, is expected to be a SMILES or a SMARTS
        :param addHs: specify if the implicit and explicit hydrogens should be added or not, default False.
        :param explicitOnly: specify if explicit hydrogen are included or not
        :return: the molecule if some conversion have been made. If the conversion fails None is returned so
                 please handle the case where the conversion fails in your code:
        :exception ValueError if the input is neither a CHem.Mol neither a string
        """
        if not isinstance(s, (str, Chem.Mol)):
            raise ValueError("Input should be a CHem.Mol or a string")

        if isinstance(s, str):
            res = Chem.MolFromSmiles(s)
        else:
            res = s

        if addHs and (res is not None):
            res = Chem.AddHs(res, explicitOnly=explicitOnly)
        return res


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


class MolecularGraphTransformer(MoleculeTransformer):
    """
    Transform a rdkit molecule or a smile into a graph structure
    """
    # label all elements in the vocab from 1 to the vocab size
    mol_element_to_int = dict(zip(MOL_ALPHABET, 1 + np.arange(len(MOL_ALPHABET))))
    vocab_size = len(mol_element_to_int)
    nb_max_neighbors = 10 * 2  # the maximum valence of an atom is . We double by two because of the edges
    unknown_atom_int = 0
    padding_idx = -1

    # print(mol_element_to_int)

    @staticmethod
    def get_node_type(variable):
        """This return the correct int representation value from the atom or bond dictionary
        :param variable, atom or bond type
        """
        if variable in ATOM_ALPHABET:
            return MolecularGraphTransformer.mol_element_to_int[variable]
        elif variable in BOND_ALPHABET:
            return MolecularGraphTransformer.mol_element_to_int[variable]
        else:
            return MolecularGraphTransformer.unknown_atom_int

    def __init__(self, returnTensor=True, mode=1):
        """
        This transformer help to transform a sdf or smiles version of a molecule into a graph
        :param returnTensor: whether or not we should return tensors (from pytorch) by opposition to arrays (from numpy).
                             default, True
        :param mode: the mode specify how the graph is outputted. Each mode is described below:
                     1 - the output is a pair of (nodes_type, neighbors_indexes) ,
                         where nodes_type is a 1-D array or tensor which contains the type of all atoms or bonds,
                         and neighbor_indexes is a matrix which indicate the indexes of the neighbors atoms
                         or bonds in the nodes_types. This matrix has 20 rows and rows which doesn't have has many
                         neighbors are filled  with -1 at the empty places.
                     2 - the output is a pair of (formula, adjacency matrix) where,
                         formula is a vector which positions correspond to an atom and the content is the number of that
                         atom in the molecule,
                         adjacency matrix,

        """
        super(MolecularGraphTransformer, self).__init__()
        self.returnTensor = returnTensor
        self.mode = mode
        self.vocab = MolecularGraphTransformer.mol_element_to_int

    def __get_repr1(self, mol: Chem.Mol):
        # list of adjacency
        nodes_type, neighbors_indexes = [], []
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            bonds = atom.GetBonds()
            nodes_type.append(MolecularGraphTransformer.get_node_type(symbol))
            neighbors = sum([[n_atoms + bond.GetIdx(),
                              bond.GetOtherAtom(atom).GetIdx()]
                             for bond in bonds], [])
            neighbors_indexes.append(neighbors)

        # bonds as nodes of the graph and connected atoms as neighbors
        for bond in mol.GetBonds():
            symbol = bond.GetBondType()
            nodes_type.append(MolecularGraphTransformer.get_node_type(symbol))
            neighbors_indexes.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])

        nodes_type = np.array(nodes_type)
        # padding the edges with -1 for a size up to self.nb_max_neighbors
        neighbors_indexes[-1] += [self.padding_idx] * (self.nb_max_neighbors - len(neighbors_indexes[-1]))
        neighbors_indexes = np.array(list(zip_longest(*neighbors_indexes, fillvalue=self.padding_idx))).T

        if self.returnTensor:
            nodes_type = torch.from_numpy(nodes_type)
            neighbors_indexes = torch.from_numpy(neighbors_indexes)

        return nodes_type, neighbors_indexes

    def __get_repr2(self, mol: Chem.Mol):
        n_atoms = mol.GetNumAtoms()
        nodes_type = np.zeros(n_atoms)
        formula = np.zeros(len(periodic_elements), dtype=np.float32)
        adjancency_matrix = np.zeros((n_atoms, n_atoms), dtype=np.int64)
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            print(symbol, end=' ')
            bonds = atom.GetBonds()
            formula[MAPPING_ATOM_TO_INT[symbol]] += 1
            nodes_type[i] = MAPPING_ATOM_TO_INT[symbol]
            idx1 = atom.GetIdx()
            for bond in bonds:
                bond_type = bond.GetBondType()
                idx2 = bond.GetOtherAtom(atom).GetIdx()
                adjancency_matrix[idx1, idx2] = MAPPING_BOND_TO_INT[bond_type]

        sorted_nodes_idx = np.argsort(nodes_type)
        adjancency_matrix = adjancency_matrix[sorted_nodes_idx]
        adjancency_matrix = adjancency_matrix[:, sorted_nodes_idx]
        adjancency_matrix = adjancency_matrix[np.triu_indices_from(adjancency_matrix, k=1)]
        # # sanity_check:
        # for i in range(n_atoms):
        #     for j in range(0, i+1):
        #         assert adjancency_matrix[i, j] == adjancency_matrix[j, i]

        if self.returnTensor:
            formula = torch.from_numpy(formula)
            adjancency_matrix = torch.from_numpy(adjancency_matrix)

        return formula, adjancency_matrix

    def __transform(self, x):
        if isinstance(x, str):
            res = self.to_mol(x)
        elif isinstance(x, Chem.Mol):
            res = x
        else:
            raise Exception('Unhandle type')
        if self.mode == 1:
            return self.__get_repr1(res)
        elif self.mode == 2:
            return self.__get_repr2(res)

    def transform(self, inputs):
        return [self.__transform(el) for el in inputs]


class MolecularTreeDecompositionTransformer(MoleculeTransformer):
    """
    Transform a rdkit molecule object to a tree structure which nodes are different functionnal groups
    """
    padding_idx = -1

    def __init__(self, vocab, returnTensor=True, mode=1):
        """
        This transformer help to transform a sdf or smiles version of a molecule into a graph
        :param returnTensor: whether or not we should return tensors (from pytorch) by opposition to arrays (from numpy).
                             default, True
        :param mode: the mode specify how the graph is outputted. Each mode is described below:
                     1 - the output is a pair of (nodes_type, neighbors_indexes) ,
                         where nodes_type is a 1-D array or tensor which contains the type of all atoms or bonds,
                         and neighbor_indexes is a matrix which indicate the indexes of the neighbors atoms
                         or bonds in the nodes_types. This matrix has 20 rows and rows which doesn't have has many
                         neighbors are filled  with -1 at the empty places.
                     2 - the output is a pair of (formula, adjacency matrix) where,
                         formula is a vector which positions correspond to an atom and the content is the number of that
                         atom in the molecule,
                         adjacency matrix,

        """
        super(MolecularTreeDecompositionTransformer, self).__init__()
        self.vocab = dict(zip(vocab, 1 + np.arange(len(vocab))))
        self.returnTensor = returnTensor
        self.mode = mode
        self.nb_max_neighbors = 0
        self.vocab_size = len(self.vocab) + 1

    def __get_repr1(self, mol: Chem.Mol):
        nodes, edges = decomposition_into_tree(mol)
        nodes_type = np.array([self.vocab.get(Chem.MolFragmentToSmiles(mol, f), 0) for f in nodes])
        neighbors_indexes = [[] for _ in range(len(nodes))]
        for node1, node2 in edges:
            neighbors_indexes[node1].append(node2)
            neighbors_indexes[node2].append(node1)

        # padding the edges with -1 for a size up to self.nb_max_neighbors
        neighbors_indexes[-1] += [self.padding_idx] * (self.nb_max_neighbors - len(neighbors_indexes[-1]))
        neighbors_indexes = np.array(list(zip_longest(*neighbors_indexes, fillvalue=self.padding_idx))).T

        if self.returnTensor:
            nodes_type = torch.from_numpy(nodes_type)
            neighbors_indexes = torch.from_numpy(neighbors_indexes)

        return nodes_type, neighbors_indexes

    def __get_repr2(self, mol: Chem.Mol):
        nodes, edges = decomposition_into_tree(mol)
        nodes_type = np.array([self.vocab.get(Chem.MolFragmentToSmiles(mol, f), 0) for f in nodes])
        n_nodes = len(nodes)
        formula = np.zeros(self.vocab_size, dtype=np.float32)
        for i in nodes_type:
            formula[i] += 1.0

        adjancency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int64)
        for i, j in edges:
            adjancency_matrix[i, j] = 1
            adjancency_matrix[j, i] = 1

        sorted_nodes_idx = np.argsort(nodes_type)
        adjancency_matrix = adjancency_matrix[sorted_nodes_idx]
        adjancency_matrix = adjancency_matrix[:, sorted_nodes_idx]
        adjancency_matrix = adjancency_matrix[np.triu_indices_from(adjancency_matrix, k=1)]

        if self.returnTensor:
            formula = torch.from_numpy(formula)
            adjancency_matrix = torch.from_numpy(adjancency_matrix)

        return formula, adjancency_matrix

    def __transform(self, mol):
        if self.mode == 1:
            return self.__get_repr1(mol)
        elif self.mode == 2:
            return self.__get_repr2(mol)

    def transform(self, inputs):
        mols = [self.to_mol(x) for x in inputs]
        self.nb_max_neighbors = max([mol.GetNumAtoms() for mol in mols])
        return [self.__transform(mol) for mol in mols]
