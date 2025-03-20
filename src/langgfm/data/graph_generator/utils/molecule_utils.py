import networkx as nx
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

chem_dict = {
    1: "Hydrogen",
    2: "Helium",
    3: "Lithium",
    4: "Beryllium",
    5: "Boron",
    6: "Carbon",
    7: "Nitrogen",
    8: "Oxygen",
    9: "Fluorine",
    10: "Neon",
    11: "Sodium",
    12: "Magnesium",
    13: "Aluminum",
    14: "Silicon",
    15: "Phosphorus",
    16: "Sulfur",
    17: "Chlorine",
    18: "Argon",
    19: "Potassium",
    20: "Calcium",
    21: "Scandium",
    22: "Titanium",
    23: "Vanadium",
    24: "Chromium",
    25: "Manganese",
    26: "Iron",
    27: "Cobalt",
    28: "Nickel",
    29: "Copper",
    30: "Zinc",
    31: "Gallium",
    32: "Germanium",
    33: "Arsenic",
    34: "Selenium",
    35: "Bromine",
    36: "Krypton",
    37: "Rubidium",
    38: "Strontium",
    39: "Yttrium",
    40: "Zirconium",
    41: "Niobium",
    42: "Molybdenum",
    43: "Technetium",
    44: "Ruthenium",
    45: "Rhodium",
    46: "Palladium",
    47: "Silver",
    48: "Cadmium",
    49: "Indium",
    50: "Tin",
    51: "Antimony",
    52: "Tellurium",
    53: "Iodine",
    54: "Xenon",
    55: "Cesium",
    56: "Barium",
    57: "Lanthanum",
    58: "Cerium",
    59: "Praseodymium",
    60: "Neodymium",
    61: "Promethium",
    62: "Samarium",
    63: "Europium",
    64: "Gadolinium",
    65: "Terbium",
    66: "Dysprosium",
    67: "Holmium",
    68: "Erbium",
    69: "Thulium",
    70: "Ytterbium",
    71: "Lutetium",
    72: "Hafnium",
    73: "Tantalum",
    74: "Tungsten",
    75: "Rhenium",
    76: "Osmium",
    77: "Iridium",
    78: "Platinum",
    79: "Gold",
    80: "Mercury",
    81: "Thallium",
    82: "Lead",
    83: "Bismuth",
    84: "Polonium",
    85: "Astatine",
    86: "Radon",
    87: "Francium",
    88: "Radium",
    89: "Actinium",
    90: "Thorium",
    91: "Protactinium",
    92: "Uranium",
    93: "Neptunium",
    94: "Plutonium",
    95: "Americium",
    96: "Curium",
    97: "Berkelium",
    98: "Californium",
    99: "Einsteinium",
    100: "Fermium",
    101: "Mendelevium",
    102: "Nobelium",
    103: "Lawrencium",
    104: "Rutherfordium",
    105: "Dubnium",
    106: "Seaborgium",
    107: "Bohrium",
    108: "Hassium",
    109: "Meitnerium",
    110: "Darmstadtium",
    111: "Roentgenium",
    112: "Ununbiium",
    113: "——",
    114: "Ununquadium"
}


allowable_features_map = {
    "possible_atomic_num_dict": chem_dict,
    "possible_chirality_dict": {
        "CHI_UNSPECIFIED": "unspecified",
        "CHI_TETRAHEDRAL_CW": "tetrahedral clockwise",
        "CHI_TETRAHEDRAL_CCW": "tetrahedral counter-clockwise",
        "CHI_OTHER": "other",
        "misc": "misc",
    },
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        "misc",
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": [
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "misc",
    ],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": [
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
        "misc",
    ],
    "possible_bond_stereo_dict": {
        "STEREONONE": "none",
        "STEREOZ": "Z",
        "STEREOE": "E",
        "STEREOCIS": "CIS",
        "STEREOTRANS": "TRANS",
        "STEREOANY": "ANY",
    },
    "possible_is_conjugated_list": [False, True],
}


def ReorderCanonicalRankAtoms(mol):
    order = tuple(
        zip(
            *sorted(
                [(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))]
            )
        )
    )[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order


def get_chem_id2name():
    file1 = open("./id2element.csv", "r")
    Lines = file1.readlines()
    chem_dict = {}
    for line in Lines:
        line_split = line.strip().split(",")
        chem_dict[line_split[0]] = line_split[2]
    return chem_dict


def atom_to_feature(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = {
        "atom": chem_dict[int(atom.GetAtomicNum())],
        "atomic_number": int(atom.GetAtomicNum()),
        "chirality": allowable_features_map["possible_chirality_dict"][
            str(atom.GetChiralTag())
        ],
        # + " chirality",
        "degree": int(atom.GetTotalDegree()),
        "formal_charge": int(atom.GetFormalCharge()),
        "num_of_hydrogen": int(atom.GetTotalNumHs()),
        "num_of_radical_electrons": int(atom.GetNumRadicalElectrons()),
        "hybridization": str(atom.GetHybridization()),
        # "is aromatic: " if atom.GetIsAromatic() else "not aromatric",
        "aromatic": str(atom.GetIsAromatic()),
        # "is in ring" if atom.IsInRing() else "not in ring",
        "in_ring": str(atom.IsInRing()),
    }
    # return "feature node. atom: " + " , ".join(atom_feature)
    return  atom_feature


def bond_to_feature(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = {
        "bond_type": str(bond.GetBondType()),
        "bond_stereo": allowable_features_map["possible_bond_stereo_dict"][
            str(bond.GetStereo())
        ],
        # "is conjugated" if bond.GetIsConjugated() else "not conjugated",
        "conjugated": str(bond.GetIsConjugated())
    }
    # return "feature edge. chemical bond. " + " , ".join(bond_feature)
    return bond_feature


def compute_cycle(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    return cycle_score


def smiles2graph(smiles_string, removeHs=True, reorder_atoms=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    # print(f"{smiles_string=}")
    # smiles_string = rdMolStandardize.StandardizeSmiles(smiles_string)
    # print(f"standard {smiles_string=}")
    mol = Chem.MolFromSmiles(smiles_string)
    # print(f"{mol=}")
    cycle_score = compute_cycle(mol)
    mol = mol if removeHs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = ReorderCanonicalRankAtoms(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature(atom))

    # bonds
    edges_list = []
    edge_features_list = []
    bonds = mol.GetBonds()
    if len(bonds) == 0:
        edge_list = np.zeros((0, 2))
    else:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_list = np.array(edges_list)

    graph = dict()
    graph["edge_list"] = edge_list
    graph["edge_feat"] = edge_features_list
    graph["node_feat"] = atom_features_list
    graph["cycle"] = cycle_score

    # return graph
    # construct nx graph
    G = nx.MultiDiGraph() # Multi

    for node_idx, desc in enumerate(graph['node_feat']):
        # 'feature node. atom: Bromine , atomic number is 35 , unspecified chirality , 
        # degree of 1 , formal charge of 0 , num of hydrogen is 1 , num of radical electrons is 0 , 
        # hybridization is SP3 , not aromatric , not in ring'
        G.add_node(node_idx, type = 'atom', **desc)

    for (src, dst), desc in zip(graph['edge_list'], graph['edge_feat']): # edge_list already contains both directions
        G.add_edge(src, dst, type='bond', **desc)

    return G
