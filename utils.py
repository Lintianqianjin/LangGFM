import yaml
import argparse
import json
from transformers import LlamaTokenizer
import os
import networkx as nx
from itertools import combinations
import random

from rdkit import Chem
from rdkit.Chem import rdmolops


def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)

def build_args():
    parser = argparse.ArgumentParser(description='GraphReasoner')
    parser.add_argument('--task', type=str, default='cycle_train', help='task to perform')
    return parser.parse_args()
    


def smiles_to_networkx(smiles):
    """ Convert the SMILES string to an RDKit molecule object
    # Example usage
    smiles = "CC(=O)O[C@@H]1C[C@H]2C(C)(C)C(=O)C=C[C@]2(C)[C@H]2CC[C@]3(C)C(=CC[C@H]3c3ccoc3)[C@@]21C"
    graph = smiles_to_networkx(smiles)

    # Print the nodes and edges of the graph
    print("Nodes:")
    for node, data in graph.nodes(data=True):
        print(f"Node {node}: {data}")

    print("\nEdges:")
    for start_node, end_node, data in graph.edges(data=True):
        print(f"Edge from {start_node} to {end_node}: {data}")
    """
    mol = Chem.MolFromSmiles(smiles)
    
    # Initialize a NetworkX graph object
    graph = nx.Graph()
    
    # Add nodes with atom symbols as attributes
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
    
    # Add edges with bond types as attributes
    for bond in mol.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        # print(bond_type)
        # print(str(bond_type))
        # print(bond_type['bond_type'])
        # print(bond_type['bond_type'].split('.'))
        # exit()
        graph.add_edge(start_atom, end_atom, bond_type=str(bond_type))
    
    return graph


def is_serializable(obj):
    """
    Check if an object is serializable to JSON.
    :param obj: The object to check.
    :return: True if the object is serializable, False otherwise.
    """
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def save_serializable_items(data_list, json_filename, txt_filename):
    """
    Save serializable items from the list to a JSON file and 
    non-serializable items as strings to a text file.
    :param data_list: List of items to check and save.
    :param json_filename: Name of the file where the JSON data will be saved.
    :param txt_filename: Name of the file where the non-serializable items will be saved as strings.
    """
    serializable_items = []
    non_serializable_items = []
    
    for item in data_list:
        if is_serializable(item):
            serializable_items.append(item)
        else:
            non_serializable_items.append(item)
    
    # Save serializable items to JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(serializable_items, json_file, indent=4)
    
    # Save non-serializable items to a text file
    with open(txt_filename, 'w') as txt_file:
        for item in non_serializable_items:
            txt_file.write(f"{str(item)}\n")
    
    if non_serializable_items:
        print("Non-serializable items were found and saved to the text file.")
    else:
        print("All items were serializable and saved successfully.")


def read_file(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file.readlines()]


def parse_graphs(dataset_name):
    # Read files
    edges = read_file(f'{dataset_name}_A.txt')
    graph_indicator = read_file(f'{dataset_name}_graph_indicator.txt')
    graph_labels = read_file(f'{dataset_name}_graph_labels.txt')

    # Optional files
    node_labels = None
    file_path = f'{dataset_name}_node_labels.txt'
    if os.path.exists(file_path):
        node_labels = read_file(file_path)

    edge_labels = None
    file_path = f'{dataset_name}_edge_labels.txt'
    if os.path.exists(file_path):
        edge_labels = read_file(file_path)

    node_attributes = None
    file_path = f'{dataset_name}_node_attributes.txt'
    if os.path.exists(file_path):
        node_attributes = read_file(file_path)

    edge_attributes = None
    file_path = f'{dataset_name}_edge_attributes.txt'
    if os.path.exists(file_path):
        edge_attributes = read_file(file_path)

    graph_attributes = None
    file_path = f'{dataset_name}_graph_attributes.txt'
    if os.path.exists(file_path):
        graph_attributes = read_file(file_path)

    # Initialize variables
    num_nodes = len(graph_indicator)
    num_edges = len(edges)
    num_graphs = len(graph_labels)

    # Create graphs
    graphs = [nx.Graph() for _ in range(num_graphs)]

    # Add nodes to graphs
    for node_id, graph_id in enumerate(graph_indicator, start=1):
        graph_idx = int(graph_id) - 1
        graphs[graph_idx].add_node(node_id)
        
        if node_labels:
            graphs[graph_idx].nodes[node_id]['label'] = node_labels[node_id - 1]

        if node_attributes:
            attributes = list(map(float, node_attributes[node_id - 1].split(',')))
            # graphs[graph_idx].nodes[node_id]['attributes'] = attributes
            graphs[graph_idx].nodes[node_id]['x'] = attributes[0]
            graphs[graph_idx].nodes[node_id]['y'] = attributes[1]

    # Add edges to graphs
    for edge_id, edge in enumerate(edges):
        node1, node2 = map(int, edge.split(','))
        graph_idx = int(graph_indicator[node1 - 1]) - 1
        graphs[graph_idx].add_edge(node1, node2)

        if edge_labels:
            graphs[graph_idx][node1][node2]['label'] = edge_labels[edge_id]

        if edge_attributes:
            attributes = list(map(float, edge_attributes[edge_id].split(',')))
            # graphs[graph_idx][node1][node2]['attributes'] = attributes
            graphs[graph_idx][node1][node2]['orient'] = attributes[0]
            graphs[graph_idx][node1][node2]['angle'] = attributes[1]

    # Add graph labels
    graph_label_map = {
        0: 'L',
        1: 'TR',
        2: 'A',
        3: 'TA',
        4: 'W',
        5: 'R',
        6: 'T',
        7: 'WR',
        8: 'TL',
        9: 'LT',
        10: 'AT',
        11: 'RT',
        12: 'WL',
        13: 'RW',
        14: 'AR'
    }

    for graph_idx, graph_id in enumerate(graph_labels, start=1):
        graphs[graph_idx - 1].graph['label'] = graph_label_map[int(graph_id)]

    # Add graph attributes
    if graph_attributes:
        for graph_idx, graph_attribute in enumerate(graph_attributes, start=1):
            graphs[graph_idx - 1].graph['attribute'] = float(graph_attribute)

    return graphs


def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            if item == 214328887:
                print("*"*20)
            flattened_list.append(item)
    return flattened_list


# build social circle datasets
def load_edges(ego_id):
    edges_path = f'./GraphData/twitter/{ego_id}.edges'
    edges = []
    with open(edges_path, 'r') as file:
        for line in file:
            nodes = line.strip().split()
            edges.append((int(nodes[0]), int(nodes[1])))
    return edges

def load_circles(ego_id):
    circles_path = f'./GraphData/twitter/{ego_id}.circles'
    circles = {}
    with open(circles_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            circle_id = int(parts[0])
            users = list(map(int, parts[1:]))
            circles[circle_id] = users
    return circles

def load_features(ego_id):
    feat_path = f'./GraphData/twitter/{ego_id}.feat'
    features = {}
    with open(feat_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            user_id = int(parts[0])
            features[user_id] = list(map(int, parts[1:]))
    return features

def load_feature_names(ego_id):
    featnames_path = f'./GraphData/twitter/{ego_id}.featnames'
    featnames = []
    with open(featnames_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            featnames.append(parts[1])
    return featnames

def load_ego_features(ego_id):
    egofeat_path = f'./GraphData/twitter/{ego_id}.egofeat'
    with open(egofeat_path, 'r') as file:
        line = file.readline()
        egofeatures = list(map(int, line.strip().split()))
    return egofeatures

def construct_hashtag_mention(features, featnames):
    return ';'.join([featnames[i] for i, flag in enumerate(features) if flag == 1])

def construct_ego_network(ego_id):
    G = nx.Graph()
    ego_node = int(ego_id)
    
    # Load edges
    edges = load_edges(ego_id)
    node_set = set([edge[0] for edge in edges] + [edge[1] for edge in edges])
    node_mapping = {raw_idx: new_idx for new_idx, raw_idx in enumerate(node_set, start=1)}
    node_mapping.update({ego_node: 0})
    new_edges = [(node_mapping[edge[0]],node_mapping[edge[1]]) for edge in edges]

    G.add_edges_from(new_edges)
    
    # Load circles
    circles = load_circles(ego_id)
    new_circles = {}
    # print(ego_node)
    for k,raw_ids in circles.items():
        new_circles[k] = [node_mapping[raw_id] for raw_id in raw_ids if raw_id in node_mapping]
    # Load characteristics
    node_features = load_features(ego_id)
    featnames = load_feature_names(ego_id)
    egofeatures = load_ego_features(ego_id)
    
    # Adding ego-node and ego-node features
    # G.add_node(ego_node)
    new_ego_node = node_mapping[ego_node]
    G.add_node(new_ego_node)
    G.nodes[new_ego_node]['hashtag_mention'] = construct_hashtag_mention(egofeatures, featnames)
    
    # Adding features to existing nodes and adding ego-node edges
    for node in G.nodes():
        if node != new_ego_node:
            if node in node_features:
                G.nodes[node]['hashtag_mention'] = construct_hashtag_mention(node_features[node], featnames)
            else:
                G.nodes[node]['hashtag_mention'] = ''
            
            # Adding edges between ego-node and other nodes
            G.add_edge(new_ego_node, node)

    # Adding circles as graph label
    # G.graph['label'] = circles
    G.graph['label'] = new_circles

    return G

# def main():
#     base_dir = "path_to_your_directory_with_files"
#     ego_networks = {}

#     for user_file in os.listdir(base_dir):
#         if user_file.endswith(".edges"):
#             user_id = user_file.split('.')[0]
#             ego_networks[user_id] = construct_ego_network(user_id)
    
#     return ego_networks

# if __name__ == "__main__":
#     ego_networks = main()
#     print("Ego networks constructed successfully.")
#     # You can now process ego_networks as needed

