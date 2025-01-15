import copy
import numpy as np
import networkx as nx


def shuffle_nodes_randomly(G):
    """
    Shuffle the nodes of a given graph randomly while preserving node and edge data.
    Parameters:
    G (networkx.Graph): The input graph to shuffle.
    Returns:
    tuple: A tuple containing:
        - copy_G (networkx.Graph): A new graph with nodes shuffled randomly.
        - node_idx_mapping_old_to_new (dict): A dictionary mapping old node indices to new node indices.
    """
    
    copy_G = copy.deepcopy(G)
    copy_G.clear()
    
    # add_nodes
    nodes_with_data = G.nodes(data=True)
    # print(f"{nodes_with_data=}")
    node_idx_mapping_old_to_new = {}
    for new_idx, old_idx in enumerate(np.random.permutation(len(nodes_with_data))):
        old_data = nodes_with_data[old_idx]
        copy_G.add_node(new_idx, **old_data)
        node_idx_mapping_old_to_new[old_idx] = new_idx
    
    # add edges
    for edge in G.edges(data=True):
        src, dst, edata = edge
        # print(src, dst, edata)
        src = node_idx_mapping_old_to_new[src]
        dst = node_idx_mapping_old_to_new[dst]
        
        copy_G.add_edge(src, dst, **edata)
    
    return copy_G, node_idx_mapping_old_to_new
