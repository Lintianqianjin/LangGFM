from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import os
import sys
import torch
from collections import defaultdict

from torch_geometric.typing import (
    DEFAULT_REL,
    EdgeTensorType,
    EdgeType,
    FeatureTensorType,
    NodeOrEdgeType,
    NodeType,
    QueryType,
    SparseTensor,
    TensorFrame,
    torch_frame,
)

import logging
from ....utils.logger import logger


def get_node_slices(num_nodes: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    r"""Returns the boundaries of each node type in a graph."""
    node_slices: Dict[NodeType, Tuple[int, int]] = {}
    cumsum = 0
    for node_type, N in num_nodes.items():
        node_slices[node_type] = (cumsum, cumsum + N)
        cumsum += N
    return node_slices


def represent_edges_with_multiplex_id(edge_index: torch.Tensor, edge_indices_candidates: torch.Tensor):
    """
    Args:
        edge_index (torch.Tensor): Shape (2, num_edges), the full set of edges in the graph.
        edge_indices_candidates (torch.Tensor): 1D tensor of candidate edge indices 
                                                (subset of [0..num_edges-1]).

    Returns:
        results (List[Tuple[int, int, int, int]]): Each element is a tuple 
            (src, dst, global_edge_idx, multiplex_id), where
            - src, dst: the endpoints of the edge
            - global_edge_idx: the global edge index of this edge
            - multiplex_id: which #edge this is among all edges with the same (src, dst)
    """
    # 1. Build a dictionary mapping (src, dst) -> sorted list of global edge indices
    #    This covers ALL edges in the entire graph, not just the candidate ones.
    pair_to_edges = defaultdict(list)

    num_edges = edge_index.shape[1]
    for e_idx in range(num_edges):
        src = edge_index[0, e_idx].item()
        dst = edge_index[1, e_idx].item()
        pair_to_edges[(src, dst)].append(e_idx)

    # Sort each list so that the order is well-defined
    for pair in pair_to_edges:
        pair_to_edges[pair].sort()

    # 2. For each candidate edge index, find its (src, dst) and multiplex_id
    results = []
    for e_idx in edge_indices_candidates:
        e_idx = int(e_idx)
        # e_idx 是整张图里这条边的全局索引
        src = edge_index[0, e_idx].item()
        dst = edge_index[1, e_idx].item()
        pair = (src, dst)

        # 找到该 (src, dst) 在字典中的所有边的全局索引列表
        edge_list = pair_to_edges[pair]
        
        # multiplex_id 即这个 e_idx 在 edge_list 里的位置
        # （因为 edge_list 已经排过序，index 即可代表此边在该组多重边中的“第几个”）
        multiplex_id = edge_list.index(e_idx)
        # if multiplex_id > 0:
        #     logger.debug(f"Found multi-edge: {src} -> {dst}, edge_list={edge_list}, multiplex_id={multiplex_id}")

        results.append((src, dst, multiplex_id))

    return results


def get_multiplex_id_by_edge_idx(edge_idx: int, edge_index: torch.Tensor) -> int:
    """
    Get the multiplex ID of an edge in a multiplex graph.

    In a multiplex graph, a node pair (src, dst) may have multiple edges (multi-edges).
    This function returns the relative index (multiplex ID) of the edge at the given 
    global edge index within the group of edges that share the same (src, dst) pair.

    Args:
        edge_idx (int): The global index of the target edge in `edge_index`.
        edge_index (torch.Tensor): A tensor of shape (2, num_edges), representing 
                                   the graph’s edge list.

    Returns:
        int: The multiplex ID of the edge within its (src, dst) group.

    Raises:
        ValueError: If `edge_index` is not a valid (2, num_edges) tensor.
        IndexError: If `edge_idx` is out of the valid range.
        ValueError: If no edges are found for the given (src, dst) pair.
    """
    
    # Validate the shape and type of `edge_index`
    if not isinstance(edge_index, torch.Tensor) or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be a torch.Tensor of shape (2, num_edges).")
    
    num_edges = edge_index.shape[1]
    
    # Ensure `edge_idx` is within the valid range
    if edge_idx < 0 or edge_idx >= num_edges:
        raise IndexError(f"edge_idx is out of range. Expected in [0, {num_edges - 1}], got {edge_idx}.")
    
    # Retrieve the source (src) and destination (dst) nodes of the target edge
    src, dst = edge_index[:, edge_idx]

    # Identify all edges that have the same (src, dst) pair
    mask = (edge_index[0] == src) & (edge_index[1] == dst)
    
    # Get the indices of all edges that match (src, dst)
    multiplex_indices = mask.nonzero(as_tuple=True)[0]

    # If no matching edges are found, raise an error
    if multiplex_indices.numel() == 0:
        raise ValueError(f"No edges found for the node pair ({src} -> {dst}) in edge_index.")
    
    # Find the relative position of the given edge within its (src, dst) group
    multiplex_id = (multiplex_indices == edge_idx).nonzero(as_tuple=True)[0].item()

    return multiplex_id


def get_edge_idx_in_graph(src, dst, edge_index, multiplex_id:int=None):
    '''
    src, dst: source and destination node index of the edge
    edge_index: edge_index of the graph ([2, num_edges])
    multiplex_id: if the graph is a multiplex graph, 
                    then the multiplex_id is required to get the edge index.
    '''
    
    mask = (edge_index[0] == src) & (edge_index[1] == dst)
    
    if mask.any():
        edge_idx = mask.nonzero(as_tuple=True)[0] #.item()
        
        if multiplex_id is not None:
            edge_idx = edge_idx[multiplex_id].item()
        elif len(edge_idx) > 1:
            edge_idx = edge_idx.tolist()
        elif len(edge_idx) == 1:
            edge_idx = edge_idx.item()
        else:
            raise ValueError(f"Error format of the `edge_idx`: {edge_idx}")
        
        return edge_idx
    else:
        raise ValueError(f"Edge ({src}, {dst}) not found in the edge_index.")
    
    
    
def get_edge_idx_in_etype(edge_idx, edge_types, return_etype=False):
    """
        edge_idx: edge_idx-th edge in graph.edge_index (2, num_edges)
        edge_type_tensor: edge_type_tensor of the graph (num_edges)
    """
    edge_type_index = edge_types[edge_idx].item()
    
    # get all the edges of the same type as the edge_idx-th edge
    edge_mask = (edge_types == edge_type_index)
    edge_indices_of_type = edge_mask.nonzero(as_tuple=True)[0]  # all the edges of the same type as the edge_idx-th edge
    idx_in_edge_type = (edge_indices_of_type == edge_idx).nonzero(as_tuple=True)[0].item()
    
    if return_etype:
        return idx_in_edge_type, edge_type_index
    else:
        return idx_in_edge_type