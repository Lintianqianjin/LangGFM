from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
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

def get_node_slices(num_nodes: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    r"""Returns the boundaries of each node type in a graph."""
    node_slices: Dict[NodeType, Tuple[int, int]] = {}
    cumsum = 0
    for node_type, N in num_nodes.items():
        node_slices[node_type] = (cumsum, cumsum + N)
        cumsum += N
    return node_slices





def get_edge_idx_in_graph(src, dst, edge_index, multiplex_id:int=None):
    '''
    Get the edge index (i-th) of an edge in the edge_index ([2, num_edges]).'''
    
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