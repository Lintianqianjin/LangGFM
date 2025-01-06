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