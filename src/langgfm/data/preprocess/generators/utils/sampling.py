from typing import List, Literal, Optional, Tuple, Union, overload

import torch
from torch import Tensor

from torch_geometric.utils.num_nodes import maybe_num_nodes


def k_hop_sampled_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    neighbor_size: Union[int, List[int]],
    random_seed: int = 42,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in k hop
    based on k_hop_graph of pyg. 
    https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/utils/subgraph.html
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx], device=row.device)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = torch.tensor(node_idx, device=row.device)
    else:
        node_idx = node_idx.to(row.device)

    # Set the random seed if specified for reproducibility
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Handle the neighbor_size parameter
    if isinstance(neighbor_size, int):
        neighbor_size = [neighbor_size] * num_hops
    elif isinstance(neighbor_size, list):
        assert len(neighbor_size) >= num_hops, "`neighbor_size` list must have at least `num_hops` elements"
        neighbor_size = neighbor_size
    else:
        raise ValueError("`neighbor_size` must be an integer or a list of integers")

    subsets = [node_idx]

    for hop in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        # subsets.append(col[edge_mask])
        # Find neighbors and sample
        neighbors = col[edge_mask]
        if len(neighbors) > neighbor_size[hop]:
            # Perform random sampling among neighbors
            sampled_neighbors = neighbors[torch.randperm(len(neighbors))[:neighbor_size[hop]]]
        else:
            sampled_neighbors = neighbors
        
        subsets.append(sampled_neighbors)

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        mapping = row.new_full((num_nodes, ), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask