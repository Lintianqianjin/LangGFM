from typing import List, Literal, Optional, Tuple, Union, overload

import torch
from torch import Tensor

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import k_hop_subgraph

from ....utils.random_control import set_seed

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
        # torch.manual_seed(random_seed)
        set_seed(random_seed)

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


def generate_node_centric_k_hop_subgraph(graph, sample_id, num_hops, neighbor_size=None, random_seed=None, sampling=False):
    """
    Generate a k-hop subgraph for a given node ID.

    Parameters:
        graph: The input graph object containing edge_index.
        sample_id: The node ID for which the subgraph is generated.
        num_hops: Number of hops for the subgraph.
        neighbor_size: (Optional) Size of neighbors to sample if sampling is enabled.
        random_seed: (Optional) Random seed for reproducibility in sampling.
        sampling: Whether to perform neighbor sampling.

    Returns:
        sub_graph_edge_index: The combined edge index of the subgraph.
        sub_graph_nodes: The set of nodes in the subgraph.
    """
    if sampling:
        # Generate k-hop subgraph with neighbor sampling
        src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_sampled_subgraph(
            node_idx=sample_id, num_hops=num_hops, edge_index=graph.edge_index,
            relabel_nodes=False, flow='source_to_target', directed=False,
            neighbor_size=neighbor_size, random_seed=random_seed
        )

        tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_sampled_subgraph(
            node_idx=sample_id, num_hops=num_hops, edge_index=graph.edge_index,
            relabel_nodes=False, flow='target_to_source', directed=False,
            neighbor_size=neighbor_size, random_seed=random_seed
        )
    else:
        # Generate k-hop subgraph without sampling
        src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_subgraph(
            node_idx=sample_id, num_hops=num_hops, edge_index=graph.edge_index,
            relabel_nodes=False, flow='source_to_target', directed=False
        )

        tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_subgraph(
            node_idx=sample_id, num_hops=num_hops, edge_index=graph.edge_index,
            relabel_nodes=False, flow='target_to_source', directed=False
        )

    # Combine edges and nodes
    sub_graph_edge_mask = torch.logical_or(src_to_tgt_edge_mask, tgt_to_src_edge_mask)
    sub_graph_edge_index = graph.edge_index.T[sub_graph_edge_mask].T
    sub_graph_nodes = set(src_to_tgt_subset.numpy().tolist()) | set(tgt_to_src_subset.numpy().tolist())

    return sub_graph_edge_index, sub_graph_nodes, sub_graph_edge_mask



def generate_edge_centric_k_hop_subgraph(edge_index, edge, num_hops, neighbor_size=None, random_seed=None, sampling=False):
    """
    Generate a k-hop subgraph for a given node ID.

    Parameters:
        graph: The input graph object containing edge_index.
        sample_id: The node ID for which the subgraph is generated.
        num_hops: Number of hops for the subgraph.
        neighbor_size: (Optional) Size of neighbors to sample if sampling is enabled.
        random_seed: (Optional) Random seed for reproducibility in sampling.
        sampling: Whether to perform neighbor sampling.

    Returns:
        sub_graph_edge_index: The combined edge index of the subgraph.
        sub_graph_nodes: The set of nodes in the subgraph.
    """
    src, dst = edge
    if sampling:
        # Generate k-hop subgraph with neighbor sampling
        src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_sampled_subgraph(
            node_idx=[src, dst], num_hops=num_hops, edge_index=edge_index,
            relabel_nodes=False, flow='source_to_target', directed=False,
            neighbor_size=neighbor_size, random_seed=random_seed
        )

        tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_sampled_subgraph(
            node_idx=[src, dst], num_hops=num_hops, edge_index=edge_index,
            relabel_nodes=False, flow='target_to_source', directed=False,
            neighbor_size=neighbor_size, random_seed=random_seed
        )
    else:
        # Generate k-hop subgraph without sampling
        src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_subgraph(
            node_idx=[src, dst], num_hops=num_hops, edge_index=edge_index,
            relabel_nodes=False, flow='source_to_target', directed=False
        )

        tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_subgraph(
            node_idx=[src, dst], num_hops=num_hops, edge_index=edge_index,
            relabel_nodes=False, flow='target_to_source', directed=False
        )

    # Combine edges and nodes
    sub_graph_edge_mask = torch.logical_or(src_to_tgt_edge_mask, tgt_to_src_edge_mask)
    # sub_graph_edge_index = edge_index.T[sub_graph_edge_mask].T
    sub_graph_nodes = set(src_to_tgt_subset.numpy().tolist()) | set(tgt_to_src_subset.numpy().tolist())

    return sub_graph_nodes, sub_graph_edge_mask # sub_graph_edge_index, 
