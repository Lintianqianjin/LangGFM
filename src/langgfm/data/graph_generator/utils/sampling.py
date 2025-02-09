from typing import List, Literal, Optional, Tuple, Union, overload

import torch
from torch import Tensor

# from torch_geometric.utils.num_nodes import maybe_num_nodes
# from torch_geometric.utils import k_hop_subgraph

from ....utils.random_control import set_seed

import logging
logger = logging.getLogger("root")

# def k_hop_sampled_subgraph(
#     node_idx: Union[int, List[int], Tensor],
#     num_hops: int,
#     edge_index: Tensor,
#     neighbor_size: Union[int, List[int]],
#     random_seed: int = 42,
#     relabel_nodes: bool = False,
#     num_nodes: Optional[int] = None,
#     flow: str = 'source_to_target',
#     directed: bool = False,
#     ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
#     r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in k hop
#     based on k_hop_graph of pyg. 
#     https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/utils/subgraph.html
#     """
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)

#     assert flow in ['source_to_target', 'target_to_source']
#     if flow == 'target_to_source':
#         row, col = edge_index
#     else:
#         col, row = edge_index

#     node_mask = row.new_empty(num_nodes, dtype=torch.bool)
#     edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

#     if isinstance(node_idx, int):
#         node_idx = torch.tensor([node_idx], device=row.device)
#     elif isinstance(node_idx, (list, tuple)):
#         node_idx = torch.tensor(node_idx, device=row.device)
#     else:
#         node_idx = node_idx.to(row.device)

#     # Set the random seed if specified for reproducibility
#     if random_seed is not None:
#         # torch.manual_seed(random_seed)
#         set_seed(random_seed)

#     # Handle the neighbor_size parameter
#     if isinstance(neighbor_size, int):
#         neighbor_size = [neighbor_size] * num_hops
#     elif isinstance(neighbor_size, list):
#         assert len(neighbor_size) >= num_hops, "`neighbor_size` list must have at least `num_hops` elements"
#         neighbor_size = neighbor_size
#     else:
#         raise ValueError("`neighbor_size` must be an integer or a list of integers")

#     subsets = [node_idx]

#     for hop in range(num_hops):
#         node_mask.fill_(False)
#         node_mask[subsets[-1]] = True
#         torch.index_select(node_mask, 0, row, out=edge_mask)
#         # subsets.append(col[edge_mask])
#         # Find neighbors and sample
#         neighbors = col[edge_mask]
#         if len(neighbors) > neighbor_size[hop]:
#             # Perform random sampling among neighbors
#             sampled_neighbors = neighbors[torch.randperm(len(neighbors))[:neighbor_size[hop]]]
#         else:
#             sampled_neighbors = neighbors
        
#         subsets.append(sampled_neighbors)

#     subset, inv = torch.cat(subsets).unique(return_inverse=True)
#     inv = inv[:node_idx.numel()]

#     node_mask.fill_(False)
#     node_mask[subset] = True

#     if not directed:
#         edge_mask = node_mask[row] & node_mask[col]

#     edge_index = edge_index[:, edge_mask]

#     if relabel_nodes:
#         mapping = row.new_full((num_nodes, ), -1)
#         mapping[subset] = torch.arange(subset.size(0), device=row.device)
#         edge_index = mapping[edge_index]

#     return subset, edge_index, inv, edge_mask

import torch
import random
from typing import Optional, Union, List

def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Converts a directed graph to an undirected one by adding reverse edges.
    For each edge (u, v), it also adds (v, u).
    If your graph is already undirected, you can return the original edge_index directly.
    """
    row, col = edge_index
    reversed_edges = torch.stack([col, row], dim=0)
    return torch.cat([edge_index, reversed_edges], dim=1)

def build_csr(edge_index: torch.Tensor, num_nodes: int):
    """
    Builds a CSR-like structure from edge_index (2, E):
      - row_ptr[i] gives the start position of node i's neighbors in col_ind
      - col_ind[row_ptr[i]: row_ptr[i+1]] contains all neighbors of node i
    The input edge_index is assumed to be undirected or will be converted by calling to_undirected().
    """
    # Convert to undirected (if your original graph is directed)
    edge_index_undirected = to_undirected(edge_index)

    row, col = edge_index_undirected
    # Sort edges by row so that all edges from the same source node are contiguous
    sorted_idx = row.argsort()
    row = row[sorted_idx]
    col = col[sorted_idx]

    # Build row_ptr: the cumulative count of edges for each node
    row_counts = torch.bincount(row, minlength=num_nodes)
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    row_ptr[1:] = torch.cumsum(row_counts, dim=0)

    return row_ptr, col

def get_khop_subgraph(
    edge_index: torch.Tensor,
    node_idx: Union[int, List[int], torch.Tensor],
    num_hops: int,
    max_neighbors_per_hop: Optional[Union[int, List[int]]] = None,
    sampling: bool = False,
    random_seed: Optional[int] = None
):
    """
    Performs a layer-wise BFS up to num_hops steps from the seed node(s) and optionally samples neighbors per node.

    Parameters
    ----------
    edge_index : torch.Tensor, shape (2, E)
        The edge list of the graph. If the graph is directed, it should be converted first or handled by to_undirected().
    node_idx : int, list[int], or 1D torch.Tensor[int]
        The starting node(s) for BFS.
    num_hops : int
        The maximum number of BFS layers.
    max_neighbors_per_hop : None or int or list[int], optional
        - When sampling=True, this parameter controls the maximum number of neighbors retained per node per hop.
          * If an int is given, the same limit is used for every hop.
          * If a list[int] is given, each hop uses a potentially different limit. For example, [15, 5] means:
            - For the 1st hop, each node keeps at most 15 neighbors.
            - For the 2nd hop, each node keeps at most 5 neighbors.
          * If None, no limit is applied, keeping all neighbors.
        - When sampling=False, this parameter is ignored and all neighbors are kept.
    sampling : bool
        Whether to perform neighbor sampling (True) or not (False).
    random_seed : int or None
        If set, a random seed is applied (both Python and PyTorch) for reproducible sampling.

    Returns
    -------
    sub_graph_nodes : set[int]
        The set of nodes visited by BFS, intersected with nodes from edge_index[0].
    sub_graph_edge_mask : torch.BoolTensor of shape (E,)
        A boolean mask for the original edges. True means the edge belongs to the k-hop subgraph (both endpoints visited).

    Notes
    -----
    This approach uses a BFS with a frontier set of nodes at each hop. When sampling=True, each node's neighbor list
    is randomly truncated (if longer than the specified limit). Otherwise, the full neighbor list is used.
    """
    # 1) Convert node_idx into a list of integers
    if isinstance(node_idx, int):
        start_nodes = [node_idx]
    elif isinstance(node_idx, list):
        start_nodes = node_idx
    elif isinstance(node_idx, torch.Tensor):
        start_nodes = node_idx.tolist()
    else:
        raise TypeError(f"Unsupported type for node_idx: {type(node_idx)}")

    # 2) Set random seed if specified
    if random_seed is not None:
        # random.seed(random_seed)
        # torch.manual_seed(random_seed)
        set_seed(random_seed)
    
    logger.debug(f"f{edge_index=}")
    
    # 3) Determine the number of nodes in the graph
    all_nodes = torch.cat([edge_index[0], edge_index[1]], dim=0)
    num_nodes = int(all_nodes.max().item()) + 1

    # 4) Build the CSR-like structure
    row_ptr, col_ind = build_csr(edge_index, num_nodes)

    # 5) Initialize BFS
    visited = set(start_nodes)
    frontier = set(start_nodes)

    # Helper function: get the sampling limit for the current hop
    def get_limit_for_hop(hop_idx: int) -> Optional[int]:
        # If sampling=False, no limit
        if not sampling:
            return None

        # If sampling=True but max_neighbors_per_hop is None, keep all
        if max_neighbors_per_hop is None:
            return None

        # If max_neighbors_per_hop is a single int, apply it to all hops
        if isinstance(max_neighbors_per_hop, int):
            return max_neighbors_per_hop

        # If max_neighbors_per_hop is a list, index by hop
        if hop_idx < len(max_neighbors_per_hop):
            return max_neighbors_per_hop[hop_idx]
        else:
            # If out of range, we could return the last or None
            return max_neighbors_per_hop[-1]

    # 6) Layer-wise BFS
    for hop in range(num_hops):
        if not frontier:
            break
        next_frontier = set()
        limit_this_hop = get_limit_for_hop(hop)

        for node in frontier:
            u = int(node)
            start = row_ptr[u].item()
            end = row_ptr[u+1].item()
            neighbors = col_ind[start:end]

            # If we have a limit for neighbors, randomly sample
            if limit_this_hop is not None and len(neighbors) > limit_this_hop:
                perm = torch.randperm(len(neighbors))[:limit_this_hop]
                neighbors = neighbors[perm]

            for nbr in neighbors.tolist():
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.add(nbr)

        frontier = next_frontier

    # 7) Build subgraph nodes and edge mask
    # Intersect visited nodes with the nodes in edge_index[0]
    # row0_nodes = set(edge_index[0].tolist())
    # logger.info(f"{visited=}")
    # # logger.info(f"{row0_nodes=}")
    # sub_graph_nodes = visited.intersection(row0_nodes)
    # logger.info(f"{sub_graph_nodes=}")
    sub_graph_nodes = visited

    # Create a boolean mask for edges whose both endpoints are visited
    visited_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for n in visited:
        visited_mask[n] = True

    edge_u = edge_index[0]
    edge_v = edge_index[1]
    sub_graph_edge_mask = visited_mask[edge_u] & visited_mask[edge_v]
    
    logger.debug(f"{sub_graph_nodes=}")
    return sub_graph_nodes, sub_graph_edge_mask


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
    logger.debug(f"{graph=}")
    if sampling:
        # Generate k-hop subgraph with neighbor sampling
    #     src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_sampled_subgraph(
    #         node_idx=sample_id, num_hops=num_hops, edge_index=graph.edge_index,
    #         relabel_nodes=False, flow='source_to_target', directed=False,
    #         neighbor_size=neighbor_size, random_seed=random_seed
    #     )
    #     tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_sampled_subgraph(
    #         node_idx=sample_id, num_hops=num_hops, edge_index=graph.edge_index,
    #         relabel_nodes=False, flow='target_to_source', directed=False,
    #         neighbor_size=neighbor_size, random_seed=random_seed
    #     )
        sub_graph_nodes, sub_graph_edge_mask = get_khop_subgraph(
            edge_index=graph.edge_index, node_idx=sample_id, num_hops=num_hops, 
            max_neighbors_per_hop=neighbor_size, sampling=True, random_seed=random_seed
        )
    else:
    #     # Generate k-hop subgraph without sampling
    #     src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_subgraph(
    #         node_idx=sample_id, num_hops=num_hops, edge_index=graph.edge_index,
    #         relabel_nodes=False, flow='source_to_target', directed=False
    #     )

    #     tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_subgraph(
    #         node_idx=sample_id, num_hops=num_hops, edge_index=graph.edge_index,
    #         relabel_nodes=False, flow='target_to_source', directed=False
    #     )
        sub_graph_nodes, sub_graph_edge_mask = get_khop_subgraph(
            edge_index=graph.edge_index, node_idx=sample_id, num_hops=num_hops, 
            sampling=False
        )
    # # Combine edges and nodes
    # sub_graph_edge_mask = torch.logical_or(src_to_tgt_edge_mask, tgt_to_src_edge_mask)
    # sub_graph_edge_index = graph.edge_index.T[sub_graph_edge_mask].T
    # sub_graph_nodes = set(src_to_tgt_subset.numpy().tolist()) | set(tgt_to_src_subset.numpy().tolist())
    # sub_graph_edge_index, 
    logger.debug(f"{sub_graph_nodes=}")
    return sub_graph_nodes, sub_graph_edge_mask



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
        # # Generate k-hop subgraph with neighbor sampling
        # src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_sampled_subgraph(
        #     node_idx=[src, dst], num_hops=num_hops, edge_index=edge_index,
        #     relabel_nodes=False, flow='source_to_target', directed=False,
        #     neighbor_size=neighbor_size, random_seed=random_seed
        # )

        # tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_sampled_subgraph(
        #     node_idx=[src, dst], num_hops=num_hops, edge_index=edge_index,
        #     relabel_nodes=False, flow='target_to_source', directed=False,
        #     neighbor_size=neighbor_size, random_seed=random_seed
        # )
        sub_graph_nodes, sub_graph_edge_mask = get_khop_subgraph(
            edge_index=edge_index, node_idx=[src, dst], num_hops=num_hops, 
            max_neighbors_per_hop=neighbor_size, sampling=True, random_seed=random_seed
        )
    else:
        # # Generate k-hop subgraph without sampling
        # src_to_tgt_subset, src_to_tgt_edge_index, _, src_to_tgt_edge_mask = k_hop_subgraph(
        #     node_idx=[src, dst], num_hops=num_hops, edge_index=edge_index,
        #     relabel_nodes=False, flow='source_to_target', directed=False
        # )

        # tgt_to_src_subset, tgt_to_src_edge_index, _, tgt_to_src_edge_mask = k_hop_subgraph(
        #     node_idx=[src, dst], num_hops=num_hops, edge_index=edge_index,
        #     relabel_nodes=False, flow='target_to_source', directed=False
        # )
        sub_graph_nodes, sub_graph_edge_mask = get_khop_subgraph(
            edge_index=edge_index, node_idx=[src, dst], num_hops=num_hops, 
            sampling=False
        )

    # # Combine edges and nodes
    # sub_graph_edge_mask = torch.logical_or(src_to_tgt_edge_mask, tgt_to_src_edge_mask)
    # # sub_graph_edge_index = edge_index.T[sub_graph_edge_mask].T
    # sub_graph_nodes = set(src_to_tgt_subset.numpy().tolist()) | set(tgt_to_src_subset.numpy().tolist())

    return sub_graph_nodes, sub_graph_edge_mask # sub_graph_edge_index, 
