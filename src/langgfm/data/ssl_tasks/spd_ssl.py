from .base_ssl import SelfSupervisedGraphTask
import networkx as nx
import random


@SelfSupervisedGraphTask.register("shortest_path_distance")
class ShortestPathDistance(SelfSupervisedGraphTask):
    def __init__(self, max_distance=None, path_details=False):
        """
        Initialize the ShortestPathDistance task.

        Args:
            max_distance (int, optional): Maximum distance to consider for path queries.
                If None, no limit is applied.
            path_details (bool): Whether to include path details in the answer.
        """
        self.max_distance = max_distance
        self.path_details = path_details

    def modify_graph(self, graph: nx.Graph) -> dict:
        """
        For this task, the graph is used as-is without modifications.

        Args:
            graph (nx.Graph): The input NetworkX graph.

        Returns:
            dict: A dictionary containing the unmodified graph.
        """
        return {"modified_graph": graph}

    def generate_query(self, modify_outputs: dict) -> dict:
        """
        Generate a query about the shortest path between two nodes.

        Args:
            modify_outputs: Output from modify_graph.

        Returns:
            dict: A dictionary containing:
                - "query_text": Natural language query about shortest path.
                - "source_node": The source node.
                - "target_node": The target node.
                - "path_details": Whether path details should be included.
        """
        graph = modify_outputs["modified_graph"]
        
        # Get list of nodes from the graph
        nodes = list(graph.nodes())
        
        # Ensure we have at least 2 nodes
        if len(nodes) < 2:
            raise ValueError("Graph must have at least 2 nodes for shortest path queries.")
        
        # Find connected components
        if graph.is_directed():
            components = list(nx.weakly_connected_components(graph))
        else:
            components = list(nx.connected_components(graph))
        
        # Choose a component with at least 2 nodes
        viable_components = [comp for comp in components if len(comp) >= 2]
        if not viable_components:
            raise ValueError("No connected component with at least 2 nodes found.")
        
        component = random.choice(viable_components)
        component_nodes = list(component)
        
        # Choose two distinct nodes from the same component
        source_node = random.choice(component_nodes)
        target_node = random.choice([n for n in component_nodes if n != source_node])
        
        # Generate query text
        if self.path_details:
            query_text = (
                f"What is the shortest path distance between node {source_node} and node {target_node} in the graph? "
                f"Please also list the nodes in the shortest path."
            )
        else:
            query_text = f"What is the shortest path distance between node {source_node} and node {target_node} in the graph?"
        
        return {
            "query_text": query_text,
            "source_node": source_node,
            "target_node": target_node,
            "path_details": self.path_details
        }

    def generate_answer(self, modify_outputs: dict, query_outputs: dict) -> str:
        """
        Generate a natural language answer about the shortest path.

        Args:
            modify_outputs: Output from modify_graph.
            query_outputs: Output from generate_query.

        Returns:
            str: A natural language answer.
        """
        graph = modify_outputs["modified_graph"]
        source_node = query_outputs["source_node"]
        target_node = query_outputs["target_node"]
        path_details = query_outputs["path_details"]
        
        try:
            # Calculate shortest path
            path = nx.shortest_path(graph, source=source_node, target=target_node)
            distance = len(path) - 1  # Distance is number of edges, which is nodes - 1
            
            if path_details:
                # Include path details
                path_str = " -> ".join(str(node) for node in path)
                return (
                    f"The shortest path distance between node {source_node} and node {target_node} is {distance}. "
                    f"The path is: {path_str}."
                )
            else:
                # Only distance
                return f"The shortest path distance between node {source_node} and node {target_node} is {distance}."
                
        except nx.NetworkXNoPath:
            return f"There is no path between node {source_node} and node {target_node} in the graph."