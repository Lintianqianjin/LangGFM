import re
import io
import os
import json
import numpy as np
import pandas as pd

import networkx as nx

from tabulate import tabulate

from collections import defaultdict

from ...utils.json_encoder import MyJsonEncoder

import logging
logger = logging.getLogger("root")

class GraphTextualizer:
    """
    A class to convert NetworkX graphs into various text formats.
    Supported formats: JSON, Markdown Table, GraphML, GML
    """

    def __init__(self):
        pass
    
    def convert_if_not_multi(self, G):
        """
        Convert a MultiDiGraph to DiGraph if it doesn't have any parallel edges.
        Otherwise, return the original MultiDiGraph.
        
        Parameters:
        -----------
        G : nx.MultiDiGraph
            Input graph to check and potentially convert
        
        Returns:
        --------
        nx.MultiDiGraph or nx.DiGraph
            The original graph or converted graph
        """
        # Check if the graph is actually a MultiDiGraph
        if not isinstance(G, nx.MultiDiGraph):
            raise TypeError("Input must be a NetworkX MultiDiGraph")
        
        # Check if there are any multi-edges
        has_multi_edges = False
        
        # Dictionary to check for duplicate edges
        edge_dict = {}
        
        # Check all edges
        for u, v, key in G.edges(keys=True):
            edge = (u, v)
            if edge in edge_dict:
                has_multi_edges = True
                break
            edge_dict[edge] = True
        
        # If no multi-edges, convert to DiGraph
        if not has_multi_edges:
            DG = nx.DiGraph()
            # Copy nodes and their attributes
            for node, data in G.nodes(data=True):
                DG.add_node(node, **data)
            # Copy edges and their attributes
            for u, v, key, data in G.edges(data=True, keys=True):
                DG.add_edge(u, v, **data)
            return DG
        else:
            # Return the original graph
            return G

    def __format_cell(self, cell):
        """
        Format a cell by replacing newline characters with <br> and 
        replacing '|' with a space to avoid Markdown table issues.
        """
        if isinstance(cell, str):
            return cell.replace('\n', '<br>').replace('|', ' ')
        return cell

    def __cleanup_markdown_table(self, md_table: str) -> str:
        """
        Clean up the Markdown table string by removing consecutive spaces 
        and hyphens. This helps keep the output more readable.
        """
        # Replace consecutive spaces with a single space
        md_table = re.sub(' +', ' ', md_table)
        # Replace consecutive hyphens with a single hyphen
        md_table = re.sub('-+', '-', md_table)
        return md_table

    def __to_markdown_table(self, graph: nx.Graph, directed=True) -> str:
        """
        Export the given NetworkX graph to multiple Markdown tables, 
        grouped by the 'type' attribute of both nodes and edges.

        :param graph: A NetworkX Graph (or DiGraph) instance.
        :param directed: If True, edges will be presented in the form "src -> tgt".
                         If False, edges will be presented in the form "src <-> tgt".
        :return: A string containing Markdown-formatted tables grouped by node/edge types.
        """
        data = nx.node_link_data(graph, edges="edges")

        # --------------------
        # Group nodes by their 'type'
        # --------------------
        node_by_type = defaultdict(list)
        for node in data["nodes"]:
            node_type = node.get("type", "Unknown_NodeType")  # Default if 'type' is missing
            new_node = {"node": node["id"]}
            for k, v in node.items():
                if k != "id":
                    new_node[k] = v
            node_by_type[node_type].append(new_node)

        # Build a list of Markdown tables for each node type
        node_tables_md = []
        for ntype, node_list in node_by_type.items():
            df_node = pd.DataFrame(node_list).fillna("Unknown")
            # Format cells to avoid Markdown conflicts
            df_node = df_node.map(self.__format_cell)

            # Convert the DataFrame to Markdown
            md_table = tabulate(df_node, tablefmt='pipe', headers='keys', showindex=False)
            md_table = self.__cleanup_markdown_table(md_table)

            # Append a labeled section for each node type
            node_tables_md.append(f"**Node Table (type = `{ntype}`)**:\n{md_table}\n")

        # --------------------
        # Group edges by their 'type'
        # --------------------
        edge_by_type = defaultdict(list)
        visited_edges_by_type = defaultdict(set)

        for edge in data["edges"]:
            edge_type = edge.get("type", "Unknown_EdgeType")  # Default if 'type' is missing
            src, tgt = edge["source"], edge["target"]
            key = edge.get("key", None)  # Support for multigraph edges with key
            
            # For undirected graphs, avoid duplicates by sorting node pairs
            if not directed:
                sorted_pair = tuple(sorted([src, tgt]))
            else:
                sorted_pair = (src, tgt)
            
            # Include the key in the visited set for multigraph support
            if key is not None:
                sorted_pair = (sorted_pair, key)
            else:
                sorted_pair = sorted_pair

            if sorted_pair not in visited_edges_by_type[edge_type]:
                visited_edges_by_type[edge_type].add(sorted_pair)

                if directed:
                    edge_str = f"{src} -> {tgt}"
                else:
                    edge_str = f"{src} <-> {tgt}"

                new_edge = {"edge": edge_str}
                for k, v in edge.items():
                    if k not in ["source", "target"]:
                        new_edge[k] = v

                edge_by_type[edge_type].append(new_edge)

        # Build a list of Markdown tables for each edge type
        edge_tables_md = []
        for etype, edge_list in edge_by_type.items():
            # if not edge_list:
            #     continue
            df_edge = pd.DataFrame(edge_list).fillna("Unknown")
            # Format cells
            df_edge = df_edge.map(self.__format_cell)

            # Convert to Markdown
            md_table = tabulate(df_edge, tablefmt='pipe', headers='keys', showindex=False)
            md_table = self.__cleanup_markdown_table(md_table)
            edge_tables_md.append(
                f"**Edge Table (type = `{etype}`, `{'<->' if not directed else '->'}`)**:\n{md_table}\n"
            )

        # --------------------
        # Combine all Markdown tables
        # --------------------
        result_sections = []
        result_sections.append("# Node Table")
        result_sections.extend(node_tables_md)

        result_sections.append("# Edge Table\n(` <-> ` means undirected and ` -> ` means directed.)")
        result_sections.extend(edge_tables_md)

        # Return the final Markdown string
        return "\n".join(result_sections)
    
    def __to_json(self, graph: nx.Graph) -> str:
        """
        Export the graph to a JSON string.

        :param graph: A NetworkX graph instance.
        :return: A JSON-formatted string representing the graph.
        """
        graph_data = nx.node_link_data(graph,edges="edges")
        graph_text = json.dumps(graph_data, cls=MyJsonEncoder, indent=2)
        return graph_text
    

    def __to_graphml(self, graph: nx.Graph) -> str:
        """
        Export the graph to a GraphML format string.

        :param graph: A NetworkX graph instance.
        :return: A GraphML-formatted string.
        """
        graph_text = '\n'.join(nx.generate_graphml(graph))
        return graph_text


    def __to_gml(self, graph: nx.Graph) -> str:
        """
        Export the graph to a GML format string.

        :param graph: A NetworkX graph instance.
        :return: A GML-formatted string.
        """
        # with io.StringIO() as output:
        #     nx.write_gml(graph, output)
        #     return output.getvalue()
        # print(graph)
        logger.debug(graph)
        logger.debug(f"{graph.nodes(data=True)=}")
        str_list = list(nx.generate_gml(graph))
        logger.debug(str_list)
        graph_text = '\n'.join(nx.generate_gml(graph))
        return graph_text

    def export(self, graph: nx.Graph, format: str, simplify_if_no_multi=False, **kwargs) -> str:
        """
        Export the graph in the specified format.

        :param graph: A NetworkX graph instance.
        :param format: The format to export the graph to. Supported formats: "json", "markdown", "graphml", "gml".
        :return: A string representing the graph in the specified format.
        :raises ValueError: If the format is unsupported.
        """
        if simplify_if_no_multi:
            graph = self.convert_if_not_multi(graph)
        
        format = format.lower()
        if format == "json":
            return self.__to_json(graph)
        elif format == "table":
            return self.__to_markdown_table(graph, **kwargs)
        elif format == "graphml":
            return self.__to_graphml(graph)
        elif format == "gml":
            return self.__to_gml(graph)
        else:
            raise ValueError(f"Unsupported format: {format}")

    
    
# 示例使用
if __name__ == "__main__":
    # 创建一个示例图
    G = nx.MultiDiGraph()
    G.add_node(1, type="person", name="Alice", age=30)
    G.add_node(2, type="person", name="Bob", age=25)
    G.add_node(3, type="school", name="wpi", loc="usa")
    G.add_node(4, type="school", name="zju", loc="prc")
    G.add_edge(1, 2, type = "friend", years=5.0)
    G.add_edge(2, 1, type = "friend", years=5.0)
    G.add_edge(1, 3, type = "affiliated", years=4.0)
    G.add_edge(2, 4, type = "affiliated", years=2.0)

    # 使用 GraphTextualizer
    textualizer = GraphTextualizer()
    # 导出为 JSON 格式
    json_text = textualizer.export(G, format="json")
    print(json_text)
    # 导出为 Markdown Table 格式
    markdown_text = textualizer.export(G, format="table")
    print(markdown_text)
    # 导出为 GraphML 格式
    graphml_text = textualizer.export(G, format="graphml")
    print(graphml_text)
    # 导出为 GML 格式
    gml_text = textualizer.export(G, format="gml")
    print(gml_text)
