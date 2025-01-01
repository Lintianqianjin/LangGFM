import re
import io
import json
import numpy as np
import pandas as pd

import networkx as nx

from tabulate import tabulate


class MyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)
    
class GraphTextualizer:
    """
    A class to convert NetworkX graphs into various text formats.
    Supported formats: JSON, Markdown Table, GraphML, GML
    """

    def __init__(self):
        pass

    def __to_json(self, graph: nx.Graph) -> str:
        """
        Export the graph to a JSON string.

        :param graph: A NetworkX graph instance.
        :return: A JSON-formatted string representing the graph.
        """
        graph_data = nx.node_link_data(graph,edges="edges")
        graph_text = json.dumps(graph_data, cls=MyJsonEncoder, indent=2)
        return graph_text

    def __to_markdown_table(self, graph: nx.Graph, undirected=False) -> str:
        """
        Export the graph to a Markdown table format.

        :param graph: A NetworkX graph instance.
        :return: A Markdown-formatted string representing the graph's nodes and edges.
        """
        def format_cell(cell):
            if isinstance(cell, str):
                return cell.replace('\n', '<br>').replace('|', ' ')
            return cell

        g = nx.node_link_data(graph,edges="edges")
        
        node_feature_list = []
        for node in g['nodes']:
            sub_node_feat = {'node': node['id']}
            keys = list(node.keys())
            for key in keys:
                if key != "id":
                    sub_node_feat[key] = node[key]
            node_feature_list.append(sub_node_feat)

        node_df = pd.DataFrame(node_feature_list)
        node_df = node_df.map(format_cell)

        node_markdown_table = tabulate(node_df, tablefmt='pipe', headers='keys', showindex=False)

        # Replace consecutive spaces with a single space
        node_markdown_table = re.sub(' +', ' ', node_markdown_table)
        # Replace consecutive hyphens with a single hyphen
        node_markdown_table = re.sub('-+', '-', node_markdown_table)
        # print(node_markdown_table)
        
        edge_feature_list = []
        added = set()
        for edge in g['edges']:
            src, tgt = edge['source'], edge['target']
            if (src, tgt) not in added: 
                # src, tgt = edge['source'], edge['target']
                keys = list(edge.keys())
                if undirected:
                    sub_edge_feat = {'edge': f"{edge['source']} ↔ {edge['target']}"}
                    for key in keys:
                        if key != "source" and key != "target":
                            sub_edge_feat[key] = edge[key]
                    
                    edge_feature_list.append(sub_edge_feat)

                    added.add((src, tgt))
                    added.add((tgt, src))
                else:
                    sub_edge_feat = {'edge': f"{edge['source']} → {edge['target']}"}
                    for key in keys:
                        if key != "source" and key != "target":
                            sub_edge_feat[key] = edge[key]
                    
                    edge_feature_list.append(sub_edge_feat)

                    added.add((src, tgt))
                    # added.add((tgt, src))

        edge_df = pd.DataFrame(edge_feature_list)
        edge_df = edge_df.fillna("Unknown")
        # 应用格式化函数
        edge_df = edge_df.map(format_cell)
        
        edge_markdown_table = tabulate(edge_df, tablefmt='pipe', headers='keys', showindex=False)

        # Replace consecutive spaces with a single space
        edge_markdown_table = re.sub(' +', ' ', edge_markdown_table)
        # Replace consecutive hyphens with a single hyphen
        edge_markdown_table = re.sub('-+', '-', edge_markdown_table)
        
        return f"Node Table:\n{node_markdown_table}\n\nEdge Table (`↔` means undirected and `→` means directed.):\n{edge_markdown_table}"


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
        graph_text = '\n'.join(nx.generate_gml(graph))
        return graph_text


    def export(self, graph: nx.Graph, format: str) -> str:
        """
        Export the graph in the specified format.

        :param graph: A NetworkX graph instance.
        :param format: The format to export the graph to. Supported formats: "json", "markdown", "graphml", "gml".
        :return: A string representing the graph in the specified format.
        :raises ValueError: If the format is unsupported.
        """
        format = format.lower()
        if format == "json":
            return self.__to_json(graph)
        elif format == "table":
            return self.__to_markdown_table(graph)
        elif format == "graphml":
            return self.__to_graphml(graph)
        elif format == "gml":
            return self.__to_gml(graph)
        else:
            raise ValueError(f"Unsupported format: {format}")

# 示例使用
if __name__ == "__main__":
    # 创建一个示例图
    G = nx.Graph()
    G.add_node(1, name="Alice", age=30)
    G.add_node(2, name="Bob", age=25)
    G.add_edge(1, 2, weight=5.0)

    # 使用 GraphTextualizer
    print("JSON format:")
    print(GraphTextualizer.to_json(G))

    print("\nMarkdown format:")
    print(GraphTextualizer.to_markdown(G))

    print("\nGraphML format:")
    print(GraphTextualizer.to_graphml(G))

    print("\nGML format:")
    print(GraphTextualizer.to_gml(G))