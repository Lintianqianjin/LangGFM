import re
import json
import tempfile
import pandas as pd
import networkx as nx


class TextualizedGraphLoader:
    """
    A class to load graphs from textual representations in different formats.
    Supported formats: JSON, Markdown Table, GraphML, GML
    """

    def __init__(self, directed=True, multigraph=True):
        """
        Initialize the loader.

        :param directed: Whether the graph should be directed.
        :param multigraph: Whether the graph should allow multiple edges between nodes.
        """
        self.directed = directed
        self.multigraph = multigraph

    def __parse_table(self, table):
        """
        Parse a Markdown-style table into a pandas DataFrame.

        :param table: The table as a string.
        :return: A pandas DataFrame with the parsed data.
        """
        lines = table.strip().split('\n')
        headers = [header.strip() for header in lines[0].split('|') if header]
        data = [
            [item.strip() for item in line.strip("|").split('|') if item]
            for line in lines[2:]
            if line.strip()
        ]
        return pd.DataFrame(data, columns=headers)

    def __read_table(self, graph_text):
        """
        Read a graph from a Markdown-style table.

        :param graph_text: The graph representation as a Markdown-style table.
        :return: A NetworkX graph instance.
        """
        node_data = re.search(r"Node Table:\n(.+)\n\nEdge Table", graph_text, re.DOTALL).group(1)
        node_df = self.__parse_table(node_data)

        edge_data = re.search(r"Edge Table .*:\n(.+)", graph_text, re.DOTALL)
        if edge_data is not None:
            edge_df = self.__parse_table(edge_data.group(1))
        else:
            edge_df = pd.DataFrame([], columns=['edge'])

        # regex to extract source, target nodes and the edge symbol
        
        pattern = r"(?P<source>\d+)\s*(?P<edge_type><->|->)\s*(?P<target>\d+)"
        
        def extract_edge_info(edge_text):
            # print(edge_text)
            match = re.search(pattern, edge_text)
            # print(match)
            if match:
                result = [
                    int(match.group("source")),   # Source node
                    int(match.group("target")),   # Target node
                    match.group("edge_type")      # Edge type
                ]
                return result
                # print(result)
            else:
                print("No match found.")
                    
        edge_df[['source', 'target', 'edge_type']] = edge_df['edge'].apply(lambda x: extract_edge_info(x)).tolist()
        
        edge_df.drop(columns=['edge'], inplace=True)

        # edge_df = edge_df.astype({'source': int, 'target': int})

        if not self.directed and not self.multigraph:
            G = nx.Graph()
        elif not self.directed and self.multigraph:
            G = nx.MultiGraph()
        elif self.directed and not self.multigraph:
            G = nx.DiGraph()
        elif self.directed and self.multigraph:
            G = nx.MultiDiGraph()

        for _, row in node_df.iterrows():
            attrs = {k: row[k] for k in row.index if k != "node"}
            G.add_node(int(row['node']), **attrs)

        for _, row in edge_df.iterrows():
            attributes = row.drop(['source', 'target', 'edge_type']).to_dict()
            # key is edge id between a same pair nodes
            if 'key' in attributes and type(attributes['key']) == str: 
                attributes['key'] = int(attributes['key'])
            # print(f"{attributes=}")
            G.add_edge(row['source'], row['target'], **attributes)
            if row['edge_type'] == '<->':
                G.add_edge(row['target'], row['source'], **attributes)

        return G

    def load_graph_from_text(self, graph_text, format_):
        """
        Load a graph from a textual representation.

        :param graph_text: The graph as a string.
        :param format_: The format of the graph. Supported: "JSON", "Table", "GraphML", "GML".
        :return: A NetworkX graph instance.
        """
        if format_.lower() == "json":
            # , directed=self.directed, multigraph=self.multigraph
            return nx.node_link_graph(json.loads(graph_text), edges="edges")
        elif format_.lower() == "table":
            return self.__read_table(graph_text)
        elif format_.lower() == "graphml":
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".graphml") as tmpfile:
                tmpfile.write(graph_text)
                tmpfile.flush()
            return nx.read_graphml(tmpfile.name, node_type=int, force_multigraph=self.multigraph)
        elif format_.lower() == "gml":
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".gml") as tmpfile:
                tmpfile.write(graph_text)
                tmpfile.flush()
            return nx.read_gml(tmpfile.name, destringizer=int)
        else:
            raise ValueError(f"Unsupported format: {format_}")


# 示例使用
if __name__ == "__main__":
    # 示例 Markdown Table
    markdown_table = """
Node Table:
| node | name  | age |
|------|-------|-----|
| 1    | Alice | 30  |
| 2    | Bob   | 25  |

Edge Table (`↔` means undirected and `→` means directed.):
| edge       | weight |
|------------|--------|
| 1 ↔ 2      | 5.0    |
| 1 → 2      | 5.0    |
"""

    loader = TextualizedGraphLoader(directed=True, multigraph=True)
    graph = loader.load_graph_from_text(markdown_table, format_="Table")

    print("Loaded graph:")
    print(graph.nodes(data=True))
    print(graph.edges(data=True))