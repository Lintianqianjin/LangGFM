import re
import json
import tempfile
import pandas as pd
import networkx as nx

import logging
logger = logging.getLogger("root")

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

    # def __read_table(self, graph_text):
    #     """
    #     Read a graph from a Markdown-style table.

    #     :param graph_text: The graph representation as a Markdown-style table.
    #     :return: A NetworkX graph instance.
    #     """
    #     node_data = re.search(r"Node Table:\n(.+)\n\nEdge Table", graph_text, re.DOTALL).group(1)
    #     node_df = self.__parse_table(node_data)

    #     edge_data = re.search(r"Edge Table .*:\n(.+)", graph_text, re.DOTALL)
    #     if edge_data is not None:
    #         edge_df = self.__parse_table(edge_data.group(1))
    #     else:
    #         edge_df = pd.DataFrame([], columns=['edge'])

    #     # regex to extract source, target nodes and the edge symbol
        
    #     pattern = r"(?P<source>\d+)\s*(?P<edge_type><->|->)\s*(?P<target>\d+)"
        
    #     def extract_edge_info(edge_text):
    #         # print(edge_text)
    #         match = re.search(pattern, edge_text)
    #         # print(match)
    #         if match:
    #             result = [
    #                 int(match.group("source")),   # Source node
    #                 int(match.group("target")),   # Target node
    #                 match.group("edge_type")      # Edge type
    #             ]
    #             return result
    #             # print(result)
    #         else:
    #             print("No match found.")
                    
    #     edge_df[['source', 'target', 'edge_type']] = edge_df['edge'].apply(lambda x: extract_edge_info(x)).tolist()
        
    #     edge_df.drop(columns=['edge'], inplace=True)

    #     # edge_df = edge_df.astype({'source': int, 'target': int})

    #     if not self.directed and not self.multigraph:
    #         G = nx.Graph()
    #     elif not self.directed and self.multigraph:
    #         G = nx.MultiGraph()
    #     elif self.directed and not self.multigraph:
    #         G = nx.DiGraph()
    #     elif self.directed and self.multigraph:
    #         G = nx.MultiDiGraph()

    #     for _, row in node_df.iterrows():
    #         attrs = {k: row[k] for k in row.index if k != "node"}
    #         G.add_node(int(row['node']), **attrs)

    #     for _, row in edge_df.iterrows():
    #         attributes = row.drop(['source', 'target', 'edge_type']).to_dict()
    #         # key is edge id between a same pair nodes
    #         if 'key' in attributes and type(attributes['key']) == str: 
    #             attributes['key'] = int(attributes['key'])
    #         # print(f"{attributes=}")
    #         G.add_edge(row['source'], row['target'], **attributes)
    #         if row['edge_type'] == '<->':
    #             G.add_edge(row['target'], row['source'], **attributes)

    #     return G
    
    
    def __parse_table(self, table_str: str) -> pd.DataFrame:
        """
        Parse a Markdown-style table into a pandas DataFrame.

        :param table_str: The raw table content as a string (without extra headings).
        :return: A pandas DataFrame with the parsed data.
        """
        lines = table_str.strip().split('\n')
        if len(lines) < 2:
            # Not a valid table (no header, no separator).
            return pd.DataFrame()

        # The first line contains the headers, e.g.: "| node | type | year |"
        headers = [h.strip() for h in lines[0].strip('|').split('|') if h.strip()]
        logging.info(f"Headers: {headers}")
        # The second line typically contains the separator of hyphens (e.g. "| --- | --- |").
        # We skip it and parse the remaining lines as table content.
        data_lines = lines[2:]

        rows = []
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
            # Split each row by '|'
            items = [item.strip() for item in line.strip('|').split('|')]
            rows.append(items)

        df = pd.DataFrame(rows, columns=headers)
        return df

    def __read_table(self, md_text: str):
        """
        Read and reconstruct a NetworkX graph from the Markdown text output 
        produced by __to_markdown_table (grouped by node/edge types).

        :param md_text: The full Markdown text (including multiple node tables 
                        and multiple edge tables).
        :return: A NetworkX Graph/DiGraph/MultiGraph/MultiDiGraph instance 
                 (depending on self.directed and self.multigraph).
        """

        # Decide which NetworkX graph type to build
        if not self.directed and not self.multigraph:
            G = nx.Graph()
        elif not self.directed and self.multigraph:
            G = nx.MultiGraph()
        elif self.directed and not self.multigraph:
            G = nx.DiGraph()
        else:  # self.directed and self.multigraph
            G = nx.MultiDiGraph()

        # ------------------------------------------------------------------
        # 1) Parse all Node Tables
        #    We look for blocks like:
        #    **Node Table (type = `SomeType`)**:
        #    | column1 | column2 | ...
        #    | ------- | ------- | ...
        #    | data    | data    | ...
        # ------------------------------------------------------------------
        node_pattern = (
            r"\*\*Node Table \(type = `([^`]+)`\)\*\*:\s*\n"  # Match the title line (with type=XXX), allowing spaces after the colon before a newline
            r"(.*?)"                                         # Non-greedy capture of the table content
            r"(?=\n\*\*Node Table|\n# Edge Table|$)"         # End at the next node table, edge table, or the end of the document
        )

        # Extract matches for the node tables
        node_tables_matches = re.findall(node_pattern, md_text, flags=re.DOTALL)

        # node_matches is a list of tuples like: [("Paper", "<table_str>"), ("Author", "<table_str>"), ... ]
        for node_type, node_table_str in node_tables_matches:
            logger.info(f"Node type: {node_type}")
            logger.info(f"Node table:\n{node_table_str}")
            node_df = self.__parse_table(node_table_str)

            # Each row represents one node. There's a column "node" that holds the ID (by default).
            # The other columns are node attributes. If you want to store the node type
            # (from the heading) explicitly, you can do so as well:
            for _, row in node_df.iterrows():
                node_id = row.get('node')
                if node_id is None:
                    continue

                # Attempt to convert node_id to int if it looks numeric; else keep as string
                try:
                    node_id = int(node_id)
                except ValueError:
                    pass  # keep as string if it's not a valid integer

                # Build the attribute dictionary
                node_attrs = {}
                for col_name, col_value in row.items():
                    if col_name == 'node':
                        continue
                    node_attrs[col_name] = col_value

                # Optionally store the node type from the heading
                node_attrs['type'] = node_type

                G.add_node(node_id, **node_attrs)

        # ------------------------------------------------------------------
        # 2) Parse all Edge Tables
        #    We look for blocks like:
        #    **Edge Table (type = `Citation`, `<->`)**:
        #    | edge            | someAttr | ...
        #    | --------------- | -------- | ...
        #    | 1 <-> 2         | ...      | ...
        # ------------------------------------------------------------------
        edge_pattern = re.compile(
            r"\*\*Edge Table \(type = `([^`]+)`, `([^`]+)`\)\*\*:\n(.*?)(?=\*\*Node Table|\*\*Edge Table|$)",
            re.DOTALL
        )
        edge_tables_matches = edge_pattern.findall(md_text)
        # edge_pattern = (
        #     r"\*\*Edge Table \(type = `([^`]+)`\)\*\*:\s*\n"  # Match the title line (with type=XXX), allowing spaces after the colon before a newline
        #     r"(.*?)"                                         # Non-greedy capture of the table content
        #     r"(?=\n\*\*Node Table|\n# Edge Table|$)"         # End at the next node table, edge table, or the end of the document
        # )

        # Extract matches for the node tables
        # edge_tables_matches = re.findall(edge_pattern, md_text, flags=re.DOTALL)
        
        # edge_matches is a list of tuples like:
        # [("Citation", "<->", "<table_str>"), ("CoAuthor", "<->", "<table_str>"), ...]

        # This arrow info might match "<->" or "->" from the heading. 
        # But each row in the table also has a column "edge" with "src -> tgt" or "src <-> tgt".
        # Typically, if the entire output was generated in "directed=True" mode, we'd see "->"
        # in the heading as well as in each row's "edge". If "directed=False", we'd see "<->".

        # A regex to extract source and target from the "edge" column text:
        edge_regex = re.compile(r"^(\S+)\s*(->|<->)\s*(\S+)$")

        for edge_type, arrow_heading, edge_table_str in edge_tables_matches:
            
            logger.info(f"Edge type: {edge_type}")
            logger.info(f"Edge table:\n{edge_table_str}")
            
            edge_df = self.__parse_table(edge_table_str)
            logger.info(f"{edge_df=}")
            # Each row has at least one column named "edge", plus possibly other attributes
            if 'edge' not in edge_df.columns:
                # If there's no 'edge' column, skip
                continue

            for _, row in edge_df.iterrows():
                edge_str = str(row['edge'])
                match = edge_regex.match(edge_str)
                if not match:
                    # If it doesn't match the pattern, skip or handle error
                    continue

                src_str, arrow_str, tgt_str = match.groups()
                logger.info(f"{src_str=}, {arrow_str=}, {tgt_str=}")
                # Attempt to convert them to int if possible
                try:
                    src = int(src_str)
                except ValueError:
                    src = src_str
                try:
                    tgt = int(tgt_str)
                except ValueError:
                    tgt = tgt_str

                # Build edge attributes (all columns except 'edge')
                edge_attrs = {}
                for col_name, col_value in row.items():
                    if col_name == 'edge':
                        continue
                    elif col_name == 'key':
                        edge_attrs[col_name] = int(col_value)
                    else:
                        edge_attrs[col_name] = col_value
                
                # Optionally store edge type from the heading
                edge_attrs['type'] = edge_type
                logger.info(f"{edge_attrs=}")

                # Add the edge(s) to G
                # If the arrow is '->', add a single directed edge
                # If the arrow is '<->', add edges in both directions, 
                #   if that is consistent with your usage. 
                # Typically in an undirected graph, you only need one edge. 
                # But if you're using a DiGraph or MultiDiGraph, you might want two separate edges.
                if arrow_str == '->':
                    G.add_edge(src, tgt, **edge_attrs)
                elif arrow_str == '<->':
                    # In an undirected Graph, add_edge once is enough. 
                    # But if the underlying G is a DiGraph, we might want two edges. 
                    # if isinstance(G, nx.Graph):
                    #     # For an undirected Graph, just one edge is needed
                    #     G.add_edge(src, tgt, **edge_attrs)
                    # else:
                        # For a DiGraph or MultiDiGraph, you may want two directed edges
                    G.add_edge(src, tgt, **edge_attrs)
                    G.add_edge(tgt, src, **edge_attrs)

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