import os
import json
import networkx as nx
import pandas as pd


from .base_generator import GraphTaskGraphGenerator


@GraphTaskGraphGenerator.register("fingerprint")
class FingerprintGraphGenerator(GraphTaskGraphGenerator):
    """
    FingerprintGraphGenerator: A generator for creating k-hop subgraphs 
    from the Fingerprint dataset using NetworkX format.
    """
    
    def load_data(self):
        """
        Load the Fingerprint dataset and preprocess required mappings.
        """
        self.root = './data/Fingerprint'
        graphs = self.__parse_graphs(self.root, dataset_name = 'Fingerprint')
        # print(f"{graphs=}")
        
        self.graph = graphs # list of networkx graphs
        
        graphs_label = [graph.graph['label'] for graph in graphs]
        num_nodes = [graph.number_of_nodes() for graph in graphs]
        graphs_node_flag = [bool(graph.nodes(data=True)) for graph in graphs]
        graphs_df = pd.DataFrame({'graph':graphs, "number_of_nodes":num_nodes, "label":graphs_label, "graphs_node_flag":graphs_node_flag})
        
        # Filter graphs with at least 5 nodes and nodes with data
        graphs_df = graphs_df[graphs_df['number_of_nodes']>=5]
        self.graphs_df = graphs_df[graphs_df.graphs_node_flag==True]
        
        self.all_samples = set(self.graphs_df.index.tolist())
        
    
    
    def __read_file(self, filename):
        with open(filename, 'r') as file:
            return [line.strip() for line in file.readlines()]
        
    def __parse_graphs(self, root, dataset_name):
        prefix = f"{root}/{dataset_name}"
        # Read files
        edges = self.__read_file(f'{prefix}_A.txt')
        graph_indicator = self.__read_file(f'{prefix}_graph_indicator.txt')
        graph_labels = self.__read_file(f'{prefix}_graph_labels.txt')

        # Optional files
        node_labels = None
        file_path = f'{prefix}_node_labels.txt'
        if os.path.exists(file_path):
            node_labels = self.__read_file(file_path)

        edge_labels = None
        file_path = f'{prefix}_edge_labels.txt'
        if os.path.exists(file_path):
            edge_labels = self.__read_file(file_path)

        node_attributes = None
        file_path = f'{prefix}_node_attributes.txt'
        if os.path.exists(file_path):
            node_attributes = self.__read_file(file_path)

        edge_attributes = None
        file_path = f'{prefix}_edge_attributes.txt'
        if os.path.exists(file_path):
            edge_attributes = self.__read_file(file_path)

        graph_attributes = None
        file_path = f'{prefix}_graph_attributes.txt'
        if os.path.exists(file_path):
            graph_attributes = self.__read_file(file_path)

        # Initialize variables
        num_nodes = len(graph_indicator)
        num_edges = len(edges)
        num_graphs = len(graph_labels)

        # Create graphs
        graphs = [nx.Graph() for _ in range(num_graphs)]

        # Add nodes to graphs
        for node_id, graph_id in enumerate(graph_indicator, start=1):
            graph_idx = int(graph_id) - 1
            graphs[graph_idx].add_node(node_id)
            
            if node_labels:
                graphs[graph_idx].nodes[node_id]['label'] = node_labels[node_id - 1]

            if node_attributes:
                attributes = list(map(float, node_attributes[node_id - 1].split(',')))
                # graphs[graph_idx].nodes[node_id]['attributes'] = attributes
                graphs[graph_idx].nodes[node_id]['x'] = attributes[0]
                graphs[graph_idx].nodes[node_id]['y'] = attributes[1]

        # Add edges to graphs
        for edge_id, edge in enumerate(edges):
            node1, node2 = map(int, edge.split(','))
            graph_idx = int(graph_indicator[node1 - 1]) - 1
            graphs[graph_idx].add_edge(node1, node2)

            if edge_labels:
                graphs[graph_idx][node1][node2]['label'] = edge_labels[edge_id]

            if edge_attributes:
                attributes = list(map(float, edge_attributes[edge_id].split(',')))
                # graphs[graph_idx][node1][node2]['attributes'] = attributes
                graphs[graph_idx][node1][node2]['orient'] = attributes[0]
                graphs[graph_idx][node1][node2]['angle'] = attributes[1]

        # Add graph labels
        graph_label_map = {
            0: 'L',
            1: 'TR',
            2: 'A',
            3: 'TA',
            4: 'W',
            5: 'R',
            6: 'T',
            7: 'WR',
            8: 'TL',
            9: 'LT',
            10: 'AT',
            11: 'RT',
            12: 'WL',
            13: 'RW',
            14: 'AR'
        }

        for graph_idx, graph_id in enumerate(graph_labels, start=1):
            graphs[graph_idx - 1].graph['label'] = graph_label_map[int(graph_id)]

        # Add graph attributes
        if graph_attributes:
            for graph_idx, graph_attribute in enumerate(graph_attributes, start=1):
                graphs[graph_idx - 1].graph['attribute'] = float(graph_attribute)

        return graphs


    def get_query(self, sample=None):
        query = ("Fingerprint patterns are traditionally classified into three broad categories: loops, whorls, and arches, each with further subdivisions. "
        "Here's a general interpretation based on standard fingerprint classifications: "
        "L: Loop, A loop pattern in fingerprint classification.\n"
        "TR: Tented Arch, A type of arch pattern where the ridges converge and thrust upward.\n"
        "A: Arch, A plain arch pattern where ridges flow in one side and exit on the opposite side.\n"
        "TA: Tented Arch (another possible notation).\n"
        "W: Whorl, A circular or spiral pattern.\n"
        "R: Radial Loop, A loop pattern that opens toward the thumb (radial side).\n"
        "T: Transverse/Horizon Loop (assuming a loop pattern that opens horizontally).\n"
        "WR: Whorl Radial, A combination or variation involving a whorl with radial loop characteristics.\n"
        "TL: Tented Loop or Transitional Loop (likely a tented arch that is inclined towards a looping form).\n"
        "LT: Loop Tented, Similar to TL, but perhaps a loop with a more pronounced tented characteristic.\n"
        "AT: Arch Tented, Another variant of tented arch notation.\n"
        "RT: Radial Tented Arch, A tented arch that opens towards the thumb.\n"
        "WL: Whorl Loop, A variant or combined characteristic involving both a whorl and loop.\n"
        "RW: Radial Whorl, A whorl that shows characteristics more aligned with a radial opening.\n"
        "AR: Arch Radial, Likely an arch pattern with some radial loop characteristics.\n"
        "Please infer which category the given fingerprint belongs to from the above, and answer with the abbreviation of the category.")
        return query
    
    def get_answer(self, sample):
        category = self.graphs_df.at[sample, 'label']
        answer = f"The given fingerprint is likely to belong to {category}."
        return answer
    
    def __reindex_graph(self, G):
        mapping = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}
        
        # create new graph and reindex nodes
        G_new = nx.MultiDiGraph()
        for old_node, new_node in mapping.items():
            G_new.add_node(new_node, **G.nodes[old_node])  # copy node attrs
        
        for u, v, data in G.edges(data=True):
            G_new.add_edge(mapping[u], mapping[v], **data)  # copy edge attrs
        
        return G_new
    
    def create_networkx_graph(self, sample):
        graph = self.graphs_df.at[sample, 'graph']
        graph = self.__reindex_graph(graph)
        
        return graph