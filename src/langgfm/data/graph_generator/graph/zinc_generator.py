import os
import json
import networkx as nx
import pandas as pd
from torch_geometric.datasets import ZINC

from ....utils.number_process import NumberFormatter
from .._base_generator import GraphTaskGraphGenerator


@GraphTaskGraphGenerator.register("zinc")
class ZincGraphGenerator(GraphTaskGraphGenerator):
    
    directed = False
    has_node_attr = True
    has_edge_attr = True
    
    def load_data(self):
        
        self.root = os.path.join(self.raw_data_dir, "zinc")
        train_dataset = ZINC(self.root,subset=True, split='train')
        valid_dataset = ZINC(self.root,subset=True, split='val')
        test_dataset = ZINC(self.root,subset=True, split='test')
        
        # print(f"{len(train_dataset)=}, {len(valid_dataset)=}, {len(test_dataset)=}")
        
        self.dataset = train_dataset + valid_dataset + test_dataset
        
        self.atom_dict = {
            0: 'C',
            1: 'O',
            2: 'N',
            3: 'F',
            4: 'C H1',
            5: 'S',
            6: 'Cl',
            7: 'O -',
            8: 'N H1 +',
            9: 'Br',
            10: 'N H3 +',
            11: 'N H2 +',
            12: 'N +',
            13: 'N -',
            14: 'S -',
            15: 'I',
            16: 'P',
            17: 'O H1 +',
            18: 'N H1 -',
            19: 'O +',
            20: 'S +',
            21: 'P H1',
            22: 'P H2',
            23: 'C H2 -',
            24: 'P +',
            25: 'S H1 +',
            26: 'C H1 -',
            27: 'P H1 +'
        }
            
        self.bond_dict = {
            1: 'SINGLE',
            2: 'DOUBLE',
            3: 'TRIPLE'
        }
        
        
        self.all_samples = list(range(len(self.dataset)))
        

    def get_query(self, **kwargs):
        query = """The objective is to **regress** the penalized \( \log P \), also referred to as **constrained solubility** in some works. The target value \( y \) is defined as:

\[
y = \log P - ext{SAS} - ext{cycles}
\]

where:  
- \( \log P \) represents the **water-octanol partition coefficient**,  
- **SAS** denotes the **synthetic accessibility score**,  
- **cycles** refers to the number of **ring structures with more than six atoms**.  

Please estimate the value of \( y \) for the given molecule.
"""
# Please format the value as follows: <number_start><digit_info>{}{}<integer_part>{integer_digits}<decimal_part>{comma_separated_decimal_digits}<number_end>
        return query

    def get_answer(self, sample_id):
        y = self.dataset[sample_id].y[0].item()
        ans = NumberFormatter.format_number(round(y, 4))
        answer = f"The estimated penalized logP (\( y \)) for the given molecule is <answer> {ans} </answer>."
        return answer
    
    def create_networkx_graph(self, sample_id):
        data = self.dataset[sample_id]
        G = nx.MultiDiGraph()
        
        for node_idx, node_x in enumerate(data.x.numpy()):
            node_type = self.atom_dict[node_x[0]]
            G.add_node(node_idx, type="heavy_atom", name=node_type)
        
        for edge_idx, (src, dst) in enumerate(data.edge_index.numpy().T):
            bond_type = self.bond_dict[data.edge_attr[edge_idx].item()]
            G.add_edge(src, dst, type="bond", name=bond_type)
        
        return G

    @property
    def graph_description(self):
        desc = "This graph is a molecular graph. Nodes represent heavy atoms and edges represent chemical bonds. The node features are the types of heavy atoms and the edge features are the types of bonds between them."
        return desc