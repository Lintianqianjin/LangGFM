# Base
from ._base_generator import InputGraphGenerator

# Node
from .node.ogbn_arxiv_generator import OgbnArxivGraphGenerator
from .node.wikics_generator import WikicsGraphGenerator
from .node.aminer_generator import AMinerGraphGenerator
from .node.twitch_generator import TwitchGraphGenerator
from .node.usa_airport import USAirportGraphGenerator
from .node.oag_scholar_interest_generator import OAGScholarInterestGraphGenerator
from .node.re_europe_generator import REEuropeGraphGenerator

# Edge
from .edge.ogbl_vessel_generator import OgblVesselGraphGenerator
from .edge.movielens_generator import MovieLens1M
from .edge.yelp_review_generator import YelpReviewGraphGenerator
from .edge.fb15k237_generator import FB15K237GraphGenerator
from .edge.stack_elec_generator import StackElecGraphGenerator

# Graph
from .graph.fingerprint_generator import FingerprintGraphGenerator
from .graph.bace_generator import BaceGraphGenerator
from .graph.esol_generator import ESOLGraphGenerator
from .graph.chebi20_generator import ChEBI20GraphGenerator
from .graph.explagraphs_generator import ExplagraphsGraphGenerator

# Structure
# 0-hop
from .structure.node_counting_generator import NodeCountingGraphGenerator
from .structure.edge_counting_generator import EdgeCountingGraphGenerator
from .structure.node_attribute_retrieval_generator import NodeAttributeRetrievalGraphGenerator
from .structure.edge_attribute_retrieval_generator import EdgeAttributeRetrievalGraphGenerator

# 1-hop
from .structure.edge_existence_generator import EdgeExistenceGraphGenerator
from .structure.degree_counting_generator import DegreeCountingGraphGenerator

# multi-hop
from .structure.connectivity_generator import ConnectivityGraphGenerator
from .structure.shortest_path_generator import ShortestPathGraphGenerator
from .structure.cycle_checking_generator import CycleCheckingGraphGenerator

# global
from .structure.hamilton_path_generator import HamiltonPathGraphGenerator
from .structure.graph_automorphic_generator import GraphAutomorphicGraphGenerator
from .structure.graph_structure_detection_generator import GraphStructureDetectionGraphGenerator
