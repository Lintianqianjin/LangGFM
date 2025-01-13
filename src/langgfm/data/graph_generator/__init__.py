# Base
from ._base_generator import InputGraphGenerator

# Node
from .ogbn_arxiv_generator import OgbnArxivGraphGenerator
from .wikics_generator import WikicsGraphGenerator
from .aminer_generator import AMinerGraphGenerator
from .twitch_generator import TwitchGraphGenerator
from .usa_airport import USAirportGraphGenerator
from .oag_scholar_interest_generator import OAGScholarInterestGraphGenerator
from .re_europe_generator import REEuropeGraphGenerator


# Edge
from .ogbl_vessel_generator import OgblVesselGraphGenerator
from .movielens_generator import MovieLens1M
from .yelp_review_generator import YelpReviewGraphGenerator

# Graph
from .fingerprint_generator import FingerprintGraphGenerator
from .bace_generator import BaceGraphGenerator
from .esol_generator import ESOLGraphGenerator
from .chebi20_generator import ChEBI20GraphGenerator

# Structure
from .node_counting_generator import NodeCountingGraphGenerator
from .edge_counting_generator import EdgeCountingGraphGenerator
from .node_attribute_retrieval_generator import NodeAttributeRetrievalGraphGenerator
from .edge_attribute_retrieval_generator import EdgeAttributeRetrievalGraphGenerator
from .edge_existence_generator import EdgeExistenceGraphGenerator
from .degree_counting_generator import DegreeCountingGraphGenerator
from .connectivity_generator import ConnectivityGraphGenerator
from .shortest_path_generator import ShortestPathGraphGenerator
from .cycle_checking_generator import CycleCheckingGraphGenerator
from .hamilton_path_generator import HamiltonPathGraphGenerator
from .graph_automorphic_generator import GraphAutomorphicGraphGenerator
from .graph_structure_detection_generator import GraphStructureDetectionGraphGenerator
