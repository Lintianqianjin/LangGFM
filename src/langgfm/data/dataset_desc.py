

class GraphDescription:

    def __init__(self):
        '''
        '''

    @property
    def KarateClub(self):
        return "This is a social network of a karate club. "\
            "Nodes correspond to club members, and edges represent friendships between them. " 

    @property
    def MUTAG(self):
        return "This graph is a chemical compound, where explicit hydrogen atoms have been removed. "\
            "Nodes represent atoms and edges represent chemical bonds. "\
            "Nodes are labeled by atom type (C, N, O, F, I, Cl, Br) and edges by bond type (single, double, triple or aromatic)."
    
    @property
    def GEDDatasetLINUX(self):
        return "This is a Program Dependence Graphs (PDG) generated from the Linux kernel. "\
            "Each graph represents a function, where "\
            "a node represents one statement and "\
            "an edge represents the dependency between the two statements. "
    
    @property
    def WikiCS(self):
        return "This graph is a hyperlink relationship graph between webpages. "\
            "Each node represents a webpage, and each edge represents a hyperlink on the source node pointing to the target node."
    
    @property
    def FB15K237(self):
        return "This graph is a subgraph from the knowledge graph FB15K237 where nodes represent entities and edges represent relationships."

    @property
    def MovieLens(self):
        return "This graph is a heterogeneous graph where users rate movies. "\
            "Each node represents a user or a movie. The titles and genres (separated by \"|\") of the movies are known. "\
            "Each edge represents a user rating for a movie. The rating value and the time of the rating are known."

    @property
    def BA2Motif(self):
        return "This is a synthetic graph. "\
            "Nodes are just nodes and "\
            "Edges are just undirected edges."
    
    @property
    def AMiner(self):
        return "This graph is a heterogeneous academic graph. "\
            "Each node represents an author or a paper or a venue."\
            "Each edge represents an \"an author writes a paper\" or \"a paper is published in a venue\"."
    
    @property
    def StanfordSentimentTreebank(self):
        return "The graph is the constituency tree, also known as a parse tree or a phrase structure tree, of a sentence. "\
            "The leaf nodes represent words. The non-leaf nodes can represent a higher-level constituents or phrases in the sentence. "\
            "The links, also known as branching lines, represent the grammatical relationships between the nodes/constituents in the tree/sentence. "\
            "These links indicate how the constituents are connected and how they function within the sentence."

    @property
    def MiniGC(self):
        return "This is a synthetic graph. "\
            "Nodes are just nodes and "\
            "Edges are just undirected edges."
    
    @property
    def OgbnArxiv(self):
        return "This is a directed graph, representing the citation network between some Computer Science (CS) arXiv papers. "\
            "Each node is an arXiv paper and "\
            "each directed edge indicates that one paper cites another one. "
    
    @property
    def OgbnArxivForAblation(self):
        return "This is a directed graph, representing the citation network between some Computer Science (CS) arXiv papers. "\
            "Each node is an arXiv paper and "\
            "each directed edge indicates that one paper cites another one. "

    @property
    def MAG240M(self):
        return "This graph is a heterogeneous academic graph. "\
            "Each node represents an author or a paper or an institution. "\
            "Each edge represents \"an author writes a paper\" or \"a paper cites a paper\" or \"an author is affiliated with an institution\"."

    @property
    def OgblVessel(self):
        return "This graph is an undirected, unweighted spatial graph of a partial mouse brain. "\
            "Nodes represent bifurcation points, "\
            "edges represent the vessels. "\
            "The node features are 3-dimensional, "\
            "representing the spatial (x, y, z) coordinates of the nodes in Allen Brain atlas reference space."
    
    @property
    def ReviewerRecommendation(self):
        return "This graph is a heterogeneous graph of peer-review records. "\
            "Each node represents a paper or a reviewer. "\
            "Each edge represents \"an reviewer reviewed a paper\"."

    @property
    def DeezerEgoNet(self):
        return "This graph is an ego-net of an Eastern European user collected from the music streaming service Deezer. "\
            "Nodes are Deezer users from European countries and "\
            "edges are mutual follower relationships between them. "

    @property
    def TwitchEgoNet(self):
        return "This graph is an ego-net of a Twitch user. "\
        "Nodes are users and "\
        "links are friendships. "
    
    @property
    def RedditThread(self):
        return "This is a graph of a thread from Reddit. "\
            "Nodes are Reddit users who participate in and "\
            "links are replies between them. "
            

    @property
    def Cora(self):
        # Done.
        return "Cora is a citation network contains papers and their citation relationship in computer science domain. "\
            "Each node in Cora represents a research paper from the computer science domain. "\
            "The raw text feature of a node is the title and abstract of the respective paper. "\
            "Every edge in the Cora dataset indicates the citation relationship between papers. "\
            "Each node's label corresponds to the category of the paper. "

    @property
    def CoraLink(self):
        # Done.
        return "Cora is a citation network contains papers and their citation relationship in computer science domain. "\
            "Each node in Cora represents a research paper from the computer science domain. "\
            "The raw text feature of a node is the title and abstract of the respective paper. "\
            "Every edge in the Cora dataset indicates the citation relationship between papers. "\
            "Each node's label corresponds to the category of the paper. "

    @property
    def HIV(self):
        # Done.
        return "This graph is a molecule graph, where explicit hydrogen atoms have been removed. "\
            "Nodes represent atoms and edges represent chemical bonds. "

    @property
    def BACE(self):
        return "This graph is a molecule graph, where explicit hydrogen atoms have been removed. "\
            "Nodes represent atoms and edges represent chemical bonds. "
    
    @property
    def BBBP(self):
        return "This graph is a molecule graph, where explicit hydrogen atoms have been removed. "\
            "Nodes represent atoms and edges represent chemical bonds. "
    
    @property
    def ESOL(self):
        return "This graph is a molecule graph, where explicit hydrogen atoms have been removed. "\
            "Nodes represent atoms and edges represent chemical bonds. "

    @property
    def FreeSolv(self):
        return "This graph is a molecule graph, where explicit hydrogen atoms have been removed. "\
            "Nodes represent atoms and edges represent chemical bonds. "
    
    @property
    def LIPO(self):
        return "This graph is a molecule graph, where explicit hydrogen atoms have been removed. "\
            "Nodes represent atoms and edges represent chemical bonds. "

    @property
    def PubMed(self):
        # TODO.
        return "PubMed is a citation network contains papers and their citation relationship in biomedical domain."\
            "Each node in PubMed represents a research paper from the biomedical domain. "\
            "The raw text feature of a node is the title and or abstract of the respective paper. "\
            "Every edge in the PubMed dataset indicates the citation relationship between papers. "\
            "Each node's label corresponds to the category of the paper. "
    
    @property
    def PubMedLink(self):
        # TODO.
        return "PubMed is a citation network contains papers and their citation relationship in biomedical domain."\
            "Each node in PubMed represents a research paper from the biomedical domain. "\
            "The raw text feature of a node is the title and or abstract of the respective paper. "\
            "Every edge in the PubMed dataset indicates the citation relationship between papers. "\
            "Each node's label corresponds to the category of the paper. "

    @property
    def NodeSizeCounting(self):
        return "Here is a undirected graph whose nodes are marked with numbers."

    @property
    def EdgeSizeCounting(self):
        return "Here is a undirected graph whose nodes are marked with numbers, and edges are <source, target> pairs."

    @property
    def NodeAttributeRetrieval(self):
        return "Here is a undirected graph whose nodes are marked with numbers, and edges are <source, target> pairs.\
                Note the nodes have attribute or weight."

    @property
    def EdgeAttributeRetrieval(self):
        return "Here is a undirected graph whose nodes are marked with numbers, and edges are <source, target> pairs.\
                Note the edges have attribute or weight."

    @property
    def NodeDegreeCounting(self):
        return "Here is a undirected graph whose nodes are marked with numbers, and edges are <source, target> pairs."

    @property
    def ShortestPath(self):
        return "Here is a undirected graph whose nodes are marked with numbers, and edges are <source, target> pairs.\
                Note the edge's weight means the distance between two nodes."

    @property
    def MaxTriangleSum(self):
        return "Here is a undirected graph whose nodes are marked with numbers, and edges are <source, target> pairs.\
                Note the nodes have attribute or weight."

    @property
    def HamiltonPath(self):
        return "Here is a undirected graph whose nodes are marked with numbers, and edges are <source, target> pairs."
               
    @property
    def SubgraphMatching(self):
        return "Here is a undirected graph whose nodes are marked with numbers, and edges are <source, target> pairs."
         
    @property
    def GraphStructure(self):
        return "Here is a undirected graph whose nodes are marked with numbers, and edges are <source, target> pairs."
   
    @property
    def GraphAutomorphic(self):
        return "Here is a undirected graph whose nodes are marked with numbers, and edges are <source, target> pairs."
   
    @property
    def Fingerprint(self):
        return """Fingerprints are converted into graphs by filtering the images and extracting regions that are relevant. 
        In order to obtain graphs from fingerprint images, the relevant regions are binarized and a noise removal and 
        thinning procedure is applied. This results in a skeletonized representation of the extracted regions. 
        Ending points and bifurcation points of the skeletonized regions are represented by nodes. 
        Additional nodes are inserted in regular intervals between ending points and bifurcation points. 
        Finally, undirected edges are inserted to link nodes that are directly connected through a ridge in the skeleton.
        Each node is labeled with a two-dimensional attribute giving its position. The edges are attributed with an angle 
        denoting the orientation of the edge with respect to the horizontal direction.
        Fingerprint patterns are traditionally classified into three broad categories: loops, whorls, and arches, each with further subdivisions. 
        Here's a general interpretation based on standard fingerprint classifications:
        L: Loop, A loop pattern in fingerprint classification.
        TR: Tented Arch, A type of arch pattern where the ridges converge and thrust upward.
        A: Arch, A plain arch pattern where ridges flow in one side and exit on the opposite side.
        TA: Tented Arch (another possible notation).
        W: Whorl, A circular or spiral pattern.
        R: Radial Loop, A loop pattern that opens toward the thumb (radial side).
        T: Transverse/Horizon Loop (assuming a loop pattern that opens horizontally).
        WR: Whorl Radial, A combination or variation involving a whorl with radial loop characteristics.
        TL: Tented Loop or Transitional Loop (likely a tented arch that is inclined towards a looping form).
        LT: Loop Tented, Similar to TL, but perhaps a loop with a more pronounced tented characteristic.
        AT: Arch Tented, Another variant of tented arch notation.
        RT: Radial Tented Arch, A tented arch that opens towards the thumb.
        WL: Whorl Loop, A variant or combined characteristic involving both a whorl and loop.
        RW: Radial Whorl, A whorl that shows characteristics more aligned with a radial opening.
        AR: Arch Radial, Likely an arch pattern with some radial loop characteristics.
        """
   
    @property
    def ChEBI20(self):
        return "This is a chemical compound from ChEBI20 where node represent atom and edge represent chemical bond."
       
    @property
    def SocialCircle(self):
        return "These are ego-networks for users from Twitter. The nodes are users, with node feature including hashtags and \
            mentions for each user. The edges represent following relationship.\
            Social circles refers to some subsets of categorized friends."
    
    @property
    def YelpReviewGeneration(self):
        return "This is a heterogeneous graph about reviews on the Yelp platform. Each node is either a business or a user. Each edge represents that two users are friends, or a user reviewed a business, or a user left a tip for a business."

    @property
    def Twitch(self):
        return "Twitch is a streaming service where users can broadcast live streams of playing computer games. Nodes are Twitch users and edges are undirected mutual follower relationships between them."

    @property
    def USAAirport(self):
        return "This is an unweighted and undirected American air-traffic network collected from the Bureau of Transportation Statistics from January to October, 2016, where nodes correspond to airports and edges indicate the existence of commercial flights. "

    @property
    def MetaQA(self):
        return "This a graph of the schema of a knowledge graph. Nodes are defined node types, edges are defined Relationships."
        
        # Here is an schema of heterogeneous graph about movie: 
        # # Node Labels

        # Movie: Represents movies.
        # Properties:
        # name (string): The name of the movie.

        # Person: Represents people involved in movies (e.g., directors, writers, actors).
        # Properties:
        # name (string): The name of the person.

        # Genre: Represents genres of movies.
        # Properties:
        # name (string): The name of the genre.

        # Tag: Represents tags associated with movies.
        # Properties:
        # name (string): The name of the tag.

        # IMDB_Votes: Represents the IMDb votes of movies.
        # Properties:
        # name (string): The name of the IMDb vote status (e.g., "famous").

        # IMDB_Rating: Represents the IMDb ratings of movies.
        # Properties:
        # name (string): The name of the IMDb rating status (e.g., "good").

        # Year: Represents the year a movie was released.
        # Properties:
        # value (integer): The release year.

        # Language: Represents the language of a movie.
        # Properties:
        # name (string): The language name.

        # # Relationships

        # DIRECTED_BY: Connects a Movie to a Person (e.g., director).
        # WRITTEN_BY: Connects a Movie to a Person (e.g., writer).
        # STARRED_ACTORS: Connects a Movie to a Person (e.g., actor).
        # HAS_GENRE: Connects a Movie to a Genre.
        # HAS_TAG: Connects a Movie to a Tag.
        # HAS_IMDB_VOTES: Connects a Movie to IMDB_Votes.
        # HAS_IMDB_RATING: Connects a Movie to IMDB_Rating.
        # RELEASED_IN: Connects a Movie to a Year.
        # IN_LANGUAGE: Connects a Movie to a Language.
        