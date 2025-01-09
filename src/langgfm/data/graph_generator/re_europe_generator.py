import os
import networkx as nx
import pandas as pd
import warnings
import torch

from ._base_generator import NodeTaskGraphGenerator


@NodeTaskGraphGenerator.register("re_europe")
class REEuropeGraphGenerator(NodeTaskGraphGenerator):
    """
    REEuropeGraphGenerator: A generator for creating subgraphs (or the entire graph)
    from the RE-Europe dataset to predict the next 24-hour peak load for a specific bus.
    """

    # 若需要，可在此处对该数据集或本任务做简要描述
    graph_description = (
        "This generator uses the RE-Europe dataset (Metadata + Nodal_TS) to build "
        "a graph where each node is a bus and each edge is a power line. "
        "It also retrieves load time series data for each bus to facilitate 24h peak load forecasts."
    )

    def load_data(self):
        """
        Load the RE-Europe dataset and build internal structures for graph construction
        and for the time series load signals.
        """
        # 你可以根据自己的目录结构修改 self.root 
        self.root = "./data/RE-Europe"  
        
        # 1) 读取网络节点与边的 metadata
        self.network_nodes = pd.read_csv(
            os.path.join(self.root, "Metadata", "network_nodes.csv")
        )
        self.network_edges = pd.read_csv(
            os.path.join(self.root, "Metadata", "network_edges.csv")
        )
        
        # # 如果有 HVDC 线路，也可以一并读取
        # hvdc_path = os.path.join(self.root, "Metadata", "network_hvdc_links.csv")
        # if os.path.exists(hvdc_path):
        #     self.network_hvdc_links = pd.read_csv(hvdc_path)
        # else:
        #     self.network_hvdc_links = pd.DataFrame()
        
        # 2) 读取负载时序数据
        #    load_signal.csv: 每个 bus 每个小时的负载值 
        #    （该文件通常包含形如 [timestamp, bus0, bus1, bus2, ...] 或 [hour_index, bus_id, load_value] 等格式，
        #     请根据实际格式进行解析）
        self.load_signal = pd.read_csv(
            os.path.join(self.root, "Nodal_TS", "load_signal.csv")
        )
        
        # 如果 load_signal.csv 列很多（每个 bus 一列），你需要知道 bus 与列的映射关系。
        # 如果是 (hour_index, bus_id, load_value) 这种长表结构，也要将其 pivot 成 (hour_index, busX,...).
        # 以下仅做示例，不同数据格式需要你实际调整：
        #
        # self.load_signal_wide = self.load_signal.pivot(
        #     index='time_idx', columns='bus_id', values='load_value'
        # ).fillna(0)
        #
        # 这里先简单假设 read_csv 后就得到类似:
        # "time, bus_0, bus_1, ..., bus_1493"
        # 其中 time 是连续的小时索引或者时间戳

        # 可以存储 node_id 与其所在行/列的映射，以方便后面调用
        self.node_id_to_idx = {}
        for idx, row in self.network_nodes.iterrows():
            self.node_id_to_idx[row["ID"]] = idx

        # 为了与 OAG 示例对应，这里可以使用： self.all_samples = set(bus_id列表)
        self.all_samples = set(self.network_nodes["ID"].unique())

    def get_query(self, target_node_idx):
        """
        构造对目标 bus 节点的预测请求文本。
        例如: "请根据历史负载数据和邻近线路，预测该 bus 在未来24小时的峰值负载。"
        """
        query = (
            f"Given the historical load data and the network context of bus {target_node_idx}, "
            f"could you predict the peak load (in MW) for the next 24 hours?"
        )
        return query

    def get_answer(self, sample_id, target_node_idx):
        """
        按照示例，只给出一个简单的占位回答。
        实际使用中，你可能在此调用预训练模型、回溯样本数据等，给出真正的预测值。
        
        sample_id:   在有些场景下，sample_id 会和 target_node_idx 相同，也可能是个训练集索引。
                     这里为了与 OAG 例子保持一致，保留此参数。
        target_node_idx: 指定需要预测的节点(bus) ID。
        """

        peak_load_prediction = 0
        
        return (
            f"The predicted peak load for bus {target_node_idx} in the next 24 hours "
            f"is approximately {peak_load_prediction:.2f} MW."
        )

    def create_networkx_graph(self):
        """
        将 bus 与线路信息导入到 NetworkX 图结构中，并添加必要的属性。
        如果数据规模过大，也可以只在需要时构建子图。
        """
        G = nx.MultiDiGraph()

        # 1) 添加节点
        for _, row in self.network_nodes.iterrows():
            bus_id = row["ID"]
            G.add_node(
                bus_id,
                type="bus",
                name=row["name"],
                country=row["country"],
                voltage=row["voltage"],
                latitude=row["latitude"],
                longitude=row["longitude"],
            )

        # 2) 添加线路边
        for _, row in self.network_edges.iterrows():
            from_node = row["fromNode"]
            to_node = row["toNode"]
            # 线路属性示例：X, Y, limit, length 等
            G.add_edge(
                from_node,
                to_node,
                type="ac_line",
                reactance=row["X"],
                susceptance=row["Y"],
                limit=row["limit"],
                length=row["length"],
            )

        # 3) 如果有 HVDC 链接，也把它加进图
        if not self.network_hvdc_links.empty:
            for _, row in self.network_hvdc_links.iterrows():
                from_node = row["fromNode"]
                to_node = row["toNode"]
                G.add_edge(
                    from_node,
                    to_node,
                    type="hvdc_line",
                    limit=row["limit"],
                    voltage=row["voltage"],
                    length=row["length"],
                )

        return G

    def create_networkx_graph_subgraph(self, target_node_idx, k=1):
        """
        (可选方法) 创建一个以某个目标 bus 为中心的 k-hop 子图。
        你可以在训练或预测时只关注目标 bus 及其 k 范围内的邻居。
        """
        if not hasattr(self, "graph") or self.graph is None:
            warnings.warn("Graph not loaded yet, building the entire graph first.")
            self.graph = self.create_networkx_graph()

        # 采用简单的nx.bfs层次搜索或类似方法来获取k跳邻居
        nodes_k_hop = nx.bfs_tree(self.graph.to_undirected(), source=target_node_idx, depth_limit=k)
        sub_nodes = list(nodes_k_hop.nodes())
        sub_G = self.graph.subgraph(sub_nodes).copy()

        return sub_G
