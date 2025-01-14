import os
import networkx as nx
import numpy as np
import pandas as pd
import warnings
import torch

from datetime import datetime, timedelta

from torch_geometric.data import HeteroData
from .._base_generator import NodeTaskGraphGenerator
from ..utils.graph_utils import get_node_slices

@NodeTaskGraphGenerator.register("re_europe")
class REEuropeGraphGenerator(NodeTaskGraphGenerator):
    """
    REEuropeLoadForecastGraphGenerator: A generator for creating subgraphs (or the entire graph)
    from the RE-Europe dataset, incorporating bus layouts features, generator data,
    and load signals to predict next 24-hour peak load for a specific bus.
    """

    graph_description = (
        "Generator uses the RE-Europe dataset (Metadata + Nodal_TS + capacity layouts + generator info) "
        "to build a graph with bus nodes, generator nodes, and AC/HVDC line edges. "
        "Bus layout features (like solar/wind capacities) are also assigned to the bus node attributes."
    )
        
    def load_data(self):
        """
        Load the RE-Europe dataset and build internal structures for graph construction,
        including bus layouts (solar/wind) features, generator info, etc.
        """
        self.root = "./data/RE-Europe"
        
        
        # bus historical load data
        self.bus_load_data = pd.read_csv(os.path.join(self.root, "Nodal_TS", "load_signal.csv"))
        # 转换时间列
        self.bus_load_data['Time'] = pd.to_datetime(self.bus_load_data['Time'])
        self.bus_load_data = self.bus_load_data.set_index('Time')
        # 确保数据按时间排序
        self.bus_load_data = self.bus_load_data.sort_index()
        
        
        # generator node data
        self.generator_info = pd.read_csv(os.path.join(self.root, "Metadata", "generator_info.csv"))
        # lines edge data
        self.network_edges = pd.read_csv(os.path.join(self.root, "Metadata", "network_edges.csv"))
            
        # bus node data
        if os.path.exists(f"{self.root}/network_nodes_with_all_features.csv"):
            self.network_nodes = pd.read_csv(f"{self.root}/network_nodes_with_all_features.csv")
        else:
            self.network_nodes = pd.read_csv(os.path.join(self.root, "Metadata", "network_nodes.csv"))
            
            self.solar_layout_cosmo = pd.read_csv(os.path.join(self.root, "Metadata", "solar_layouts_COSMO.csv"))
            self.solar_layout_cosmo = self.solar_layout_cosmo.add_suffix("_solar_COSMO")

            self.wind_layout_cosmo = pd.read_csv(os.path.join(self.root, "Metadata", "wind_layouts_COSMO.csv"))
            self.wind_layout_cosmo = self.wind_layout_cosmo.add_suffix("_wind_COSMO")
            
            self.solar_layout_ecmwf = pd.read_csv(os.path.join(self.root, "Metadata", "solar_layouts_ECMWF.csv"))
            self.solar_layout_ecmwf = self.solar_layout_ecmwf.add_suffix("_solar_ECMWF")
            
            self.wind_layout_ecmwf = pd.read_csv(os.path.join(self.root, "Metadata", "wind_layouts_ECMWF.csv"))
            self.wind_layout_ecmwf = self.wind_layout_ecmwf.add_suffix("_wind_ECMWF")
            
            self.network_nodes = self.network_nodes.merge(self.solar_layout_cosmo, how="left", left_on="ID", right_on='node_solar_COSMO')
            self.network_nodes = self.network_nodes.merge(self.wind_layout_cosmo, how="left", left_on="ID", right_on='node_wind_COSMO')
            self.network_nodes = self.network_nodes.merge(self.solar_layout_ecmwf, how="left", left_on="ID", right_on='node_solar_ECMWF')
            self.network_nodes = self.network_nodes.merge(self.wind_layout_ecmwf, how="left", left_on="ID", right_on='node_wind_ECMWF')
            
            self.network_nodes = self.network_nodes.drop(columns=['node_solar_COSMO', 'node_wind_COSMO', 'node_solar_ECMWF', 'node_wind_ECMWF'])
            
            self.network_nodes.to_csv(f"{self.root}/network_nodes_with_all_features.csv")
        
        # bus node mapping
        self.bus_id_to_nid = {row["ID"]: idx for idx, row in self.network_nodes.iterrows()}
        
        # reindex bus node
        self.network_edges['fromNode'] = self.network_edges['fromNode'].astype(int).map(self.bus_id_to_nid)
        self.network_edges['toNode'] = self.network_edges['toNode'].astype(int).map(self.bus_id_to_nid)
        
        # reindex generator
        self.generator_info['origin'] = self.generator_info['origin'].map(self.bus_id_to_nid)
        self.generator_info['nid'] = self.generator_info.index
            
        if os.path.exists(f"{self.root}/RE-Europe.pt"):
            # load pytorch geometric graph
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                data = torch.load(f"{self.root}/RE-Europe.pt")
        else:
            # change bus id to node id
            # print(f"{self.network_edges['fromNode'].dtype=}") # int64
            # print(f"{type(list(self.bus_id_to_nid.keys())[0])=}")
            # print(f"{type(list(self.bus_id_to_nid.values())[0])=}")
            # self.network_edges['fromNode'] = self.network_edges['fromNode'].astype(int).map(self.bus_id_to_nid)
            # self.network_edges['toNode'] = self.network_edges['toNode'].astype(int).map(self.bus_id_to_nid)
            # get edge index and save in torch tensor
            lines_edge_index = torch.tensor(self.network_edges[['fromNode', 'toNode']].values.T, dtype=torch.long)
            # get edge index and save in torch tensor
            generator_edge_index = torch.tensor(self.generator_info[['origin','nid']].values.T, dtype=torch.long)
            # print(f"{generator_edge_index.min()=}, {generator_edge_index.max()=}")
            # create pytorch geometric graph
            data = HeteroData()
            data['bus'].num_nodes = len(self.network_nodes)
            data['generator'].num_nodes = len(self.generator_info)
            
            data["bus","close_to","generator"].edge_index = generator_edge_index
            data["bus","transmission_line","bus"].edge_index = lines_edge_index
            
            torch.save(data, f"{self.root}/RE-Europe.pt")
        
        # print(f"{data=}")
        # print(f'{data["bus","close_to","generator"].edge_index.max()=}')
        # print(f'{data["bus","close_to","generator"].edge_index[1].max()=}')
        # print(f'{data["bus","transmission_line","bus"].edge_index.max()=}')
        self.node_slices = get_node_slices(data.num_nodes_dict)
        self.node_type_mapping = {0: "bus", 1: "generator"}
        self.edge_type_mapping = {0: "close_to", 1: "transmission_line"}
        self.graph = data.to_homogeneous()
        # print(f"{self.graph.edge_index.max()=}")
        
        # bus_idx: sample_idx % number_of_bus_nodes
        # time_idx: sample_idx // number_of_bus_nodes (delta days after 2012-01-08)

        num_days = len(np.unique(self.bus_load_data.index.date))
        num_days = num_days - 7  # 7 days of past data
        self.all_samples = set(range(len(self.network_nodes) * num_days))
        
    def __load_bus_day_avg_load(self, df: pd.DataFrame, target_date: datetime):
        """
        计算指定日期当天所有bus 24小时load的平均值，以及过去七天的每天的平均值。
        
        :param df: pd.DataFrame, 传入的 load_signal 数据框
        :param date_str: str, 指定的日期，格式为 'YYYY-MM-DD'
        :return: tuple (pd.Series, list[pd.Series])，返回当天各bus的平均load，过去七天每天各bus的平均load list
        """
        
        # 计算指定日期的平均值
        target_day_data = df[df.index.date == target_date.date()]
        daily_avg = target_day_data.mean()
        
        return daily_avg
    
    def __load_bus_past_week_avg_load(self, df: pd.DataFrame, target_date: datetime) -> pd.DataFrame:
        """
        计算过去七天的每个bus的load均值，并返回一个DataFrame：
        - index: bus
        - columns: 过去七天的日期（从 target_date-1 到 target_date-7）
        
        Parameters:
        df (pd.DataFrame): 输入数据，index 为 datetime，columns 为 bus，每个值是对应 bus 在该时间点的 load。
        target_date (datetime): 目标日期

        Returns:
        pd.DataFrame: 以 bus 为 index，过去七天的 load 均值为 columns 的 DataFrame
        """
        past_load_dict = {}

        for i in range(1, 8):  # 过去7天
            past_date = target_date - timedelta(days=i)
            past_day_data = df[df.index.date == past_date.date()]  # 选取过去某天的数据
            
            if not past_day_data.empty:
                past_day_avg = past_day_data.mean(axis=0)  # 计算所有时间点的平均负载
            else:
                past_day_avg = pd.Series(index=df.columns, dtype=float)  # 无数据，返回 NaN

            past_load_dict[past_date.strftime('%Y-%m-%d')] = past_day_avg  # 以日期为列名

        # 组合为 DataFrame
        past_load_df = pd.DataFrame(past_load_dict)
        past_load_df.index = past_load_df.index.astype(int)
        # print(f"{past_load_df.index=}")
        # print(f"{self.bus_id_to_nid=}")
        past_load_df = past_load_df.rename(index=self.bus_id_to_nid)  # 进行 index 映射转换
        past_load_df.index.name = "nid"  # 重新命名 index

        return past_load_df

    def generate_graph(self, sample_id):
        '''
        convert sample_id to bus_idx and time_idx
        '''
        bus_idx = sample_id % len(self.network_nodes)
        time_idx = sample_id // len(self.network_nodes)
        # time_idx days after 2012-01-08
        target_date = datetime.strptime("2012-01-08", "%Y-%m-%d") + timedelta(days=time_idx)
        self.historical_load_data = self.__load_bus_past_week_avg_load(self.bus_load_data, target_date)
        # print(f"{self.historical_load_data=}")
        new_G, metadata = super().generate_graph(bus_idx)

        metadata['raw_sample_id'] = sample_id
        
        return new_G, metadata
        
    def get_query(self, target_node_idx):
        """
        构造对目标 bus 节点的预测请求文本。
        """
        query = (
            f"Given the historical load data and the network context of bus {target_node_idx}, "
            f"could you predict the average load (in MW) for the next 24 hours?"
        )
        return query

    def get_answer(self, sample_id, target_node_idx):
        """
        在此处用模型推理得到未来24小时峰值负载预测值。
        这里给出一个示例占位。
        """
        bus_idx = sample_id % len(self.network_nodes)
        time_idx = sample_id // len(self.network_nodes)
        
        # get date string
        target_date = datetime.strptime("2012-01-08", "%Y-%m-%d") + timedelta(days=time_idx)

        daily_avg = self.__load_bus_day_avg_load(self.bus_load_data, target_date)
        avg_load = daily_avg.iloc[bus_idx]
        
        return (
            f"The predicted average load for bus {target_node_idx} in the next 24 hours "
            f"is around {avg_load:.2f} MW."
        )

    def create_networkx_graph(self, node_mapping, sub_graph_edge_mask=None):
        """
        生成包含 bus + generator + line 的 NetworkX 图结构，并将 layouts 特征等合并为 bus 的属性。
        """
       # Create a NetworkX graph
        G = nx.MultiDiGraph()
        for raw_node_idx, new_node_idx in node_mapping.items():
            node_type = self.node_type_mapping[self.graph.node_type[raw_node_idx].item()]
            if node_type == 'bus':
                bus_idx = raw_node_idx - self.node_slices['bus'][0]
                
                bus_info = self.network_nodes.iloc[bus_idx]

                past_load_data = self.historical_load_data.loc[bus_idx]
                # print(f"{past_load_data=}")
                
                feautres = {
                    # "voltage (kV)": bus_info['voltage'].item(), most are 380
                    "COSMO_solar_capacity_proportional (MWh)": bus_info['Proportional_solar_COSMO'].item(),
                    "COSMO_wind_capacity_proportional (MWh)": bus_info['Proportional_wind_COSMO'].item(),
                    "daily_avg_load_last_7_days_recent_to_old (MW)": past_load_data.values.round(2).tolist(),
                }
                
                G.add_node(
                    new_node_idx, type = 'bus', 
                    **feautres
                )
                
            elif node_type == 'generator':
                generator_idx = raw_node_idx - self.node_slices['generator'][0]
                generator_info = self.generator_info.iloc[generator_idx]
                # print(f"{generator_info=}")
                feautres = {
                    "capacity (MW)": generator_info['capacity'].item(),
                    "marginal_cost ($/MWh)": generator_info['lincost'].item(),
                    "cycle_cost ($)": generator_info['cyclecost'].item(),
                    "minimal_up_time (hours)": generator_info['minuptime'].item(),
                    "minimal_down_time (hours)": generator_info['mindowntime'].item(),
                    "minimal_production (MW)": generator_info['minonlinecapacity'].item(),
                }
                G.add_node(
                    new_node_idx, type = 'generator',
                    **feautres
                )

        for edge_idx in sub_graph_edge_mask.nonzero(as_tuple=True)[0]:
            
            raw_src, raw_dst = self.graph.edge_index.T[edge_idx]
            raw_src, raw_dst = raw_src.item(), raw_dst.item()
            
            src = node_mapping[raw_src]
            dst = node_mapping[raw_dst]
            
            # check edge_type
            edge_type = self.graph.edge_type[edge_idx].item()
            
            if edge_type == 0:
                edge_type = 'close_to'
                G.add_edge(src, dst, type="close_to")
            
            elif edge_type == 1:
                edge_type = 'transmission_line'
                
                # get original edge data
                raw_src, raw_dst = raw_src - self.node_slices['bus'][0], raw_dst - self.node_slices['bus'][0]
                # print(f"{raw_src=}, {raw_dst=}")
                # print(
                #     f"{self.network_edges.loc[(self.network_edges['fromNode'] == raw_src)]=}"
                # )
                edge_data = self.network_edges.loc[(self.network_edges['fromNode'] == raw_src) \
                    & (self.network_edges['toNode'] == raw_dst)]
                
                # print(f"{raw_src=}, {raw_dst=}")
                # print(f"{edge_data=}")
                
                feautres = {
                    "reactance_per_unit": edge_data['X'].values[0].item(),
                    "susceptance_per_unit": edge_data['Y'].values[0].item(),
                    # "thermal_limit (MW)": self.network_edges.at[edge_idx, 'limit'], # most are zeros
                    "great_circle_distance (km)": edge_data['length'].values[0].item()
                }
                
                G.add_edge(src, dst, type="transmission_line", **feautres)
            
        return G