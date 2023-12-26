from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, GCNConv, HGTConv, Linear, GatedGraphConv, AttentionalAggregation, SAGEConv

import torch.nn.functional as F
from src.datas.vocabulary import Vocabulary
from src.models.modules.common_layers import STEncoder


class SAGEEncoder(torch.nn.Module):

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(SAGEEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        # self.conv1 = SAGEConv((-1, -1), config.n_hidden_layers)
        # self.conv2 = SAGEConv((-1, -1), config.hidden_size)
        # self.convs = torch.nn.ModuleList()

        # self.input_GCL = HeteroConv({
        #         ('vul', 'c', 'vul'): GCNConv(-1, config.hidden_size, add_self_loops=False),
        #         ('vul', 'd', 'vul'): GCNConv(-1, config.hidden_size, add_self_loops=False),
        #         ('safe', 'c', 'safe'): GCNConv(-1, config.hidden_size, add_self_loops=False),
        #         ('safe', 'd', 'safe'): GCNConv(-1, config.hidden_size, add_self_loops=False),
        #         ('vul', 'c', 'safe'): GCNConv(-1, config.hidden_size, add_self_loops=False),
        #         ('vul', 'd', 'safe'): GCNConv(-1, config.hidden_size, add_self_loops=False),
        #         ('safe', 'c', 'vul'): GCNConv(-1, config.hidden_size, add_self_loops=False),
        #         ('safe', 'd', 'vul'): GCNConv(-1, config.hidden_size, add_self_loops=False),
        #     }, aggr='mean')
        self.input_GCL = HGTConv(config.rnn.hidden_size, config.hidden_size)
        #
        # self.input_GPL = TopKPooling(config.hidden_size,
        #                              ratio=config.pooling_ratio)
        #
        # for i in range(config.n_hidden_layers - 1):
        #     setattr(self, f"hidden_GCL{i}",
        #             SAGEConv(config.hidden_size, config.hidden_size))
        #     setattr(
        #         self, f"hidden_GPL{i}",
        #         TopKPooling(config.hidden_size,
        #                     ratio=config.pooling_ratio))
        self.lin = Linear(config.hidden_size, config.hidden_size)
        # self.attpool = AttentionalAggregation(torch.nn.Linear(config.hidden_size, 1))

    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden]
        # edge_index = batched_graph.edge_index
        # batch = batched_graph.batch
        # print(batched_graph.x_dict)
        # print(batched_graph.edge_index_dict)
        # print("batched_graph:", batched_graph)
        # # print("x_dict:", batched_graph.x_dict)
        # # print("edge_index_dict:", batched_graph.edge_index_dict.shape)
        # x_dict = self.input_GCL(batched_graph.x_dict, batched_graph.edge_index_dict)
        # # print(x_dict)
        # x_dict = {key: x.relu() for key, x in x_dict.items()}
        # x = self.conv1(batched_graph.x, edge_index).relu()
        # x = self.conv2(x, edge_index)
        # node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        # node_embedding, edge_index, _, batch, _, _ = self.input_GPL(node_embedding, edge_index, None,
        #                                                             batch)
        # [n_XFG; XFG hidden dim]
        # torch.use_deterministic_algorithms(False)
        # out = self.attpool(node_embedding, batch)
        # torch.use_deterministic_algorithms(True, warn_only=True)


        # for i in range(self.__config.n_hidden_layers - 1):
        #     node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index))
        #     node_embedding, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
        #         node_embedding, edge_index, None, batch)
        #     out += self.attpool(node_embedding, batch)
        # [n_XFG; XFG hidden dim]

        x_dict = batched_graph.x_dict
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.__st_embedding(x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, batched_graph.edge_index_dict)

        return self.lin(x_dict['author'])
        print("x_dict['vul'] shape: ", x_dict['vul'].shape)
        print("x_dict['safe'] shape: ", x_dict['safe'].shape)
        result = self.lin(x_dict['vul'])
        print("result shape: ", result.shape)
        return result
