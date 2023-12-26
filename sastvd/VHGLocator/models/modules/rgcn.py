from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, RGCNConv, AttentionalAggregation

import torch.nn.functional as F
from src.datas.vocabulary import Vocabulary
from src.models.modules.common_layers import STEncoder

class RelationConvEncoder(torch.nn.Module):
    """

    Schlichtkrull and Kipf: Modeling Relational Data with Graph Convolutional Networks
    (https://arxiv.org/abs/1703.06103)

    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(RelationConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = RGCNConv(in_channels=config.rnn.hidden_size,
                                  out_channels=config.hidden_size, num_relations=config.rgcn.num_rel)
        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)

        for i in range(config.n_hidden_layers - 1):
            setattr(self, f"hidden_GCL{i}",
                    RGCNConv(in_channels=config.hidden_size,
                             out_channels=config.hidden_size, num_relations=config.rgcn.num_rel))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(config.hidden_size,
                            ratio=config.pooling_ratio))

        self.attpool = AttentionalAggregation(torch.nn.Linear(config.hidden_size, 1))

    def forward(self, batched_graph: Batch):
        # print("batched_g:", batched_graph.x.shape)
        node_embedding = self.__st_embedding(batched_graph.x)
        # print("node_embedding 1:", node_embedding.shape)

        edge_index = batched_graph.edge_index
        edge_type = batched_graph.edge_type
        batch = batched_graph.batch
        # print("batch:", node_embedding.size())

        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index, edge_type))
        # print("input1:", node_embedding.size(), edge_index.size(), edge_type.size())
        # node_embedding, edge_index, edge_type, batch, _, _ = self.input_GPL(node_embedding, edge_index, edge_type,
        #                                                             batch)
        # print("node_embedding 2:", node_embedding.shape)

        # print("input2:", node_embedding.size(), edge_index.size(), edge_type.size())
        # torch.use_deterministic_algorithms(False)
        # out = self.attpool(node_embedding, batch)
        out = node_embedding
        # torch.use_deterministic_algorithms(True, warn_only=True)

        for i in range(self.__config.n_hidden_layers - 1):
            node_embedding = getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index, edge_type)
            # print(f"hidden_GCL{i}", node_embedding)
            node_embedding = F.relu(node_embedding)
            # node_embedding, edge_index, edge_type, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
            #     node_embedding, edge_index, edge_type, batch)
            # print(f"node_embedding {i}:", node_embedding.shape)

            out += node_embedding
        return out

# class RGCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2, dropout=0.5):
#         super().__init__()
#         self.convs = torch.nn.ModuleList()
#         self.relu = F.relu
#         self.dropout = dropout
#         self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
#         for i in range(n_layers - 2):
#             self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
#         self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))
#
#     def forward(self, x, edge_index, edge_type):
#         for conv, norm in zip(self.convs, self.norms):
#             x = norm(conv(x, edge_index, edge_type))
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training
#         return x
