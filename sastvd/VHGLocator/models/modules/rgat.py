from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, RGATConv, AttentionalAggregation

import torch.nn.functional as F
from src.datas.vocabulary import Vocabulary
from src.models.modules.common_layers import STEncoder

class RGATConvEncoder(torch.nn.Module):
    """

    Schlichtkrull and Kipf: Modeling Relational Data with Graph Convolutional Networks
    (https://arxiv.org/abs/1703.06103)

    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(RGATConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = RGATConv(in_channels=config.rnn.hidden_size,
                                  out_channels=config.hidden_size, num_relations=config.rgcn.num_rel)
        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)

        for i in range(config.n_hidden_layers - 1):
            setattr(self, f"hidden_GCL{i}",
                    RGATConv(in_channels=config.hidden_size,
                             out_channels=config.hidden_size, num_relations=config.rgcn.num_rel))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(config.hidden_size,
                            ratio=config.pooling_ratio))

        self.attpool = AttentionalAggregation(torch.nn.Linear(config.hidden_size, 1))

    def forward(self, batched_graph: Batch):
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        edge_type = batched_graph.edge_type
        batch = batched_graph.batch
        # print("batch:", node_embedding.size())

        torch.use_deterministic_algorithms(False)
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index, edge_type))
        torch.use_deterministic_algorithms(True, warn_only=True)
        # print("input1:", node_embedding.size(), edge_index.size(), edge_type.size())
        node_embedding, edge_index, edge_type, batch, _, _ = self.input_GPL(node_embedding, edge_index, edge_type,
                                                                    batch)
        # print("input2:", node_embedding.size(), edge_index.size(), edge_type.size())
        torch.use_deterministic_algorithms(False)
        out = self.attpool(node_embedding, batch)
        torch.use_deterministic_algorithms(True, warn_only=True)

        for i in range(self.__config.n_hidden_layers - 1):
            node_embedding = getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index, edge_type)
            # print(f"hidden_GCL{i}", node_embedding)
            node_embedding = F.relu(node_embedding)
            node_embedding, edge_index, edge_type, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, edge_type, batch)
            out += self.attpool(node_embedding, batch)
        return out