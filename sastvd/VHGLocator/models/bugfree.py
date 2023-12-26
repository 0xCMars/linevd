import torch
from pytorch_lightning import LightningModule
from src.models.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder
from src.models.modules.rgcn import RelationConvEncoder
from src.models.modules.rgat import RGATConvEncoder
from src.datas.vocabulary import Vocabulary
from torch import nn
from torch_geometric.data import Batch
from torch.optim import Adam, SGD, Adamax, RMSprop
from typing import Dict
from src.datas.samples import XFGBatch
import torch.nn.functional as F
from src.metrics import Statistic

class BugFree(LightningModule):
    _encoders = {
        "gcn": GraphConvEncoder,
        "ggnn": GatedGraphConvEncoder,
        "rgcn": RelationConvEncoder,
        "rgat": RGATConvEncoder
    }

    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int):
        super().__init__()
        self.save_hyperparameters()
        self.__config = config
        hidden_size = config.classifier.hidden_size
        self.__graph_encoder = self._encoders[config.gnn.name](config.gnn, vocab, vocabulary_size, pad_idx)

        layers = [
            nn.Linear(config.gnn.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier.drop_out)
        ]

        if config.classifier.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({config.classifier.n_hidden_layers})")

        for _ in range(config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.classifier.drop_out)
            ]
        self.__hidden_layers = nn.Sequential(*layers)
        self.__classifier = nn.Linear(hidden_size, config.classifier.n_classes)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch: Batch) -> torch.Tensor:
        """

        Args:
            batch:

        Returns:

        """
        # [n_XFG, hidden size]
        graph_hid = self.__graph_encoder(batch)
        # print("graph:", graph_hid.shape)
        hiddens = self.__hidden_layers(graph_hid)
        # print("hidder:", hiddens.shape)
        # [n_XFG; n_classes]
        return self.__classifier(hiddens)

    def _get_optimizer(self, name: str) -> torch.nn.Module:
        if name in self._optimizers:
            return self._optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")

    def configure_optimizers(self) -> Dict:
        parameters = [self.parameters()]
        optimizer = self._get_optimizer(
            self.__config.hyper_parameters.optimizer)(
            [{
                "params": p
            } for p in parameters],
            self.__config.hyper_parameters.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: self.__config.hyper_parameters.decay_gamma
                                    ** epoch
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _log_training_step(self, results: Dict):
        self.log_dict(results, on_step=True, on_epoch=False)

    def training_step(self, batch: XFGBatch,
                      batch_idx: int) -> torch.Tensor:
        logits = self(batch.graphs)
        # print("logits:", logits)
        # print("batch label: ", batch.labels)
        labels = batch.labels.to(torch.int64)

        # torch.autograd.set_detect_anomaly(True)
        loss = F.cross_entropy(logits, labels)

        result: Dict = {"train_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )

            batch_metric = statistic.calculate_metrics(group="train")
            result.update(batch_metric)
            self._log_training_step(result)
            self.log("F1",
                     batch_metric["train_f1"],
                     prog_bar=True,
                     logger=False)
        pred = {"loss": loss, "statistic": statistic}
        self.training_step_outputs.append(pred)
        return pred

    def validation_step(self, batch: XFGBatch,
                        batch_idx: int) -> torch.Tensor:
        logits = self(batch.graphs)
        # print("logits:", logits.shape)
        # print("batch label: ", batch.labels.shape)
        labels = batch.labels.to(torch.int64)
        # print("type", batch.labels)

        # torch.autograd.set_detect_anomaly(True)
        loss = F.cross_entropy(logits, labels)

        result: Dict = {"val_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="val")
            result.update(batch_metric)
        pred = {"loss": loss, "statistic": statistic}
        self.validation_step_outputs.append(pred)
        return pred

    def test_step(self, batch: XFGBatch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        torch.autograd.set_detect_anomaly(True)
        labels = batch.labels.to(torch.int64)

        loss = F.cross_entropy(logits, labels)

        result: Dict = {"test_loss", loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="test")
            result.update(batch_metric)
        pred = {"loss": loss, "statistic": statistic}
        self.test_step_outputs.append(pred)
        return pred

    def _prepare_epoch_end_log(self, step_outputs,
                               step: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
        return {f"{step}_loss": mean_loss}

    def _shared_epoch_end(self, step_outputs, group: str):
        log = self._prepare_epoch_end_log(step_outputs, group)
        statistic = Statistic.union_statistics(
            [out["statistic"] for out in step_outputs])
        log.update(statistic.calculate_metrics(group))
        self.log_dict(log, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        training_step_output = self.training_step_outputs
        self._shared_epoch_end(training_step_output, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        validation_step_output = self.validation_step_outputs
        self._shared_epoch_end(validation_step_output, "val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        test_step_output = self.test_step_outputs
        self._shared_epoch_end(test_step_output, "test")
        self.test_step_outputs.clear()
