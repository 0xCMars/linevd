from pytorch_lightning import LightningModule
from sastvd.VHGLocator.models.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder
from sastvd.VHGLocator.models.modules.rgcn import RelationConvEncoder
from torch.optim import Adam, Adamax
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, AUROC, MatthewsCorrCoef
import torch
from dgl.nn.pytorch import RelGraphConv
import torch.nn.functional as F
import dgl
import sastvd.helpers.ml as ml
import sastvd.ivdetect.evaluate as ivde
import sastvd.helpers.rank_eval as svdr

class VHGLocator(LightningModule):

    def __init__(
            self,
            hfeat: int = 512,
            embtype: str = "codebert",
            embfeat: int = -1,  # Keep for legacy purposes
            num_heads: int = 4,
            lr: float = 1e-3,
            hdropout: float = 0.2,
            mlpdropout: float = 0.2,
            dropout: float = 0.2,
            methodlevel: bool = False,
            nsampling: bool = False,
            model: str = "rgcn",
            loss: str = "ce",
            multitask: str = "linemethod",
            stmtweight: int = 5,
            gnntype: str = "gat",
            random: bool = False,
            scea: float = 0.7,
    ):
        super().__init__()
        self.lr = lr
        self.random = random
        self.save_hyperparameters()
        print(self.hparams)

        if self.hparams.embtype == "codebert":
            self.hparams.embfeat = 768
            self.EMBED = "_CODEBERT"
        if self.hparams.embtype == "glove":
            self.hparams.embfeat = 200
            self.EMBED = "_GLOVE"
        if self.hparams.embtype == "doc2vec":
            self.hparams.embfeat = 300
            self.EMBED = "_DOC2VEC"

        self.loss = CrossEntropyLoss(
            weight=torch.Tensor([1, self.hparams.stmtweight]).cuda()
        )
        self.accuracy = Accuracy()
        self.auroc = AUROC(compute_on_step=False)
        self.mcc = MatthewsCorrCoef(2)

        hfeat = self.hparams.hfeat
        drop = self.hparams.dropout
        embfeat = self.hparams.embfeat
        gnn_args = {"out_feats": hfeat}

        gnn = RelGraphConv
        gnn1_args = {"in_feats": embfeat, **gnn_args}
        gnn2_args = {"in_feats": hfeat, **gnn_args}
        self.gcl = gnn(**gnn1_args)
        self.gcl2 = gnn(**gnn2_args)
        self.fc = torch.nn.Linear(hfeat, self.hparams.hfeat)
        self.fconly = torch.nn.Linear(embfeat, self.hparams.hfeat)
        self.mlpdropout = torch.nn.Dropout(self.hparams.mlpdropout)

        self.codebertfc = torch.nn.Linear(768, self.hparams.hfeat)
        self.fch = []
        for _ in range(8):
            self.fch.append(torch.nn.Linear(self.hparams.hfeat, self.hparams.hfeat))
        self.hidden = torch.nn.ModuleList(self.fch)
        self.hdropout = torch.nn.Dropout(self.hparams.hdropout)
        self.fc2 = torch.nn.Linear(self.hparams.hfeat, 2)

    def forward(self, g, test=False):

        if self.hparams.nsampling and not test:
            hdst = g[2][-1].dstdata[self.EMBED]
            h_func = g[2][-1].dstdata["_FUNC_EMB"]
            g2 = g[2][1]
            g = g[2][0]

            h = g.srcdata[self.EMBED]

        if self.random:
            return torch.rand((h.shape[0], 2)).to(self.device), torch.rand(
                h_func.shape[0], 2
            ).to(self.device)

        # if self.hparams.embfeat != 768:
        #     h_func = self.codebertfc(h_func)

        h = self.gcl(g, h)
        h = self.gcl2(g2, h)
        h = self.mlpdropout(F.elu(self.fc(h)))
        # h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        for idx, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h)))
            # h_func = self.hdropout(F.elu(hlayer(h_func)))
        h = self.fc2(h)
        # h_func = self.fc2(
        #     h_func
        # )  # Share weights between method-level and statement-level tasks

        return h

    def shared_step(self, batch, test=False):
        logits = self(batch, test)
        if self.hparams.nsampling and not test:
            labels = batch[2][-1].dstdata["_VULN"].long()
            # labels_func = batch[2][-1].dstdata["_FVULN"].long()
        else:
            labels = batch.ndata["_VULN"].long()

        return logits, labels

    def training_step(self, batch, batch_idx):
        logit, labels = self.shared_step(batch)
        loss = self.loss(logit, labels)

        pred = F.softmax(logit, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        mcc = self.mcc(pred.argmax(1), labels)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        self.log("train_mcc", mcc, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logit, labels = self.shared_step(batch)
        loss = self.loss(logit, labels)
        pred = F.softmax(logit, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        mcc = self.mcc(pred.argmax(1), labels)

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.auroc.update(logit[:, 1], labels)
        self.log("val_auroc", self.auroc, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits, labels  = self.shared_step(
            batch, True
        )

        batch.ndata["pred"] = F.softmax(logits, dim=1)
        # batch.ndata["pred_func"] = F.softmax(logits[1], dim=1)
        preds = []

        for i in dgl.unbatch(batch):
            preds.append(
                [
                    list(i.ndata["pred"].detach().cpu().numpy()),
                    list(i.ndata["_VULN"].detach().cpu().numpy()),
                    list(i.ndata["_LINE"].detach().cpu().numpy()),
                ]
            )
        return logits, labels, preds

    def test_epoch_end(self, outputs):
        all_pred = torch.empty((0, 2)).long().cuda()
        all_true = torch.empty((0)).long().cuda()
        all_funcs = []

        from importlib import reload
        reload(ml)

        for out in outputs:
            # logits[0]
            all_pred = torch.cat([all_pred, out[0].cuda()])
            # label
            all_true = torch.cat([all_true, out[1].cuda()])
            all_funcs += out[2]
        all_pred = F.softmax(all_pred, dim=1)
        self.all_funcs = all_funcs
        self.all_true = all_true
        self.all_pred = all_pred

        # Custom ranked accuracy (inc negatives)
        self.res1 = ivde.eval_statements_list(all_funcs)
        # Custom ranked accuracy (only positives)
        self.res1vo = ivde.eval_statements_list(all_funcs, vo=True, thresh=0)

        # Regular metrics
        self.res2 = ml.get_metrics_logits(all_true, all_pred)

        # Ranked metrics
        rank_metrs = []
        rank_metrs_vo = []
        for af in all_funcs:
            rank_metr_calc = svdr.rank_metr([i[1] for i in af[0]], af[1], 0)
            if max(af[1]) > 0:
                rank_metrs_vo.append(rank_metr_calc)
            rank_metrs.append(rank_metr_calc)
        try:
            self.res3 = ml.dict_mean(rank_metrs)
        except Exception as E:
            print(E)
            pass
        self.res3vo = ml.dict_mean(rank_metrs_vo)

        return

    def configure_optimizers(self):
        """Configure optimizer."""
        return Adam(self.parameters(), lr=self.lr)
