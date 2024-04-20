import os
from glob import glob

import dgl
import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.codebert as cb
import sastvd.helpers.dclass as svddc
import sastvd.helpers.doc2vec as svdd2v
import sastvd.helpers.glove as svdg
import sastvd.helpers.joern as svdj
import sastvd.helpers.losses as svdloss
import sastvd.helpers.ml as ml
import sastvd.helpers.rank_eval as svdr
import sastvd.helpers.sast as sast
import sastvd.ivdetect.evaluate as ivde
import sastvd.linevd.gnnexplainer as lvdgne
import torch as th
import torch.nn.functional as F
import torchmetrics
import sastvd.VHGLocator.helpers.InputDataset as idataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, GraphConv
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from tqdm import tqdm

def ne_groupnodes(n, e):
    """Group nodes with same line number."""
    nl = n[n.lineNumber != ""].copy()
    nl.lineNumber = nl.lineNumber.astype(int)
    nl = nl.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    nl = nl.groupby("lineNumber").head(1)
    el = e.copy()
    el.innode = el.line_in
    el.outnode = el.line_out
    nl.id = nl.lineNumber
    nl = svdj.drop_lone_nodes(nl, el)
    el = el.drop_duplicates(subset=["innode", "outnode", "etype"])
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    return nl, el

def feature_extraction(_id, graph_type="cfgcdg", return_nodes=False):
    """Extract graph feature (basic).

    _id = svddc.BigVulDataset.itempath(177775)
    _id = svddc.BigVulDataset.itempath(180189)
    _id = svddc.BigVulDataset.itempath(178958)

    return_nodes arg is used to get the node information (for empirical evaluation).
    """
    # Get CPG
    n, e = svdj.get_node_edges(_id)
    n, e = ne_groupnodes(n, e)

    # Return node metadata
    if return_nodes:
        return n

    # Filter nodes
    e = svdj.rdg(e, graph_type.split("+")[0])
    n = svdj.drop_lone_nodes(n, e)

    # Plot graph
    # svdj.plot_graph_node_edge_df(n, e)

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index()
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # Map edge types
    etypes = e.etype.tolist()
    d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    etypes = [d[i] for i in etypes]

    # Append function name to code
    if "+raw" not in graph_type:
        try:
            func_name = n[n.lineNumber == 1].name.item()
        except:
            print(_id)
            func_name = ""
        n.code = func_name + " " + n.name + " " + "</s>" + " " + n.code
    else:
        n.code = "</s>" + " " + n.code

    # Return plain-text code, line number list, innodes, outnodes
    return n.code.tolist(), n.id.tolist(), e.innode.tolist(), e.outnode.tolist(), etypes


class VHGLocatorDataset(idataset.InputDatsset):
    """IVDetect version of BigVul."""

    def __init__(self, gtype="pdg", feat="all", **kwargs):
        """Init."""
        super(VHGLocatorDataset, self).__init__(**kwargs)
        print("VHGLocatorDataset process")
        # lines = ivde.get_dep_add_lines_bigvul()
        # lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
        self.lines = []
        # print(len(self.lines))
        self.graph_type = gtype
        # use the alreay train model
        glove_path = svd.processed_dir() / "bigvul/glove_False/vectors.txt"
        self.glove_dict, _ = svdg.glove_dict(glove_path)
        self.d2v = svdd2v.D2V(svd.processed_dir() / "bigvul/d2v_False")
        self.feat = feat

    def item(self, _id, codebert=None):
        """Cache item."""
        savedir = svd.get_dir(
            svd.cache_dir() / f"test_case_VHGLocatore_codebert_{self.graph_type}"
        ) / str(_id)
        if os.path.exists(savedir):
            g = load_graphs(str(savedir))[0][0]
            # print(g)
            # g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
            # if "_SASTRATS" in g.ndata:
            #     g.ndata.pop("_SASTRATS")
            #     g.ndata.pop("_SASTCPP")
            #     g.ndata.pop("_SASTFF")
            #     g.ndata.pop("_GLOVE")
            #     g.ndata.pop("_DOC2VEC")
            if "_CODEBERT" in g.ndata:
                if self.feat == "codebert":
                    for i in ["_GLOVE", "_DOC2VEC", "_RANDFEAT"]:
                        g.ndata.pop(i, None)
                if self.feat == "glove":
                    for i in ["_CODEBERT", "_DOC2VEC", "_RANDFEAT"]:
                        g.ndata.pop(i, None)
                if self.feat == "doc2vec":
                    for i in ["_CODEBERT", "_GLOVE", "_RANDFEAT"]:
                        g.ndata.pop(i, None)
                return g
        code, lineno, ei, eo, et = feature_extraction(
            idataset.InputDatsset.itempath(_id), self.graph_type
        )
        if _id in self.lines:
            vuln = [1 if i in self.lines[_id] else 0 for i in lineno]
        else:
            vuln = [0 for _ in lineno]
        g = dgl.graph((eo, ei))
        gembeds = th.Tensor(svdg.get_embeddings_list(code, self.glove_dict, 200))
        g.ndata["_GLOVE"] = gembeds
        g.ndata["_DOC2VEC"] = th.Tensor([self.d2v.infer(i) for i in code])
        if codebert:
            code = [c.replace("\\t", "").replace("\\n", "") for c in code]
            chunked_batches = svd.chunks(code, 128)
            features = [codebert.encode(c).detach().cpu() for c in chunked_batches]
            g.ndata["_CODEBERT"] = th.cat(features)
            # print("g CODEBERT" + str(g.ndata["_CODEBERT"]))
        g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
        g.ndata["_LINE"] = th.Tensor(lineno).int()
        g.ndata["_VULN"] = th.Tensor(vuln).float()
        node_type = [1 for i in lineno]
        g.ndata["_NTYPE"] = th.Tensor(node_type).long()

        # Get SAST labels
        # s = sast.get_sast_lines(svd.processed_dir() / f"test_case/before/{_id}.c.sast.pkl")
        # rats = [1 if i in s["rats"] else 0 for i in g.ndata["_LINE"]]
        # cppcheck = [1 if i in s["cppcheck"] else 0 for i in g.ndata["_LINE"]]
        # flawfinder = [1 if i in s["flawfinder"] else 0 for i in g.ndata["_LINE"]]
        # g.ndata["_SASTRATS"] = th.tensor(rats).long()
        # g.ndata["_SASTCPP"] = th.tensor(cppcheck).long()
        # g.ndata["_SASTFF"] = th.tensor(flawfinder).long()

        g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
        g.edata["_ETYPE"] = th.Tensor(et).long()

        emb_path = svd.cache_dir() / f"input_codebert_method_level/{_id}.pt"
        g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
        g = dgl.add_self_loop(g)
        save_graphs(str(savedir), [g])
        return g

    def cache_items(self, codebert):
        """Cache all items."""
        desc = "cache items:"
        for i in tqdm(self.df.sample(len(self.df)).id.tolist(), desc=desc):
            try:
                self.item(i, codebert)
            except Exception as E:
                print(E)

    def cache_codebert_method_level(self, codebert):
        """Cache method-level embeddings using Codebert.

        ONLY NEEDS TO BE RUN ONCE.
        """
        savedir = svd.get_dir(svd.cache_dir() / "input_codebert_method_level")
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        done = set(done)
        batches = svd.chunks((range(len(self.df))), 128)
        for idx_batch in tqdm(batches):
            batch_texts = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].before.tolist()
            batch_ids = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].id.tolist()
            if set(batch_ids).issubset(done):
                continue
            texts = ["</s> " + ct for ct in batch_texts]
            embedded = codebert.encode(texts).detach().cpu()
            assert len(batch_texts) == len(batch_ids)
            for i in range(len(batch_texts)):
                th.save(embedded[i], savedir / f"{batch_ids[i]}.pt")

    def __getitem__(self, idx):
        """Override getitem."""
        return self.item(self.idx2id[idx])

class VHGLocatorDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for InputData."""

    def __init__(
        self,
        batch_size: int = 32,
        sample: int = -1,
        methodlevel: bool = False,
        nsampling: bool = False,
        nsampling_hops: int = 1,
        gtype: str = "cfgcdg",
        splits: str = "default",
        feat: str = "all",
    ):
        """Init class from VHGLocator dataset."""
        super().__init__()
        self.dataargs = {"sample": sample, "gtype": gtype, "splits": splits, "feat": feat}
        # print("VHGLocatorDataModule Train")
        # self.train = BigVulDatasetLineVD(partition="train", **dataargs)
        # print("VHGLocatorDataModule val")
        # self.val = BigVulDatasetLineVD(partition="val", **dataargs)
        # print("VHGLocatorDataModule test")
        # self.test = BigVulDatasetLineVD(partition="test", **dataargs)
        self.codebert = cb.CodeBert()
        # self.train.cache_codebert_method_level(codebert)
        # self.val.cache_codebert_method_level(codebert)
        # self.test.cache_codebert_method_level(codebert)
        # self.train.cache_items(codebert)
        # self.val.cache_items(codebert)
        # self.test.cache_items(codebert)
        self.batch_size = batch_size
        # print(self.batch_size)
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops

    def node_dl(self, g, shuffle=False):
        """Return node dataloader."""
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nsampling_hops)
        # print("node_dl:", self.batch_size)
        return dgl.dataloading.DataLoader(
            g,
            g.nodes(),
            sampler,
            device="cuda",
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            # pin_memory=True,
            num_workers=2,
        )

    def train_dataloader(self):
        """Return train dataloader."""
        print("VHGLocatorDataModule Train")
        train = VHGLocatorDataset(partition="train", **self.dataargs)
        train.cache_codebert_method_level(self.codebert)
        train.cache_items(self.codebert)
        # print(len(train))
        if self.nsampling:
            g = next(iter(GraphDataLoader(train, batch_size=len(train))))
            return self.node_dl(g, shuffle=True)
        return GraphDataLoader(train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return val dataloader."""
        print("VHGLocatorDataModule val")
        val = VHGLocatorDataset(partition="val", **self.dataargs)
        val.cache_codebert_method_level(self.codebert)
        val.cache_items(self.codebert)
        if self.nsampling:
            g = next(iter(GraphDataLoader(val, batch_size=len(val))))
            return self.node_dl(g)
        return GraphDataLoader(val, pin_memory=True, batch_size=self.batch_size)

    def val_graph_dataloader(self):
        """Return test dataloader."""
        # print("VHGLocatorDataModule val")
        val = VHGLocatorDataset(partition="val", **self.dataargs)
        val.cache_codebert_method_level(self.codebert)
        val.cache_items(self.codebert)

        return GraphDataLoader(val, batch_size=32, pin_memory=True)

    def test_dataloader(self):
        """Return test dataloader."""
        print("VHGLocatorDataModule test")
        test = VHGLocatorDataset(partition="test", **self.dataargs)
        test.cache_codebert_method_level(self.codebert)
        test.cache_items(self.codebert)
        return GraphDataLoader(test, batch_size=1, pin_memory=True)
