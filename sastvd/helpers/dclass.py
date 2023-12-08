import json
from glob import glob
from pathlib import Path

import pandas as pd
import sastvd as svd
import sastvd.helpers.datasets as svdds
import sastvd.helpers.glove as svdglove


class BigVulDataset:
    """Represent BigVul as graph dataset."""

    def __init__(self, partition="train", vulonly=False, sample=-1, splits="default"):
        """Init class."""
        # Get finished samples
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(svd.processed_dir() / "bigvul/before/*nodes*"))
        ]
        # print(len(self.finished))
        self.df = svdds.bigvul(splits=splits)
        # print("1",self.df.keys())
        # print(self.df.tail(5))

        self.partition = partition
        self.df = self.df[self.df.label == partition]
        self.df = self.df[self.df.id.isin(self.finished)]
        # print("2",self.df.keys())
        # print(self.df.tail(5))

        # Balance training set
        if partition == "train":
            # print(self.df["vul"].head(100))
            vul = self.df[self.df.vul == 1]
            # print("vul :", vul.shape)
            # x = self.df[self.df.vul == 0]
            # print("nonvul :", x.shape)
            vul = vul.sample(1024, random_state=0)
            nonvul = self.df[self.df.vul == 0]
            nonvul = nonvul.sample(min(len(vul), len(nonvul)), random_state=0)
            # print("nonvul :", nonvul.head())
            self.df = pd.concat([vul, nonvul])
        if partition == "val":
            # vul = self.df[self.df.vul == 1].sample(128, random_state=0)
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]
            nonvul = nonvul.sample(min(len(vul), len(nonvul)), random_state=0)
            self.df = pd.concat([vul, nonvul])

        # Correct ratio for test set
        if partition == "test":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]
            nonvul = nonvul.sample(min(len(nonvul), len(vul) * 20), random_state=0)
            self.df = pd.concat([vul, nonvul])

        # print("3",self.df.keys())
        # print(self.df.head(5))

        # Small sample (for debugging):
        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)
        # print("3.1",self.df.keys())

        # Filter only vulnerable
        if vulonly:
            self.df = self.df[self.df.vul == 1]
        # print("3.2",self.df.keys())

        # Filter out samples with no lineNumber from Joern output
        self.df["valid"] = svd.dfmp(
            self.df, BigVulDataset.check_validity, 'id', desc="Validate Samples: "
        )
        # print("3.3",self.df.keys())
        # print(self.df["valid"].head(5))
        # print(self.df["valid"])
        self.df = self.df[self.df.valid]
        # print("4",self.df.shape)

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        # print("5",self.df.shape)

        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        # Load Glove vectors.
        glove_path = svd.processed_dir() / "bigvul/glove_False/vectors.txt"
        self.emb_dict, _ = svdglove.glove_dict(glove_path)
        print("finish")

    def itempath(_id):
        """Get itempath path from item id."""
        return svd.processed_dir() / f"bigvul/before/{_id}.c"

    def check_validity(_id):
        """Check whether sample with id=_id has node/edges.

        Example:
        _id = 1320
        with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
            nodes = json.load(f)
        """
        valid = 0
        try:
            with open(str(BigVulDataset.itempath(_id)) + ".nodes.json", "r") as f:
                # print(str(BigVulDataset.itempath(_id)) + ".nodes.json")
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n.keys():
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:
                            valid = 1
                            break
                if valid == 0:
                    return False
            with open(str(BigVulDataset.itempath(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception as E:
            print(E, str(BigVulDataset.itempath(_id)))
            return False

    def get_vuln_indices(self, _id):
        """Obtain vulnerable lines from sample ID."""
        df = self.df[self.df.id == _id]
        removed = df.removed.item()
        return dict([(i, 1) for i in removed])

    def stats(self):
        """Print dataset stats."""
        print(self.df.groupby(["label", "vul"]).count()[["id"]])

    def __getitem__(self, idx):
        """Must override."""
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __repr__(self):
        """Override representation."""
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)
        return f"BigVulDataset(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"
