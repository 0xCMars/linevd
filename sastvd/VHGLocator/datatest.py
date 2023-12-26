import sastvd.linevd as lvd

config = {
    "hfeat": 512,
    "embtype": "codebert",
    "stmtweight": 1,
    "hdropout": 0.3,
    "gatdropout": 0.2,
    "modeltype": "rgcn",
    "gnntype": "gat",
    "loss": "ce",
    "scea": 0.5,
    "gtype": "rel+raw",
    "batch_size": 1024,
    "multitask": "linemethod",
    "splits": "default",
    "lr": 1e-4,
}
gtype=config["gtype"]
splits=config["splits"]
feat = "all"
dataargs = {"sample": -1, "gtype": gtype, "splits": splits, "feat": feat}

test = lvd.BigVulDatasetLineVD(partition="train", **dataargs)
a = test.item(1)
print(len(a.edata["_ETYPE"]))
print(a)