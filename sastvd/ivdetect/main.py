"""Implementation of IVDetect."""
import pickle as pkl
from importlib import reload

import dgl
import sastvd as svd
import sastvd.helpers.ml as ml
import sastvd.helpers.rank_eval as svdr
import sastvd.ivdetect.evaluate as ivde
import sastvd.ivdetect.gnnexplainer as ge
import sastvd.ivdetect.helpers as ivd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from warnings import filterwarnings
import torch.nn.functional as F

def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore",
                   category=FutureWarning)
filter_warnings()
# Load data
reload(ivd)
# train_ds = ivd.BigVulDatasetIVDetect(partition="train")
# val_ds = ivd.BigVulDatasetIVDetect(partition="val")
dl_args = {"drop_last": False, "shuffle": True, "num_workers": 0}
# train_dl = GraphDataLoader(train_ds, batch_size=16, **dl_args)
# val_dl = GraphDataLoader(val_ds, batch_size=16, **dl_args)

# %% Create model
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
svd.debug(dev)
model = ivd.IVDetect(200, 64)
model.to(dev)

# Debugging a single sample
# batch = next(iter(train_dl))
# print(batch)
# batch = batch.to(dev)
# logits = model(batch, train_ds)

# # %% Optimiser
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train loop
# ID = svd.get_run_id({})
# ID = "202312081151_2903285_line_model"
# ID = "202401041707_f1c089f_use_the_new_model_to_test"
ID = "202401091344_3dfcc11_new_ivdetect"

logger = ml.LogWriter(
    model, svd.processed_dir() / "ivdetect" / ID, max_patience=50, val_every=30
)
logger.load_logger()
# while True:
#     for batch in train_dl:
#
#         # Training
#         model.train()
#         batch = batch.to(dev)
#         logits = model(batch, train_ds)
#         labels = batch.ndata["_VULN"].long()
#         # labels = dgl.max_nodes(batch, "_VULN").long()
#         # print(len(logits))
#         # print(len(labels))
#         loss = F.cross_entropy(logits, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # Evaluation
#         train_mets = ml.get_metrics_logits(labels, logits)
#         val_mets = train_mets
#         if logger.log_val():
#             model.eval()
#             with torch.no_grad():
#                 all_pred = torch.empty((0, 2)).long().to(dev)
#                 all_true = torch.empty((0)).long().to(dev)
#                 for val_batch in val_dl:
#                     val_batch = val_batch.to(dev)
#                     # val_labels = dgl.max_nodes(val_batch, "_VULN").long()
#                     val_labels = val_batch.ndata["_VULN"].long()
#
#                     val_logits = model(val_batch, val_ds)
#                     # print(len(val_labels))
#                     # print(len(val_logits))
#                     all_pred = torch.cat([all_pred, val_logits])
#                     all_true = torch.cat([all_true, val_labels])
#                 val_mets = ml.get_metrics_logits(all_true, all_pred)
#         logger.log(train_mets, val_mets)
#         logger.save_logger()
#
#     # Early Stopping
#     if logger.stop():
#         break
#     logger.epoch()


test_ds = ivd.BigVulDatasetIVDetect(partition="test")
test_dl = GraphDataLoader(test_ds, batch_size=16, **dl_args)

# Print test results
logger.load_best_model()
model.eval()
all_pred = torch.empty((0, 2)).long().to(dev)
all_true = torch.empty((0)).long().to(dev)
with torch.no_grad():
    for test_batch in test_dl:
        test_batch = test_batch.to(dev)
        # test_labels = dgl.max_nodes(test_batch, "_VULN").long()
        test_labels = test_batch.ndata["_VULN"].long()
        # print("test label", test_labels)
        test_logits = model(test_batch, test_ds)
        all_pred = torch.cat([all_pred, test_logits])
        all_true = torch.cat([all_true, test_labels])
        test_mets = ml.get_metrics_logits(all_true, all_pred)
        logger.test(test_mets)
logger.test(test_mets)
all_pred = F.softmax(all_pred, dim=1).argmax(1)
all_pred = all_pred.detach().cpu().numpy()
all_true = all_true.detach().cpu().numpy()
print(all_pred)
print(all_true)
rank_metr_test = ml.met_dict_to_str(svdr.rank_metr(all_pred, all_true))

# %% Statement-level through GNNExplainer
# correct_lines = ivde.get_dep_add_lines_bigvul()
# pred_lines = dict()
# for batch in test_dl:
#     for g in dgl.unbatch(batch):
#         sampleid = g.ndata["_SAMPLE"].max().int().item()
#         if sampleid not in correct_lines:
#             continue
#         if sampleid in pred_lines:
#             continue
#         try:
#             lines = ge.gnnexplainer(model, g.to(dev), test_ds)
#         except Exception as E:
#             print(E)
#         pred_lines[sampleid] = lines

# with open(svd.cache_dir() / "pred_lines.pkl", "wb") as f:
#     pkl.dump(pred_lines, f)
#
# MFR = []
# for sampleid, pred in pred_lines.items():
#     true = correct_lines[sampleid]
#     true = list(true["removed"]) + list(true["depadd"])
#     for i, p in enumerate(pred):
#         if p in true:
#             MFR += [i]
#             break
# print(sum(MFR) / len(MFR))
