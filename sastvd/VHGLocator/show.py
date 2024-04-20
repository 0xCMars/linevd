import re
from pycparser import c_ast, parse_file
import pandas as pd
import os
import pickle as pkl
import sastvd.VHGLocator.helpers.datasets as dshelper
import sastvd.helpers.datasets as svdd
import sastvd as svd
import sastvd.helpers.tokenise as svdt
import sastvd.helpers.glove as svdglove
import sastvd.helpers.joern as svdj
import sastvd.helpers.sast as sast
import sastvd.VHGLocator.helpers.VHGLocatorInputModule as vhgModule
import sastvd.VHGLocator as vhg
import torch
from pytorch_lightning import Trainer

# 获取文件的函数划分，写入c文件
filepath = "./storage/data/gcc/source-code/iaopre.c"
filepath2 = "./storage/data/gcc/source-code/mlf.c"
filepath3 = "./storage/data/gcc/source-code/bigvul1.c"
files = [filepath, filepath2, filepath3]

headers = ['Access Gained', 'Attack Origin', 'Authentication Required', 'Availability', 'CVE ID', 'CVE Page', 'CWE ID', 'Complexity', 'Confidentiality', 'Integrity', 'Known Exploits', 'Publish Date', 'Score', 'Summary', 'Update Date', 'Vulnerability Classification', 'add_lines', 'codeLink', 'commit_id', 'commit_message', 'del_lines', 'file_name', 'files_changed', 'func_after', 'func_before', 'lang', 'lines_after', 'lines_before', 'parentID', 'patch', 'project', 'project_after', 'project_before', 'vul', 'vul_func_with_fix']

target = "./storage/external/test_case.csv"
# bigvul = pd.read_csv('./storage/external/MSR_data_cleaned.csv')
# print(bigvul)
# 将文件转为csv格式
# 新建创建表头
# with open('./storage/external/test_case.csv', 'w') as file:
#     # 创建 writer 对象
#     writer = csv.writer(file)
#     # 写入表头
#     writer.writerow(headers)
data = []
for file in files:

    with open(file, "r") as file:
        lines = file.readlines()

        info = {}
        for head in headers:
            info[head] = " "
        info['func_before'] = ''.join(lines)

        data.append(info)
    file.close()
code = pd.DataFrame(data)
code.to_csv(target, header=headers)

# 读取csv格式中的代码，构造类似BigVul的datafram，格式为parquet
df = dshelper.InputData(minimal=False)
# print(df['func_before'])

# 执行joern，svdj.full_run_joern获得xxx.edgs.json

# def preprocess(row):
#     """Parallelise svdj functions.
#
#     Example:
#     df = svdd.bigvul()
#     row = df.iloc[180189]  # PAPER EXAMPLE
#     row = df.iloc[177860]  # EDGE CASE 1
#     preprocess(row)
#     """
#     # print("process", row["id"])
#     savedir_before = svd.get_dir(svd.processed_dir() / row["dataset"] / "before")
#     savedir_after = svd.get_dir(svd.processed_dir() / row["dataset"] / "after")
#     # print(row)
#     # Write C Files
#     fpath1 = savedir_before / f"{row['id']}.c"
#     with open(fpath1, "w") as f:
#         f.write(row["before"])
#     fpath2 = savedir_after / f"{row['id']}.c"
#     if len(row["diff"]) > 0:
#         with open(fpath2, "w") as f:
#             f.write(row["after"])
#
#     # Run Joern on "before" code
#     if not os.path.exists(f"{fpath1}.edges.json"):
#         svdj.full_run_joern(fpath1, verbose=3)
#
#     # Run Joern on "after" code
#     if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
#         svdj.full_run_joern(fpath2, verbose=3)
#
#     # Run SAST extraction
#     fpath3 = savedir_before / f"{row['id']}.c.sast.pkl"
#     if not os.path.exists(fpath3):
#         sast_before = sast.run_sast(row["before"])
#         with open(fpath3, "wb") as f:
#             pkl.dump(sast_before, f)
#
# svd.dfmp(df, preprocess, ordr=False, workers=8)

# 创建类似 BigVulDatasetLineVDDataModule 来读取新的数据。test_dataloader
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
    "batch_size": 1,
    "multitask": "line",
    "splits": "default",
    "lr": 1e-4,
}
samplesz = -1

checkpoint_path = "./ts_logger/VHGLocator/BigVul/version_20/checkpoints/epoch=27-step=7056-val_loss=0.3388.ckpt"
model = vhg.VHGLocator.load_from_checkpoint(checkpoint_path=checkpoint_path)
model = model.cuda()
print(next(model.parameters()).device)
gpu = 1 if torch.cuda.is_available() else None
print("gpu", gpu)
data_module = vhgModule.VHGLocatorDataModule(
        batch_size=config["batch_size"],
        sample=samplesz,
        methodlevel=False,
        nsampling=False,
        nsampling_hops=2,
        gtype=config["gtype"],
        splits=config["splits"],
    )
print(data_module)
trainer = Trainer(accelerator="gpu", devices=1)
trainer.test(model, datamodule=data_module)
# print(model.predict_res)
print(model.predict_lab)

# todo 针对predict_lab作研究

# [
#     [
#         [
#             [array([0.97330713, 0.02669284], dtype=float32), array([0.9646954 , 0.03530466], dtype=float32), array([0.9675698 , 0.03243014], dtype=float32), array([0.976474  , 0.02352599], dtype=float32), array([0.9779413, 0.0220588], dtype=float32), array([0.9690206 , 0.03097936], dtype=float32), array([0.9772381 , 0.02276187], dtype=float32), array([0.97831374, 0.02168625], dtype=float32), array([0.9730185 , 0.02698146], dtype=float32), array([0.9730185 , 0.02698146], dtype=float32), array([0.97158176, 0.02841827], dtype=float32), array([0.9749034 , 0.02509658], dtype=float32), array([0.9743872 , 0.02561278], dtype=float32)]
#             , [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             [15, 6, 24, 20, 3, 8, 4, 18, 9, 11, 26, 27, 17]
#         ],
#         [
#             [array([0.96775013, 0.03224992], dtype=float32), array([0.9609788 , 0.03902121], dtype=float32), array([0.92608833, 0.07391171], dtype=float32), array([0.9356923 , 0.06430763], dtype=float32), array([0.9265828, 0.0734172], dtype=float32)], [0.0, 0.0, 0.0, 0.0, 0.0],
#             [4, 2, 6, 8, 5]
#         ],
#         [
#             [array([0.88544   , 0.11456002], dtype=float32), array([0.82395935, 0.17604066], dtype=float32), array([0.90185213, 0.09814785], dtype=float32), array([0.9136708 , 0.08632925], dtype=float32), array([0.7345577 , 0.26544225], dtype=float32), array([0.82868874, 0.17131129], dtype=float32), array([0.8197748 , 0.18022524], dtype=float32), array([0.91170555, 0.08829443], dtype=float32), array([0.920597  , 0.07940295], dtype=float32), array([0.92372304, 0.07627699], dtype=float32), array([0.86191523, 0.13808481], dtype=float32), array([0.8680353 , 0.13196476], dtype=float32), array([0.9220632 , 0.07793684], dtype=float32), array([0.85765934, 0.14234066], dtype=float32), array([0.9121295 , 0.08787047], dtype=float32), array([0.98061633, 0.01938367], dtype=float32), array([0.9181236 , 0.08187639], dtype=float32), array([0.98294556, 0.0170544 ], dtype=float32), array([0.89700234, 0.10299763], dtype=float32), array([0.83286154, 0.16713844], dtype=float32), array([0.9074535 , 0.09254648], dtype=float32), array([0.82869476, 0.17130525], dtype=float32), array([0.96825635, 0.03174368], dtype=float32), array([0.9138945 , 0.08610552], dtype=float32), array([0.9779553 , 0.02204475], dtype=float32), array([0.92198867, 0.07801127], dtype=float32), array([0.97794   , 0.02205995], dtype=float32), array([0.9135108 , 0.08648922], dtype=float32), array([0.9774179, 0.0225821], dtype=float32), array([0.9050176 , 0.09498243], dtype=float32), array([0.9691945 , 0.03080549], dtype=float32), array([0.9774828 , 0.02251714], dtype=float32), array([0.92205626, 0.0779437 ], dtype=float32), array([0.8773442 , 0.12265584], dtype=float32), array([0.97620845, 0.02379151], dtype=float32), array([0.9691817 , 0.03081833], dtype=float32), array([0.91149235, 0.08850762], dtype=float32), array([0.93068403, 0.06931592], dtype=float32), array([0.9057979 , 0.09420209], dtype=float32), array([0.9088614 , 0.09113859], dtype=float32), array([0.9013582, 0.0986418], dtype=float32), array([0.9033413 , 0.09665873], dtype=float32), array([0.9088851 , 0.09111492], dtype=float32), array([0.980993  , 0.01900705], dtype=float32), array([0.93315625, 0.06684373], dtype=float32), array([0.8787826 , 0.12121744], dtype=float32), array([0.9284054 , 0.07159458], dtype=float32), array([0.96681064, 0.03318939], dtype=float32)],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             [36, 44, 11, 14, 1, 58, 66, 12, 10, 16, 21, 42, 34, 51, 23, 49, 53, 57, 29, 60, 30, 38, 45, 7, 27, 25, 32, 64, 40, 62, 59, 65, 54, 47, 63, 61, 8, 18, 39, 26, 37, 17, 31, 19, 56, 68, 48, 67]
#         ]
#     ]
# ]
