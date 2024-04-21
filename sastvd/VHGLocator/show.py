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
import numpy as np
import glob
# 获取文件的函数划分，写入c文件
# filepath = "./storage/data/gcc/source-code/iaopre.c"
# filepath2 = "./storage/data/gcc/source-code/mlf.c"
# filepath3 = "./storage/data/gcc/source-code/bigvul1.c"
# files = [filepath, filepath2, filepath3]

# 在导入新的项目前，删除./storage/cache/下的所有test_case前缀文件夹和input_codebert

source_code_path = "./storage/data/gcc/source-code"
c_files = glob.glob(source_code_path + "/*.c")

headers = ['Access Gained', 'Attack Origin', 'Authentication Required', 'Availability', 'CVE ID', 'CVE Page', 'CWE ID', 'Complexity', 'Confidentiality', 'Integrity', 'Known Exploits', 'Publish Date', 'Score', 'Summary', 'Update Date', 'Vulnerability Classification', 'add_lines', 'codeLink', 'commit_id', 'commit_message', 'del_lines', 'file_name', 'files_changed', 'func_after', 'func_before', 'lang', 'lines_after', 'lines_before', 'parentID', 'patch', 'project', 'project_after', 'project_before', 'vul', 'vul_func_with_fix']

target = "./storage/external/test_case.csv"

# 将文件及其代码转为存储在csv中，产出的文件在`target`中
print("文件转为csv数据库")
data = []
for file in c_files:

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


# 读取csv格式中的代码，构造类似BigVul数据库的datafram，格式为parquet
# 文件会保存在 ./storage/cache/test_case/和./storage/cache/test_datasets/test_datasets_Fasle.pq中
print("-------------------------------------------------")
print("读取csv转化为InputData")
df = dshelper.InputData(minimal=False)
# print(df['func_before'])

# 执行joern，svdj.full_run_joern获得xxx.edgs.json，产出对代码的图信息
# 生成的文件存在./storage/processed/test_case/中
def preprocess(row):
    """Parallelise joern functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    # print("process", row["id"])
    savedir_before = svd.get_dir(svd.processed_dir() / row["dataset"] / "before")
    savedir_after = svd.get_dir(svd.processed_dir() / row["dataset"] / "after")
    # print(row)
    # Write C Files
    fpath1 = savedir_before / f"{row['id']}.c"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    fpath2 = savedir_after / f"{row['id']}.c"
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])

    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        svdj.full_run_joern(fpath1, verbose=3)

    # Run Joern on "after" code
    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        svdj.full_run_joern(fpath2, verbose=3)

    # Run SAST extraction
    fpath3 = savedir_before / f"{row['id']}.c.sast.pkl"
    if not os.path.exists(fpath3):
        sast_before = sast.run_sast(row["before"])
        with open(fpath3, "wb") as f:
            pkl.dump(sast_before, f)
print("-------------------------------------------------")
print("将InputData中的数据利用joern分析得出图数据")
svd.dfmp(df, preprocess, ordr=False, workers=8)

# 创建类似 BigVulDatasetLineVDDataModule 来读取新的数据，test_dataloader
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
print("-------------------------------------------------")

# 读取模型
checkpoint_path = "./ts_logger/VHGLocator/BigVul/version_20/checkpoints/epoch=27-step=7056-val_loss=0.3388.ckpt"
model = vhg.VHGLocator.load_from_checkpoint(checkpoint_path=checkpoint_path)
model = model.cuda()
# 检查是否在gpu上
print(next(model.parameters()).device)
gpu = 1 if torch.cuda.is_available() else None
print("gpu", gpu)

# 读取数据
data_module = vhgModule.VHGLocatorDataModule(
        batch_size=config["batch_size"],
        sample=samplesz,
        methodlevel=False,
        nsampling=False,
        nsampling_hops=2,
        gtype=config["gtype"],
        splits=config["splits"],
    )
# print(data_module)
trainer = Trainer(accelerator="gpu", devices=1)
trainer.test(model, datamodule=data_module)
# print(model.predict_res)
# print(model.predict_lab)
#
print("-------------------------------------------------")
print("Result:")
for i in range(len(model.predict_lab)):
    result = model.predict_lab[i]
    predics = np.array(result[0][0])[:, 1]
    line = np.array(result[0][2])
    # print(predics)
    # print(line)
    sorted_indeices = np.argsort(predics, axis=0)[::-1]
    sorted_predict = predics[sorted_indeices]
    sorted_line = line[sorted_indeices]

    print("目标文件是：" + c_files[i])
    showNo = 10
    if len(sorted_line) < 10:
        showNo = len(sorted_line)
    for i in range(showNo):
        print(f"第{i + 1}位有可能的漏洞行号是{sorted_line[i]}")
