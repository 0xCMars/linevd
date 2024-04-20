import sastvd.VHGLocator as vhg
from os.path import basename, join
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from commode_utils.callback import PrintEpochResultCallback, UploadCheckpointCallback
import torch
import sastvd.linevd as lvd
from pytorch_lightning import Trainer
import sastvd as svd
import pandas as pd
from warnings import filterwarnings

def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore",
                   category=FutureWarning)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.utilities.data",
                   lineno=84)
samplesz = -1

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
    "batch_size": 256,
    "multitask": "line",
    "splits": "default",
    "lr": 1e-4,
}

sp = svd.get_dir(svd.processed_dir() / f"raytune_best_VHG")
# codebert rel + raw linemethod
# checkpoint_path="./sastvd/VHGLocator/ts_logger/VHGLocator/BigVul/version_26/checkpoints/epoch=31-step=4960-val_loss=0.1517.ckpt"

# doc2vec rel + raw linemethod
# checkpoint_path="./sastvd/VHGLocator/ts_logger/VHGLocator/BigVul/version_28/checkpoints/epoch=09-step=12320-val_loss=0.1294.ckpt"
# 202401051910_f029eca_modify_dclass_and_sastvd_scripts.csv

# codebert pdg + raw line
# checkpoint_path="./sastvd/VHGLocator/ts_logger/VHGLocator/BigVul/version_29/checkpoints/epoch=25-step=31772-val_loss=0.3680.ckpt"
# 202401052050_f029eca_modify_dclass_and_sastvd_scripts.csv

# codebert rel+raw line
# checkpoint_path="./sastvd/VHGLocator/ts_logger/VHGLocator/BigVul/version_30/checkpoints/epoch=85-step=105952-val_loss=0.1086.ckpt"

# codebert all+raw line
# checkpoint_path="./sastvd/VHGLocator/ts_logger/VHGLocator/BigVul/version_31/checkpoints/epoch=29-step=41340-val_loss=0.3320.ckpt"

# codebert rel+raw linemethod 3layer
# checkpoint_path="./sastvd/VHGLocator/ts_logger/VHGLocator/BigVul/version_36/checkpoints/epoch=103-step=128128-val_loss=0.1132.ckpt"

# codebert rel+raw line 3layer
# checkpoint_path="./sastvd/VHGLocator/ts_logger/VHGLocator/BigVul/version_37/checkpoints/epoch=55-step=68992-val_loss=0.1129.ckpt"

# codebert rel+raw line 2layer HGTConv vuln节点
# checkpoint_path="./sastvd/VHGLocator/ts_logger/VHGLocator/BigVul/version_58/checkpoints/epoch=07-step=39416-val_loss=0.0000.ckpt"

# codebert rel+raw line 2layer HGTConv all 1 202401090024_c213214_train_new_model.csv
# checkpoint_path="./sastvd/VHGLocator/ts_logger/VHGLocator/BigVul/version_63/checkpoints/epoch=29-step=147810-val_loss=0.3769.ckpt"

# codebert rel+raw line rgcn no sampling best
# checkpoint_path = "./ts_logger/VHGLocator/BigVul/version_10/checkpoints/epoch=21-step=11088-val_loss=0.3409.ckpt"
# checkpoint_path = "./ts_logger/VHGLocator/BigVul/version_11/checkpoints/epoch=115-step=58464-val_loss=0.3209.ckpt"
# checkpoint_path = "./ts_logger/VHGLocator/BigVul/version_12/checkpoints/epoch=221-step=111888-val_loss=0.3152.ckpt"

# codebert cfgcdg+raw line hgat no sampling
# checkpoint_path = "./ts_logger/VHGLocator/BigVul/version_13/checkpoints/epoch=127-step=32256-val_loss=0.3372.ckpt"

# codebert pdg+raw line hgat no sampling
# checkpoint_path = "./ts_logger/VHGLocator/BigVul/version_17/checkpoints/epoch=19-step=5040-val_loss=0.3478.ckpt"
# checkpoint_path = "./ts_logger/VHGLocator/BigVul/version_19/checkpoints/epoch=75-step=19152-val_loss=0.3474.ckpt"

# codebert rel+raw line hgat no sampling
checkpoint_path = "./ts_logger/VHGLocator/BigVul/version_20/checkpoints/epoch=27-step=7056-val_loss=0.3388.ckpt"

def test(
        config, savepath, samplesz=-1, max_epochs=130, num_gpus=1, checkpoint_dir=None
):
    model = vhg.VHGLocator.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model = model.cuda()
    print(next(model.parameters()).device)

    gpu = 1 if torch.cuda.is_available() else None
    print("gpu", gpu)
    # Load data
    data_module = lvd.BigVulDatasetLineVDDataModule(
        batch_size=config["batch_size"],
        sample=samplesz,
        methodlevel=False,
        nsampling=True,
        nsampling_hops=2,
        gtype=config["gtype"],
        splits=config["splits"],
    )
    trainer = Trainer(accelerator="gpu", devices=1)
    trainer.test(model, datamodule=data_module)
    # print("model:", model.res2)
    res = [
        "VHGLocator",
        "linemethod",
        model.res1vo,
        model.res2,
        model.res2,
        model.res3vo,
        model.res2,
        model.lr,
    ]
    main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_VHGLocator_test")
    print(model.res2)
    mets = lvd.get_relevant_metrics(res)
    res_df = pd.DataFrame.from_records([mets])
    res_df.to_csv(str(main_savedir / svd.get_run_id()) + ".csv", index=0)

if __name__ == '__main__':
    filter_warnings()
    test(config, savepath=sp, max_epochs=130, samplesz=samplesz)
