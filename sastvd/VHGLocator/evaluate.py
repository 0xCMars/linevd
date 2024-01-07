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
checkpoint_path="./sastvd/VHGLocator/ts_logger/VHGLocator/BigVul/version_30/checkpoints/epoch=85-step=105952-val_loss=0.1086.ckpt"

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
    trainer = Trainer(devices=gpu)
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
    main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_VHGLocator")
    print(model.res2)
    mets = lvd.get_relevant_metrics(res)
    res_df = pd.DataFrame.from_records([mets])
    res_df.to_csv(str(main_savedir / svd.get_run_id()) + ".csv", index=0)

if __name__ == '__main__':
    filter_warnings()
    test(config, savepath=sp, max_epochs=130, samplesz=samplesz)
