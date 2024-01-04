from os.path import basename, join
from pytorch_lightning.loggers import TensorBoardLogger
import sastvd.linevd as lvd
import sastvd as svd
import sastvd.linevd.run as lvdrun
from ray import tune
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from commode_utils.callback import PrintEpochResultCallback, UploadCheckpointCallback
import torch
import pandas as pd

config = {
    "hfeat": 512,
    "embtype": "codebert",
    "stmtweight": 1,
    "hdropout": 0.3,
    "gatdropout": 0.2,
    "modeltype": "gat2layer",
    "gnntype": "gat",
    "loss": "ce",
    "scea": 0.5,
    "gtype": "pdg+raw",
    "batch_size": 64,
    "multitask": "line",
    "splits": "default",
    "lr": 1e-4,
}
samplesz = -1
run_id = svd.get_run_id()
sp = svd.get_dir(svd.processed_dir() / f"raytune_best_{samplesz}" / run_id)
# checkpoint_path="./ts_logger/LitGNN/BigVul/version_6/checkpoints/epoch=117-step=982940-val_loss=0.1041.ckpt"
checkpoint_path="./ts_logger/LitGNN/BigVul/version_8/checkpoints/epoch=29-step=147120-val_loss=0.1087.ckpt"
def test(
        config, savepath, samplesz=-1, max_epochs=130, num_gpus=1, checkpoint_dir=None
):
    model = lvd.LitGNN.load_from_checkpoint(checkpoint_path=checkpoint_path)
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
        "linemethod",
        "linemethod",
        model.res1vo,
        model.res2mt,
        model.res2f,
        model.res3vo,
        model.res2,
        model.lr,
    ]
    main_savedir = svd.get_dir(svd.outputs_dir() / "rq_results_methodonly")

    mets = lvd.get_relevant_metrics(res)
    res_df = pd.DataFrame.from_records([mets])
    res_df.to_csv(str(main_savedir / svd.get_run_id()) + ".csv", index=0)

if __name__ == '__main__':
    test(config, savepath=sp, max_epochs=130, samplesz=samplesz)