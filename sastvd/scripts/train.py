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
import numpy as np
from warnings import filterwarnings

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
    "batch_size": 1024,
    "multitask": "linemethod",
    "splits": "default",
    "lr": 1e-4,
}

samplesz = -1
run_id = svd.get_run_id()
sp = svd.get_dir(svd.processed_dir() / f"raytune_best_{samplesz}" / run_id)

def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore",
                   category=FutureWarning)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.utilities.data",
                   lineno=84)
def train(
    config, savepath, samplesz=-1, max_epochs=130, num_gpus=1, checkpoint_dir=None
):
    model = lvd.LitGNN(
        hfeat=config["hfeat"],
        embtype=config["embtype"],
        methodlevel=False,
        nsampling=True,
        model=config["modeltype"],
        loss=config["loss"],
        hdropout=config["hdropout"],
        gatdropout=config["gatdropout"],
        num_heads=2,
        multitask=config["multitask"],
        stmtweight=config["stmtweight"],
        gnntype=config["gnntype"],
        scea=config["scea"],
        lr=config["lr"],
    )

    # Define logger
    model_name = model.__class__.__name__
    dataset_name = basename("BigVul")
    # tensorboard logger
    tensorlogger = TensorBoardLogger(join("./ts_logger", model_name),
                                     dataset_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=join(tensorlogger.log_dir, "checkpoints"),
        monitor="val_loss",
        filename="{epoch:02d}-{step:02d}-{val_loss:.4f}",
        every_n_epochs=2,
        save_top_k=5,
    )

    upload_weights = UploadCheckpointCallback(
        join(tensorlogger.log_dir, "checkpoints"))

    # early_stopping_callback = EarlyStopping(patience=10,
    #                                         monitor="val_loss",
    #                                         verbose=True,
    #                                         mode="min")

    lr_logger = LearningRateMonitor("step")
    print_epoch_results = PrintEpochResultCallback(split_symbol="_",
                                                   after_test=False)

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

    # Train model
    metrics = ["train_loss", "val_loss", "val_auroc"]

    trainer = Trainer(
        devices=gpu,
        accelerator="auto",
        # precision="16-mixed",
        max_epochs=max_epochs,
        default_root_dir=savepath,
        num_sanity_val_steps=0,
        val_check_interval=1.0,
        log_every_n_steps=10,
        logger=[tensorlogger],
        callbacks=[
            checkpoint_callback, lr_logger,
            print_epoch_results, upload_weights
        ],
    )

    print("Begin Train")
    trainer.fit(model, data_module)
    print("Begin Test")
    # trainer.test(model=model, datamodule=data_module)


if __name__ == '__main__':
    filter_warnings()
    train(config, savepath=sp, max_epochs=130, samplesz=samplesz)
