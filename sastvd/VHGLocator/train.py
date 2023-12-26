import sastvd.VHGLocator as vhg
from os.path import basename, join
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from commode_utils.callback import PrintEpochResultCallback, UploadCheckpointCallback
import torch
import sastvd.linevd as lvd
from pytorch_lightning import Trainer
import sastvd as svd

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
    "multitask": "linemethod",
    "splits": "default",
    "lr": 1e-4,
}

sp = svd.get_dir(svd.processed_dir() / f"raytune_best_VHG")

def train(
        config, savepath, samplesz=-1, max_epochs=130
):
    model = vhg.VHGLocator(
        hfeat=config["hfeat"],
        embtype=config["embtype"],
        nsampling=True,
        model=config["modeltype"],
        loss=config["loss"],
        hdropout=config["hdropout"],
        stmtweight=config["stmtweight"],
        scea=config["scea"],
        lr=config["lr"],
    )

    model_name = model.__class__.__name__
    dataset_name = basename("BigVul")
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
    lr_logger = LearningRateMonitor("step")
    print_epoch_results = PrintEpochResultCallback(split_symbol="_",
                                                   after_test=False)
    gpu = 1 if torch.cuda.is_available() else None
    print("gpu", gpu)
    data_module = lvd.BigVulDatasetLineVDDataModule(
        batch_size=config["batch_size"],
        sample=samplesz,
        methodlevel=False,
        nsampling=True,
        nsampling_hops=2,
        gtype=config["gtype"],
        splits=config["splits"],
    )

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
    # tradin = data_module.train_dataloader()
    # val = data_module.val_dataloader()
    trainer.fit(model, data_module)

if __name__ == '__main__':
    filter_warnings()
    train(config, sp)