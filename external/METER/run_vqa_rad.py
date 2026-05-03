"""Training and evaluation script for VQA-RAD with BiomedCLIP + METER.

Usage:
    python run_vqa_rad.py with task_finetune_vqa_rad_biomedclip \
        data_root=<path/to/arrow/dir> \
        per_gpu_batchsize=32 \
        num_workers=4

To run test-only with a saved checkpoint:
    python run_vqa_rad.py with task_finetune_vqa_rad_biomedclip \
        data_root=<path/to/arrow/dir> \
        per_gpu_batchsize=32 \
        load_path=<checkpoint.ckpt> \
        test_only=True
"""
import os
import copy
import pytorch_lightning as pl

from meter.config import ex
from meter.modules import METERTransformerSS
from meter.datamodules.multitask_datamodule import MTDataModule

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # Single-node, no distributed sampler for VQA-RAD
    dm = MTDataModule(_config, dist=False)

    model = METERTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * max(num_gpus, 1) * _config["num_nodes"]
    ), 1)

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    # PL 2.x API: use devices= and accelerator= instead of gpus=/accelerator="ddp"
    if num_gpus > 0:
        accelerator = "gpu"
        devices = num_gpus
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps if max_steps is not None else -1,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        # Run test with best checkpoint after training (skip for fast_dev_run)
        if not _config["fast_dev_run"]:
            trainer.test(model, datamodule=dm, ckpt_path="best")
    else:
        trainer.test(model, datamodule=dm)
