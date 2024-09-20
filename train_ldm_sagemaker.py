from diffgar.models.ldm.diffusion import LightningDiffGar
from diffgar.dataloading.dataloaders import TextAudioDataModule
from pytorch_lightning.cli import SaveConfigCallback, LightningCLI
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import os
from jsonargparse import lazy_instance
from pytorch_lightning.strategies import DDPStrategy
import wandb
import boto3
from botocore.exceptions import NoCredentialsError

import logging

from sagemaker_training.sagemaker_training import launch_sagemaker_training
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self) -> None:
        
            config = self.parser.dump(self.config, skip_none=False)
            
            #dump to config.py
            
            print(type(config))
            
            # with open(self.config_filename, "w") as config_file:
            #     config_file.write(config)
                
            
            
            
            

class MyLightningCLI(LightningCLI):
    
    trainer_defaults = {
        "strategy": lazy_instance(DDPStrategy, find_unused_parameters=False),
    }
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--log", default=False)
        parser.add_argument("--log_model", default=False)
        parser.add_argument("--ckpt_path", default="checkpoints")
        parser.add_argument("--resume_id", default=None)
        parser.add_argument("--resume_from_checkpoint", default=None)
        parser.add_argument("--project", default="DiffGAR-LDM")

    def instantiate_classes(self) -> None:
        pass

if __name__ == "__main__":


    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    
    cli = MyLightningCLI(model_class=LightningDiffGar, datamodule_class=TextAudioDataModule, seed_everything_default=123,
                         run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True},trainer_defaults=MyLightningCLI.trainer_defaults)
    
    cli.parser.save(cli.config, "config.yaml", skip_none=False, overwrite=True)
    
    cfg = OmegaConf.to_container(OmegaConf.load("sagemaker_training/configs/config.yaml"))
    
    launch_sagemaker_training(cfg)
    
    