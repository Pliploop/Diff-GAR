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
        parser.add_argument("--save_dir", default=None)
        parser.add_argument("--root_path", default=None)
        parser.add_argument("--extract_method", default='get_audio_embedding_from_data')
        parser.add_argument("--out_key", default='embedding_proj')
        parser.add_argument("--hop", default=48000)
        parser.add_argument("--limit_n", default=None)
        parser.add_argument("--save", default=False)
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument('--extracted_at', default=None)

    def instantiate_classes(self) -> None:
        pass

if __name__ == "__main__":


    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    
    cli = MyLightningCLI(model_class=LightningDiffGar, datamodule_class=TextAudioDataModule, seed_everything_default=123,
                         run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True},trainer_defaults=MyLightningCLI.trainer_defaults)
    
    cli.parser.save(cli.config, "preprocessing_config.yaml", skip_none=False, overwrite=True)
    
    cfg = OmegaConf.to_container(OmegaConf.load("sagemaker_training/configs/preprocessing_config.yaml"))
    
    
    upload_cfg_to = cfg['pull_config']
    
    s3_client = boto3.client('s3')
    # copy the config file to the s3 bucket
    try:
        #upload cfg to is of the form s3://bucket/key
        bucket, key = upload_cfg_to.replace("s3://", "").split("/", 1)
        s3_client.upload_file("preprocessing_config.yaml", bucket,key)
        # remove the local config file
        os.remove("preprocessing_config.yaml")
    except NoCredentialsError:
        print("No AWS credentials found. Please set up your AWS credentials.")
    
    # launch_sagemaker_processing(cfg)
    
    