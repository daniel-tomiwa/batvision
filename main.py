from utils.dataset import BatvisionV2Dataset
from utils.utils_general import get_optimizer
from trainer import Trainer
from models.unet import MyUNet

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torchsummaryX import summary

import os
import warnings
warnings.filterwarnings("ignore", module="torchaudio")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    working_dir = hydra.utils.get_original_cwd()
    print(f"Working directory: {working_dir}")

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # ------------ GPU config ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    print("{} {} device is used".format(n_GPU, device))


    # ------------ Create dataset loaders -----------
    if cfg.dataset.name == 'batvisionv2':

        if cfg.mode.mode == 'train':
            train_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_train) 
            print(f'Train Dataset of {len(train_set)} instances')
            train_loader = DataLoader(train_set, batch_size=cfg.mode.batch_size, shuffle=cfg.mode.shuffle, num_workers=cfg.mode.num_threads)
            if cfg.mode.validation:
                val_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_val) 
                print(f'Validation Dataset of {len(val_set)} instances')
                val_loader = DataLoader(val_set, batch_size=cfg.mode.batch_size, shuffle=cfg.mode.shuffle, num_workers=cfg.mode.num_threads)

        elif cfg.mode.mode == 'test':
            if cfg.mode.eval_on == 'val':
                eval_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_val) 
            else:
                eval_set = BatvisionV2Dataset(cfg, cfg.dataset.annotation_file_test) 
            print(f'Eval Dataset of {len(eval_set)} instances')
            eval_loader = DataLoader(eval_set, batch_size = cfg.mode.batch_size, shuffle=False, num_workers=cfg.mode.num_threads)

    else:
        raise Exception('Training can be done only on BV1 and BV2')
    

    # ------------ Create model ------------
    if cfg.model.name == 'unet_baseline':
        model = MyUNet(
            dim=cfg.model.dim,
            init_dim=cfg.model.init_dim,
            out_dim=cfg.model.out_dim,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.channels
        )
        print(f'Model: {cfg.model.name} created')
        summary(model.to(device), torch.zeros((1, 2, 256, 256)).to(device))
        # os.exit(0)
    else:
        raise Exception('Only unet_baseline is supported for now')
    
    # ------------ Optimizer ------------
    optimizer = get_optimizer(
        model, 
        lr=cfg.mode.learning_rate, 
        weight_decay=cfg.mode.weight_decay, 
        optimizer_type=cfg.mode.optimizer
    )

    # ------------ Trainer ------------
    if cfg.mode.mode == 'train':
        trainer = Trainer(
            cfg=cfg,
            device=device,
            model=model,
            data_loader=train_loader,
            val_loader=val_loader if cfg.mode.validation else None,
            optimizer=optimizer,
            use_tensorboard=cfg.mode.use_tensorboard
        )
        trainer.train()

    elif cfg.mode.mode == 'test':
        trainer = Trainer(
            cfg=cfg,
            device=device,
            model=model,
            data_loader=eval_loader,
            optimizer=optimizer,
            use_tensorboard=cfg.mode.use_tensorboard
        )
        trainer.test()

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")