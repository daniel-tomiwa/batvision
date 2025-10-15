import torch
from torch.utils.tensorboard import SummaryWriter
from utils.utils_general import compute_errors, visualize_predictions
from utils.utils_tensorboard import tensorboard_display_input_pred
from omegaconf import DictConfig, OmegaConf
import time
import numpy as np
import pandas as pd
import os
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig

class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        model: torch.nn.Module, 
        data_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        optimizer: torch.optim.Optimizer = None, 
        use_tensorboard: bool = True,
    ):

        super().__init__()

        self.cfg = cfg
        self.device = device
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.use_tensorboard = use_tensorboard
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        if self.cfg.mode.mode == "train":
            self.experiment_name = cfg.model.name + '_' +  cfg.dataset.name + '_' + 'BS' + str(cfg.mode.batch_size) + '_' + 'Lr' + str(cfg.mode.learning_rate) + '_' + cfg.mode.optimizer + '_' + cfg.mode.experiment_name
        self.criterion = None

        if self.use_tensorboard:

            self.writer = SummaryWriter(f"./logs/{self.experiment_name}/")
            print(f"TensorBoard logging enabled. Logs will be saved to ./logs/{self.experiment_name}/")

            file = open("./logs/"+ self.experiment_name +'/'+"architecture.txt","w")
            # dataloader parameters
            file.write("Dataset name: {}\n".format(self.cfg.dataset.name))
            file.write("Batch size: {}\n".format(self.cfg.mode.batch_size))
            file.write("Image processing: {}\n".format(self.cfg.dataset.preprocess))
            file.write("Image resize: {}\n".format(self.cfg.dataset.images_size))
            file.write("Depth norm: {}\n".format(self.cfg.dataset.depth_norm))
            file.write("Audio type used for training: {}\n".format(self.cfg.dataset.audio_format))
            
            # parameters
            file.write("Learning rate: {}\n".format(self.cfg.mode.learning_rate))
            file.write("Optimize used : {}\n".format(self.cfg.mode.optimizer))

            # net architecture
            file.write("Generator: {}\n".format(self.cfg.model.generator))

            file.write(str(self.model))
            file.close()

    def _save_architecture(self):
        
        # Make sure the output architectures directory exists
        architecture_path = os.path.join('outputs', 'architecture')
        if not os.path.exists(architecture_path):
            os.makedirs(architecture_path)

        model_arch_path = os.path.join(architecture_path, self.experiment_name)

        if not os.path.exists(model_arch_path):
            os.makedirs(model_arch_path)

        model_arch_file = os.path.join(model_arch_path, 'model_architecture.txt')
        model_arch = str(self.model)
        try:
            with open(model_arch_file, 'w') as f:
                f.write(model_arch)
        except Exception as e:
            print(f"Error saving model architecture: {e}")

        if self.cfg.mode.use_wandb:
            wandb.save(model_arch_file)

    def _load_checkpoint(self, checkpoint_path: str):

        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1

    def _initialize_criterion(self):
        if self.cfg.mode.criterion == "L1":
            self.criterion = torch.nn.L1Loss().to(self.device)
        else:
            raise ValueError(f"Unsupported loss type: {self.cfg.mode.criterion}")
        
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        print('Save network at epoch {}'.format(epoch))
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        run_dir = HydraConfig.get().run.dir
        print("Run directory:", run_dir)
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        print("Checkpoint directory:", checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth')
        best_model_path = os.path.join(checkpoint_dir, 'model_best.pth')


        # path_check = './outputs/checkpoints/' + self.experiment_name + '/'
        # isExist = os.path.exists(path_check)
        # if not isExist:
        #     os.makedirs(path_check)
        # torch.save(state, './outputs/checkpoints/' + self.experiment_name + '/checkpoint_' + str(epoch) + '.pth')
        torch.save(state, checkpoint_path)
        if is_best:
            # torch.save(state, './outputs/checkpoints/' + self.experiment_name + '/model_best.pth')
            torch.save(state, best_model_path)
        if self.cfg.mode.use_wandb:
            # wandb.save('./outputs/checkpoints/' + self.experiment_name + '/checkpoint_' + str(epoch) + '.pth')
            wandb.save(checkpoint_path)
            if is_best:
                # wandb.save('./outputs/checkpoints/' + self.experiment_name + '/model_best.pth')
                wandb.save(best_model_path)
                

    def train(self):

        if self.cfg.mode.use_wandb:
            wandb.login(key=self.cfg.mode.wandb_api_key)

        if self.cfg.mode.checkpoint_epoch is None:
            self.start_epoch = 1

            if self.cfg.mode.use_wandb:
                run = wandb.init(
                    project="batvision",
                    name=self.experiment_name,
                    config=OmegaConf.to_container(self.cfg, resolve=True)
                )

            self._save_architecture() # save architecture at the beginning of training

        else: # NOTE: specify the checkpoint path and epoch in the config file
            load_epoch = self.cfg.mode.checkpoint_epoch
            checkpoint_path = self.cfg.mode.checkpoint_path
            if checkpoint_path is None or load_epoch is None:
                raise ValueError("checkpoint_path and checkpoint_epoch must be provided for resuming training.")
            
            checkpoint_path = checkpoint_path + '/checkpoint_' + str(load_epoch) + '.pth'

            self._load_checkpoint(checkpoint_path)

            self.start_epoch = load_epoch + 1

            if self.cfg.mode.use_wandb:

                id = self.cfg.mode.wandb_id

                if id is None:
                    raise ValueError("wandb_id must be provided for resuming wandb run.")

                run = wandb.init(
                    project="batvision",
                    name=self.experiment_name,
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                    resume="allow",
                    id=id
                )

        if self.cfg.mode.use_wandb:
            wandb.watch(self.model, log="all")

        for param_group in self.optimizer.param_groups:
            print("Learning rate used: {}".format(param_group['lr']))

        # Initialize loss function
        self._initialize_criterion()

        num_epochs = self.cfg.mode.epochs
        is_best = False

        for epoch in range(self.start_epoch, num_epochs + 1):

            t0 = time.time()

            batch_loss = [] 
            batch_loss_val = [] 

            # ------ Training ---------
            self.model.train()  

            batch_bar = tqdm(
                total=len(self.data_loader),
                desc=f"Epoch {epoch}/{num_epochs}",
                dynamic_ncols=True,
                leave=False,
                position=0                
            )

            for i, (audio, gtdepth) in enumerate(self.data_loader):

                audio = audio.to(self.device)
                gtdepth = gtdepth.to(self.device)   

                # set parameter gradients to zero
                self.optimizer.zero_grad()

                # forward
                depth_pred = self.model(audio)
                
                # compute loss
                loss = self.criterion(depth_pred[gtdepth!=0], gtdepth[gtdepth!=0]) 
                batch_loss.append(loss.item()) 
                
                # optimize
                loss.backward()  # backward-pass
                self.optimizer.step()  # update weights

                batch_bar.set_postfix(loss=np.mean(batch_loss))
                batch_bar.update()
                
            if self.use_tensorboard:
                self.writer.add_scalar('Train/Loss', np.mean(batch_loss), epoch)

            if self.cfg.mode.use_wandb:
                wandb.log({"Train/Loss": np.mean(batch_loss), "epoch": epoch})
        
            epoch_time = time.time()-t0
            print(' - epoch time: {:.1f}'.format(epoch_time))

            if self.use_tensorboard:
                self.writer.add_scalar('Epoch_time',epoch_time,epoch)

            # ------- Validation ------------
            if self.cfg.mode.validation and epoch % self.cfg.mode.validation_iter == 0:
                self.model.eval() 

                errors = []

                with torch.no_grad():

                    for audio_val, gtdepth_val in self.val_loader:
                        audio_val = audio_val.to(self.device)
                        gtdepth_val = gtdepth_val.to(self.device)        

                        depth_pred_val = self.model(audio_val)
                        
                        loss_val = self.criterion(depth_pred_val[gtdepth_val!=0], gtdepth_val[gtdepth_val!=0])
                        batch_loss_val.append(loss_val.item()) 

                        for idx in range(depth_pred_val.shape[0]):
                            if self.cfg.dataset.depth_norm:
                                # if normalization, return to true range for metrics computation
                                errors.append(compute_errors(gtdepth_val[idx].cpu().numpy() * self.cfg.dataset.max_depth, 
                                        depth_pred_val[idx].cpu().numpy() * self.cfg.dataset.max_depth))
                            else:
                                errors.append(compute_errors(gtdepth_val[idx].cpu().numpy(), 
                                        depth_pred_val[idx].cpu().numpy()))
        
                    mean_errors = np.array(errors).mean(0)	
                    print('RMSE: {:.3f}'.format(mean_errors[1]))
                    
                    val_errors = {}
                    val_errors['ABS_REL'] = mean_errors[0]
                    val_errors['RMSE'] = mean_errors[1]
                    val_errors['DELTA1'] = mean_errors[2] 
                    val_errors['DELTA2'] = mean_errors[3]
                    val_errors['DELTA3'] = mean_errors[4]

                    # save best model
                    if np.mean(batch_loss_val) < self.best_val_loss:
                        self.best_val_loss = np.mean(batch_loss_val)
                        is_best = True
                    else:
                        is_best = False

                    if self.use_tensorboard:
                        self.writer.add_scalar('Val/Loss', np.mean(batch_loss_val), epoch)
                        self.writer.add_scalar('Val/RMSE', val_errors['RMSE'], epoch)
                        self.writer.add_scalar('Val/ABS_REL', val_errors['ABS_REL'], epoch)
                        self.writer.add_scalar('Val/DELTA1', val_errors['DELTA1'], epoch)
                        self.writer.add_scalar('Val/DELTA2', val_errors['DELTA2'], epoch)
                        self.writer.add_scalar('Val/DELTA3', val_errors['DELTA3'], epoch)

                        if epoch % self.cfg.mode.validation_display == 0:
                            tensorboard_display_input_pred(self.writer, audio_val, depth_pred_val, gtdepth_val, 'Val', epoch)

                    if self.cfg.mode.use_wandb:
                        wandb.log({
                            "Val/Loss": np.mean(batch_loss_val), 
                            "Val/RMSE": val_errors['RMSE'], 
                            "Val/ABS_REL": val_errors['ABS_REL'], 
                            "Val/DELTA1": val_errors['DELTA1'], 
                            "Val/DELTA2": val_errors['DELTA2'], 
                            "Val/DELTA3": val_errors['DELTA3'], 
                            "epoch": epoch
                        })

            # ------- Display tensorboard ------------
            if self.use_tensorboard and epoch % self.cfg.mode.print_tensorboard == 0:
                tensorboard_display_input_pred(self.writer, audio, depth_pred, gtdepth, 'Train', epoch)

            # ------- Save ------------
            if epoch % self.cfg.mode.saving_checkpoints == 0:
                self._save_checkpoint(epoch, is_best)

        if self.cfg.mode.use_wandb:
            run.finish()

    def test(self):

        if self.cfg.mode.checkpoint_epoch is None or self.cfg.mode.checkpoint_path is None:
            raise ValueError("Checkpoint must be provided for testing.")
        else:
            load_epoch = self.cfg.mode.checkpoint_epoch
            checkpoint_path = self.cfg.mode.checkpoint_path

            checkpoint_path = os.path.join(checkpoint_path, 'checkpoint_' + str(load_epoch) + '.pth')

            self._load_checkpoint(checkpoint_path)

            if self.cfg.mode.use_wandb:
                wandb.login(key=self.cfg.mode.wandb_api_key)
                id = self.cfg.mode.wandb_id
                if id is None:
                    raise ValueError("wandb_id must be provided for resuming wandb run.")
                run = wandb.init(
                    project="batvision",
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                    resume="allow",
                    id=id
                )

        # Initialize loss function
        self._initialize_criterion()
        
        # Set model to evaluation mode
        self.model.eval()

        gt_imgs_to_save = []
        pred_imgs_to_save = []
        image_imgs_to_save = []
        loss_list = []
        # errors = []
        rmse_list = []
        abs_rel_list = []
        log10_list = []
        delta1_list = []
        delta2_list = []
        delta3_list = [] 
        mae_list = []

        batch_bar = tqdm(
                total=len(self.data_loader),
                dynamic_ncols=True,
                leave=False,
                position=0                
            )

        with torch.no_grad():

            for audio, depth, image in self.data_loader:

                audio = audio.to(self.device)
                depth = depth.to(self.device)
                image = image.to(self.device)

                depth_pred = self.model(audio)

                loss_test = self.criterion(depth_pred[depth !=0], depth[depth !=0])
                loss_list.append(loss_test.cpu().item())

                for idx in range(depth_pred.shape[0]):
                    gt_imgs_to_save.append(depth[idx].detach().cpu().numpy())
                    pred_imgs_to_save.append(depth_pred[idx].detach().cpu().numpy())
                    image_imgs_to_save.append(image[idx].detach().cpu().numpy())
                    if self.cfg.dataset.depth_norm:
                        unscaledgt = depth[idx].detach().cpu().numpy() * self.cfg.dataset.max_depth
                        unscaledpred = depth_pred[idx].detach().cpu().numpy() * self.cfg.dataset.max_depth
                        abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(unscaledgt, 
                            unscaledpred)
                    else:   
                        abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(depth[idx].cpu().numpy(), 
                                depth_pred[idx].cpu().numpy())
                        
                    # errors.append((abs_rel, rmse, a1, a2, a3, log_10, mae))
                    rmse_list.append(rmse)
                    abs_rel_list.append(abs_rel)
                    log10_list.append(log_10)
                    delta1_list.append(a1)
                    delta2_list.append(a2)
                    delta3_list.append(a3)
                    mae_list.append(mae)

                batch_bar.set_postfix(loss=np.mean(loss_list))
                batch_bar.update()

        # Save evaluation
        d = {
            'loss': loss_list, 
            'abs_rel': abs_rel_list, 
            'rmse': rmse_list, 
            'log10': log10_list, 
            'delta1': delta1_list, 
            'delta2': delta2_list, 
            'delta3': delta3_list, 
            'mae': mae_list, 
            'gt_images': gt_imgs_to_save, 
            'pred_imgs': pred_imgs_to_save,
            'input_images': image_imgs_to_save
        }

        stats_df = pd.DataFrame(data=d)

        save_path = "./outputs/eval/" + self.cfg.mode.checkpoint_path.split('/')[-1] + '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.cfg.mode.eval_on == 'test':
            stats_df.to_pickle(
                save_path + "test.plk"
            )
        elif self.cfg.mode.eval_on == 'val':
            stats_df.to_pickle(
                save_path + "val.plk"
            )
        elif self.cfg.mode.eval_on == 'train':
            stats_df.to_pickle(
                save_path + "train.plk"
            )
        else:
            raise ValueError("eval_on must be 'test', 'val' or 'train'.")
        
        pred_fig = visualize_predictions(stats_df, num_samples=20, random_seed=42, show=False)

        if self.cfg.mode.use_wandb:

            wandb.log({"Test/examples": wandb.Image(pred_fig)})

            wandb.log({
                "Test/loss_mean": np.mean(loss_list),
                "Test/abs_rel_mean": np.mean(abs_rel_list),
                "Test/rmse_mean": np.mean(rmse_list),
                "Test/log10_mean": np.mean(log10_list),
                "Test/delta1_mean": np.mean(delta1_list),
                "Test/delta2_mean": np.mean(delta2_list),
                "Test/delta3_mean": np.mean(delta3_list),
                "Test/mae_mean": np.mean(mae_list),
            })

            run.finish() 

        plt.close(pred_fig)