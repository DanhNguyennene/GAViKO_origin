import torchio as tio
import pandas as pd
from data.dataset import CustomDataset
from torch.utils.data import DataLoader, DistributedSampler
from model.gaviko import Gaviko
from model.adaptformer import AdaptFormer
from model.vision_transformer import VisionTransformer
from model.dvpt import DynamicVisualPromptTuning
from model.evp import ExplicitVisualPrompting
from model.ssf import ScalingShiftingFeatures
from model.melo import MeLO
from model.vpt import PromptedVisionTransformer
from utils.logging import MemoryUsageLogger, analyze_model_computation
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import os
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
import deepspeed
from deepspeed import DeepSpeedEngine
from thop import profile
from torchprofile import profile_macs
import wandb
from omegaconf import OmegaConf
from utils.logging import CSVLogger, setup_logging

def setup_ddp(rank, world_size, master_port="12355"):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the GPU device
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()

class DataPreprocessor:
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size

    def preprocess(self, df):
        spatial_augment = {
            tio.RandomAffine(degrees=15, p=0.5),
            tio.RandomFlip(axes=(0), flip_probability=0.5)
        }

        intensity_augment = {
            tio.RandomNoise(): 0.25,
            tio.RandomBiasField(): 0.25,
            tio.RandomBlur(std=(0,1.5)): 0.25,
            tio.RandomMotion(): 0.25,
        }

        train_transforms = tio.Compose([
            tio.Compose(spatial_augment, p=1),
            tio.RescaleIntensity(out_min_max=(0,1)),
        ])

        val_transforms = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0,1)),
        ])

        test_transforms = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0,1)),
        ])

        df = pd.read_csv(self.config['data']['data_path'])

        train_df = df[df['subset'] == 'train'].reset_index(drop=True)
        val_df = df[df['subset'] == 'val'].reset_index(drop=True)
        test_df = df[df['subset'] == 'test'].reset_index(drop=True)

        train_ds = CustomDataset(train_df, transforms=train_transforms, image_folder=self.config['data']['image_folder'])
        val_ds = CustomDataset(val_df, transforms=val_transforms, image_folder=self.config['data']['image_folder'])
        test_ds = CustomDataset(test_df, transforms=test_transforms, image_folder=self.config['data']['image_folder'])

        # Use DistributedSampler for DDP
        train_sampler = DistributedSampler(
            train_ds, 
            num_replicas=self.world_size, 
            rank=self.rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_ds, 
            num_replicas=self.world_size, 
            rank=self.rank,
            shuffle=False
        )
        
        test_sampler = DistributedSampler(
            test_ds, 
            num_replicas=self.world_size, 
            rank=self.rank,
            shuffle=False
        )

        # Adjust num_workers for multiple processes
        num_workers = max(1, self.config['data']['num_workers'] // self.world_size)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.config['data']['batch_size'], 
            sampler=train_sampler,
            num_workers=num_workers, 
            pin_memory=True,
            drop_last=True  # Important for DDP consistency
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.config['data']['batch_size'], 
            sampler=val_sampler,
            num_workers=num_workers, 
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_ds, 
            batch_size=self.config['data']['batch_size'], 
            sampler=test_sampler,
            num_workers=num_workers, 
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, train_sampler, val_sampler

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def train_ddp(rank, world_size, config):
    # Setup DDP
    setup_ddp(rank, world_size, config.get('ddp', {}).get('master_port', "12355"))
    
    # Only setup logging on rank 0
    if rank == 0:
        setup_logging(log_dir=config['utils']['log_dir'])
        logging.info(f"Starting DDP training on {world_size} GPUs")
        
        # Initialize WandB only on rank 0
        os.makedirs(config['utils']['log_dir'], exist_ok=True)
        time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        if config['wandb']['enable']:
            logging.info("Initializing WandB...")
            wandb.init(
                project=config['wandb']['project'],
                config=OmegaConf.to_container(config, resolve=True),
                name=config['wandb'].get('name', f"ddp_run_{time_stamp}"),
                dir=config['utils']['log_dir'],
                save_code=True,
            )

        model_name = config['model']['method']
        csv_logger = CSVLogger(
            log_dir=config['utils']['log_dir'], 
            filename_prefix=f'{model_name}_ddp_training_log', 
            fields=['epoch', 'train_step_acc', 'train_step_loss', 'train_epoch_loss', 
                   'val_step_acc', 'val_step_loss', 'val_epoch_loss', 'lr', 
                   'best_epoch', 'best_val_acc', 'time_stamp', 'train_step', 'val_step',
                   'train_epoch_acc', 'val_epoch_acc']
        )
    
    device = torch.device(f'cuda:{rank}')
    
    # Preprocess data with DDP support
    data_preprocessor = DataPreprocessor(config, rank, world_size)
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, train_sampler, val_sampler = data_preprocessor.preprocess(pd.read_csv(config['data']['data_path']))

    # Initialize model
    if config['model']['method'] == 'gaviko':
        model = Gaviko(**config['model'])
    elif config['model']['method'] == 'linear':
        model = VisionTransformer(**config['model'])
        for key, value in model.named_parameters():
            if "head" in key:
                value.requires_grad = True
            else:
                value.requires_grad = False
    elif config['model']['method'] == 'fft':
        model = VisionTransformer(**config['model'])
    elif config['model']['method'] == 'adaptformer':
        model = AdaptFormer(**config['model'])
    elif config['model']['method'] == 'bitfit':
        model = VisionTransformer(**config['model'])
        for key, value in model.named_parameters():
            if "bias" in key or "head" in key:
                value.requires_grad = True
            else:
                value.requires_grad = False
    elif config['model']['method'] == 'dvpt':
        model = DynamicVisualPromptTuning(**config['model'])
    elif config['model']['method'] == 'evp':
        model = ExplicitVisualPrompting(**config['model'])
    elif config['model']['method'] == 'ssf':
        model = ScalingShiftingFeatures(**config['model'])
    elif config['model']['method'] == 'melo':
        vit_model = VisionTransformer(**config['model'])
        model = MeLO(vit=vit_model, **config['model'])
    elif config['model']['method'] in ['deep_vpt', 'shallow_vpt']:
        model = PromptedVisionTransformer(**config['model'])

    # Move model to device and set precision
    model = model.to(device)
    if config['train']['fp16']:
        model = model.half()

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        count_freeze = sum(1 for p in model.parameters() if not p.requires_grad)
        count_tuning = sum(1 for p in model.parameters() if p.requires_grad)
        tuning_params = [name for name, param in model.named_parameters() if param.requires_grad]
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info(f'There are {count_tuning} trainable params.')
        logging.info(f'There are {count_freeze} freeze params')
        logging.info(f'Total trainable parameters: {total_params}')

    # Loss function
    if config['train']['loss_fn'] == 'focal_loss':
        criterion = FocalLoss(gamma=1.2)
    else:
        criterion = CrossEntropyLoss()

    # Optimizer and scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not config['train']['deepspeed']['enabled']:
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=config['train']['lr'],
            eps=1e-4 if config['train']['fp16'] else 1e-8,
        )

        steps_per_epoch = len(train_loader)
        num_epochs = config['train']['num_epochs']
        total_steps = steps_per_epoch * num_epochs
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config['train']['scheduler']['max_lr'],
            total_steps=total_steps,
            pct_start=config['train']['scheduler']['pct_start'],
            div_factor=config['train']['scheduler']['div_factor'],
            final_div_factor=config['train']['scheduler']['final_div_factor'],
            anneal_strategy=config['train']['scheduler']['anneal_strategy'],
            three_phase=config['train']['scheduler']['three_phase']
        )

    # Training variables
    val_acc_max = 0
    current_epoch = 0
    patience = config['train']['patience']
    epoch_since_improvement = 0
    best_epoch = 0

    # DeepSpeed initialization (if enabled)
    if config['train']['deepspeed']['enabled']:
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model.module,  # Use .module to get the underlying model
            config=config['train']['deepspeed']['config'],
            model_parameters=trainable_params
        )
        if rank == 0:
            logging.info("DeepSpeed initialized with DDP.")

    # Training loop
    for epoch in range(num_epochs):
        # Set epoch for sampler (important for proper shuffling in DDP)
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        num_acc = 0.0
        running_loss = 0.0
        total_samples = 0
        
        if rank == 0:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        else:
            train_pbar = train_loader
            
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            if config['train']['deepspeed']['enabled']:
                inputs = inputs.to(engine.device, dtype=torch.float16 if config['train']['fp16'] else torch.float32)
                labels = labels.to(engine.device)
                
                outputs = engine(inputs)
                loss = criterion(outputs, labels)
                
                engine.backward(loss)
                engine.step()
                
                current_lr = engine.get_lr()[0] if hasattr(engine, 'get_lr') else config['train']['lr']
            else:
                optimizer.zero_grad()
                
                inputs = inputs.to(device, dtype=torch.float16 if config['train']['fp16'] else torch.float32)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']

            # Reduce metrics across all processes
            batch_loss = reduce_tensor(loss.detach(), world_size)
            batch_acc = reduce_tensor((torch.argmax(outputs, dim=1) == labels).float().mean(), world_size)
            
            running_loss += batch_loss.item() * inputs.size(0)
            num_acc += batch_acc.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Log only on rank 0
            if rank == 0:
                train_step = epoch * len(train_loader) + batch_idx + 1
                train_step_acc = num_acc / total_samples
                train_step_loss = running_loss / total_samples
                
                if config['wandb']['enable'] and batch_idx % 10 == 0:  # Log every 10 batches
                    wandb.log({
                        'train_step_acc': train_step_acc,
                        'train_step_loss': train_step_loss,
                        'lr': current_lr,
                        'epoch': epoch,
                        'train_step': train_step,
                    }, step=train_step)

        # Validation phase
        model.eval()
        num_val_acc = 0.0
        running_val_loss = 0.0
        total_val_samples = 0
        
        with torch.no_grad():
            if rank == 0:
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            else:
                val_pbar = val_loader
                
            for batch_idx, (inputs, labels) in enumerate(val_pbar):
                if config['train']['deepspeed']['enabled']:
                    inputs = inputs.to(engine.device, dtype=torch.float16 if config['train']['fp16'] else torch.float32)
                    labels = labels.to(engine.device)
                    outputs = engine(inputs)
                else:
                    inputs = inputs.to(device, dtype=torch.float16 if config['train']['fp16'] else torch.float32)
                    labels = labels.to(device)
                    outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                
                # Reduce metrics across all processes
                batch_loss = reduce_tensor(loss.detach(), world_size)
                batch_acc = reduce_tensor((torch.argmax(outputs, dim=1) == labels).float().mean(), world_size)
                
                running_val_loss += batch_loss.item() * inputs.size(0)
                num_val_acc += batch_acc.item() * inputs.size(0)
                total_val_samples += inputs.size(0)

        # Calculate epoch metrics
        train_loss = running_loss / len(train_ds)
        train_acc = num_acc / len(train_ds)
        val_loss = running_val_loss / len(val_ds)
        val_acc = num_val_acc / len(val_ds)

        current_epoch += 1
        
        # Log and save only on rank 0
        if rank == 0:
            logging.info(f"Epoch {epoch + 1}")
            logging.info(f"Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
            logging.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")
            
            if config['wandb']['enable']:
                wandb.log({
                    'train_epoch_loss': train_loss,
                    'train_epoch_acc': train_acc,
                    'val_epoch_loss': val_loss,
                    'val_epoch_acc': val_acc,
                    'best_val_acc': val_acc_max,
                }, step=train_step)
            
            # Save best model
            if val_acc > val_acc_max:
                logging.info(f'Validation accuracy increased ({val_acc_max:.6f} --> {val_acc:.6f}).')
                val_acc_max = val_acc
                best_epoch = current_epoch
                
                if val_acc_max > config['train']['save_threshold']:
                    logging.info("Saving model ...")
                    model_name = config['model']['method']
                    save_dir = os.path.join(config['train']['save_dir'], 'experiments', model_name)
                    os.makedirs(save_dir, exist_ok=True)
                    backbone = config['model']['backbone'].replace('-', '_')
                    checkpoint_path = os.path.join(save_dir, f'{model_name}_{backbone}_ddp_best_model_epoch{current_epoch}_acc{val_acc:.4f}.pt')
                    
                    # Save only trainable parameters
                    if config['train']['deepspeed']['enabled']:
                        state_dict = engine.state_dict()
                    else:
                        state_dict = model.module.state_dict()  # Use .module to get underlying model
                    
                    tuning_params = [name for name, param in model.named_parameters() if param.requires_grad]
                    filtered_state_dict = {k: v for k, v in state_dict.items() if k in tuning_params}
                    torch.save(filtered_state_dict, checkpoint_path)
                    logging.info(f"Model saved to {checkpoint_path}")
                
                epoch_since_improvement = 0
            else:
                epoch_since_improvement += 1
                logging.info(f"There's no improvement for {epoch_since_improvement} epochs.")
                if epoch_since_improvement >= patience:
                    logging.info("The training halted by early stopping criterion.")
                    break

    # Cleanup
    if rank == 0:
        logging.info("Training completed.")
        if config['wandb']['enable']:
            wandb.finish()
    
    cleanup_ddp()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DDP Training script for Gaviko model")
    parser.add_argument('--config', type=str, default='configs/original_gaviko.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--method', type=str, default='gaviko',
                        choices=['gaviko', 'fft', 'linear', 'adaptformer', 'bitfit', 'dvpt', 'evp', 'ssf', 'melo', 'deep_vpt','shallow_vpt'],
                        help='Model to train')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--world_size', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--master_port', type=str, default="12355",
                        help='Master port for DDP communication')
    
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config['model']['method'] = args.method
    
    if config['model']['method'] == 'deep_vpt':
        config['model']['deep_prompt'] = True
    elif config['model']['method'] == 'shallow_vpt':
        config['model']['deep_prompt'] = False
    
    config['train']['save_dir'] = args.results_dir if args.results_dir is not None else config['train']['save_dir']
    
    # Set DDP configuration
    world_size = args.world_size if args.world_size is not None else torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs available for training")
    
    config['ddp'] = {
        'master_port': args.master_port,
        'world_size': world_size
    }
    
    print(f"Starting DDP training on {world_size} GPUs")
    print(f"Config: {config}")
    
    # Spawn processes for DDP
    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()