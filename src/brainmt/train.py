import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from scipy.stats import pearsonr
import wandb

from data.datasets import fMRIDataset
from models.brain_mt import BrainMT
from utils.distributed import init_distributed_mode, get_rank, get_world_size, cleanup
from utils.misc import set_seed, seed_worker
from utils.lr_scheduler import CosineAnnealingWarmUpRestarts
from utils.optim_factory import get_parameter_groups, LayerDecayValueAssigner, create_optimizer

log = logging.getLogger(__name__)

def train_one_epoch(model, criteria, data_loader, optimizer, scaler, device, epoch, cfg):
    model.train()
    data_loader.sampler.set_epoch(epoch)
    
    losses = []
    train_outputs = []
    train_targets = []
    progress_bar = None
    if get_rank() == 0:
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1} Training")

    for i, (inputs, targets) in (progress_bar if progress_bar else enumerate(data_loader)):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criteria(outputs.squeeze(1), targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        train_outputs.append(outputs.squeeze(1).detach().cpu())
        train_targets.append(targets.detach().cpu())
        
        # --- Task-specific metric calculation ---
        if cfg.task.name == "classification":
            acc = ((outputs.squeeze(1) >= 0) == targets).float().mean()
            if progress_bar:
                progress_bar.set_postfix(loss=loss.item(), acc=acc.item())
        else:
            if progress_bar:
                progress_bar.set_postfix(loss=loss.item())

    train_outputs = torch.cat(train_outputs, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    
    return np.mean(losses), train_outputs, train_targets

@torch.no_grad()
def evaluate(model, criteria, data_loader, device):
    model.eval()
    val_losses = []
    val_outputs = []
    val_targets = []
    
    val_loader_iter = data_loader
    if get_rank() == 0:
        val_loader_iter = tqdm(data_loader, total=len(data_loader), desc="Validation")
        
    for inputs, targets in val_loader_iter:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criteria(outputs.squeeze(1), targets)
        val_losses.append(loss.item())
        val_outputs.append(outputs.squeeze(1).cpu())
        val_targets.append(targets.cpu())
        
    val_outputs = torch.cat(val_outputs, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    
    return np.mean(val_losses), val_outputs, val_targets

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # --- DDP and Seed Setup ---
    init_distributed_mode(cfg)
    set_seed(cfg.run.seed + get_rank())
    cudnn.benchmark = True
    device = torch.device(cfg.run.device)

    # --- Dataset and Dataloader ---
    if get_rank() == 0:
        log.info("Loading dataset...")
    dataset = fMRIDataset(
        img_path=cfg.dataset.img_path,
        target_path=cfg.dataset.target_path,
        target_col=cfg.dataset.target_col,
        id_col=cfg.dataset.id_col,
        num_frames_slice=cfg.dataset.num_frames_slice
    )
    # Split dataset: 70% train, 15% validation, 15% test
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(cfg.run.seed)
    )
    
    if get_rank() == 0:
        log.info(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg.run.batch_size, num_workers=cfg.run.num_workers, pin_memory=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=cfg.run.batch_size, num_workers=cfg.run.num_workers, pin_memory=True, worker_init_fn=seed_worker)
    
    # --- Model ---
    if get_rank() == 0:
        log.info(f"Creating model: {cfg.model.name}")
        log.info("Step 1: Preparing model config...")
    model_config = {k: v for k, v in cfg.model.items() if k != 'name'}
    if get_rank() == 0:
        log.info("Step 2: About to instantiate BrainMT...")
    model = BrainMT(**model_config)
    if get_rank() == 0:
        log.info("Step 3: Model created, moving to device...")
    model = model.to(device)
    if get_rank() == 0:
        log.info("Step 4: About to wrap with DDP...")
    
    if get_rank() == 0:
        log.info("Step 4a: All processes ready, creating DDP...")
    
    # Get the current device ID for this process
    device_id = torch.cuda.current_device()
    
    if get_rank() == 0:
        log.info(f"Creating DDP with device_id: {device_id}")
    
    # Add a barrier to ensure all processes are ready
    torch.distributed.barrier()
    
    if get_rank() == 0:
        log.info("All processes synchronized, creating DDP...")
    
    try:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False,
            broadcast_buffers=True,
        )
        
        if get_rank() == 0:
            log.info("Step 5: DDP creation successful!")
            
    except Exception as e:
        log.error(f"DDP creation failed on rank {get_rank()}: {e}")
        raise
    
    if get_rank() == 0:
        log.info("Step 6: About to check model parameters...")
    
    if get_rank() == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Number of trainable parameters: {n_params / 1e6:.2f}M")
        log.info("Step 7: Parameter counting complete.")

    if cfg.task.loss_fn == "mse":
        criteria = nn.MSELoss()
        if get_rank() == 0:
            log.info("Using Mean Squared Error (MSE) loss for regression.")
    elif cfg.task.loss_fn == "bce_with_logits":
        criteria = nn.BCEWithLogitsLoss()
        if get_rank() == 0:
            log.info("Using Binary Cross-Entropy with Logits for classification.")
    else:
        raise ValueError(f"Unsupported loss function: {cfg.task.loss_fn}")

    if get_rank() == 0:
        log.info("Step 8: About to create optimizer...")

    # --- Optimizer and Scheduler ---
    num_layers = model.module.get_num_layers()
    assigner = LayerDecayValueAssigner(list(cfg.optimizer.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    skip_weight_decay_list = model.module.no_weight_decay()
    
    optimizer = create_optimizer(
        model, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id, 
        get_layer_scale=assigner.get_scale
    )
    
    total_steps = len(train_loader) * cfg.run.epochs
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, 
        first_cycle_steps=len(train_loader) * cfg.scheduler.first_cycle_epochs, 
        cycle_mult=cfg.scheduler.cycle_mult, 
        max_lr=cfg.optimizer.lr, 
        min_lr=cfg.scheduler.min_lr, 
        warmup_steps=len(train_loader) * cfg.scheduler.warmup_epochs, 
        gamma=cfg.scheduler.gamma
    )
    
    scaler = torch.amp.GradScaler()

    # --- Logging (W&B) ---
    if get_rank() == 0:
        name = f"{cfg.model.name}_{cfg.dataset.ds_name}_{cfg.task.name}_seed{cfg.run.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=cfg.logging.project_name, name=name, config=OmegaConf.to_container(cfg, resolve=True))

    # --- Training Loop ---
    best_val_loss = float('inf')
    for epoch in range(cfg.run.epochs):
        train_loss, train_outputs, train_targets = train_one_epoch(model, criteria, train_loader, optimizer, scaler, device, epoch, cfg)
        val_loss, val_outputs, val_targets = evaluate(model, criteria, val_loader, device)
        scheduler.step()

        # Gather results from all processes
        gathered_train_outputs, gathered_train_targets = [None] * get_world_size(), [None] * get_world_size()
        gathered_val_outputs, gathered_val_targets = [None] * get_world_size(), [None] * get_world_size()
        
        torch.distributed.all_gather_object(gathered_train_outputs, train_outputs)
        torch.distributed.all_gather_object(gathered_train_targets, train_targets)
        torch.distributed.all_gather_object(gathered_val_outputs, val_outputs)
        torch.distributed.all_gather_object(gathered_val_targets, val_targets)
        
        if get_rank() == 0:
            all_train_outputs = torch.cat(gathered_train_outputs, dim=0)
            all_train_targets = torch.cat(gathered_train_targets, dim=0)
            all_val_outputs = torch.cat(gathered_val_outputs, dim=0)
            all_val_targets = torch.cat(gathered_val_targets, dim=0)
            
            wandb_log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]['lr']}
            log_str = f"Epoch {epoch+1}/{cfg.run.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"

            # --- Task-Specific Evaluation & Logging ---
            if cfg.task.name == 'classification':
                # Training metrics
                train_acc_func = BinaryAccuracy().to("cpu")
                train_auroc_func = BinaryAUROC().to("cpu")
                
                train_acc = train_acc_func((all_train_outputs >= 0).float(), all_train_targets.int())
                train_bal_acc = balanced_accuracy_score(all_train_targets.numpy(), (all_train_outputs.numpy() >= 0))
                train_auroc = train_auroc_func(torch.sigmoid(all_train_outputs), all_train_targets.int())
                
                # Validation metrics
                val_acc_func = BinaryAccuracy().to("cpu")
                val_auroc_func = BinaryAUROC().to("cpu")
                
                val_acc = val_acc_func((all_val_outputs >= 0).float(), all_val_targets.int())
                val_bal_acc = balanced_accuracy_score(all_val_targets.numpy(), (all_val_outputs.numpy() >= 0))
                val_auroc = val_auroc_func(torch.sigmoid(all_val_outputs), all_val_targets.int())
                
                wandb_log.update({
                    "train_acc": train_acc.item(),
                    "train_balanced_accuracy": train_bal_acc,
                    "train_auroc": train_auroc.item(),
                    "val_accuracy": val_acc.item(),
                    "val_balanced_accuracy": val_bal_acc,
                    "val_auroc": val_auroc.item()
                })
                log_str += f" | Train Acc: {train_acc:.4f} | Train AUROC: {train_auroc:.4f} | Val Acc: {val_acc:.4f} | Val AUROC: {val_auroc:.4f}"

            elif cfg.task.name == 'regression':
                # Training metrics
                all_train_outputs_np = all_train_outputs.numpy()
                all_train_targets_np = all_train_targets.numpy()
                
                train_mse = mean_squared_error(all_train_targets_np, all_train_outputs_np)
                train_mae = mean_absolute_error(all_train_targets_np, all_train_outputs_np)
                train_corr, _ = pearsonr(all_train_targets_np.flatten(), all_train_outputs_np.flatten())
                
                # Validation metrics
                all_val_outputs_np = all_val_outputs.numpy()
                all_val_targets_np = all_val_targets.numpy()
                
                val_mse = mean_squared_error(all_val_targets_np, all_val_outputs_np)
                val_mae = mean_absolute_error(all_val_targets_np, all_val_outputs_np)
                val_corr, _ = pearsonr(all_val_targets_np.flatten(), all_val_outputs_np.flatten())

                wandb_log.update({
                    "train_mse": train_mse, 
                    "train_mae": train_mae, 
                    "train_corr": train_corr,
                    "val_mse": val_mse, 
                    "val_mae": val_mae, 
                    "val_corr": val_corr
                })
                log_str += f" | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Val R: {val_corr:.4f}"
            
            log.info(log_str)
            wandb.log(wandb_log)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(cfg.checkpoint.dir, f"{name}_best.pth")
                os.makedirs(cfg.checkpoint.dir, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'config': OmegaConf.to_container(cfg, resolve=True)
                }, checkpoint_path)
                log.info(f"Best model saved to {checkpoint_path} at epoch {epoch+1}")
    
    if get_rank() == 0:
        wandb.finish()
    cleanup()

if __name__ == '__main__':
    main()
