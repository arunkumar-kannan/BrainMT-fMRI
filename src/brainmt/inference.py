import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import random_split, DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score, classification_report, confusion_matrix
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score
from scipy.stats import pearsonr
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from data.datasets import fMRIDataset
from models.brain_mt import BrainMT
from utils.misc import set_seed, seed_worker

log = logging.getLogger(__name__)

@torch.no_grad()
def inference(model, criterion, data_loader, device, cfg):
    """Run inference on test dataset"""
    model.eval()
    test_losses = []
    test_outputs = []
    test_targets = []
    
    log.info("Running inference on test dataset...")
    progress_bar = tqdm(data_loader, desc="Inference", total=len(data_loader))
    
    for inputs, targets in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if cfg.training.use_amp:
            with amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(1), targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), targets)
        
        test_losses.append(loss.item())
        test_outputs.append(outputs.squeeze(1).cpu())
        test_targets.append(targets.cpu())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    test_outputs = torch.cat(test_outputs, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    return np.mean(test_losses), test_outputs, test_targets

def calculate_classification_metrics(outputs, targets):
    """Calculate comprehensive classification metrics"""
    probs = torch.sigmoid(outputs)
    predictions = (probs > 0.5).float()
    
    # Convert to numpy for sklearn metrics
    targets_np = targets.numpy().astype(int)
    predictions_np = predictions.numpy().astype(int)
    probs_np = probs.numpy()
    
    # Basic metrics
    acc_func = BinaryAccuracy()
    precision_func = BinaryPrecision()
    recall_func = BinaryRecall()
    f1_func = BinaryF1Score()
    auroc_func = BinaryAUROC()
    
    accuracy = acc_func(predictions, targets.int()).item()
    precision = precision_func(predictions, targets.int()).item()
    recall = recall_func(predictions, targets.int()).item()
    f1_score = f1_func(predictions, targets.int()).item()
    auroc = auroc_func(probs, targets.int()).item()
    balanced_acc = balanced_accuracy_score(targets_np, predictions_np)
    
    # Classification report
    class_report = classification_report(targets_np, predictions_np, output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(targets_np, predictions_np)
    
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'auroc': auroc,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'predictions': predictions_np,
        'probabilities': probs_np
    }
    
    return metrics

def calculate_regression_metrics(outputs, targets):
    """Calculate comprehensive regression metrics"""
    outputs_np = outputs.numpy()
    targets_np = targets.numpy()
    
    # Basic regression metrics
    mse = mean_squared_error(targets_np, outputs_np)
    mae = mean_absolute_error(targets_np, outputs_np)
    
    # Correlation metrics
    pearson_r, pearson_p = pearsonr(targets_np.flatten(), outputs_np.flatten())
    
    # R-squared
    ss_res = np.sum((targets_np - outputs_np) ** 2)
    ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'r_squared': r_squared,
        'predictions': outputs_np,
        'targets': targets_np
    }
    
    return metrics

def plot_results(metrics, cfg, save_dir):
    """Create visualization plots for results"""
    os.makedirs(save_dir, exist_ok=True)
    
    if cfg.task.name == 'classification':
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curve (simplified - just the AUC score is calculated above)
        log.info(f"Classification plots saved to {save_dir}")
        
    elif cfg.task.name == 'regression':
        # Scatter plot of predictions vs targets
        plt.figure(figsize=(10, 8))
        plt.scatter(metrics['targets'], metrics['predictions'], alpha=0.6)
        plt.plot([metrics['targets'].min(), metrics['targets'].max()], 
                [metrics['targets'].min(), metrics['targets'].max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'Predictions vs True Values (R² = {metrics["r_squared"]:.3f})')
        plt.savefig(os.path.join(save_dir, 'predictions_vs_targets.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Residuals plot
        residuals = metrics['targets'] - metrics['predictions']
        plt.figure(figsize=(10, 6))
        plt.scatter(metrics['predictions'], residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predictions')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.savefig(os.path.join(save_dir, 'residuals.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Regression plots saved to {save_dir}")

def save_results(metrics, cfg, save_path):
    """Save detailed results to file"""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"INFERENCE RESULTS - {cfg.task.name.upper()}\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {cfg.model.name}\n")
        f.write(f"Dataset: {cfg.dataset.ds_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        if cfg.task.name == 'classification':
            f.write("CLASSIFICATION METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"AUROC: {metrics['auroc']:.4f}\n")
            f.write("\n")
            
            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 20 + "\n")
            f.write(str(metrics['confusion_matrix']) + "\n")
            f.write("\n")
            
        elif cfg.task.name == 'regression':
            f.write("REGRESSION METRICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"MSE: {metrics['mse']:.6f}\n")
            f.write(f"MAE: {metrics['mae']:.6f}\n")
            f.write(f"R²: {metrics['r_squared']:.4f}\n")
            f.write(f"Pearson R: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})\n")
    
    log.info(f"Results saved to {save_path}")

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Print configuration
    log.info("Inference Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(cfg.run.seed)
    cudnn.benchmark = True
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join("inference_results", f"{cfg.model.name}_{cfg.task.name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset and create the same split as in training
    log.info("Loading dataset...")
    dataset = fMRIDataset(
        img_path=cfg.dataset.img_path,
        target_path=cfg.dataset.target_path,
        target_col=cfg.dataset.target_col,
        id_col=cfg.dataset.id_col,
        num_frames_slice=cfg.dataset.num_frames_slice
    )
    
    # Use the same split as in training: 70% train, 15% validation, 15% test
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(cfg.run.seed)
    )
    
    log.info(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.run.batch_size,
        shuffle=False,
        num_workers=cfg.run.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    
    # Load model
    log.info(f"Loading model: {cfg.model.name}")
    model_config = {k: v for k, v in cfg.model.items() if k != 'name'}
    model = BrainMT(**model_config).to(device)
    
    # Load trained weights
    checkpoint_path = cfg.get('inference', {}).get('checkpoint_path', 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        # Try to find the checkpoint in the checkpoint directory
        potential_paths = [
            os.path.join(cfg.checkpoint.dir, f"{cfg.model.name}_{cfg.dataset.ds_name}_{cfg.task.name}_best.pth"),
            os.path.join(cfg.checkpoint.dir, "best_model.pth"),
            "best_model.pth"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        else:
            raise FileNotFoundError(f"Could not find model checkpoint. Tried: {potential_paths}")
    
    log.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model loaded with {n_params / 1e6:.2f}M parameters")
    
    # Set up loss function
    if cfg.task.loss_fn == "mse":
        criterion = nn.MSELoss()
        log.info("Using Mean Squared Error (MSE) loss for regression.")
    elif cfg.task.loss_fn == "bce_with_logits":
        criterion = nn.BCEWithLogitsLoss()
        log.info("Using Binary Cross-Entropy with Logits for classification.")
    else:
        raise ValueError(f"Unsupported loss function: {cfg.task.loss_fn}")
    
    # Run inference
    test_loss, test_outputs, test_targets = inference(model, criterion, test_loader, device, cfg)
    
    log.info(f"Test Loss: {test_loss:.6f}")
    
    # Calculate metrics
    if cfg.task.name == 'classification':
        metrics = calculate_classification_metrics(test_outputs, test_targets)
        
        log.info("CLASSIFICATION RESULTS:")
        log.info(f"Accuracy: {metrics['accuracy']:.4f}")
        log.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        log.info(f"Precision: {metrics['precision']:.4f}")
        log.info(f"Recall: {metrics['recall']:.4f}")
        log.info(f"F1-Score: {metrics['f1_score']:.4f}")
        log.info(f"AUROC: {metrics['auroc']:.4f}")
        
        # WandB logging for classification
        wandb_log = {
            "test_loss": test_loss,
            "test_accuracy": metrics['accuracy'],
            "test_balanced_accuracy": metrics['balanced_accuracy'],
            "test_precision": metrics['precision'],
            "test_recall": metrics['recall'],
            "test_f1_score": metrics['f1_score'],
            "test_auroc": metrics['auroc']
        }
        
    elif cfg.task.name == 'regression':
        metrics = calculate_regression_metrics(test_outputs, test_targets)
        
        log.info("REGRESSION RESULTS:")
        log.info(f"MSE: {metrics['mse']:.6f}")
        log.info(f"MAE: {metrics['mae']:.6f}")
        log.info(f"R²: {metrics['r_squared']:.4f}")
        log.info(f"Pearson R: {metrics['pearson_r']:.4f}")
        
        # WandB logging for regression
        wandb_log = {
            "test_loss": test_loss,
            "test_mse": metrics['mse'],
            "test_mae": metrics['mae'],
            "test_r_squared": metrics['r_squared'],
            "test_pearson_r": metrics['pearson_r'],
        }
    
    # Initialize wandb for logging results
    if cfg.logging.get('use_wandb', False):
        wandb.init(
            project=cfg.logging.project_name + "_inference",
            name=f"inference_{cfg.model.name}_{cfg.task.name}_{timestamp}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        wandb.log(wandb_log)
    
    # Create plots
    plot_results(metrics, cfg, output_dir)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "inference_results.txt")
    save_results(metrics, cfg, results_file)
    
    # Save predictions and targets for further analysis
    predictions_file = os.path.join(output_dir, "predictions.npz")
    if cfg.task.name == 'classification':
        np.savez(predictions_file, 
                predictions=metrics['predictions'],
                probabilities=metrics['probabilities'],
                targets=test_targets.numpy(),
                outputs=test_outputs.numpy())
    else:
        np.savez(predictions_file,
                predictions=metrics['predictions'],
                targets=metrics['targets'],
                outputs=test_outputs.numpy())
    
    log.info(f"Predictions saved to {predictions_file}")
    
    # Cleanup wandb
    if cfg.logging.get('use_wandb', False):
        wandb.finish()
    
    log.info(f"Inference completed! Results saved in: {output_dir}")

if __name__ == '__main__':
    main()
