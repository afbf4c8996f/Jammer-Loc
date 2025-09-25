# train_utils.py

import os
import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_auc_score
from metrics import compute_metrics
from tqdm import tqdm
import logging
from typing import Tuple, Optional, Dict, Any


from typing import Optional, Any, Tuple, Dict
import os
import torch
import numpy as np
from tqdm import tqdm
import logging
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_auc_score
from metrics import compute_metrics


# Obtain a module-specific logger
logger = logging.getLogger(__name__)


def save_checkpoint(
    filepath: str,
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    scheduler_state: Dict[str, Any],
    epoch: int,
    best_val_loss: float,
    no_improvement_count: int
) -> None:
    """
    Save the training checkpoint.

    Args:
        filepath (str): Path to save the checkpoint.
        model_state (dict): State dictionary of the model.
        optimizer_state (dict): State dictionary of the optimizer.
        scheduler_state (dict): State dictionary of the scheduler.
        epoch (int): Current epoch number.
        best_val_loss (float): Best validation loss achieved.
        no_improvement_count (int): Number of epochs with no improvement.
    """
    try:
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "no_improvement_count": no_improvement_count
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {e}")

def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None) -> Tuple[Optional[int], Optional[float], Optional[int]]:
    """
    Load the training checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
        scheduler (Any, optional): The scheduler to load the state into.

    Returns:
        tuple: (epoch, best_val_loss, no_improvement_count)
    """
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model state loaded from {filepath}")

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded.")

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Scheduler state loaded.")

        epoch = checkpoint.get("epoch", None)
        best_val_loss = checkpoint.get("best_val_loss", None)
        no_improvement_count = checkpoint.get("no_improvement_count", None)

        return epoch, best_val_loss, no_improvement_count
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filepath}: {e}")
        return None, None, None

def evaluate_model_on_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    split_name: str = ""
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a given data loader.
    Returns average loss, true labels, predictions, and probabilities.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to perform computations on.
        criterion (torch.nn.Module): Loss function.

    Returns:
        tuple: (avg_loss, all_labels, all_preds, all_probs)
    """
    logger.info("Starting model evaluation.")

    model.to(device)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_preds = []
    all_probs = []

    bar_desc = f"Evaluating ({split_name})" if split_name else "Evaluating"
    pbar = tqdm(loader, desc=bar_desc, leave=False)
   
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(pbar, 1):
            try:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                batch_size = y_batch.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                _, predicted = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                all_labels.extend(y_batch.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs)

                pbar.set_postfix({'Batch Loss': loss.item()})
                logger.debug(f"Processed batch {batch_idx}: Loss={loss.item():.4f}")
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    pbar.close()

    if total_samples == 0:
        logger.error("No samples were processed during evaluation.")
        avg_loss = float('nan')
    else:
        avg_loss = total_loss / total_samples
        logger.info(f"Evaluation completed. Average Loss: {avg_loss:.4f}")

    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)




def train_and_select_best_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int,
    scheduler: Any,
    patience: int,
    output_dir: str,
    model_name: str,
    cfg: Any,
    task: str = "classification",
    writer: Optional[Any] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, list]]:
    """
    Train the model and select the best epoch based on validation performance.

    This function now uses a unified `monitor_val` metric for both
    learning-rate adjustments and checkpointing:
      - Classification: monitors `val_accuracy` (higher is better).
      - Regression: monitors either `val_median_distance` or `val_loss`,
        per `cfg.training.scheduler_monitor` (lower is better).
    """
    # Initialize best_monitor and best_model_state based on task
    if task == "classification":
        best_monitor = float('-inf')
    else:
        best_monitor = float('inf')
    best_model_state = None
    no_improvement = 0

    # Initialize metrics containers
    if task == "classification":
        epoch_metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1_weighted": [],
        }
    else:  # regression
        epoch_metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_median_distance": [],
        }

    for epoch in range(1, num_epochs + 1):
        model.train()
        logger.info(f"Epoch {epoch}/{num_epochs} - Training started.")

        total_loss = 0.0
        total_samples = 0
        if task == "classification":
            train_preds, train_labels = [], []

        # ── Training loop ────────────────────────────────────────────
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            if task == "classification":
                _, preds = torch.max(out, 1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(yb.cpu().numpy())

        # Compute train metrics
        avg_train_loss = total_loss / total_samples if total_samples else float('nan')
        epoch_metrics["train_loss"].append(avg_train_loss)
        if task == "classification":
            train_acc = compute_metrics(train_labels, train_preds).get("accuracy", float('nan'))
            epoch_metrics["train_accuracy"].append(train_acc)
            logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Accuracy={train_acc:.4f}")
        else:
            logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}")

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        if task == "classification":
            val_loss, val_labels, val_preds, _ = evaluate_model_on_loader(
                model, val_loader, device, criterion, split_name="val"
            )
            cm_val = compute_metrics(val_labels, val_preds) if len(val_preds) > 0 else {}
            epoch_metrics["val_loss"].append(val_loss)
            epoch_metrics["val_accuracy"].append(cm_val.get("accuracy", float('nan')))
            epoch_metrics["val_f1_weighted"].append(cm_val.get("f1_weighted", float('nan')))
            logger.info(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Accuracy={cm_val.get('accuracy', float('nan')):.4f}")
        else:
            total_val_loss = 0.0
            total_val_samples = 0
            preds_list, true_list = [], []
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                l = criterion(out, yb)
                bs = yb.size(0)
                total_val_loss += l.item() * bs
                total_val_samples += bs
                preds_list.append(out.detach().cpu().numpy())
                true_list.append(yb.detach().cpu().numpy())

            val_loss = total_val_loss / total_val_samples if total_val_samples else float('nan')
            epoch_metrics["val_loss"].append(val_loss)
            preds = np.vstack(preds_list)
            trues = np.vstack(true_list)
            dists = np.sqrt((preds - trues) ** 2).sum(axis=1)
            epoch_metrics["val_median_distance"].append(float(np.median(dists)))
            logger.info(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Median Dist={epoch_metrics['val_median_distance'][-1]:.4f}")

        # ── Scheduler step & checkpoint ──────────────────────────────
        if task == "classification":
            monitor_name = "val_accuracy"
            monitor_val = epoch_metrics["val_accuracy"][-1]
        else:
            if cfg.training.scheduler_monitor == "median":
                monitor_name = "val_median_distance"
                monitor_val = epoch_metrics["val_median_distance"][-1]
            else:
                monitor_name = "val_loss"
                monitor_val = epoch_metrics["val_loss"][-1]

        scheduler.step(monitor_val)
        logger.info(f"Scheduler.step() on {monitor_name}={monitor_val:.4f} LR now: {scheduler.get_last_lr()}")

        # Check for improvement on the same metric
        improved = (monitor_val > best_monitor) if task == "classification" else (monitor_val < best_monitor)
        if improved:
            best_monitor = monitor_val
            best_model_state = model.state_dict()
            no_improvement = 0
            os.makedirs(output_dir, exist_ok=True)
            ckpt_path = os.path.join(output_dir, f"{model_name}_best_checkpoint.pth")
            save_checkpoint(
                filepath=ckpt_path,
                model_state=best_model_state,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                epoch=epoch,
                best_val_loss=best_monitor,
                no_improvement_count=no_improvement,
            )
            logger.info(f"Epoch {epoch}: Improved {monitor_name} to {monitor_val:.4f}; saved checkpoint.")
        else:
            no_improvement += 1
            logger.info(f"Epoch {epoch}: No improvement ({no_improvement}/{patience}).")
            if no_improvement >= patience:
                logger.warning(f"Early stopping at epoch {epoch}.")
                break

    logger.info("Training complete.")
    return best_model_state, epoch_metrics


def evaluate_on_test(model: torch.nn.Module, criterion: torch.nn.Module, test_loader: torch.utils.data.DataLoader, 
                     device: torch.device, num_classes: int, split_name: str = "test") -> tuple:
    """
    Evaluate the model on the test set and compute metrics.
    Returns test_loss, test_metrics, confusion matrix, overall AUC, and additional variables.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        criterion (torch.nn.Module): Loss function.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device to perform computations on.
        num_classes (int): Number of classes for classification.

    Returns:
        tuple: (test_loss, test_metrics, cm, overall_auc, test_labels, test_preds, test_probs)
    """
    try:
        test_loss, test_labels, test_preds, test_probs = evaluate_model_on_loader(model, test_loader, device, criterion,split_name=split_name)
    except Exception as e:
        logger.error(f"Failed to evaluate on {split_name} loader: {e}")
        return float('nan'), {}, np.array([]), 'N/A', np.array([]), np.array([]), np.array([])

    if len(test_labels) == 0:
        logger.error(f"No samples were processed on {split_name} set. Evaluation aborted.")
        return test_loss, {}, np.array([]), 'N/A', test_labels, test_preds, test_probs

    # Compute test metrics
    test_metrics = compute_metrics(test_labels, test_preds)
    logger.info("\n--- Final Test Results ---")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics.get('accuracy', float('nan')):.4f}")
    logger.info(f"Test F1 Macro: {test_metrics.get('f1_macro', float('nan')):.4f}")
    logger.info(f"Test F1 Weighted: {test_metrics.get('f1_weighted', float('nan')):.4f}")
    logger.info(f"Test Precision: {test_metrics.get('precision', float('nan')):.4f}")
    logger.info(f"Test Recall: {test_metrics.get('recall', float('nan')):.4f}")

    # Compute confusion matrix
    try:
        cm = confusion_matrix(test_labels, test_preds, labels=np.arange(num_classes))
        logger.info("Confusion matrix computed.")
    except Exception as e:
        logger.error(f"Failed to compute confusion matrix: {e}")
        cm = np.array([])

    # Compute AUC
    try:
        unique_classes_in_test = np.unique(test_labels)
        if len(unique_classes_in_test) < 2:
            overall_auc = 'N/A'
            logger.warning("Only one class present in the test set; AUC is undefined.")
        else:
            y_true_binarized = label_binarize(test_labels, classes=np.arange(num_classes))
            # If our test set is missing some classes entirely
            if y_true_binarized.shape[1] < num_classes:
                overall_auc = 'N/A'
                logger.warning("Some classes missing in test set, skipping AUC.")
            else:
                overall_auc = roc_auc_score(y_true_binarized, test_probs, multi_class='ovr')
                logger.info(f"Overall AUC: {overall_auc:.4f}")
    except Exception as e:
        overall_auc = 'N/A'
        logger.error(f"Failed to compute AUC: {e}")

    return test_loss, test_metrics, cm, overall_auc, test_labels, test_preds, test_probs
