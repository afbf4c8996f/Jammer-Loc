import os
import logging
import matplotlib
matplotlib.use('Agg')  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
from typing import List, Dict
from typing import Union
import logging

logger = logging.getLogger(__name__)

# Set a global theme; adjust as desired
sns.set_theme(style="whitegrid")

import os
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def plot_confusion_matrix(
    cm: Union[pd.DataFrame, np.ndarray],
    inverse_label_mapping: dict,
    title: str,
    output_dir: str
):
    """
    Plots and saves a confusion matrix (PDF + SVG at 300 dpi).

    - cm:                    square DataFrame of counts
    - inverse_label_mapping: dict index → original label string
    - title:                 e.g. "Confusion Matrix (Test) – convmixer1d"
    - output_dir:            directory to save the figures
    """
    # If someone passed us an ndarray, wrap it in a DataFrame
    if isinstance(cm, np.ndarray):
        # Build row/col labels from the mapping
        labels = [inverse_label_mapping[i] for i in range(cm.shape[0])]
        cm = pd.DataFrame(cm, index=labels, columns=labels)

    try:
        # Clean up title
        clean_title = title.replace("(Test)", "").strip()
       

        # Build figure
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

        # Draw heatmap (no in-cell annotation)
        sns.heatmap(
            cm,
            annot=False,
            cmap="viridis",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            ax=ax,
        )

        # Titles and labels
        ax.set_title(clean_title, fontsize=14)
        ax.set_xlabel("Predicted Labels", fontsize=12)
        ax.set_ylabel("True Labels", fontsize=12)

        # Rotate tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=10)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save paths
        pdf_path = os.path.join(output_dir, f"{clean_title.replace(' ', '_')}.pdf")
        svg_path = os.path.join(output_dir, f"{clean_title.replace(' ', '_')}.svg")

        # Save at 300 dpi for high resolution
        fig.savefig(pdf_path, dpi=800)
        fig.savefig(svg_path, dpi=800)
        plt.close(fig)

        logger.info(f"Confusion matrix saved to:\n • {pdf_path}\n • {svg_path}")
    except Exception as e:
        logger.error(f"Failed to plot/save confusion matrix '{title}': {e}")



# --- Training Curves ---
def plot_training_curves(
    epoch_metrics: dict,
    model_name: str,
    output_dir: str
):
    try:
        epochs = range(1, len(epoch_metrics.get("train_loss", [])) + 1)
        has_accuracy = (
            "train_accuracy" in epoch_metrics and
            "val_accuracy" in epoch_metrics
        )

        if has_accuracy:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ax_loss, ax_acc = axes
        else:
            fig, ax_loss = plt.subplots(1, 1, figsize=(8, 5))
            ax_acc = None

        # Plot Loss
        ax_loss.plot(
            epochs,
            epoch_metrics.get("train_loss", []),
            marker='o',
            label='Training Loss'
        )
        ax_loss.plot(
            epochs,
            epoch_metrics.get("val_loss", []),
            marker='o',
            label='Validation Loss'
        )
        ax_loss.set_title('Training and Validation Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()

        # Plot Accuracy if available
        if has_accuracy and ax_acc is not None:
            ax_acc.plot(
                epochs,
                epoch_metrics.get("train_accuracy", []),
                marker='o',
                label='Training Accuracy'
            )
            ax_acc.plot(
                epochs,
                epoch_metrics.get("val_accuracy", []),
                marker='o',
                label='Validation Accuracy'
            )
            ax_acc.set_title('Training and Validation Accuracy')
            ax_acc.set_xlabel('Epochs')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.legend()

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"Training curves saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to plot training curves: {e}")

# --- Model Comparison Bar Chart ---
def plot_metric_comparison(
    metrics_dict: Dict[str, Dict[str, Dict[str, float]]],
    metric_name: str,
    output_dir: str
) -> None:
    """
    Bar chart of a single metric on the *test* split across multiple models.

    Args:
        metrics_dict: Nested dict {model: {split: {metric: value}}}.
        metric_name: Metric key to plot.
        output_dir: Directory to save the plot.
    """
    try:
        models = list(metrics_dict.keys())
        scores = [metrics_dict[m]['test'][metric_name] for m in models]
        plt.figure(figsize=(8, 5))
        plt.bar(models, scores)
        plt.title(f"Test {metric_name.upper()} Comparison")
        plt.xlabel('Model')
        plt.ylabel(metric_name)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"model_comparison_{metric_name}.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved model comparison bar chart → {save_path}")
    except Exception as e:
        logger.error(f"Failed to plot metric comparison for {metric_name}: {e}")
