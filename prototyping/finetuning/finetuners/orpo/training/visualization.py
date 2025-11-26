"""
Visualization utilities for ORPO training metrics.
"""

import matplotlib.pyplot as plt
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """
    Generates plots for ORPO training metrics.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_metrics(self, log_history: List[Dict[str, Any]]):
        """
        Plot rewards, margins, and accuracies from log history.
        """
        logger.info("Generating training plots...")
        
        # Extract metrics
        epochs = []
        steps = []
        rewards_chosen = []
        rewards_rejected = []
        margins = []
        accuracies = []
        log_odds_chosen = []
        
        for entry in log_history:
            if "epoch" in entry:
                # Check if we have the relevant metrics in this entry
                if "rewards/chosen" in entry:
                    epochs.append(entry["epoch"])
                    steps.append(entry.get("step", 0))
                    rewards_chosen.append(entry["rewards/chosen"])
                    rewards_rejected.append(entry["rewards/rejected"])
                    margins.append(entry["rewards/margins"])
                    accuracies.append(entry["rewards/accuracies"])
                    log_odds_chosen.append(entry.get("log_odds_chosen", 0)) # might not be in all logs

        if not epochs:
            logger.warning("No metrics found to plot.")
            return

        # Plot Rewards
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, rewards_chosen, label="Rewards Chosen")
        plt.plot(epochs, rewards_rejected, label="Rewards Rejected")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.title("ORPO Rewards over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "rewards_plot.png"))
        plt.close()

        # Plot Margins
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, margins, label="Reward Margin")
        plt.xlabel("Epoch")
        plt.ylabel("Margin")
        plt.title("ORPO Reward Margins over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "margins_plot.png"))
        plt.close()
        
        # Plot Accuracies
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accuracies, label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("ORPO Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "accuracy_plot.png"))
        plt.close()

        logger.info(f"Plots saved to {self.output_dir}")
