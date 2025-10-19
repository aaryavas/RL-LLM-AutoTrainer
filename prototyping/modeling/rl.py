"""
RL integration for iterative PEFT fine-tuning.
This module provides interfaces for incorporating reinforcement learning
into the iterative training process.
"""
from typing import Dict, List, Optional, Any, Callable, Tuple
from abc import ABC, abstractmethod
import numpy as np
import torch
from pathlib import Path

from .config import FrameworkConfig
from .peft import PEFTFineTuner


class IterationRewardFunction(ABC):
    """Abstract base class for reward functions that evaluate training iterations."""

    @abstractmethod
    def __call__(self, iteration_results: Dict[str, Any], previous_results: List[Dict[str, Any]]) -> float:
        """Calculate reward for a training iteration.

        Args:
            iteration_results: Results from the current iteration
            previous_results: Results from all previous iterations

        Returns:
            Reward value (higher is better)
        """
        pass


class TrainingLossReward(IterationRewardFunction):
    """Reward based on training loss improvement."""

    def __init__(self, improvement_weight: float = 1.0, absolute_weight: float = 0.1):
        self.improvement_weight = improvement_weight
        self.absolute_weight = absolute_weight

    def __call__(self, iteration_results: Dict[str, Any], previous_results: List[Dict[str, Any]]) -> float:
        current_loss = iteration_results.get("train_loss", float('inf'))

        # Base reward from current loss (lower loss = higher reward)
        base_reward = -current_loss * self.absolute_weight

        # Improvement reward
        if previous_results:
            prev_loss = previous_results[-1].get("train_loss", float('inf'))
            improvement = prev_loss - current_loss
            improvement_reward = improvement * self.improvement_weight
        else:
            improvement_reward = 0.0

        return base_reward + improvement_reward


class EvaluationReward(IterationRewardFunction):
    """Reward based on evaluation metrics."""

    def __init__(self, eval_metric: str = "eval_loss", metric_weight: float = 1.0):
        self.eval_metric = eval_metric
        self.metric_weight = metric_weight

    def __call__(self, iteration_results: Dict[str, Any], previous_results: List[Dict[str, Any]]) -> float:
        eval_results = iteration_results.get("eval_results", {})
        current_metric = eval_results.get(self.eval_metric, float('inf'))

        if self.eval_metric == "eval_loss":
            # Lower loss is better
            return -current_metric * self.metric_weight
        else:
            # Higher metric is better (e.g., accuracy, f1)
            return current_metric * self.metric_weight


class CombinedReward(IterationRewardFunction):
    """Combine multiple reward functions."""

    def __init__(self, reward_functions: List[Tuple[IterationRewardFunction, float]]):
        self.reward_functions = reward_functions

    def __call__(self, iteration_results: Dict[str, Any], previous_results: List[Dict[str, Any]]) -> float:
        total_reward = 0.0
        for reward_fn, weight in self.reward_functions:
            total_reward += weight * reward_fn(iteration_results, previous_results)
        return total_reward


class HyperparameterAction:
    """Represents a hyperparameter modification action."""

    def __init__(self, param_name: str, param_value: Any, param_section: str = "training"):
        self.param_name = param_name
        self.param_value = param_value
        self.param_section = param_section  # 'training', 'peft', 'data', etc.

    def apply_to_config(self, config: FrameworkConfig) -> FrameworkConfig:
        """Apply this action to a configuration."""
        # Create a copy to avoid modifying the original
        import copy
        config_copy = copy.deepcopy(config)

        section = getattr(config_copy, self.param_section)
        if hasattr(section, self.param_name):
            setattr(section, self.param_name, self.param_value)

        return config_copy

    def __repr__(self) -> str:
        return f"HyperparameterAction({self.param_section}.{self.param_name} = {self.param_value})"


class TrainingActionSpace:
    """Defines the action space for training modifications."""

    def __init__(self):
        self.learning_rates = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
        self.batch_sizes = [1, 2, 4, 8, 16]
        self.gradient_accumulation_steps = [1, 2, 4, 8, 16]
        self.peft_ranks = [8, 16, 32, 64]
        self.warmup_ratios = [0.0, 0.05, 0.1, 0.15, 0.2]

    def get_action_space(self) -> List[HyperparameterAction]:
        """Get all possible actions."""
        actions = []

        # Learning rate actions
        for lr in self.learning_rates:
            actions.append(HyperparameterAction("learning_rate", lr, "training"))

        # Batch size actions
        for bs in self.batch_sizes:
            actions.append(HyperparameterAction("per_device_train_batch_size", bs, "training"))

        # Gradient accumulation actions
        for gas in self.gradient_accumulation_steps:
            actions.append(HyperparameterAction("gradient_accumulation_steps", gas, "training"))

        # PEFT rank actions
        for r in self.peft_ranks:
            actions.append(HyperparameterAction("r", r, "peft"))

        # Warmup ratio actions
        for wr in self.warmup_ratios:
            actions.append(HyperparameterAction("warmup_ratio", wr, "training"))

        return actions

    def sample_random_action(self) -> HyperparameterAction:
        """Sample a random action from the action space."""
        actions = self.get_action_space()
        return np.random.choice(actions)


class RLTrainer:
    """RL-based PEFT trainer that learns optimal training strategies."""

    def __init__(
        self,
        reward_function: IterationRewardFunction,
        action_space: TrainingActionSpace,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.1,
        learning_rate: float = 1e-3
    ):
        self.reward_function = reward_function
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate

        # Q-table for storing action values (action -> value)
        self.q_table: Dict[str, float] = {}
        self.visit_counts: Dict[str, int] = {}

        # Training history
        self.training_history: List[Dict[str, Any]] = []

    def choose_action(self, state: Dict[str, Any]) -> HyperparameterAction:
        """Choose an action using epsilon-greedy policy."""
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            return self.action_space.sample_random_action()
        else:
            # Exploit: best action based on Q-values
            actions = self.action_space.get_action_space()
            action_values = []

            for action in actions:
                action_key = self._action_to_key(action)
                q_value = self.q_table.get(action_key, 0.0)
                action_values.append((action, q_value))

            # Return action with highest Q-value
            best_action = max(action_values, key=lambda x: x[1])[0]
            return best_action

    def update_q_value(self, action: HyperparameterAction, reward: float, next_state: Optional[Dict[str, Any]] = None):
        """Update Q-value for an action using the reward."""
        action_key = self._action_to_key(action)

        # Initialize Q-value if first visit
        if action_key not in self.q_table:
            self.q_table[action_key] = 0.0
            self.visit_counts[action_key] = 0

        # Q-learning update
        current_q = self.q_table[action_key]
        self.visit_counts[action_key] += 1

        # Calculate target (simplified: just the reward since we're looking at immediate rewards)
        target = reward

        # Update Q-value
        self.q_table[action_key] = current_q + self.learning_rate * (target - current_q)

    def train_iteration_with_rl(
        self,
        tuner: PEFTFineTuner,
        iteration: int,
        previous_results: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], float]:
        """Run one training iteration with RL-based hyperparameter selection."""

        # Choose action (hyperparameter configuration)
        state = {"iteration": iteration, "previous_results": previous_results}
        action = self.choose_action(state)

        print(f"RL chose action: {action}")

        # Apply action to config
        modified_config = action.apply_to_config(tuner.config)
        tuner.config = modified_config

        # Run training iteration
        iteration_result = tuner.train_iteration(iteration)

        # Calculate reward
        reward = self.reward_function(iteration_result, previous_results)

        # Update Q-value
        self.update_q_value(action, reward)

        # Store in history
        self.training_history.append({
            "iteration": iteration,
            "action": action.__dict__,
            "reward": reward,
            "results": iteration_result
        })

        print(f"Iteration {iteration} reward: {reward}")

        return iteration_result, reward

    def _action_to_key(self, action: HyperparameterAction) -> str:
        """Convert action to a hashable key for Q-table."""
        return f"{action.param_section}.{action.param_name}:{action.param_value}"

    def get_best_action_sequence(self) -> List[HyperparameterAction]:
        """Get the sequence of actions that were taken (for logging)."""
        actions = []
        for history_item in self.training_history:
            action_dict = history_item["action"]
            action = HyperparameterAction(**action_dict)
            actions.append(action)
        return actions

    def save_policy(self, filepath: str) -> None:
        """Save the learned policy/Q-table."""
        policy_data = {
            "q_table": self.q_table,
            "visit_counts": self.visit_counts,
            "training_history": self.training_history
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(policy_data, f, indent=2, default=str)

        print(f"Policy saved to: {filepath}")

    def load_policy(self, filepath: str) -> None:
        """Load a pretrained policy."""
        import json
        with open(filepath, 'r') as f:
            policy_data = json.load(f)

        self.q_table = policy_data.get("q_table", {})
        self.visit_counts = policy_data.get("visit_counts", {})
        self.training_history = policy_data.get("training_history", [])

        print(f"Policy loaded from: {filepath}")


class RLPEFTFineTuner(PEFTFineTuner):
    """PEFT Fine-tuner with RL optimization."""

    def __init__(self, config: FrameworkConfig, rl_trainer: RLTrainer):
        super().__init__(config)
        self.rl_trainer = rl_trainer

    def iterative_train_with_rl(self) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Run iterative training with RL optimization."""
        results = []
        rewards = []

        for iteration in range(1, self.config.iterative.max_iterations + 1):
            # Run iteration with RL
            iteration_result, reward = self.rl_trainer.train_iteration_with_rl(
                self, iteration, results
            )

            results.append(iteration_result)
            rewards.append(reward)

            # Check for early stopping
            if self._should_stop_iteration(iteration, iteration_result):
                break

        # Save results
        self._save_iterative_results(results)
        self._save_rl_results(results, rewards)

        return results, rewards

    def _save_rl_results(self, results: List[Dict[str, Any]], rewards: List[float]) -> None:
        """Save RL-specific results."""
        rl_results = {
            "iterations": results,
            "rewards": rewards,
            "best_action_sequence": [action.__dict__ for action in self.rl_trainer.get_best_action_sequence()],
            "q_table_size": len(self.rl_trainer.q_table),
            "total_visits": sum(self.rl_trainer.visit_counts.values())
        }

        results_file = Path(self.config.training.output_dir) / "rl_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(results_file, 'w') as f:
            json.dump(rl_results, f, indent=2, default=str)

        # Save RL policy
        policy_file = Path(self.config.training.output_dir) / "rl_policy.json"
        self.rl_trainer.save_policy(str(policy_file))

        print(f"RL results saved to: {results_file}")
        print(f"RL policy saved to: {policy_file}")


def create_rl_trainer(reward_function: Optional[IterationRewardFunction] = None) -> RLTrainer:
    """Create a default RL trainer."""
    if reward_function is None:
        # Default reward: combination of training loss and evaluation metrics
        reward_function = CombinedReward([
            (TrainingLossReward(improvement_weight=1.0, absolute_weight=0.1), 1.0),
            (EvaluationReward(eval_metric="eval_loss", metric_weight=0.5), 1.0)
        ])

    action_space = TrainingActionSpace()

    return RLTrainer(
        reward_function=reward_function,
        action_space=action_space,
        discount_factor=0.95,
        exploration_rate=0.1,
        learning_rate=1e-3
    )


def rl_train(config: Optional[FrameworkConfig] = None) -> Dict[str, Any]:
    """Run PEFT training with RL optimization."""
    if config is None:
        from .config import FrameworkConfig
        config = FrameworkConfig()

    # Create RL trainer
    rl_trainer = create_rl_trainer()

    # Create RL-enhanced tuner
    tuner = RLPEFTFineTuner(config, rl_trainer)

    # Run RL training
    results, rewards = tuner.iterative_train_with_rl()

    return {
        "results": results,
        "rewards": rewards,
        "model_path": config.training.output_dir,
        "config": config,
        "rl_trainer": rl_trainer
    }


if __name__ == "__main__":
    # Example RL training
    from .config import FrameworkConfig

    config = FrameworkConfig()
    config.iterative.max_iterations = 2  # Shorter for demo

    results = rl_train(config)
    print(f"RL training completed. Results: {len(results['results'])} iterations")
    print(f"Rewards: {results['rewards']}")
