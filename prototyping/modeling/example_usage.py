"""
Example usage of the PEFT fine-tuning framework.
This file demonstrates how to use the framework for both standard and RL-enhanced training.
"""

from config import FrameworkConfig
from peft import PEFTFineTuner, quick_train
from rl import rl_train, create_rl_trainer, TrainingLossReward, EvaluationReward, CombinedReward


def example_standard_training():
    """Example of standard iterative PEFT training."""
    print("="*60)
    print("STANDARD PEFT TRAINING EXAMPLE")
    print("="*60)

    # Create default config or customize
    config = FrameworkConfig()
    config.training.num_train_epochs = 1  # Short epochs for demo
    config.iterative.max_iterations = 2   # Just 2 iterations for demo
    config.data.dataset_name = "timdettmers/openassistant-guanaco"  # Small dataset

    # Initialize tuner
    tuner = PEFTFineTuner(config)

    # Prepare model
    tuner.prepare_model_for_training()

    # Load and prepare data
    train_dataset, eval_dataset = tuner.load_and_prepare_data()

    # Run iterative training
    print(f"Starting {config.iterative.max_iterations} iterations of training...")
    results = tuner.iterative_train()

    # Test generation
    test_prompt = "Write a short story about a robot that learns to paint:"
    generated = tuner.generate_sample(test_prompt, max_length=100)
    print(f"\nGenerated text:\n{generated}")

    return results


def example_custom_config():
    """Example showing how to customize the configuration."""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("="*60)

    # Create custom configuration
    config = FrameworkConfig()

    # Model settings
    config.model.model_name = "HuggingFaceTB/SmolLM-1.7B-Instruct"

    # PEFT settings
    config.peft.r = 32  # Higher rank for potentially better performance
    config.peft.lora_alpha = 64
    config.peft.lora_dropout = 0.1

    # Training settings
    config.training.learning_rate = 1e-4
    config.training.per_device_train_batch_size = 2
    config.training.gradient_accumulation_steps = 8
    config.training.num_train_epochs = 1
    config.training.warmup_ratio = 0.1

    # Iterative settings
    config.iterative.max_iterations = 3
    config.iterative.iteration_save_prefix = "custom_iter_"

    # Data settings (using a smaller subset)
    config.data.max_length = 1024  # Shorter sequences
    config.data.test_size = 0.1

    # Use the quick training function
    result = quick_train(config)

    print("Custom training completed!")
    return result


def example_rl_training():
    """Example of RL-enhanced PEFT training."""
    print("\n" + "="*60)
    print("RL-ENHANCED PEFT TRAINING EXAMPLE")
    print("="*60)

    # Create config
    config = FrameworkConfig()
    config.training.num_train_epochs = 1
    config.iterative.max_iterations = 3

    # Custom reward function combining loss and eval metrics
    reward_function = CombinedReward([
        (TrainingLossReward(improvement_weight=2.0, absolute_weight=0.5), 1.0),
        (EvaluationReward(eval_metric="eval_loss", metric_weight=1.0), 0.8)
    ])

    # Run RL training
    result = rl_train(config)

    print(f"RL training completed with {len(result['rewards'])} iterations")
    print(f"Rewards received: {result['rewards']}")

    return result


def example_hyperparameter_search():
    """Example of predefined hyperparameter search."""
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH EXAMPLE")
    print("="*60)

    # Define different hyperparameter configs to try
    hp_configs = [
        # Config 1: High learning rate, low rank
        {
            "learning_rate": 5e-4,
            "r": 8
        },
        # Config 2: Low learning rate, high rank
        {
            "learning_rate": 1e-4,
            "r": 32
        },
        # Config 3: Medium settings
        {
            "learning_rate": 2e-4,
            "r": 16
        }
    ]

    config = FrameworkConfig()
    config.iterative.hyperparameter_search = True
    config.iterative.hyperparameter_configs = hp_configs
    config.iterative.max_iterations = len(hp_configs)  # One iteration per HP config
    config.training.num_train_epochs = 1

    tuner = PEFTFineTuner(config)
    results = tuner.iterative_train()

    print("Hyperparameter search completed!")
    print(f"Tested {len(results)} different configurations")

    return results


def example_resume_training():
    """Example of resuming training from a checkpoint."""
    print("\n" + "="*60)
    print("RESUME TRAINING EXAMPLE")
    print("="*60)

    # First, do a short training run
    config = FrameworkConfig()
    config.iterative.max_iterations = 1
    config.training.output_dir = "./resume_example_checkpoint"

    tuner = PEFTFineTuner(config)
    tuner.prepare_model_for_training()
    tuner.load_and_prepare_data()
    results1 = tuner.train_iteration(1)

    print(f"Completed iteration 1. Loss: {results1['train_loss']:.4f}")

    # Now resume from checkpoint
    new_config = FrameworkConfig()
    new_config.iterative.max_iterations = 2  # Continue for 2 more iterations
    new_config.training.output_dir = "./resume_example_checkpoint"
    new_config.iterative.enable_checkpoint_resume = True

    new_tuner = PEFTFineTuner(new_config)
    new_tuner.load_checkpoint("./resume_example_checkpoint")
    results2 = new_tuner.iterative_train()

    print("Resume training completed!")
    return results2


if __name__ == "__main__":
    # Run all examples (comment out as needed)
    try:
        # Standard training example
        results_standard = example_standard_training()

        # Custom config example
        results_custom = example_custom_config()

        # Hyperparameter search example
        results_hp = example_hyperparameter_search()

        # Resume training example
        results_resume = example_resume_training()

        # RL training (commented out by default as it may take longer)
        # results_rl = example_rl_training()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)

    except Exception as e:
        print(f"Example failed with error: {e}")
        import traceback
        traceback.print_exc()
