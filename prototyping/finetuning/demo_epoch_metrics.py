#!/usr/bin/env python3
"""
Test script to demonstrate epoch metrics display during fine-tuning.
This creates a quick test with minimal epochs to show the metrics feature.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_demo_data():
    """Create a small demonstration dataset."""
    print("📝 Creating demonstration dataset...")
    
    np.random.seed(42)
    
    # Create more varied data for better metrics demonstration
    positive_texts = [
        "This product is absolutely amazing and I love it",
        "Excellent quality and fantastic customer service",
        "Outstanding performance and highly recommended",
        "Great value for money and very satisfied",
        "Perfect solution for my needs and works perfectly",
        "Impressive results and exceeded expectations",
        "Top-notch quality and professional service",
        "Brilliant design and user-friendly interface"
    ]
    
    negative_texts = [
        "This product is terrible and completely disappointing",
        "Poor quality and awful customer service experience",
        "Horrible performance and would not recommend",
        "Overpriced and not worth the money at all",
        "Broken functionality and unreliable system",
        "Frustrating experience and major quality issues",
        "Unacceptable performance and poor design",
        "Complete waste of money and time"
    ]
    
    neutral_texts = [
        "This product meets basic requirements and expectations",
        "Average quality for the price range offered",
        "Standard functionality without special features",
        "Acceptable performance for general use cases",
        "Typical solution that works as advertised",
        "Basic features with room for improvement",
        "Adequate performance for standard needs",
        "Regular quality meeting minimum standards"
    ]
    
    data = []
    
    # Create 150 samples (50 each label) for good training
    for i in range(50):
        # Positive samples
        text = np.random.choice(positive_texts) + f" (sample {i})"
        data.append({
            'text': text,
            'label': 'positive',
            'model': 'demo_generator',
            'reasoning': 'Positive sentiment example'
        })
        
        # Negative samples
        text = np.random.choice(negative_texts) + f" (sample {i})"
        data.append({
            'text': text,
            'label': 'negative', 
            'model': 'demo_generator',
            'reasoning': 'Negative sentiment example'
        })
        
        # Neutral samples
        text = np.random.choice(neutral_texts) + f" (sample {i})"
        data.append({
            'text': text,
            'label': 'neutral',
            'model': 'demo_generator',
            'reasoning': 'Neutral sentiment example'
        })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    output_path = "epoch_metrics_demo_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Created demo dataset: {output_path}")
    print(f"📊 Dataset info:")
    print(f"  • Total samples: {len(df)}")
    print(f"  • Label distribution: {df['label'].value_counts().to_dict()}")
    
    return output_path

def demo_epoch_metrics():
    """Demonstrate the epoch metrics feature."""
    
    print("🎬 EPOCH METRICS DEMONSTRATION")
    print("=" * 50)
    print("This demo shows how classification metrics are")
    print("displayed at each epoch during fine-tuning.")
    print("=" * 50)
    
    # Create demo data
    data_path = create_demo_data()
    
    print(f"\n🚀 Starting demo fine-tuning with epoch metrics...")
    print(f"This will run 2 epochs with the smallest model for demonstration.")
    
    # Import after creating data
    from finetuning import SmolLM2FineTuner
    
    try:
        # Initialize with smallest model for quick demo
        finetuner = SmolLM2FineTuner(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            output_dir="./demo_models"
        )
        
        print(f"\n📋 Demo Configuration:")
        print(f"  • Model: SmolLM2-135M (fastest)")
        print(f"  • Epochs: 2 (quick demo)")
        print(f"  • Batch size: 4 (small for demo)")
        print(f"  • Data: {data_path} (150 samples)")
        print(f"  • Epoch metrics: ENABLED")
        
        # Run a quick fine-tuning demo
        model_path, eval_results = finetuner.finetune_pipeline(
            data_path=data_path,
            num_train_epochs=2,  # Just 2 epochs for demo
            batch_size=4,        # Small batch size
            learning_rate=5e-4,  # Slightly higher for visible changes
            run_name="epoch_metrics_demo",
            show_epoch_metrics=True  # Enable epoch metrics display
        )
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"\n✨ What you saw:")
        print(f"  📊 Detailed metrics displayed after each epoch")
        print(f"  📈 Progress bars showing metric improvements") 
        print(f"  🏆 Best performance tracking")
        print(f"  📋 Training summary at the end")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print(f"\n💡 This is expected if you don't have the required dependencies.")
        print(f"   The important thing is seeing how the epoch metrics work!")
        return False

def show_metrics_features():
    """Show what the epoch metrics feature provides."""
    
    print("\n📊 EPOCH METRICS FEATURES:")
    print("-" * 30)
    
    features = [
        "📈 Real-time classification metrics at each epoch",
        "📊 Visual progress bars for each metric",
        "🏆 Best performance tracking across epochs", 
        "📉 Loss progression monitoring",
        "🎯 Accuracy, Precision, Recall, F1 Score display",
        "📋 Training summary with best epoch identification",
        "📝 Epoch-by-epoch metrics saved to JSON",
        "⚡ Configurable via CLI and interactive mode"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print(f"\n💻 USAGE OPTIONS:")
    print(f"  CLI: python3 cli.py finetune data.csv --epochs 3")
    print(f"       (metrics enabled by default)")
    print(f"  CLI: python3 cli.py finetune data.csv --no-epoch-metrics")
    print(f"       (disable metrics if preferred)")
    print(f"  Interactive: Will ask if you want epoch metrics")
    print(f"  Python API: show_epoch_metrics=True (default)")

def main():
    """Main demo function."""
    
    print("🎯 EPOCH METRICS FEATURE DEMO")
    print("=" * 40)
    
    show_metrics_features()
    
    print(f"\n" + "?" * 40)
    run_demo = input("Run live demo with actual fine-tuning? (y/n): ").strip().lower()
    
    if run_demo in ['y', 'yes']:
        print(f"\n⚠️ Note: This will download the SmolLM2-135M model (~270MB)")
        print(f"and run actual fine-tuning. It may take a few minutes.")
        
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            demo_epoch_metrics()
        else:
            print(f"Demo cancelled. The feature is ready to use!")
    else:
        print(f"\n✅ The epoch metrics feature is implemented and ready!")
        print(f"Run any fine-tuning command to see it in action.")

if __name__ == "__main__":
    main()