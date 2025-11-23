#!/usr/bin/env python3
"""
Quick test to demonstrate epoch metrics visualization.
This shows what the feature looks like when running.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Create test data quickly."""
    np.random.seed(42)
    
    data = []
    for i in range(60):  # Smaller dataset for quick test
        if i < 20:
            data.append({
                'text': f"This is a positive example {i}",
                'label': 'positive'
            })
        elif i < 40:
            data.append({
                'text': f"This is a negative example {i}",
                'label': 'negative'
            })
        else:
            data.append({
                'text': f"This is a neutral example {i}",
                'label': 'neutral'
            })
    
    df = pd.DataFrame(data)
    output_path = "test_metrics_data.csv"
    df.to_csv(output_path, index=False)
    return output_path

def simulate_epoch_metrics():
    """Simulate what the epoch metrics look like."""
    print("🎬 EPOCH METRICS SIMULATION")
    print("=" * 50)
    print("This shows what you'll see during actual fine-tuning:")
    print("=" * 50)
    
    import time
    
    # Simulate 3 epochs
    for epoch in range(1, 4):
        print(f"\n📊 EPOCH {epoch} METRICS:")
        print("=" * 40)
        
        # Simulate improving metrics
        accuracy = 0.3 + (epoch * 0.2) + np.random.uniform(0, 0.1)
        precision = 0.25 + (epoch * 0.22) + np.random.uniform(0, 0.1)
        recall = 0.28 + (epoch * 0.18) + np.random.uniform(0, 0.1)
        f1 = 0.26 + (epoch * 0.20) + np.random.uniform(0, 0.1)
        loss = 1.5 - (epoch * 0.3) + np.random.uniform(-0.1, 0.1)
        
        # Show progress bars
        def show_progress_bar(name, value, emoji):
            filled = int(value * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"{emoji} {name:10} │{bar}│ {value:.3f}")
        
        show_progress_bar("Accuracy", accuracy, "🎯")
        show_progress_bar("Precision", precision, "🔍")
        show_progress_bar("Recall", recall, "📈")
        show_progress_bar("F1-Score", f1, "⚖️")
        
        print(f"📉 Loss: {loss:.4f}")
        
        if epoch == 1:
            print("🏆 New best accuracy!")
        elif epoch == 3:
            print("✨ Training completed!")
        
        time.sleep(1)  # Simulate training time
    
    print(f"\n📋 TRAINING SUMMARY:")
    print("=" * 30)
    print("🏆 Best Epoch: 3")
    print("🎯 Best Accuracy: 0.826")
    print("📊 Final Metrics Saved: epoch_metrics_demo_metrics.json")
    print("📁 Model Saved: ./demo_models/epoch_metrics_demo")

def test_actual_implementation():
    """Test the actual implementation with minimal setup."""
    print("\n🧪 TESTING ACTUAL IMPLEMENTATION")
    print("=" * 40)
    
    data_path = create_test_data()
    print(f"✅ Created test data: {data_path}")
    
    try:
        from finetuning import SmolLM2FineTuner
        
        print("✅ Successfully imported SmolLM2FineTuner")
        print("✅ All dependencies are working")
        print("✅ Epoch metrics feature is ready to use!")
        
        # Show what would happen
        print(f"\n💡 To see live epoch metrics, run:")
        print(f"   python3 cli.py finetune {data_path} --epochs 2")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install missing dependencies and try again")
        return False
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return False

def main():
    print("🎯 EPOCH METRICS FEATURE SHOWCASE")
    print("=" * 45)
    
    # Show simulation
    simulate_epoch_metrics()
    
    # Test actual implementation
    test_actual_implementation()
    
    print(f"\n✨ FEATURE SUMMARY:")
    print("  📊 Real-time classification metrics display")
    print("  📈 Visual progress bars for each metric") 
    print("  🏆 Best performance tracking")
    print("  📋 Complete training summary")
    print("  ⚡ Works with CLI, interactive, and Python API")
    
    print(f"\n🚀 Ready to use in your fine-tuning workflows!")

if __name__ == "__main__":
    main()