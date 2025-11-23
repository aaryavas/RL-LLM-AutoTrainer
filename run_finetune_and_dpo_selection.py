import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Setup paths to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Add the finetuning directory to path so internal imports (like 'from config import...') work
sys.path.append(os.path.join(current_dir, "finetuning"))

# Now we can import from the module
try:
    from finetuning.finetuning import SmolLM2VBLoRAFineTuner
    from finetuning.training.metrics import MetricsComputer
except ImportError:
    # Fallback if the sys.path append didn't work as expected for some reason
    from finetuning.finetuning import SmolLM2VBLoRAFineTuner
    from finetuning.training.metrics import MetricsComputer

def main():
    print("🚀 Starting Fine-tuning and DPO Candidate Selection Pipeline")
    print("=" * 60)

    # Configuration
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
    DATA_PATH = os.path.join(current_dir, "finetuning", "data.csv")
    OUTPUT_DIR = os.path.join(current_dir, "dpo_pipeline_output")
    DPO_OUTPUT_FILE = "dpo_candidates.csv"
    
    # 1. Fine-tune the model
    print(f"\n📦 Step 1: Fine-tuning {MODEL_NAME} on {DATA_PATH}")
    
    finetuner = SmolLM2VBLoRAFineTuner(
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR
    )
    
    # Run fine-tuning
    # We use a small number of epochs for this pipeline demonstration
    model_path, eval_results = finetuner.finetune_pipeline(
        data_path=DATA_PATH,
        num_train_epochs=1,  # Keep it fast for the pipeline test
        batch_size=2,
        learning_rate=2e-4,
        run_name="dpo_pipeline_run",
        show_epoch_metrics=True
    )
    
    print(f"✅ Fine-tuning complete. Model saved to: {model_path}")
    
    # 2. Load the fine-tuned model for generation
    print(f"\n🔄 Step 2: Loading fine-tuned model for generation")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load adapter
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
        print("   Adapter loaded successfully.")
    except Exception as e:
        print(f"⚠️ Could not load adapter from {model_path}: {e}")
        print("   Falling back to base model (this might happen if no adapter was saved or path is wrong).")
        model = base_model

    model.eval()
    
    # 3. Generate predictions on the dataset
    print(f"\n🔮 Step 3: Generating predictions for DPO selection")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"   Processing {len(df)} examples...")
    
    predictions = []
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_text = row['text']
        
        # Prepare input
        messages = [{"role": "user", "content": input_text}]
        
        if tokenizer.chat_template is not None:
            try:
                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
            except Exception:
                # Fallback if template fails
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        else:
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
            
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=128, 
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Decode
        # Skip the input tokens
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        predictions.append(generated_text)
        
    df['prediction'] = predictions
    
    # 4. Calculate Metrics and Identify Candidates
    print(f"\n🧮 Step 4: Calculating metrics and identifying DPO candidates")
    
    metrics_computer = MetricsComputer(tokenizer=tokenizer)
    
    # This function will calculate BERTScore/CodeBERTScore and filter
    failed_df = metrics_computer.identify_dpo_candidates(
        df,
        prediction_col='prediction',
        reference_col='label',
        threshold=0.82
    )
    
    # 5. Save results
    print(f"\n💾 Step 5: Saving results")
    
    if not failed_df.empty:
        failed_df.to_csv(DPO_OUTPUT_FILE, index=False)
        print(f"   ✅ Saved {len(failed_df)} DPO candidates to {DPO_OUTPUT_FILE}")
        print("\n   Preview:")
        print(failed_df[['text', 'label', 'prediction', 'bertscore_f1', 'codebertscore_f1']].head())
    else:
        print("   🎉 No candidates found (all scores above threshold or dataset empty).")
        
    print("\n✨ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
