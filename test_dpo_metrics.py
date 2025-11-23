import pandas as pd
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm

# Add the current directory to the path so we can import the metrics module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add the finetuning directory to path so internal imports work
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuning"))

from finetuning.training.metrics import MetricsComputer

def test_dpo_candidate_identification():
    print("🚀 Starting DPO Candidate Identification Test")
    print("=" * 50)

    # 1. Load the baseline data
    data_path = "finetuning/data.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return

    print(f"📊 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # For testing purposes, let's use a smaller subset if the dataset is huge
    # But data.csv seems small enough based on the read_file output
    print(f"   Loaded {len(df)} rows.")

    # 2. Load the model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M"
    print(f"🤖 Loading model: {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("   Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Generate predictions for the dataset
    print("🔮 Generating predictions...")
    
    predictions = []
    
    # We'll use a simple generation loop
    # In a real scenario, you might want to batch this
    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_text = row['text']
        
        # Format input as chat/instruction if needed, but for this base model test
        # we'll just feed the text. SmolLM2 is an instruct model usually, so let's format it.
        messages = [{"role": "user", "content": input_text}]
        if tokenizer.chat_template is not None:
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
        else:
            # Fallback for base models without chat template
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=100, 
                do_sample=False, # Greedy decoding for reproducibility
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the new tokens
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        predictions.append(generated_text)

    df['prediction'] = predictions
    print("   Predictions generated.")

    # 4. Initialize MetricsComputer
    print("🧮 Initializing MetricsComputer...")
    metrics_computer = MetricsComputer(tokenizer=tokenizer)

    # 5. Run identify_dpo_candidates
    print("🕵️ Identifying DPO candidates (Threshold: 0.82)...")
    failed_df = metrics_computer.identify_dpo_candidates(
        df, 
        prediction_col='prediction', 
        reference_col='label', 
        threshold=0.82
    )

    # 6. Output results
    print("=" * 50)
    print(f"📉 Found {len(failed_df)} failed examples out of {len(df)} total.")
    
    if not failed_df.empty:
        output_file = "failed_dpo_candidates.csv"
        failed_df.to_csv(output_file, index=False)
        print(f"💾 Failed examples saved to: {output_file}")
        
        # Show a preview
        print("\n👀 Preview of failed examples:")
        print(failed_df[['text', 'label', 'prediction', 'bertscore_f1', 'codebertscore_f1']].head())
    else:
        print("🎉 No examples failed the threshold check!")

    print("=" * 50)
    print("✅ Test completed.")

if __name__ == "__main__":
    test_dpo_candidate_identification()
