#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Ensure we can import from the local project
sys.path.append(os.getcwd())

try:
    from finetuning.training.metrics import MetricsComputer
    from finetuning.utils.merge_adapter import merge_adapter
except ImportError:
    print("Could not import required modules. Make sure you are in the project root.")
    sys.exit(1)

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=True, text=True)
    return result

def main():
    parser = argparse.ArgumentParser(description="Run VB-LoRA pipeline: SFT -> Merge -> ORPO -> Merge -> Predict -> Metrics")
    parser.add_argument("--data_path", type=str, default="data.csv", help="Path to data.csv")
    parser.add_argument("--output_dir", type=str, default="./output/pipeline_run", help="Base output directory")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to predict on")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found.")
        sys.exit(1)

    # Directories
    sft_output_dir = os.path.join(args.output_dir, "sft_adapter")
    sft_merged_dir = os.path.join(args.output_dir, "sft_merged")
    orpo_output_dir = os.path.join(args.output_dir, "orpo_adapter")
    final_model_dir = os.path.join(args.output_dir, "final_model")

    # 1. Run VB-LoRA SFT CLI
    print("\n" + "="*50)
    print("Step 1: Running VB-LoRA SFT Fine-tuning")
    print("="*50)
    
    cmd_sft = (
        f"python3 -m finetuning.cli finetune {args.data_path} "
        f"--output-dir {sft_output_dir} "
        f"--model {args.model_name} "
        f"--epochs {args.epochs} "
        f"--batch-size {args.batch_size}"
    )
    
    try:
        run_command(cmd_sft)
    except subprocess.CalledProcessError as e:
        print(f"SFT Training failed: {e}")
        sys.exit(1)

    # 2. Merge SFT Adapter
    print("\n" + "="*50)
    print("Step 2: Merging SFT Adapter")
    print("="*50)
    
    try:
        merge_adapter(
            adapter_path=sft_output_dir,
            output_path=sft_merged_dir,
            base_model_name=args.model_name
        )
    except Exception as e:
        print(f"Merging SFT adapter failed: {e}")
        sys.exit(1)

    # 3. Run ORPO CLI
    print("\n" + "="*50)
    print("Step 3: Running ORPO Fine-tuning")
    print("="*50)
    
    # ORPO CLI arguments based on README
    # Assuming data.csv has prompt/chosen/rejected or we map them
    # If data.csv is generic text/label, ORPO might fail if it expects preference columns.
    # The user asked to run on "data.csv". 
    # If data.csv is SFT data (text/label), it might not work for ORPO (prompt/chosen/rejected).
    # However, assuming the user provides a CSV that works for both or has the columns.
    # Let's assume standard column names or defaults.
    
    cmd_orpo = (
        f"python3 finetuning/orpo/cli.py "
        f"--model_name {sft_merged_dir} "
        f"--data_path {args.data_path} "
        f"--output_dir {orpo_output_dir} "
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        # Add column mappings if needed, using defaults for now
    )
    
    try:
        run_command(cmd_orpo)
    except subprocess.CalledProcessError as e:
        print(f"ORPO Training failed: {e}")
        print("Note: ORPO requires preference data (prompt, chosen, rejected). Ensure data.csv has these columns.")
        sys.exit(1)

    # 4. Merge ORPO Adapter
    print("\n" + "="*50)
    print("Step 4: Merging ORPO Adapter to create Final Model")
    print("="*50)
    
    try:
        merge_adapter(
            adapter_path=orpo_output_dir,
            output_path=final_model_dir,
            base_model_name=sft_merged_dir
        )
    except Exception as e:
        print(f"Merging ORPO adapter failed: {e}")
        sys.exit(1)

    # 5. Load Final Model & Predict
    print("\n" + "="*50)
    print("Step 5: Running Predictions on Final Model")
    print("="*50)
    
    print(f"Loading final model from: {final_model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            final_model_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        print(f"Failed to load final model: {e}")
        sys.exit(1)

    # ... (Prediction logic remains similar) ...
    
    df = pd.read_csv(args.data_path)
    
    # Determine columns
    text_col = 'text'
    label_col = 'label'
    
    # Adjust for preference data if that's what we have
    if 'prompt' in df.columns: text_col = 'prompt'
    if 'chosen' in df.columns: label_col = 'chosen' # Use chosen as reference
    
    print(f"Using columns - Input: '{text_col}', Target: '{label_col}'")

    
    if text_col not in df.columns:
        print(f"Error: Could not find input column '{text_col}' in data.")
        sys.exit(1)

    # Sample data
    eval_df = df.sample(min(len(df), args.num_samples), random_state=42)
    
    predictions = []
    references = []
    prompts = []
    
    print(f"Generating responses for {len(eval_df)} samples...")
    
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        prompt = row[text_col]
        reference = row[label_col] if label_col in df.columns else ""
        
        # Apply chat template if available/appropriate
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": str(prompt)}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = str(prompt)
            
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Try to extract just the response
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        prompts.append(prompt)
        predictions.append(response.strip())
        references.append(str(reference))

    # 4. Compute Metrics
    print("\n" + "="*50)
    print("Step 6: Computing Metrics")
    print("="*50)
    
    metrics_computer = MetricsComputer(tokenizer=tokenizer)
    results = {}
    
    # BERTScore
    if metrics_computer.metrics.get('bertscore'):
        print("Computing BERTScore...")
        try:
            bert_score = metrics_computer.metrics['bertscore'].compute(
                predictions=predictions, 
                references=references, 
                lang="en"
            )
            results['bertscore_f1'] = sum(bert_score['f1']) / len(bert_score['f1'])
            results['bertscore_precision'] = sum(bert_score['precision']) / len(bert_score['precision'])
            results['bertscore_recall'] = sum(bert_score['recall']) / len(bert_score['recall'])
        except Exception as e:
            print(f"Error computing BERTScore: {e}")

    # ROUGE
    if metrics_computer.metrics.get('rouge'):
        print("Computing ROUGE...")
        try:
            rouge_score = metrics_computer.metrics['rouge'].compute(
                predictions=predictions, 
                references=references
            )
            results.update(rouge_score)
        except Exception as e:
            print(f"Error computing ROUGE: {e}")

    print("\nFinal Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # 5. Save Predictions
    print("\n" + "="*50)
    print("Step 7: Saving Results")
    print("="*50)
    
    output_csv = os.path.join(args.output_dir, "predictions_with_metrics.csv")
    out_df = pd.DataFrame({
        "prompt": prompts,
        "reference": references,
        "prediction": predictions
    })
    out_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, "final_metrics.txt")
    with open(metrics_file, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved metrics to {metrics_file}")

if __name__ == "__main__":
    main()
