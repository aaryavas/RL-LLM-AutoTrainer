#!/usr/bin/env python3
"""
Interactive CLI Wrapper for data-gen.py

This script provides a super interactive interface for users to generate synthetic data
by walking them through all the necessary configuration steps and then calling data-gen.py
with the appropriate parameters.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class InteractiveDataGenerator:
    """Interactive wrapper for the synthetic data generator."""
    
    def __init__(self):
        self.config = {}
        self.labels = []
        self.categories_types = {}
        self.use_case = ""
        self.label_descriptions = ""
        self.prompt_examples = ""
        self.model = "meta-llama/Llama-3.2-3B-Instruct"
        self.sample_size = 100
        self.max_new_tokens = 256
        self.batch_size = 20
        self.output_dir = "./"
        self.save_reasoning = False
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        print("\n🔍 CHECKING DEPENDENCIES")
        print("=" * 70)
        
        required_packages = [
            ('dotenv', 'python-dotenv'),
            ('pandas', 'pandas'),
            ('huggingface_hub', 'huggingface-hub'),
            ('transformers', 'transformers'),
            ('torch', 'torch'),
            ('outlines', 'outlines'),
        ]
        
        missing_packages = []
        dotenv_available = False
        
        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
                print(f"✅ {package_name:30} - installed")
                if import_name == 'dotenv':
                    dotenv_available = True
            except ImportError:
                print(f"❌ {package_name:30} - MISSING")
                missing_packages.append(package_name)
        
        # Check for data-gen.py file
        data_gen_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data-gen.py')
        if os.path.exists(data_gen_path):
            print(f"✅ {'data-gen.py':30} - found")
        else:
            print(f"❌ {'data-gen.py':30} - MISSING")
            missing_packages.append('data-gen.py (file)')
        
        # Check for Hugging Face token only if dotenv is available
        hf_token = None
        if dotenv_available:
            from dotenv import load_dotenv
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
        else:
            hf_token = os.getenv("HF_TOKEN")
        
        if hf_token:
            print(f"✅ {'HF_TOKEN':30} - configured")
        else:
            print(f"⚠️  {'HF_TOKEN':30} - NOT SET (required for model access)")
            print("\n   Set your token by creating a .env file with:")
            print("   HF_TOKEN=your_huggingface_token_here")
            print("\n   Or export it as an environment variable:")
            print("   export HF_TOKEN=your_huggingface_token_here")
        
        print("=" * 70)
        
        if missing_packages:
            print("\n❌ DEPENDENCY CHECK FAILED")
            print("\nMissing packages:")
            for pkg in missing_packages:
                print(f"  - {pkg}")
            print("\nTo install missing packages, run:")
            print("  pip install " + " ".join([p for p in missing_packages if not p.endswith('(file)')]))
            print("\nOr using uv:")
            print("  uv pip install " + " ".join([p for p in missing_packages if not p.endswith('(file)')]))
            return False
        
        if not hf_token:
            proceed = input("\n⚠️  No HF_TOKEN found. Continue anyway? (y/n): ").strip().lower()
            if proceed not in ['y', 'yes']:
                return False
        
        print("\n✅ ALL DEPENDENCIES SATISFIED")
        print()
        return True
        
    def print_banner(self):
        """Print a welcome banner."""
        print("=" * 70)
        print("🚀 INTERACTIVE SYNTHETIC DATA GENERATOR 🚀")
        print("=" * 70)
        print("This tool will guide you through creating synthetic datasets using AI models.")
        print("We'll ask you a series of questions to configure your data generation.")
        print("=" * 70)
        print()

    def get_user_input(self, prompt: str, default: Optional[str] = None, 
                      input_type: str = "string", required: bool = True) -> str:
        """Get validated user input with optional default values."""
        while True:
            if default:
                display_prompt = f"{prompt} (default: {default}): "
            else:
                display_prompt = f"{prompt}: "
            
            user_input = input(display_prompt).strip()
            
            # Use default if no input provided
            if not user_input and default:
                return default
            
            # Check if required field is empty
            if required and not user_input:
                print("❌ This field is required. Please enter a value.")
                continue
            
            # Validate input type
            if input_type == "integer":
                try:
                    value = int(user_input)
                    if value <= 0:
                        print("❌ Please enter a positive integer.")
                        continue
                    return str(value)
                except ValueError:
                    print("❌ Please enter a valid integer.")
                    continue
            elif input_type == "boolean":
                if user_input.lower() in ['y', 'yes', 'true', '1']:
                    return "true"
                elif user_input.lower() in ['n', 'no', 'false', '0']:
                    return "false"
                else:
                    print("❌ Please enter 'y' for yes or 'n' for no.")
                    continue
            
            return user_input

    def get_multiline_input(self, prompt: str, end_marker: str = "END") -> str:
        """Get multiline input from user."""
        print(f"{prompt}")
        print(f"(Type '{end_marker}' on a new line to finish)")
        print("-" * 50)
        
        lines = []
        while True:
            line = input()
            if line.strip() == end_marker:
                break
            lines.append(line)
        
        return "\n".join(lines)

    def configure_use_case(self):
        """Configure the use case for data generation."""
        print("\n📋 STEP 1: USE CASE CONFIGURATION")
        print("-" * 40)
        print("What is the main purpose of your synthetic data?")
        print("Examples:")
        print("  - Customer service chatbot training")
        print("  - Content moderation")
        print("  - Sentiment analysis")
        print("  - Text classification")
        print()
        
        self.use_case = self.get_user_input(
            "Describe your use case",
            default="text classification",
            required=True
        )
        print(f"✅ Use case set: {self.use_case}")

    def configure_labels(self):
        """Configure labels and their descriptions."""
        print("\n🏷️  STEP 2: LABELS CONFIGURATION")
        print("-" * 40)
        print("Labels are the categories you want to classify your data into.")
        print("Examples for sentiment analysis: 'positive', 'negative', 'neutral'")
        print("Examples for politeness: 'polite', 'impolite', 'neutral'")
        print()
        
        while True:
            label = self.get_user_input("Enter a label name (or 'done' to finish)")
            if label.lower() == 'done':
                if not self.labels:
                    print("❌ You must define at least one label.")
                    continue
                break
            
            if label in self.labels:
                print(f"⚠️  Label '{label}' already exists.")
                continue
                
            self.labels.append(label)
            print(f"✅ Added label: {label}")
        
        print(f"\n📝 Now provide descriptions for your labels.")
        label_desc_parts = []
        for label in self.labels:
            description = self.get_user_input(
                f"Describe what '{label}' means",
                required=True
            )
            label_desc_parts.append(f"{label}: {description}")
        
        self.label_descriptions = "\n".join(label_desc_parts)
        print("✅ Label descriptions configured.")

    def configure_categories(self):
        """Configure categories and their types."""
        print("\n🗂️  STEP 3: CATEGORIES CONFIGURATION")
        print("-" * 40)
        print("Categories help diversify your data by specifying different contexts.")
        print("Each category can have multiple types/subcategories.")
        print("Example: Category 'customer_service' might have types: 'complaint', 'inquiry', 'compliment'")
        print()
        
        while True:
            category = self.get_user_input("Enter a category name (or 'done' to finish)")
            if category.lower() == 'done':
                if not self.categories_types:
                    print("❌ You must define at least one category.")
                    continue
                break
            
            if category in self.categories_types:
                print(f"⚠️  Category '{category}' already exists.")
                continue
            
            print(f"Now enter types/subcategories for '{category}':")
            types = []
            while True:
                type_name = self.get_user_input(f"Enter a type for '{category}' (or 'done' to finish)")
                if type_name.lower() == 'done':
                    if not types:
                        print("❌ Each category must have at least one type.")
                        continue
                    break
                types.append(type_name)
                print(f"  ✅ Added type: {type_name}")
            
            self.categories_types[category] = types
            print(f"✅ Category '{category}' configured with {len(types)} types.")
        
        print("✅ Categories configuration complete.")

    def configure_examples(self):
        """Configure prompt examples for few-shot learning."""
        print("\n💡 STEP 4: EXAMPLE CONFIGURATION")
        print("-" * 40)
        print("Provide examples that show the AI how to generate data in your desired format.")
        print("Each example should include: LABEL, CATEGORY, TYPE, OUTPUT, and REASONING.")
        print("IMPORTANT: Examples must be in JSON format.")
        print()
        print("Example format:")
        print("LABEL: positive")
        print("CATEGORY: customer_service")
        print("TYPE: compliment")
        print('''{
    "output": "Thank you so much for the excellent service!",
    "reasoning": "This expresses gratitude and praise, indicating positive sentiment."
}''')
        print()
        
        num_examples = int(self.get_user_input(
            "How many examples would you like to provide?",
            default="2",
            input_type="integer"
        ))
        
        examples = []
        for i in range(num_examples):
            print(f"\n--- Example {i+1} ---")
            example = self.get_multiline_input(f"Enter example {i+1} (following the format above)")
            examples.append(example)
            print(f"✅ Example {i+1} added.")
        
        self.prompt_examples = "\n\n".join(examples)
        print("✅ Examples configuration complete.")

    def configure_model_settings(self):
        """Configure model and generation settings."""
        print("\n🤖 STEP 5: MODEL & GENERATION SETTINGS")
        print("-" * 40)
        
        print("Available models:")
        models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "google/gemma-3-1b-it",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "microsoft/DialoGPT-medium",
            "custom"
        ]
        
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        choice = self.get_user_input(
            "Select a model (1-5)",
            default="1",
            input_type="integer"
        )
        
        choice = int(choice)
        if 1 <= choice <= len(models) - 1:
            self.model = models[choice - 1]
        elif choice == len(models):
            self.model = self.get_user_input("Enter custom model name (HuggingFace format)")
        else:
            self.model = models[0]  # Default
        
        print(f"✅ Model selected: {self.model}")
        
        # Generation parameters
        self.sample_size = int(self.get_user_input(
            "Number of samples to generate",
            default="100",
            input_type="integer"
        ))
        
        self.max_new_tokens = int(self.get_user_input(
            "Maximum tokens per sample",
            default="256",
            input_type="integer"
        ))
        
        self.batch_size = int(self.get_user_input(
            "Batch size for processing",
            default="20",
            input_type="integer"
        ))
        
        self.output_dir = self.get_user_input(
            "Output directory",
            default="./generated_data",
            required=False
        )
        
        save_reasoning_input = self.get_user_input(
            "Save reasoning for each generated sample? (y/n)",
            default="y",
            input_type="boolean"
        )
        self.save_reasoning = save_reasoning_input == "true"
        
        print("✅ Model and generation settings configured.")

    def create_config_file(self) -> str:
        """Create a temporary configuration file."""
        config_content = f'''# Auto-generated configuration file
labels = {self.labels}

label_descriptions = """{self.label_descriptions}"""

categories_types = {self.categories_types}

use_case = "{self.use_case}"

prompt_examples = """{self.prompt_examples}"""
'''
        
        # Create temporary config file
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "auto_config.py")
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path

    def display_summary(self):
        """Display configuration summary."""
        print("\n📊 CONFIGURATION SUMMARY")
        print("=" * 50)
        print(f"Use case: {self.use_case}")
        print(f"Labels: {', '.join(self.labels)}")
        print(f"Categories: {', '.join(self.categories_types.keys())}")
        print(f"Model: {self.model}")
        print(f"Sample size: {self.sample_size}")
        print(f"Max tokens: {self.max_new_tokens}")
        print(f"Batch size: {self.batch_size}")
        print(f"Output directory: {self.output_dir}")
        print(f"Save reasoning: {'Yes' if self.save_reasoning else 'No'}")
        print("=" * 50)

    def run_data_generation(self):
        """Execute the data generation process."""
        print("\n🚀 STARTING DATA GENERATION")
        print("-" * 40)
        
        # Create config file
        config_path = self.create_config_file()
        
        try:
            # Build command using uv run to ensure correct Python environment
            cmd = [
                "uv", "run", "python", "data-gen.py",
                "--config", config_path,
                "--sample_size", str(self.sample_size),
                "--model", self.model,
                "--max_new_tokens", str(self.max_new_tokens),
                "--batch_size", str(self.batch_size),
                "--output_dir", self.output_dir
            ]
            
            if self.save_reasoning:
                cmd.append("--save_reasoning")
            
            print(f"Executing: {' '.join(cmd)}")
            print()
            
            # Run the command
            result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            
            if result.returncode == 0:
                print("\n✅ Data generation completed successfully!")
                print(f"Check your output directory: {self.output_dir}")
            else:
                print(f"\n❌ Data generation failed with return code: {result.returncode}")
                
        except Exception as e:
            print(f"\n❌ Error running data generation: {e}")
        finally:
            # Clean up temporary config file
            try:
                os.remove(config_path)
                os.rmdir(os.path.dirname(config_path))
            except:
                pass

    def run(self):
        """Main execution flow."""
        self.print_banner()
        
        # Check dependencies first
        if not self.check_dependencies():
            print("\n❌ Please install missing dependencies and try again.")
            sys.exit(1)
        
        try:
            self.configure_use_case()
            self.configure_labels()
            self.configure_categories()
            self.configure_examples()
            self.configure_model_settings()
            
            self.display_summary()
            
            confirm = self.get_user_input(
                "\nProceed with data generation? (y/n)",
                default="y",
                input_type="boolean"
            )
            
            if confirm == "true":
                self.run_data_generation()
            else:
                print("❌ Data generation cancelled.")
                
        except KeyboardInterrupt:
            print("\n\n❌ Process interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    generator = InteractiveDataGenerator()
    generator.run()


if __name__ == "__main__":
    main()