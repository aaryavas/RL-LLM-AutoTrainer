"""This is the replication of the intel tool"""

import argparse
import importlib.util
import os
import random
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import outlines
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login, whoami
from pydantic import BaseModel, Field

# TODO: make sure to setup pyproject.toml config file to install the packages


def read_token() -> None:
    """
    This will read the token from the user, it necessary, call this function in the CLI tool
    """
    load_dotenv()
    try:
        w = whoami()
        print(f"Logged in as {w['name']}")
    except Exception:
        token = os.getenv("HF_TOKEN")
        login(token)


"""Validator for command line arguments"""


def validate_positive_integer(value: str) -> int:
    """
    Validate that the input is a positive integer.

    Args:
        value: The input string from argparse

    Returns:
        int: The validated integer value

    Raises:
        argparse.ArgumentTypeError: If validation fails
    """
    try:
        int_value = int(value)
        if int_value <= 0:
            raise argparse.ArgumentTypeError(
                f"The input value must be positive, got {int_value}"
            )
        return int_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}")


def parse_string(input_string: str) -> Tuple[str, str]:
    """
    Parses a string containing `OUTPUT:` and `REASONING:` sections and extracts their values.

    Args:
        input_string (str): The input string containing `OUTPUT:` and `REASONING:` labels.

    Returns:
        Tuple[str, str]: A tuple containing two strings:
                         - The content following `OUTPUT:`.
                         - The content following `REASONING:`.

    Raises:
        ValueError: If the input string does not match the expected format with both `OUTPUT:` and `REASONING:` sections.

    Note:
        - The function is case-sensitive and assumes `OUTPUT:` and `REASONING:` are correctly capitalized.
        - If the format is not found, it will attempt fallback parsing or use the raw input.
    """
    # Use regular expressions to extract OUTPUT and REASONING
    match = re.search(r"OUTPUT:\s*(.+?)\s*REASONING:\s*(.+)", input_string, re.DOTALL)

    if match:
        # Extract the matched groups: output and reasoning
        output = match.group(1).strip()
        reasoning = match.group(2).strip()
        return output, reasoning

    # Fallback: Try case-insensitive matching
    match = re.search(
        r"output:\s*(.+?)\s*reasoning:\s*(.+)", input_string, re.DOTALL | re.IGNORECASE
    )
    if match:
        output = match.group(1).strip()
        reasoning = match.group(2).strip()
        print(f"⚠️  Warning: Model used lowercase format. Output parsed successfully.")
        return output, reasoning

    # Fallback: Check if only OUTPUT is present
    match = re.search(r"OUTPUT:\s*(.+)", input_string, re.DOTALL)
    if match:
        output = match.group(1).strip()
        reasoning = "No reasoning provided by model"
        print(f"⚠️  Warning: No REASONING found. Using output only.")
        return output, reasoning

    # Final fallback: Use the entire response as output
    print(f"⚠️  Warning: Response format not recognized. Using raw output.")
    print(
        f"Raw response: {input_string[:200]}..."
    )  # Print first 200 chars for debugging
    return input_string.strip(), "Format not recognized - raw output used"


def sdg(
    sample_size: int,
    labels: List[str],
    label_descriptions: str,
    categories_types: Dict[str, str],
    use_case: str,
    prompt_examples: str,
    model: str,
    max_new_tokens: int,
    batch_size: int,
    output_dir: str,
    save_reasoning: bool,
) -> None:
    """
    Generates synthetic data based on specified categories and labels.

    Args:
        sample_size (int): The number of synthetic data samples to generate.
        labels (List[str]): The labels used to classify the synthetic data.
        label_descriptions (str): A description of the meaning of each label.
        categories_types (Dict[str, str]): The categories and their types for data generation and diversification.
        use_case (str): The use case of the synthetic data to provide context for the language model.
        prompt_examples (str): The examples used in the Few-Shot or Chain-of-Thought prompting.
        model (str): The large language model used for generating the synthetic data.
        max_new_tokens (int): The maximum number of new tokens to generate for each sample.
        batch_size (int): The number of samples per batch to append to the output file.
        output_dir (str): The directory path where the output file will be saved.
        save_reasoning (bool): Whether to save the reasoning or explanation behind the generated data.
    """

    categories = list(categories_types.keys())

    # Generate filename with current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{timestamp}.csv")

    # Prepare all prompts first
    prompts = []
    batch_metadata = []

    print(f"\U0001f680  Generating {sample_size} prompts...")

    # If sample_size is not divisible by batch_size, an extra batch is added
    num_batches = (sample_size + batch_size - 1) // batch_size

    for batch in range(num_batches):
        # Calculate the start and end indices for the current batch
        start = batch * batch_size
        end = min(start + batch_size, sample_size)

        # Assign random labels to the current batch
        batch_random_labels = random.choices(labels, k=end - start)

        # Assign random categories to the current batch
        batch_random_categories = random.choices(categories, k=end - start)

        for i in range(start, end):
            # Assign a random type to the ith category
            random_type = random.choices(
                categories_types[batch_random_categories[i - start]]
            )
            prompt_text = f"""You should create synthetic data for specified labels and categories. 
            This is especially useful for {use_case}.

            *Label Descriptions*
            {label_descriptions}

            *Examples*
            {prompt_examples}

            ####################

            Generate one output for the classification below.
            You may use the examples I have provided as a guide, but you cannot simply modify or rewrite them.
            Only return the OUTPUT and REASONING. 
            Do not return the LABEL, CATEGORY, or TYPE.

            LABEL: {batch_random_labels[i - start]}
            CATEGORY: {batch_random_categories[i - start]}
            TYPE: {random_type}
            OUTPUT:
            REASONING:
            """

            # Format prompt for the model (using apply_chat_template logic if needed,
            # but vLLM usually takes raw text or tokens. For chat models, we might need to format it manually
            # or use the tokenizer. Here we'll construct the chat format manually as vLLM's generate takes string)
            # However, vLLM's LLM.generate takes a prompt string.
            # If the model expects a chat template, we should format it.
            # For simplicity and consistency with previous code, we'll use the same structure.
            # But vLLM doesn't automatically apply chat templates in `generate` unless we use `chat` method (if available) or format it ourselves.
            # Let's assume we pass the raw prompt or a formatted string.
            # The previous code used `pipeline("text-generation")` with a list of messages.
            # We can use the tokenizer to apply the chat template if we want to be robust,
            # but for now let's stick to the prompt construction.

            # Actually, vLLM supports `chat` method in newer versions or we can just format it.
            # Let's use the `messages` format and apply the template using the tokenizer if possible,
            # or just construct the prompt string if we know the template.
            # Given the previous code used `messages`, let's try to stick to that if vLLM supports it,
            # otherwise we might need to load the tokenizer.

            # To keep it simple and efficient, let's just construct the prompt string.
            # But wait, the previous code used `messages` list.
            # We should probably use `llm.chat` if available or format it.
            # Since I don't want to overcomplicate with tokenizer loading just for template,
            # I will check if I can use `llm.chat`.
            # If not, I will assume the model can handle the prompt or I'll just format it as a standard chat prompt.

            # Let's use the `LLM` class. It has a `generate` method.
            # We can pass a list of prompts.
            # We need to format the messages into a single string.
            # Since we don't have the tokenizer easily accessible without loading it,
            # and we want to be hardware agnostic/optimized.

            # Let's use a simple formatting for now, or better, use the tokenizer from vllm's engine if accessible.
            # Actually, vLLM's `LLM` class can take `prompt_token_ids` or `prompt`.

            # Let's just use the raw prompt text for now, but formatted as a user/system message if the model requires it.
            # The previous code used:
            # messages = [
            #     {"role": "system", "content": ...},
            #     {"role": "user", "content": ...},
            # ]
            # generator(messages)

            # I'll use a simple formatting that works for Llama 3 (which is the default model).
            # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n

            system_content = f"You are a helpful assistant designed to generate synthetic data for {use_case} with labels {labels} in categories {categories}."

            # We will rely on vLLM's ability to handle this or just pass the text.
            # Ideally we should use the tokenizer.
            # I will add a helper to apply chat template using the model's tokenizer after initializing LLM.

            prompts.append(prompt_text)  # We will format this later after loading LLM

            batch_metadata.append(
                {
                    "label": batch_random_labels[i - start],
                    "system_content": system_content,
                }
            )

    print(f"\U0001f680  Initializing vLLM engine with model: {model}")

    # Initialize vLLM
    from vllm import LLM, SamplingParams

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,  # Default, can be adjusted
        top_p=0.9,
        max_tokens=max_new_tokens,
    )

    # Initialize the LLM
    # Note: tensor_parallel_size should be set based on available GPUs.
    # For now we default to 1, but user can change it via args if we add it.
    # We'll rely on vLLM's default or env vars.
    llm = LLM(model=model, max_model_len=8192, max_num_batched_tokens=8192,gpu_memory_utilization=.8,)

    # Apply chat template to prompts
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": batch_metadata[i]["system_content"]},
            {"role": "user", "content": prompt},
        ]
        # apply_chat_template returns a string
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    print(f"\U0001f680  Starting generation...")
    outputs = llm.generate(formatted_prompts, sampling_params, )

    print(f"\U0001f680  Processing outputs and saving to {output_path}...")

    all_data = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text

        # Debug: Print first output
        if i == 0:
            print(f"\n📝 Sample model output:")
            print(f"{'=' * 50}")
            print(generated_text[:300] if len(generated_text) > 300 else generated_text)
            print(f"{'=' * 50}\n")

        text, reasoning = parse_string(generated_text)

        entry = {
            "text": text,
            "label": batch_metadata[i]["label"],
            "model": model,
        }

        if save_reasoning:
            entry["reasoning"] = reasoning

        all_data.append(entry)

    # Save all data
    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)
    print(f"\U000026a1  Saved {len(all_data)} samples to {output_path}")


""""This is the main function and this is how the user can interact with the code through the command line"""


def main() -> None:
    """
    Main entry point for running the synthetic data generator.

    This function performs the following steps:
    1. Reads the Hugging Face authentication token from the token file.
    2. Sets up and parses command-line arguments.
    3. Invokes the `sdg` function with the parsed arguments to generate synthetic data.

    Raises:
        SystemExit: If an error occurs during token reading or argument parsing.
    """

    read_token()

    parser = argparse.ArgumentParser(
        description="Run the synthetic data generator (sdg function)."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./config/polite-guard-config.py",
        help="The configuration file for the sdg function containing labels, categories, and examples (default: ./config/polite-guard-config.py)",
    )
    parser.add_argument(
        "--sample_size",
        type=validate_positive_integer,
        default=100,
        help="The number of samples generated by the language model (default: 100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="The language model for data generation (default: meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=validate_positive_integer,
        default=256,
        help="The maximum number of new tokens to generate for each sample (default: 256)",
    )
    parser.add_argument(
        "--batch_size",
        type=validate_positive_integer,
        default=20,
        help="The batch size for saving generated samples to file (default: 20)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="The output directory (default: ./)",
    )
    parser.add_argument(
        "--save_reasoning",
        action="store_true",
        help="Enable save reasoning (default: False)",
    )

    args = parser.parse_args()

    # Dynamically load the configuration module
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    try:
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module at {config_path}")
        sdg_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sdg_config)
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)

    sdg(
        sample_size=args.sample_size,
        labels=sdg_config.labels,
        label_descriptions=sdg_config.label_descriptions,
        categories_types=sdg_config.categories_types,
        use_case=sdg_config.use_case,
        prompt_examples=sdg_config.prompt_examples,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_reasoning=args.save_reasoning,
    )


if __name__ == "__main__":
    main()
