# RL-LLM-AutoTrainer 🤖

***Note: This product is a proof of concept and is extremely early in a developmental phase, there may be bugs and flaws in the overall implementation and delivery of the system.***

***Additional Note: ReadME Documentation is not complete, more detailed documentation is soon to come 😊***

https://github.com/user-attachments/assets/445bf62c-b9f6-4d2e-a638-2353fc71c880


## Overview
Wecolme to RL-LLM-AutoTrainer a tool specifically designed to make training LLMs easier for all software developers. We reccomend trying out our **CLI TUI tool** for a nice streamlined experience. With our tool we allow the ability to:
 - Generate synthetic data of the basis of prompt engineering to match the users use case
 - Finetune an LLM to match a use case using the baseline data
 - For further training and to avoid an issue of overfitting we further allign to the desired use case with **ORPO Reinforcement Learning**



## Getting Started/Installing Dependencies

### Prerequisites
Before you begin, ensure you have:
- Node.js 18+ and npm installed
- Python 3.8+ (3.10+ recommended)
- A Hugging Face account and access token

### Setup and run with the CLI TUI 

1: Install TypeScript CLI

Navigate to the CLI package and install dependencies:

```bash
cd prototyping/cli-wrapper
npm install
```

Build the TypeScript code:

```bash
npm run build
```

Install globally for the `synth-data` command:

```bash
npm link
```

**Alternative: Use the convenience script**

```bash
./install.sh
```


**For fine-tuning, also install:**

```bash
pip install peft accelerate bitsandbytes datasets evaluate scikit-learn nltk rouge_score
```

**Or install all from requirements:**

```bash
pip install -r prototyping/vblora/requirements.txt
```

### Step 3: Configure Hugging Face Token (Optional for TUI Tool)

Set your HF_TOKEN environment variable:

```bash
export HF_TOKEN=your_huggingface_token_here
```

**Or create a .env file:**

```bash
echo "HF_TOKEN=your_huggingface_token_here" > prototyping/.env
```

### Step 4: Verify Installation

Check that everything is working:

```bash
synth-data --version
synth-data --help
```

### Recommended: Global Command

After installation, run from anywhere:

```bash
synth-data
```


### Setup and run without the CLI (manual, minimal)

1. Prepare Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install Python dependencies needed for data generation

```bash
pip install python-dotenv pandas huggingface-hub transformers torch
```

3. Configure Hugging Face token (one of these)

```bash
export HF_TOKEN=your_hf_token_here
# or create a dotenv file used by scripts
echo "HF_TOKEN=your_hf_token_here" > prototyping/.env
```

4. Run the interactive Python data generator (no Node/CLI required)

```bash
cd prototyping
python interactive-data-gen.py
# or call the lower-level script directly
python data-gen.py --help
```

5. Fine-tune / run VB-LoRA

Follow the `finetuning` README for VB-LoRA usage and options: [prototyping/finetuning/finetuners/README.md](prototyping/finetuning/finetuners/README.md)

6. Use ORPO workflows

See `orpo` README for ORPO-specific training and utilities: [prototyping/finetuning/finetuners/orpo/README.md](prototyping/finetuning/finetuners/orpo/README.md)

B. Setup and run with the CLI (recommended for interactive users)

1. Install Node dependencies and build the CLI

```bash
cd prototyping/cli-wrapper
npm install
npm run build
# (optional) install globally for `synth-data` convenience
npm link
```

2. Verify CLI install

```bash
# if linked
synth-data --version
synth-data --help

# or run from package directory without linking
cd prototyping/cli-wrapper
npm start
```

3. Use the CLI to generate synthetic data

```bash
# run the interactive flow; follow prompts
synth-data

# or invoke the local npm start command
cd prototyping/cli-wrapper
npm start
```

Common commands (summary)

```bash
# Manual / non-CLI
python prototyping/interactive-data-gen.py

# CLI (after npm link)
- See prototyping/cli-wrapper/README.md for CLI-specific usage details and screenshots/examples.

# VB-LoRA (see detailed README)
python finetuning/cli.py --help

# ORPO (see detailed README)
python orpo/cli.py --help
```

### Synthetic Data Generation
Our synthetic data generation tool reverse engineers the intel synthetic data generation tool 
(https://huggingface.co/spaces/Intel/synthetic-data-generator)



For synthetic data generation we will need the following to build **high quality** data:
- Use Case 

- Purpose and application context

- Label Defintions

- Categories for Variation

- Few Shot Examples


#### Example Use Case 

Soon to come ... 

## Finetuning

For finetuning we employ the following pipeline:
 
 - First we do supervised finetuning (VB-LoRA) training of the synthetic data
 - Once supervised finetuning is complete if the user is not satisfied with metrics we continue onwards to reinforcment learning based finetuning (ORPO) 
 - We aim to further improve metrics with reinforcement learning while avoiding overfitting


### VB-Lora FineTuning 

Please reference the following documentation for understanding of the utlilty of this tool. 

VB-LoRA fine-tuning: [prototyping/finetuning/finetuners/README.md](prototyping/finetuning/finetuners/README.md)

#### Overview

### ORPO Reinforcement Learning Finetuning

Please reference the following documentation for understanding the utility of this tool.

- ORPO trainer and tools: [prototyping/finetuning/finetuners/orpo/README.md](prototyping/finetuning/finetuners/orpo/README.md)


Related documentation
- VB-LoRA fine-tuning: [prototyping/finetuning/finetuners/README.md](prototyping/finetuning/finetuners/README.md)
- ORPO trainer and tools: [prototyping/finetuning/finetuners/orpo/README.md](prototyping/finetuning/finetuners/orpo/README.md)
- CLI wrapper (detailed CLI docs): [prototyping/cli-wrapper/README.md](prototyping/cli-wrapper/README.md)

Prerequisites
- Node.js 18+ and npm (only required for CLI UI tool)
- Python 3.8+ (3.10+ recommended)
- Git and internet access for model downloads
- Hugging Face token: set `HF_TOKEN` or create `prototyping/.env`
- **Highly Reccomended:** CUDA compatible GPU

Quick overview
- Generate synthetic data using the Python generator.
- Optionally fine-tune with VB-LoRA and/or run ORPO workflows (see linked READMEs).

A. Setup and run without the CLI (manual, minimal)

1. Prepare Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install Python dependencies needed for data generation

```bash
pip install python-dotenv pandas huggingface-hub transformers torch
```

3. Configure Hugging Face token (one of these)

```bash
export HF_TOKEN=your_hf_token_here
# or create a dotenv file used by scripts
echo "HF_TOKEN=your_hf_token_here" > prototyping/.env
```

4. Run the interactive Python data generator (no Node/CLI required)

```bash
cd prototyping
python interactive-data-gen.py
# or call the lower-level script directly
python data-gen.py --help
```

5. Optional: Fine-tune / run VB-LoRA

Follow the `finetuning` README for VB-LoRA usage and options: [prototyping/finetuning/finetuners/README.md](prototyping/finetuning/finetuners/README.md)

6. Optional: Use ORPO workflows

See `orpo` README for ORPO-specific training and utilities: [prototyping/finetuning/finetuners/orpo/README.md](prototyping/finetuning/finetuners/orpo/README.md)

B. Setup and run with the CLI (recommended for interactive users)

1. Install Node dependencies and build the CLI

```bash
cd prototyping/cli-wrapper
npm install
npm run build
# (optional) install globally for `synth-data` convenience
npm link
```

2. Verify CLI install

```bash
# if linked
synth-data --version
synth-data --help

# or run from package directory without linking
cd prototyping/cli-wrapper
npm start
```

3. Use the CLI to generate synthetic data

```bash
# run the interactive flow; follow prompts
synth-data

# or invoke the local npm start command
cd prototyping/cli-wrapper
npm start
```

CLI behavior and automation
- The CLI collects labels, examples, and generation parameters then calls the Python generator (`prototyping/interactive-data-gen.py` / `prototyping/data-gen.py`).
- The CLI checks for Python availability and `HF_TOKEN`. It will continue without `HF_TOKEN` but some models require authentication.

Common commands (summary)

```bash
# Manual / non-CLI
python prototyping/interactive-data-gen.py

# CLI (after npm link)
- See prototyping/cli-wrapper/README.md for CLI-specific usage details and screenshots/examples.

# VB-LoRA (see detailed README)
python finetuning/cli.py --help

# ORPO (see detailed README)
python orpo/cli.py --help
```

Notes and troubleshooting (MVP)
- If `synth-data: command not found`, run `npm link` from `prototyping/cli-wrapper`.
- If Python packages are missing: `pip install python-dotenv pandas huggingface-hub transformers torch`.
- If Hugging Face auth errors occur, ensure `HF_TOKEN` is exported or present in `prototyping/.env`.

Where to go next
- For VB-LoRA details and training presets: open [prototyping/finetuning/finetuners/README.md](prototyping/finetuning/finetuners/README.md).
- For ORPO-specific training and utilities: open [prototyping/finetuning/finetuners/orpo/README.md](prototyping/finetuning/finetuners/orpo/README.md).
- For CLI UX and options: open [prototyping/cli-wrapper/README.md](prototyping/cli-wrapper/README.md).

Acknowledgments
- Uses Hugging Face transformers and related tooling for model access and text generation.
- Uses VLLM for optimized CUDA training
