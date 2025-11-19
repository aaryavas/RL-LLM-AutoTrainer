# RL-LLM-AutoTrainer

## 🚀 Quick Start

> **📖 Looking for a quick reference?** See [SETUP.md](SETUP.md) for a one-page setup guide.

Want to generate synthetic data immediately? Run this:

```bash
# 1. Navigate to the CLI directory
cd prototyping/cli-wrapper

# 2. Install and build (first time only)
npm install && npm run build && npm link

# 3. Run the tool (from anywhere after linking)
synth-data
```

That's it! The interactive CLI will guide you through the rest.

**Verify Installation:**
```bash
synth-data --version  # Should output: 1.1.0
synth-data --help     # Shows usage information
./test.sh             # Run test suite (from cli-wrapper directory)
```

**What You'll See:**
```
   ╭───────────────────────────────────────────────────────────╮
   │                                                           │
   │   🚀 SYNTHETIC DATA GENERATOR 🚀                          │
   │                                                           │
   │   TypeScript CLI Wrapper for AI-Powered Data Generation   │
   │                                                           │
   ╰───────────────────────────────────────────────────────────╯

✔ All dependencies satisfied

📋 STEP 1: Use Case Configuration
──────────────────────────────────────────────────
? What is the main purpose of your synthetic data? ›
```

---

## Overview
- This repository contains prototyping tools for generating synthetic training data and fine-tuning LLMs. The current prototype includes:
  - A TypeScript/Node.js CLI wrapper that provides an interactive, user-friendly terminal interface.
  - Python scripts for data generation using the Hugging Face ecosystem.
  - VB-LoRA fine-tuning module for efficient model adaptation.

### Workflow
1. **Generate synthetic data** - Configure labels, categories, and examples
2. **Optional: Fine-tune a model** - Use VB-LoRA for memory-efficient fine-tuning on your generated data

Stack and Tooling
- Languages: TypeScript (Node.js), Python 3
- Frameworks/Libraries (Node): commander, @inquirer/prompts, chalk, ora, boxen
- Build Tooling (Node): TypeScript (tsc), ts-node (dev)
- Package Manager (Node): npm (package-lock.json present)
- Python Libraries used (inferred from code): python-dotenv, pandas, huggingface-hub, transformers, torch
- Model access: Hugging Face Hub (requires HF_TOKEN)

Repository Structure
- prototyping/
  - cli-wrapper/ — TypeScript CLI package
    - src/
      - index.ts — CLI entry point
      - cli.ts — main orchestrator
      - config/ — types and defaults
      - steps/ — configuration step handlers
      - runners/ — process spawners
      - utils/ — display and dependency utilities
    - package.json — npm package with scripts and bin
    - tsconfig.json — TypeScript config
  - vblora/ — VB-LoRA fine-tuning module
    - cli.py — fine-tuning CLI
    - finetuning.py — main fine-tuning API
    - config/ — configuration dataclasses
    - core/ — data processing, model loading
    - training/ — trainer and callbacks
  - data-gen.py — Python generator for synthetic data

Entry Points
- synth-data (global command if linked) → prototyping/cli-wrapper/dist/index.js
  - npm bin defined in prototyping/cli-wrapper/package.json
- Node development entry: prototyping/cli-wrapper/src/index.ts (run via ts-node)
- Python interactive entry: prototyping/interactive-data-gen.py (invoked by the Node CLI)
- Python generator: prototyping/data-gen.py (called by interactive-data-gen.py)

Requirements
- Node.js 18+ and npm
- Python 3.8+ (3.10+ recommended for better torch/transformers support)
- Internet access for Hugging Face model downloads
- Environment variable: HF_TOKEN (Hugging Face token) for model access

Environment Variables
- HF_TOKEN: Your Hugging Face access token
  - You can export it directly: export HF_TOKEN=your_token
  - Or put it in a .env file (python-dotenv is used when available):
    - HF_TOKEN=your_token

## 📦 Setup and Installation

### Prerequisites
Before you begin, ensure you have:
- Node.js 18+ and npm installed
- Python 3.8+ (3.10+ recommended)
- A Hugging Face account and access token

### Step 1: Install TypeScript CLI

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

### Step 2: Install Python Dependencies

Install required Python packages for data generation:

```bash
pip install python-dotenv pandas huggingface-hub transformers torch
```

**For fine-tuning, also install:**

```bash
pip install peft accelerate bitsandbytes datasets evaluate scikit-learn nltk rouge_score
```

**Or install all from requirements:**

```bash
pip install -r prototyping/vblora/requirements.txt
```

### Step 3: Configure Hugging Face Token

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

Run the test suite:

```bash
cd prototyping/cli-wrapper
./test.sh
```

## 🚀 Running the Tool

### Recommended: Global Command

After installation, run from anywhere:

```bash
synth-data
```

✨ **This is the easiest way!** The interactive CLI will guide you through all configuration steps.

### Alternative: Run from CLI Directory

If you haven't run `npm link`, you can run from the CLI package directory:

```bash
cd prototyping/cli-wrapper
npm start
```

### Development Mode

For development with live TypeScript compilation:

```bash
cd prototyping/cli-wrapper
npm run dev
```

### Direct Python Execution

Bypass the Node CLI and run Python directly:

```bash
cd prototyping
python3 interactive-data-gen.py
```

## 🤖 Fine-Tuning Configuration

After generating synthetic data, the CLI will prompt you to optionally fine-tune a model. Here are the configurable options:

### Model Selection
- **Model Family**: Currently SmolLM2 (extensible for future models)
- **Variants**: 135M, 360M, 1.7B
- **Presets**: minimal, standard, aggressive

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| Epochs | 3 | Number of training epochs |
| Learning Rate | 2e-4 | Base learning rate |
| Batch Size | 4 | Batch size per device |
| Early Stopping | 3 | Patience in epochs |

### VB-LoRA Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| Num Vectors | 90 | Vectors in vector bank |
| Vector Length | 64 | Length of each vector |
| LoRA Rank | 4 | Rank for low-rank adaptation |
| LR Vector Bank | 1e-3 | Learning rate for vector bank |
| LR Logits | 1e-2 | Learning rate for logits |

### Hardware Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| Bits | 4 | Quantization (4/8/16/32) |
| BF16 | false | Use bfloat16 precision |
| FP16 | false | Use float16 precision |

### Direct Fine-Tuning (Python)

You can also run fine-tuning directly:

```bash
cd prototyping/vblora
python cli.py finetune data.csv --variant SmolLM2-360M --epochs 5
```

## 🛠️ Available Scripts

All commands should be run from `prototyping/cli-wrapper`:

**Development:**
```bash
npm run dev        # Run with ts-node (hot reload)
npm run watch      # Watch mode for TypeScript compilation
```

**Build:**
```bash
npm run build      # Compile TypeScript to JavaScript
```

**Run:**
```bash
npm start          # Run the compiled CLI
synth-data         # Run globally (after npm link)
```

**Testing:**
```bash
./test.sh          # Run the test suite
```

Notes on Behavior
- The Node CLI checks for Python 3 availability and for prototyping/interactive-data-gen.py. It then launches the Python interactive flow, which:
  - Verifies the presence of key Python packages.
  - Checks for HF_TOKEN and offers to continue if missing (some models will require it).
  - Guides you through defining labels, categories, examples, and generation parameters.
  - Invokes data-gen.py with the collected configuration.

## Troubleshooting

**`synth-data: command not found`**
```bash
cd prototyping/cli-wrapper
npm link
```

**Build errors with TypeScript:**
```bash
cd prototyping/cli-wrapper
npm install
npm run build
```

**Python dependencies missing:**
```bash
pip install python-dotenv pandas huggingface-hub transformers torch
# or with uv:
uv pip install python-dotenv pandas huggingface-hub transformers torch
```

**HuggingFace authentication errors:**
```bash
export HF_TOKEN=your_token_here
# or create a .env file in prototyping/ with:
echo "HF_TOKEN=your_token_here" > prototyping/.env
```

**To reset and reinstall everything:**
```bash
cd prototyping/cli-wrapper
rm -rf node_modules dist
npm install
npm run build
npm link
synth-data --version  # verify it works
```

## ✅ Testing

### Quick Verification

Check that the CLI is installed correctly:

```bash
synth-data --version
```

View help documentation:

```bash
synth-data --help
```

### Run the Automated Test Suite

```bash
cd prototyping/cli-wrapper
./test.sh
```

The test suite verifies:
- ✓ Build succeeded
- ✓ Help command works
- ✓ Version check works
- ✓ Python script exists
- ✓ Dependencies installed
- ✓ TypeScript compiles without errors

### Test the Full Flow

Generate a small test dataset:

```bash
synth-data
```

Follow the interactive prompts and set a small sample size (e.g., 5 samples) for a quick test.

### Future Improvements
- Add unit tests for configuration assembly
- Add integration tests with mock models
- Add CI/CD pipeline for automated testing

Project Status and Roadmap
- Current focus: interactive synthetic data generation prototype.
- TODO: Document data-gen.py parameters/outputs in detail.
- TODO: Add example outputs and a small sample dataset for verification.
- TODO: Wire into any RL training loop if applicable to the broader project goals.

License
- The prototyping/cli-wrapper package declares license: MIT (see its package.json).
- Top-level repository license file not found.
- TODO: Add a LICENSE file at the repository root clarifying the license for the entire project.

Additional Documentation
- See prototyping/cli-wrapper/README.md for CLI-specific usage details and screenshots/examples.

Acknowledgments
- Uses Hugging Face transformers and related tooling for model access and text generation.
