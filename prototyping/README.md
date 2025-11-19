# RL-LLM-AutoTrainer Prototyping

Interactive CLI for synthetic data generation and VB-LoRA fine-tuning.

## Quick Start

### One-Command Setup

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Unix/macOS/Linux:**
```bash
chmod +x setup.sh && ./setup.sh
```

This installs all Python and Node.js dependencies and builds the CLI.

### Manual Installation

1. **Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **CLI wrapper:**
   ```bash
   cd cli-wrapper
   npm install
   npm run build
   ```

## Usage

```bash
cd cli-wrapper
npm start
```

Or directly:
```bash
node cli-wrapper/dist/index.js
```

## Project Structure

```
prototyping/
├── requirements.txt      # All Python dependencies (consolidated)
├── setup.ps1            # Windows setup script
├── setup.sh             # Unix setup script
├── cli-wrapper/         # TypeScript CLI (Commander + Inquirer)
├── modeling/            # PEFT RL training modules
├── vblora/              # VB-LoRA fine-tuning implementation
├── data-gen.py          # Synthetic data generation
└── interactive-data-gen.py
```

## Requirements

- Python 3.9+
- Node.js LTS (v20+)
- CUDA-compatible GPU (recommended for training)
