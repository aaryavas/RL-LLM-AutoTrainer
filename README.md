# RL-LLM-AutoTrainer

Overview
- This repository contains prototyping tools for generating synthetic training data for LLM/RL workflows. The current prototype includes:
  - A TypeScript/Node.js CLI wrapper that provides an interactive, user-friendly terminal interface.
  - Python scripts that actually perform (or orchestrate) the data generation, using the Hugging Face ecosystem.

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
    - src/index.ts — main CLI source (compiled to dist/index.js)
    - package.json — npm package with scripts and bin
    - tsconfig.json — TypeScript config
    - install.sh — helper script to install/build/link CLI
    - README.md — CLI-specific documentation
  - interactive-data-gen.py — interactive Python front-end that guides the user and calls data-gen.py
  - data-gen.py — Python generator invoked by the interactive script (see code for parameters)

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

Setup and Installation
1) TypeScript CLI (recommended path)
- Navigate to the CLI package:
  - cd prototyping/cli-wrapper
- Install dependencies:
  - npm install
- Build the TypeScript code:
  - npm run build
- Optional: Install globally for a synth-data command:
  - npm link
- Alternatively, use the convenience script:
  - ./install.sh

2) Python environment
- Ensure Python dependencies are installed (choose one of the following):
  - pip install python-dotenv pandas huggingface-hub transformers torch
  - or with uv: uv pip install python-dotenv pandas huggingface-hub transformers torch
- Set your HF_TOKEN as described above.

Running
- Using the globally linked CLI (if you ran npm link inside prototyping/cli-wrapper):
  - synth-data
- Running from the CLI package directory without linking:
  - npm start        # runs dist/index.js
  - npm run dev      # runs src/index.ts via ts-node
- Running the Python interactive script directly (bypassing the Node CLI):
  - python3 prototyping/interactive-data-gen.py

Available Scripts (Node package: prototyping/cli-wrapper)
- npm run dev     — ts-node src/index.ts
- npm run build   — tsc
- npm start       — node dist/index.js
- npm run watch   — tsc --watch

Notes on Behavior
- The Node CLI checks for Python 3 availability and for prototyping/interactive-data-gen.py. It then launches the Python interactive flow, which:
  - Verifies the presence of key Python packages.
  - Checks for HF_TOKEN and offers to continue if missing (some models will require it).
  - Guides you through defining labels, categories, examples, and generation parameters.
  - Invokes data-gen.py with the collected configuration.

Testing
- No automated tests were found in this repository.
- TODO: Add tests for:
  - Node CLI argument parsing and path handling.
  - Python interactive flow (unit tests for configuration assembly).
  - Integration test invoking data-gen.py with a small sample size.

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
