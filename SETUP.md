# 🚀 Quick Setup Guide - Synthetic Data Generator

## One-Command Installation

```bash
cd prototyping/cli-wrapper && npm install && npm run build && npm link
```

## Verify Installation

```bash
synth-data --version  # Should show: 1.0.0
synth-data --help     # Show usage info
```

## Run the Tool

```bash
synth-data
```

---

## Complete Setup (Step-by-Step)

### 1️⃣ Install Node.js Dependencies

```bash
cd prototyping/cli-wrapper
npm install
npm run build
npm link
```

### 2️⃣ Install Python Dependencies

```bash
pip install python-dotenv pandas huggingface-hub transformers torch
```

### 3️⃣ Set Hugging Face Token

```bash
export HF_TOKEN=your_token_here
```

Or create a `.env` file:

```bash
echo "HF_TOKEN=your_token_here" > prototyping/.env
```

### 4️⃣ Test Everything Works

```bash
cd prototyping/cli-wrapper
./test.sh
```

### 5️⃣ Run the Generator

```bash
synth-data
```

---

## Troubleshooting

### Command not found

```bash
cd prototyping/cli-wrapper
npm link
```

### Build errors

```bash
cd prototyping/cli-wrapper
rm -rf node_modules dist
npm install
npm run build
npm link
```

### Python dependencies missing

```bash
pip install python-dotenv pandas huggingface-hub transformers torch
```

---

## What You'll See

When you run `synth-data`, you'll get an interactive CLI:

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

The tool will guide you through:
1. 📋 Use Case Configuration
2. 🏷️ Labels Configuration  
3. 🗂️ Categories Configuration
4. 💡 Example Configuration
5. 🤖 Model & Generation Settings

---

## Additional Commands

```bash
# View version
synth-data --version

# View help
synth-data --help

# Run in development mode
cd prototyping/cli-wrapper
npm run dev

# Run tests
cd prototyping/cli-wrapper
./test.sh
```

---

**For complete documentation, see [README.md](README.md)**
