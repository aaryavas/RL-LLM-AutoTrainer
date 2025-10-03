# Synthetic Data Generator - TypeScript CLI

A modern, type-safe TypeScript CLI wrapper for the interactive synthetic data generator.

## Features

✨ **Beautiful CLI Interface** with colors and formatting  
🎯 **Type-Safe** TypeScript implementation  
📦 **Easy Installation** with npm  
🚀 **Interactive Prompts** for configuration  
✅ **Dependency Checking** before execution  
🎨 **Enhanced UX** with spinners, boxes, and progress indicators  

## Installation

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- `uv` package manager (for Python)

### Setup

1. **Navigate to the CLI wrapper directory:**
   ```bash
   cd prototyping/cli-wrapper
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Build the TypeScript code:**
   ```bash
   npm run build
   ```

4. **Link for global use (optional):**
   ```bash
   npm link
   ```

## Usage

### Development Mode

Run directly with ts-node:
```bash
npm run dev
```

### Production Mode

After building:
```bash
npm start
```

Or if globally linked:
```bash
synth-data
```

## How It Works

The TypeScript CLI provides an enhanced interface that:

1. **Checks Dependencies** - Verifies Python, required packages, and scripts
2. **Interactive Configuration** - Guides you through 5 setup steps:
   - Use Case Configuration
   - Labels Setup
   - Categories Configuration
   - Examples Input
   - Model & Generation Settings
3. **Displays Summary** - Shows your configuration before proceeding
4. **Executes Generation** - Calls the Python backend with proper parameters

## Configuration Steps

### Step 1: Use Case
Define the purpose of your synthetic data generation.

### Step 2: Labels
Configure classification labels with descriptions.

### Step 3: Categories
Set up data diversification categories and types.

### Step 4: Examples
Provide few-shot learning examples for the AI.

### Step 5: Model Settings
Choose model and configure generation parameters.

## Available Scripts

```bash
npm run dev        # Run in development mode with ts-node
npm run build      # Compile TypeScript to JavaScript
npm start          # Run compiled JavaScript
npm run watch      # Watch mode for development
```

## Technologies Used

- **TypeScript** - Type-safe JavaScript
- **Commander.js** - CLI framework
- **Inquirer** - Interactive prompts
- **Chalk** - Terminal colors
- **Ora** - Loading spinners
- **Boxen** - Terminal boxes

## Project Structure

```
cli-wrapper/
├── src/
│   └── index.ts          # Main CLI application
├── dist/                 # Compiled JavaScript (generated)
├── package.json          # Node.js dependencies
├── tsconfig.json         # TypeScript configuration
└── README.md            # This file
```

## Example Session

```bash
$ synth-data

╭────────────────────────────────────────────────────────╮
│                                                        │
│   🚀 SYNTHETIC DATA GENERATOR 🚀                      │
│                                                        │
│   TypeScript CLI Wrapper for AI-Powered Data         │
│   Generation                                          │
│                                                        │
╰────────────────────────────────────────────────────────╯

✔ Checking dependencies... done

📋 STEP 1: Use Case Configuration
──────────────────────────────────────────────────
? What is the main purpose of your synthetic data? › customer support chatbot
✅ Use case set: customer support chatbot

🏷️ STEP 2: Labels Configuration
──────────────────────────────────────────────────
? Enter a label name: › helpful
? Describe what 'helpful' means: › Provides clear answers to customer questions
✅ Added label: helpful
...
```

## Advantages Over Python CLI

1. **Better UX** - Enhanced visual feedback with colors and formatting
2. **Type Safety** - TypeScript prevents runtime errors
3. **Modern Tooling** - npm ecosystem and JavaScript tooling
4. **Cross-Platform** - Better Windows support
5. **Async/Await** - Cleaner asynchronous code
6. **Package Distribution** - Easy to distribute via npm

## Troubleshooting

### "Cannot find Python script"
Ensure the Python `interactive-data-gen.py` is in the correct location relative to the CLI wrapper.

### "Python3 not found"
Install Python 3.8 or higher and ensure it's in your PATH.

### Build Errors
Make sure all dependencies are installed:
```bash
npm install
```

## Contributing

Feel free to submit issues or pull requests to improve the CLI!

## License

Same as parent project.
