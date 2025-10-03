#!/usr/bin/env node
"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const child_process_1 = require("child_process");
const commander_1 = require("commander");
const chalk_1 = __importDefault(require("chalk"));
const boxen_1 = __importDefault(require("boxen"));
const ora_1 = __importDefault(require("ora"));
const inquirer = __importStar(require("@inquirer/prompts"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
class SyntheticDataCLI {
    constructor() {
        this.config = {};
        this.pythonScript = path.join(__dirname, '../../interactive-data-gen.py');
    }
    printBanner() {
        const banner = (0, boxen_1.default)(chalk_1.default.bold.cyan('🚀 SYNTHETIC DATA GENERATOR 🚀') + '\n\n' +
            chalk_1.default.white('TypeScript CLI Wrapper for AI-Powered Data Generation'), {
            padding: 1,
            margin: 1,
            borderStyle: 'round',
            borderColor: 'cyan'
        });
        console.log(banner);
    }
    async checkDependencies() {
        const spinner = (0, ora_1.default)('Checking dependencies...').start();
        try {
            // Check if Python script exists
            if (!fs.existsSync(this.pythonScript)) {
                spinner.fail(chalk_1.default.red('Python script not found!'));
                console.log(chalk_1.default.yellow(`Expected location: ${this.pythonScript}`));
                return false;
            }
            // Check if Python is available
            const pythonCheck = (0, child_process_1.spawn)('python3', ['--version']);
            await new Promise((resolve, reject) => {
                pythonCheck.on('close', (code) => {
                    if (code === 0) {
                        resolve(true);
                    }
                    else {
                        reject(new Error('Python3 not found'));
                    }
                });
            });
            spinner.succeed(chalk_1.default.green('All dependencies satisfied'));
            return true;
        }
        catch (error) {
            spinner.fail(chalk_1.default.red('Dependency check failed'));
            console.log(chalk_1.default.yellow('Please ensure Python 3 is installed'));
            return false;
        }
    }
    async configureUseCase() {
        console.log(chalk_1.default.bold.blue('\n📋 STEP 1: Use Case Configuration'));
        console.log(chalk_1.default.gray('─'.repeat(50)));
        this.config.useCase = await inquirer.input({
            message: 'What is the main purpose of your synthetic data?',
            default: 'text classification',
            validate: (input) => input.length > 0 || 'Use case is required'
        });
        console.log(chalk_1.default.green(`✅ Use case set: ${this.config.useCase}`));
    }
    async configureLabels() {
        console.log(chalk_1.default.bold.blue('\n🏷️  STEP 2: Labels Configuration'));
        console.log(chalk_1.default.gray('─'.repeat(50)));
        this.config.labels = [];
        let addMore = true;
        while (addMore) {
            const labelName = await inquirer.input({
                message: 'Enter a label name:',
                validate: (input) => input.length > 0 || 'Label name is required'
            });
            const labelDesc = await inquirer.input({
                message: `Describe what '${labelName}' means:`,
                validate: (input) => input.length > 0 || 'Description is required'
            });
            this.config.labels.push({ name: labelName, description: labelDesc });
            console.log(chalk_1.default.green(`✅ Added label: ${labelName}`));
            addMore = await inquirer.confirm({
                message: 'Add another label?',
                default: true
            });
        }
    }
    async configureCategories() {
        console.log(chalk_1.default.bold.blue('\n🗂️  STEP 3: Categories Configuration'));
        console.log(chalk_1.default.gray('─'.repeat(50)));
        this.config.categories = [];
        let addMore = true;
        while (addMore) {
            const categoryName = await inquirer.input({
                message: 'Enter a category name:',
                validate: (input) => input.length > 0 || 'Category name is required'
            });
            const typesInput = await inquirer.input({
                message: `Enter types for '${categoryName}' (comma-separated):`,
                validate: (input) => input.length > 0 || 'At least one type is required'
            });
            const types = typesInput.split(',').map(t => t.trim());
            this.config.categories.push({ name: categoryName, types });
            console.log(chalk_1.default.green(`✅ Added category: ${categoryName} with ${types.length} types`));
            addMore = await inquirer.confirm({
                message: 'Add another category?',
                default: true
            });
        }
    }
    async configureExamples() {
        console.log(chalk_1.default.bold.blue('\n💡 STEP 4: Example Configuration'));
        console.log(chalk_1.default.gray('─'.repeat(50)));
        const numExamples = await inquirer.number({
            message: 'How many examples would you like to provide?',
            default: 2,
            validate: (input) => (input && input > 0) || 'Must provide at least 1 example'
        });
        const examples = [];
        for (let i = 0; i < numExamples; i++) {
            console.log(chalk_1.default.yellow(`\n--- Example ${i + 1} ---`));
            console.log(chalk_1.default.gray('Enter your example (press Ctrl+D or type END on a new line to finish):'));
            const example = await inquirer.editor({
                message: `Example ${i + 1}:`,
                default: `LABEL: \nCATEGORY: \nTYPE: \nOUTPUT: \nREASONING: `
            });
            examples.push(example);
            console.log(chalk_1.default.green(`✅ Example ${i + 1} added`));
        }
        this.config.examples = examples.join('\n\n');
    }
    async configureModelSettings() {
        console.log(chalk_1.default.bold.blue('\n🤖 STEP 5: Model & Generation Settings'));
        console.log(chalk_1.default.gray('─'.repeat(50)));
        const modelChoices = [
            { name: 'Llama 3.2 3B Instruct', value: 'meta-llama/Llama-3.2-3B-Instruct' },
            { name: 'Gemma 3 1B', value: 'google/gemma-3-1b-it' },
            { name: 'SmolLM2 1.7B Instruct', value: 'HuggingFaceTB/SmolLM2-1.7B-Instruct' },
            { name: 'Custom model', value: 'custom' }
        ];
        const modelChoice = await inquirer.select({
            message: 'Select a model:',
            choices: modelChoices
        });
        if (modelChoice === 'custom') {
            this.config.model = await inquirer.input({
                message: 'Enter custom model name (HuggingFace format):',
                validate: (input) => input.length > 0 || 'Model name is required'
            });
        }
        else {
            this.config.model = modelChoice;
        }
        this.config.sampleSize = await inquirer.number({
            message: 'Number of samples to generate:',
            default: 100,
            validate: (input) => (input && input > 0) || 'Must be a positive number'
        });
        this.config.maxTokens = await inquirer.number({
            message: 'Maximum tokens per sample:',
            default: 256,
            validate: (input) => (input && input > 0) || 'Must be a positive number'
        });
        this.config.batchSize = await inquirer.number({
            message: 'Batch size for processing:',
            default: 20,
            validate: (input) => (input && input > 0) || 'Must be a positive number'
        });
        this.config.outputDir = await inquirer.input({
            message: 'Output directory:',
            default: './generated_data'
        });
        this.config.saveReasoning = await inquirer.confirm({
            message: 'Save reasoning for each generated sample?',
            default: true
        });
        console.log(chalk_1.default.green('✅ Model and generation settings configured'));
    }
    displaySummary() {
        console.log(chalk_1.default.bold.cyan('\n📊 CONFIGURATION SUMMARY'));
        console.log(chalk_1.default.gray('═'.repeat(50)));
        console.log(chalk_1.default.white(`Use case: ${this.config.useCase}`));
        console.log(chalk_1.default.white(`Labels: ${this.config.labels?.map(l => l.name).join(', ')}`));
        console.log(chalk_1.default.white(`Categories: ${this.config.categories?.map(c => c.name).join(', ')}`));
        console.log(chalk_1.default.white(`Model: ${this.config.model}`));
        console.log(chalk_1.default.white(`Sample size: ${this.config.sampleSize}`));
        console.log(chalk_1.default.white(`Max tokens: ${this.config.maxTokens}`));
        console.log(chalk_1.default.white(`Batch size: ${this.config.batchSize}`));
        console.log(chalk_1.default.white(`Output directory: ${this.config.outputDir}`));
        console.log(chalk_1.default.white(`Save reasoning: ${this.config.saveReasoning ? 'Yes' : 'No'}`));
        console.log(chalk_1.default.gray('═'.repeat(50)));
    }
    createTempConfigFile() {
        const tempDir = fs.mkdtempSync('/tmp/synth-data-');
        const configPath = path.join(tempDir, 'auto_config.py');
        const labelDescriptions = this.config.labels
            .map(l => `${l.name}: ${l.description}`)
            .join('\n');
        const categoriesTypes = this.config.categories.reduce((acc, cat) => {
            acc[cat.name] = cat.types;
            return acc;
        }, {});
        const configContent = `# Auto-generated configuration file
labels = ${JSON.stringify(this.config.labels.map(l => l.name))}

label_descriptions = """${labelDescriptions}"""

categories_types = ${JSON.stringify(categoriesTypes, null, 2).replace(/"/g, "'")}

use_case = "${this.config.useCase}"

prompt_examples = """${this.config.examples}"""
`;
        fs.writeFileSync(configPath, configContent);
        return configPath;
    }
    async runDataGeneration() {
        console.log(chalk_1.default.bold.green('\n🚀 STARTING DATA GENERATION'));
        console.log(chalk_1.default.gray('─'.repeat(50)));
        const configPath = this.createTempConfigFile();
        // Get the absolute path to data-gen.py (in prototyping folder)
        // __dirname will be cli-wrapper/dist, so we go up to prototyping
        const dataGenPath = path.join(__dirname, '../../data-gen.py');
        const args = [
            'run', 'python', dataGenPath,
            '--config', configPath,
            '--sample_size', this.config.sampleSize.toString(),
            '--model', this.config.model,
            '--max_new_tokens', this.config.maxTokens.toString(),
            '--batch_size', this.config.batchSize.toString(),
            '--output_dir', this.config.outputDir
        ];
        if (this.config.saveReasoning) {
            args.push('--save_reasoning');
        }
        // Set working directory to prototyping folder
        const prototypingDir = path.join(__dirname, '../..');
        console.log(chalk_1.default.cyan('\n📝 Note: You may be prompted for your Hugging Face token if not already configured.\n'));
        // Use spawn with full stdio inheritance to allow interactive prompts
        const dataGen = (0, child_process_1.spawn)('uv', args, {
            cwd: prototypingDir,
            stdio: 'inherit', // This passes stdin, stdout, stderr to the child process
            shell: false
        });
        return new Promise((resolve, reject) => {
            dataGen.on('close', (code) => {
                // Clean up temp config
                try {
                    fs.unlinkSync(configPath);
                    fs.rmdirSync(path.dirname(configPath));
                }
                catch (error) {
                    // Ignore cleanup errors
                }
                if (code === 0) {
                    console.log(chalk_1.default.green('\n✅ Data generation completed successfully!'));
                    console.log(chalk_1.default.cyan(`📁 Check your output directory: ${this.config.outputDir}`));
                    resolve();
                }
                else {
                    console.log(chalk_1.default.red(`\n❌ Data generation failed with code: ${code}`));
                    reject(new Error(`Process exited with code ${code}`));
                }
            });
            dataGen.on('error', (error) => {
                console.log(chalk_1.default.red(`\n❌ Error running data generation: ${error.message}`));
                reject(error);
            });
        });
    }
    async run() {
        this.printBanner();
        const depsOk = await this.checkDependencies();
        if (!depsOk) {
            process.exit(1);
        }
        try {
            await this.configureUseCase();
            await this.configureLabels();
            await this.configureCategories();
            await this.configureExamples();
            await this.configureModelSettings();
            this.displaySummary();
            const proceed = await inquirer.confirm({
                message: 'Proceed with data generation?',
                default: true
            });
            if (proceed) {
                await this.runDataGeneration();
            }
            else {
                console.log(chalk_1.default.yellow('❌ Data generation cancelled.'));
            }
        }
        catch (error) {
            if (error.name === 'ExitPromptError') {
                console.log(chalk_1.default.yellow('\n\n❌ Process interrupted by user.'));
                process.exit(1);
            }
            throw error;
        }
    }
}
// CLI Setup
const program = new commander_1.Command();
program
    .name('synth-data')
    .description('Interactive CLI for synthetic data generation')
    .version('1.0.0')
    .action(async () => {
    const cli = new SyntheticDataCLI();
    await cli.run();
});
program.parse();
//# sourceMappingURL=index.js.map