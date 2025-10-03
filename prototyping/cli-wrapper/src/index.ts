#!/usr/bin/env node

import { spawn } from 'child_process';
import { Command } from 'commander';
import chalk from 'chalk';
import boxen from 'boxen';
import ora from 'ora';
import * as inquirer from '@inquirer/prompts';
import * as path from 'path';
import * as fs from 'fs';

interface Label {
  name: string;
  description: string;
}

interface Category {
  name: string;
  types: string[];
}

interface Config {
  useCase: string;
  labels: Label[];
  categories: Category[];
  examples: string;
  model: string;
  sampleSize: number;
  maxTokens: number;
  batchSize: number;
  outputDir: string;
  saveReasoning: boolean;
}

class SyntheticDataCLI {
  private config: Partial<Config> = {};
  private pythonScript: string;

  constructor() {
    this.pythonScript = path.join(__dirname, '../../interactive-data-gen.py');
  }

  private printBanner(): void {
    const banner = boxen(
      chalk.bold.cyan('🚀 SYNTHETIC DATA GENERATOR 🚀') + '\n\n' +
      chalk.white('TypeScript CLI Wrapper for AI-Powered Data Generation'),
      {
        padding: 1,
        margin: 1,
        borderStyle: 'round',
        borderColor: 'cyan'
      }
    );
    console.log(banner);
  }

  private async checkDependencies(): Promise<boolean> {
    const spinner = ora('Checking dependencies...').start();

    try {
      // Check if Python script exists
      if (!fs.existsSync(this.pythonScript)) {
        spinner.fail(chalk.red('Python script not found!'));
        console.log(chalk.yellow(`Expected location: ${this.pythonScript}`));
        return false;
      }

      // Check if Python is available
      const pythonCheck = spawn('python3', ['--version']);
      
      await new Promise((resolve, reject) => {
        pythonCheck.on('close', (code) => {
          if (code === 0) {
            resolve(true);
          } else {
            reject(new Error('Python3 not found'));
          }
        });
      });

      spinner.succeed(chalk.green('All dependencies satisfied'));
      return true;
    } catch (error) {
      spinner.fail(chalk.red('Dependency check failed'));
      console.log(chalk.yellow('Please ensure Python 3 is installed'));
      return false;
    }
  }

  private async configureUseCase(): Promise<void> {
    console.log(chalk.bold.blue('\n📋 STEP 1: Use Case Configuration'));
    console.log(chalk.gray('─'.repeat(50)));

    this.config.useCase = await inquirer.input({
      message: 'What is the main purpose of your synthetic data?',
      default: 'text classification',
      validate: (input) => input.length > 0 || 'Use case is required'
    });

    console.log(chalk.green(`✅ Use case set: ${this.config.useCase}`));
  }

  private async configureLabels(): Promise<void> {
    console.log(chalk.bold.blue('\n🏷️  STEP 2: Labels Configuration'));
    console.log(chalk.gray('─'.repeat(50)));

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

      this.config.labels!.push({ name: labelName, description: labelDesc });
      console.log(chalk.green(`✅ Added label: ${labelName}`));

      addMore = await inquirer.confirm({
        message: 'Add another label?',
        default: true
      });
    }
  }

  private async configureCategories(): Promise<void> {
    console.log(chalk.bold.blue('\n🗂️  STEP 3: Categories Configuration'));
    console.log(chalk.gray('─'.repeat(50)));

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
      this.config.categories!.push({ name: categoryName, types });
      console.log(chalk.green(`✅ Added category: ${categoryName} with ${types.length} types`));

      addMore = await inquirer.confirm({
        message: 'Add another category?',
        default: true
      });
    }
  }

  private async configureExamples(): Promise<void> {
    console.log(chalk.bold.blue('\n💡 STEP 4: Example Configuration'));
    console.log(chalk.gray('─'.repeat(50)));

    const numExamples = await inquirer.number({
      message: 'How many examples would you like to provide?',
      default: 2,
      validate: (input) => (input && input > 0) || 'Must provide at least 1 example'
    });

    const examples: string[] = [];

    for (let i = 0; i < numExamples!; i++) {
      console.log(chalk.yellow(`\n--- Example ${i + 1} ---`));
      console.log(chalk.gray('Enter your example (press Ctrl+D or type END on a new line to finish):'));
      
      const example = await inquirer.editor({
        message: `Example ${i + 1}:`,
        default: `LABEL: \nCATEGORY: \nTYPE: \nOUTPUT: \nREASONING: `
      });

      examples.push(example);
      console.log(chalk.green(`✅ Example ${i + 1} added`));
    }

    this.config.examples = examples.join('\n\n');
  }

  private async configureModelSettings(): Promise<void> {
    console.log(chalk.bold.blue('\n🤖 STEP 5: Model & Generation Settings'));
    console.log(chalk.gray('─'.repeat(50)));

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
    } else {
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

    console.log(chalk.green('✅ Model and generation settings configured'));
  }

  private displaySummary(): void {
    console.log(chalk.bold.cyan('\n📊 CONFIGURATION SUMMARY'));
    console.log(chalk.gray('═'.repeat(50)));
    console.log(chalk.white(`Use case: ${this.config.useCase}`));
    console.log(chalk.white(`Labels: ${this.config.labels?.map(l => l.name).join(', ')}`));
    console.log(chalk.white(`Categories: ${this.config.categories?.map(c => c.name).join(', ')}`));
    console.log(chalk.white(`Model: ${this.config.model}`));
    console.log(chalk.white(`Sample size: ${this.config.sampleSize}`));
    console.log(chalk.white(`Max tokens: ${this.config.maxTokens}`));
    console.log(chalk.white(`Batch size: ${this.config.batchSize}`));
    console.log(chalk.white(`Output directory: ${this.config.outputDir}`));
    console.log(chalk.white(`Save reasoning: ${this.config.saveReasoning ? 'Yes' : 'No'}`));
    console.log(chalk.gray('═'.repeat(50)));
  }

  private createTempConfigFile(): string {
    const tempDir = fs.mkdtempSync('/tmp/synth-data-');
    const configPath = path.join(tempDir, 'auto_config.py');

    const labelDescriptions = this.config.labels!
      .map(l => `${l.name}: ${l.description}`)
      .join('\n');

    const categoriesTypes = this.config.categories!.reduce((acc, cat) => {
      acc[cat.name] = cat.types;
      return acc;
    }, {} as Record<string, string[]>);

    const configContent = `# Auto-generated configuration file
labels = ${JSON.stringify(this.config.labels!.map(l => l.name))}

label_descriptions = """${labelDescriptions}"""

categories_types = ${JSON.stringify(categoriesTypes, null, 2).replace(/"/g, "'")}

use_case = "${this.config.useCase}"

prompt_examples = """${this.config.examples}"""
`;

    fs.writeFileSync(configPath, configContent);
    return configPath;
  }

  private async runDataGeneration(): Promise<void> {
    console.log(chalk.bold.green('\n🚀 STARTING DATA GENERATION'));
    console.log(chalk.gray('─'.repeat(50)));

    const configPath = this.createTempConfigFile();

    // Get the absolute path to data-gen.py (in prototyping folder)
    // __dirname will be cli-wrapper/dist, so we go up to prototyping
    const dataGenPath = path.join(__dirname, '../../data-gen.py');

    const args = [
      'run', 'python', dataGenPath,
      '--config', configPath,
      '--sample_size', this.config.sampleSize!.toString(),
      '--model', this.config.model!,
      '--max_new_tokens', this.config.maxTokens!.toString(),
      '--batch_size', this.config.batchSize!.toString(),
      '--output_dir', this.config.outputDir!
    ];

    if (this.config.saveReasoning) {
      args.push('--save_reasoning');
    }

    // Set working directory to prototyping folder
    const prototypingDir = path.join(__dirname, '../..');

    console.log(chalk.cyan('\n📝 Note: You may be prompted for your Hugging Face token if not already configured.\n'));

    // Use spawn with full stdio inheritance to allow interactive prompts
    const dataGen = spawn('uv', args, {
      cwd: prototypingDir,
      stdio: 'inherit',  // This passes stdin, stdout, stderr to the child process
      shell: false
    });

    return new Promise<void>((resolve, reject) => {
      dataGen.on('close', (code) => {
        // Clean up temp config
        try {
          fs.unlinkSync(configPath);
          fs.rmdirSync(path.dirname(configPath));
        } catch (error) {
          // Ignore cleanup errors
        }

        if (code === 0) {
          console.log(chalk.green('\n✅ Data generation completed successfully!'));
          console.log(chalk.cyan(`📁 Check your output directory: ${this.config.outputDir}`));
          resolve();
        } else {
          console.log(chalk.red(`\n❌ Data generation failed with code: ${code}`));
          reject(new Error(`Process exited with code ${code}`));
        }
      });

      dataGen.on('error', (error) => {
        console.log(chalk.red(`\n❌ Error running data generation: ${error.message}`));
        reject(error);
      });
    });
  }

  async run(): Promise<void> {
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
      } else {
        console.log(chalk.yellow('❌ Data generation cancelled.'));
      }
    } catch (error) {
      if ((error as any).name === 'ExitPromptError') {
        console.log(chalk.yellow('\n\n❌ Process interrupted by user.'));
        process.exit(1);
      }
      throw error;
    }
  }
}

// CLI Setup
const program = new Command();

program
  .name('synth-data')
  .description('Interactive CLI for synthetic data generation')
  .version('1.0.0')
  .action(async () => {
    const cli = new SyntheticDataCLI();
    await cli.run();
  });

program.parse();
