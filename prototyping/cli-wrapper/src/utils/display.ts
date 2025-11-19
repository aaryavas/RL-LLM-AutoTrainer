/**
 * Display utilities for CLI output formatting.
 */

import chalk from 'chalk';
import boxen from 'boxen';
import { DataGenerationConfig, FineTuneConfig } from '../config/types';

/**
 * Print the application banner
 */
export function printBanner(): void {
  const banner = boxen(
    chalk.bold.cyan('SYNTHETIC DATA GENERATOR') +
      '\n\n' +
      chalk.white('TypeScript CLI Wrapper for AI-Powered Data Generation'),
    {
      padding: 1,
      margin: 1,
      borderStyle: 'round',
      borderColor: 'cyan',
    }
  );
  console.log(banner);
}

/**
 * Print a section header
 */
export function printSectionHeader(icon: string, title: string, step?: number): void {
  const stepText = step ? `STEP ${step}: ` : '';
  console.log(chalk.bold.blue(`\n${icon} ${stepText}${title}`));
  console.log(chalk.gray('─'.repeat(50)));
}

/**
 * Print a success message
 */
export function printSuccess(message: string): void {
  console.log(chalk.green(`✅ ${message}`));
}

/**
 * Print an error message
 */
export function printError(message: string): void {
  console.log(chalk.red(`❌ ${message}`));
}

/**
 * Print a warning message
 */
export function printWarning(message: string): void {
  console.log(chalk.yellow(`⚠️  ${message}`));
}

/**
 * Print an info message
 */
export function printInfo(message: string): void {
  console.log(chalk.cyan(`ℹ️  ${message}`));
}

/**
 * Display data generation configuration summary
 */
export function displayDataGenSummary(config: Partial<DataGenerationConfig>): void {
  console.log(chalk.bold.cyan('\n📊 DATA GENERATION SUMMARY'));
  console.log(chalk.gray('═'.repeat(50)));
  console.log(chalk.white(`Use case: ${config.useCase}`));
  console.log(chalk.white(`Labels: ${config.labels?.map((l) => l.name).join(', ')}`));
  console.log(chalk.white(`Categories: ${config.categories?.map((c) => c.name).join(', ')}`));
  console.log(chalk.white(`Model: ${config.model}`));
  console.log(chalk.white(`Sample size: ${config.sampleSize}`));
  console.log(chalk.white(`Max tokens: ${config.maxTokens}`));
  console.log(chalk.white(`Batch size: ${config.batchSize}`));
  console.log(chalk.white(`Output directory: ${config.outputDir}`));
  console.log(chalk.white(`Save reasoning: ${config.saveReasoning ? 'Yes' : 'No'}`));
  console.log(chalk.gray('═'.repeat(50)));
}

/**
 * Display fine-tuning configuration summary
 */
export function displayFineTuneSummary(config: Partial<FineTuneConfig>): void {
  console.log(chalk.bold.cyan('\n🔧 FINE-TUNING SUMMARY'));
  console.log(chalk.gray('═'.repeat(50)));

  // Model
  console.log(chalk.white(`Model: ${config.modelFamily}-${config.modelVariant}`));
  if (config.preset) {
    console.log(chalk.white(`Preset: ${config.preset}`));
  }

  // Training
  console.log(chalk.gray('\nTraining:'));
  console.log(chalk.white(`  Epochs: ${config.training?.epochs}`));
  console.log(chalk.white(`  Learning rate: ${config.training?.learningRate}`));
  console.log(chalk.white(`  Batch size: ${config.training?.batchSize}`));
  console.log(chalk.white(`  Early stopping: ${config.training?.earlyStoppingPatience}`));

  // VB-LoRA
  console.log(chalk.gray('\nVB-LoRA:'));
  console.log(chalk.white(`  Num vectors: ${config.vblora?.numVectors}`));
  console.log(chalk.white(`  Vector length: ${config.vblora?.vectorLength}`));
  console.log(chalk.white(`  LoRA rank: ${config.vblora?.loraR}`));

  // Hardware
  console.log(chalk.gray('\nHardware:'));
  console.log(chalk.white(`  Bits: ${config.hardware?.bits}`));
  console.log(chalk.white(`  BF16: ${config.hardware?.bf16 ? 'Yes' : 'No'}`));
  console.log(chalk.white(`  FP16: ${config.hardware?.fp16 ? 'Yes' : 'No'}`));

  // Output
  console.log(chalk.gray('\nOutput:'));
  console.log(chalk.white(`  Directory: ${config.outputDir}`));
  if (config.runName) {
    console.log(chalk.white(`  Run name: ${config.runName}`));
  }

  console.log(chalk.gray('═'.repeat(50)));
}
