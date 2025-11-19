/**
 * Fine-tuning runner.
 * Spawns the VB-LoRA CLI with configured parameters.
 */

import { spawn } from 'child_process';
import * as path from 'path';
import chalk from 'chalk';
import { FineTuneConfig } from '../config/types';
import { getHuggingFaceModelName } from '../config/defaults';
import { printSuccess, printError } from '../utils/display';

/**
 * Build CLI arguments from config
 */
function buildCliArgs(config: FineTuneConfig): string[] {
  const args = [
    'finetune',
    config.dataPath,
    '--model',
    getHuggingFaceModelName(config.modelFamily, config.modelVariant),
    // Training parameters
    '--epochs',
    config.training.epochs.toString(),
    '--lr',
    config.training.learningRate.toString(),
    '--batch-size',
    config.training.batchSize.toString(),
    '--early-stopping',
    config.training.earlyStoppingPatience.toString(),
    // VB-LoRA parameters
    '--num-vectors',
    config.vblora.numVectors.toString(),
    '--vector-length',
    config.vblora.vectorLength.toString(),
    '--lora-r',
    config.vblora.loraR.toString(),
    '--lr-vector-bank',
    config.vblora.lrVectorBank.toString(),
    '--lr-logits',
    config.vblora.lrLogits.toString(),
    // Hardware
    '--bits',
    config.hardware.bits.toString(),
    // Output
    '--output-dir',
    config.outputDir,
    // Data splitting
    '--test-size',
    config.testSize.toString(),
    '--val-size',
    config.valSize.toString(),
    '--text-column',
    config.textColumn,
    '--label-column',
    config.labelColumn,
  ];

  // Optional flags
  if (config.hardware.bf16) {
    args.push('--bf16');
  }
  if (config.hardware.fp16) {
    args.push('--fp16');
  }
  if (config.runName) {
    args.push('--run-name', config.runName);
  }
  if (config.verbose) {
    args.push('--verbose');
  }
  if (config.dryRun) {
    args.push('--dry-run');
  }
  if (!config.showEpochMetrics) {
    args.push('--no-epoch-metrics');
  }

  return args;
}

/**
 * Run fine-tuning
 */
export async function runFineTuning(
  config: FineTuneConfig,
  vbloraCliPath: string,
  vbloraDir: string
): Promise<void> {
  console.log(chalk.bold.green('\n🚀 STARTING FINE-TUNING'));
  console.log(chalk.gray('─'.repeat(50)));

  const args = buildCliArgs(config);

  console.log(chalk.cyan('\n📋 Running VB-LoRA fine-tuning with:'));
  console.log(chalk.gray(`  python ${path.basename(vbloraCliPath)} ${args.join(' ')}\n`));

  return new Promise<void>((resolve, reject) => {
    const fineTune = spawn('python', [vbloraCliPath, ...args], {
      cwd: vbloraDir,
      stdio: 'inherit',
      shell: false,
    });

    fineTune.on('close', (code) => {
      if (code === 0) {
        printSuccess('Fine-tuning completed successfully!');
        console.log(chalk.cyan(`📁 Model saved to: ${config.outputDir}`));
        resolve();
      } else {
        printError(`Fine-tuning failed with code: ${code}`);
        reject(new Error(`Process exited with code ${code}`));
      }
    });

    fineTune.on('error', (error) => {
      printError(`Error running fine-tuning: ${error.message}`);
      reject(error);
    });
  });
}
