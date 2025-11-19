/**
 * Main CLI orchestrator.
 * Coordinates data generation and fine-tuning workflows.
 */

import * as path from 'path';
import * as inquirer from '@inquirer/prompts';
import chalk from 'chalk';
import { DataGenerationConfig, FineTuneConfig } from './config/types';
import {
  printBanner,
  printWarning,
  displayDataGenSummary,
  displayFineTuneSummary,
} from './utils/display';
import {
  checkDataGenDependencies,
  checkFineTuneDependencies,
} from './utils/dependencies';
import { configureDataGeneration } from './steps/data-generation';
import { configureFineTuning } from './steps/fine-tuning';
import { runDataGeneration } from './runners/data-runner';
import { runFineTuning } from './runners/finetune-runner';

/**
 * Main CLI application class
 */
export class SyntheticDataCLI {
  private dataGenConfig: DataGenerationConfig | null = null;
  private fineTuneConfig: FineTuneConfig | null = null;
  private generatedDataPath: string | null = null;

  // Paths
  private readonly dataGenScript: string;
  private readonly prototypingDir: string;
  private readonly vbloraCliPath: string;
  private readonly vbloraDir: string;

  constructor() {
    // __dirname will be cli-wrapper/dist after compilation
    this.prototypingDir = path.join(__dirname, '../..');
    this.dataGenScript = path.join(this.prototypingDir, 'data-gen.py');
    this.vbloraDir = path.join(this.prototypingDir, 'vblora');
    this.vbloraCliPath = path.join(this.vbloraDir, 'cli.py');
  }

  /**
   * Run the complete CLI workflow
   */
  async run(): Promise<void> {
    printBanner();

    // Check dependencies
    const depsOk = await checkDataGenDependencies(this.dataGenScript);
    if (!depsOk) {
      process.exit(1);
    }

    try {
      // Phase 1: Data Generation
      await this.runDataGenerationPhase();

      // Phase 2: Fine-tuning (optional)
      await this.runFineTuningPhase();
    } catch (error) {
      if ((error as Error).name === 'ExitPromptError') {
        console.log(chalk.yellow('\n\n❌ Process interrupted by user.'));
        process.exit(1);
      }
      throw error;
    }
  }

  /**
   * Run the data generation phase
   */
  private async runDataGenerationPhase(): Promise<void> {
    // Configure data generation
    this.dataGenConfig = await configureDataGeneration();

    // Display summary
    displayDataGenSummary(this.dataGenConfig);

    // Confirm and run
    const proceed = await inquirer.confirm({
      message: 'Proceed with data generation?',
      default: true,
    });

    if (!proceed) {
      console.log(chalk.yellow('❌ Data generation cancelled.'));
      process.exit(0);
    }

    // Run data generation and capture output path
    this.generatedDataPath = await runDataGeneration(
      this.dataGenConfig,
      this.dataGenScript,
      this.prototypingDir
    );
  }

  /**
   * Run the fine-tuning phase (optional)
   */
  private async runFineTuningPhase(): Promise<void> {
    // Ask if user wants to fine-tune
    const wantFineTune = await inquirer.confirm({
      message: '\n🤖 Would you like to fine-tune a model with this data?',
      default: true,
    });

    if (!wantFineTune) {
      console.log(chalk.cyan('\n✨ All done! Your synthetic data is ready.'));
      return;
    }

    // Check fine-tuning dependencies
    const ftDepsOk = await checkFineTuneDependencies(this.vbloraCliPath);
    if (!ftDepsOk) {
      printWarning('Fine-tuning dependencies not satisfied. Skipping fine-tuning.');
      return;
    }

    // Use the actual generated data path
    const defaultDataPath = this.generatedDataPath || './generated_data/synthetic_data.csv';

    // Configure fine-tuning
    this.fineTuneConfig = await configureFineTuning(defaultDataPath);

    // Display summary
    displayFineTuneSummary(this.fineTuneConfig);

    // Confirm and run
    const proceedFt = await inquirer.confirm({
      message: 'Proceed with fine-tuning?',
      default: true,
    });

    if (!proceedFt) {
      console.log(chalk.yellow('❌ Fine-tuning cancelled.'));
      return;
    }

    // Run fine-tuning
    await runFineTuning(
      this.fineTuneConfig,
      this.vbloraCliPath,
      this.vbloraDir
    );

    console.log(chalk.cyan('\n✨ All done! Your model is fine-tuned and ready.'));
  }
}
