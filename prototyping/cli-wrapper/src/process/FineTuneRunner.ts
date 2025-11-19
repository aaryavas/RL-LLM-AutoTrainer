/**
 * FineTuneRunner - Executes VB-LoRA fine-tuning with buffered output.
 * Single Responsibility: Configure and run fine-tuning process.
 */

import { FineTuneConfig } from '../config/types';
import { getHuggingFaceModelName } from '../config/defaults';
import { BufferedProcessSpawner, ProcessEventEmitter } from './BufferedProcessSpawner';
import { ProcessOutputParser } from './ProcessOutputParser';

export class FineTuneRunner {
  private processSpawner: BufferedProcessSpawner;
  private outputParser: ProcessOutputParser;
  private vbloraCliPath: string;

  constructor(
    vbloraDirectory: string,
    vbloraCliPath: string
  ) {
    this.processSpawner = new BufferedProcessSpawner(vbloraDirectory);
    this.outputParser = new ProcessOutputParser();
    this.vbloraCliPath = vbloraCliPath;
  }

  /**
   * Execute fine-tuning with the given configuration.
   */
  execute(config: FineTuneConfig): ProcessEventEmitter {
    const commandArguments = this.buildCommandArguments(config);

    return this.processSpawner.spawnPythonScript(
      this.vbloraCliPath,
      commandArguments
    );
  }

  /**
   * Get the output parser for interpreting process output.
   */
  getOutputParser(): ProcessOutputParser {
    return this.outputParser;
  }

  /**
   * Build command line arguments for VB-LoRA CLI.
   */
  private buildCommandArguments(config: FineTuneConfig): string[] {
    const modelName = getHuggingFaceModelName(
      config.modelFamily,
      config.modelVariant
    );

    const commandArguments = [
      'finetune',
      config.dataPath,
      '--model', modelName,
      // Training parameters
      '--epochs', config.training.epochs.toString(),
      '--lr', config.training.learningRate.toString(),
      '--batch-size', config.training.batchSize.toString(),
      '--early-stopping', config.training.earlyStoppingPatience.toString(),
      // VB-LoRA parameters
      '--num-vectors', config.vblora.numVectors.toString(),
      '--vector-length', config.vblora.vectorLength.toString(),
      '--lora-r', config.vblora.loraR.toString(),
      '--lr-vector-bank', config.vblora.lrVectorBank.toString(),
      '--lr-logits', config.vblora.lrLogits.toString(),
      // Hardware
      '--bits', config.hardware.bits.toString(),
      // Output
      '--output-dir', config.outputDir,
      // Data splitting
      '--test-size', config.testSize.toString(),
      '--val-size', config.valSize.toString(),
      '--text-column', config.textColumn,
      '--label-column', config.labelColumn,
    ];

    // Optional flags
    if (config.hardware.bf16) {
      commandArguments.push('--bf16');
    }

    if (config.hardware.fp16) {
      commandArguments.push('--fp16');
    }

    if (config.runName) {
      commandArguments.push('--run-name', config.runName);
    }

    if (config.verbose) {
      commandArguments.push('--verbose');
    }

    if (config.dryRun) {
      commandArguments.push('--dry-run');
    }

    if (!config.showEpochMetrics) {
      commandArguments.push('--no-epoch-metrics');
    }

    return commandArguments;
  }
}
