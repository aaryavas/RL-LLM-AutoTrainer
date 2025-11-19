/**
 * DataGenerationRunner - Executes data-gen.py with buffered output.
 * Single Responsibility: Configure and run data generation process.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DataGenerationConfig } from '../config/types';
import { BufferedProcessSpawner, ProcessEventEmitter } from './BufferedProcessSpawner';
import { ProcessOutputParser } from './ProcessOutputParser';

/**
 * Result of data generation execution.
 */
export interface DataGenerationResult {
  outputFilePath: string;
  configFilePath: string;
}

export class DataGenerationRunner {
  private processSpawner: BufferedProcessSpawner;
  private outputParser: ProcessOutputParser;
  private dataGenScriptPath: string;

  constructor(
    prototypingDirectory: string,
    dataGenScriptPath: string
  ) {
    this.processSpawner = new BufferedProcessSpawner(prototypingDirectory);
    this.outputParser = new ProcessOutputParser();
    this.dataGenScriptPath = dataGenScriptPath;
  }

  /**
   * Execute data generation with the given configuration.
   */
  execute(config: DataGenerationConfig): ProcessEventEmitter {
    const configFilePath = this.createTemporaryConfigFile(config);
    const commandArguments = this.buildCommandArguments(config, configFilePath);

    const processEmitter = this.processSpawner.spawnPythonWithUv(
      this.dataGenScriptPath,
      commandArguments
    );

    // Clean up temp config on process exit
    processEmitter.on('exit', () => {
      this.cleanupTemporaryConfigFile(configFilePath);
    });

    return processEmitter;
  }

  /**
   * Get the output parser for interpreting process output.
   */
  getOutputParser(): ProcessOutputParser {
    return this.outputParser;
  }

  /**
   * Calculate the expected output file path.
   */
  calculateOutputFilePath(outputDirectory: string): string {
    const timestamp = this.generateTimestamp();
    return path.join(outputDirectory, `${timestamp}.csv`);
  }

  /**
   * Create temporary configuration file for Python script.
   */
  private createTemporaryConfigFile(config: DataGenerationConfig): string {
    const temporaryDirectory = fs.mkdtempSync(
      path.join(os.tmpdir(), 'synth-data-')
    );
    const configFilePath = path.join(temporaryDirectory, 'auto_config.py');

    const labelDescriptions = config.labels
      .map((label) => `${label.name}: ${label.description}`)
      .join('\n');

    const categoriesTypes = config.categories.reduce(
      (accumulator, category) => {
        accumulator[category.name] = category.types;
        return accumulator;
      },
      {} as Record<string, string[]>
    );

    const configFileContent = `# Auto-generated configuration file
labels = ${JSON.stringify(config.labels.map((label) => label.name))}

label_descriptions = """${labelDescriptions}"""

categories_types = ${JSON.stringify(categoriesTypes, null, 2).replace(/"/g, "'")}

use_case = "${config.useCase}"

prompt_examples = """${config.examples}"""
`;

    fs.writeFileSync(configFilePath, configFileContent);
    return configFilePath;
  }

  /**
   * Clean up temporary configuration file.
   */
  private cleanupTemporaryConfigFile(configFilePath: string): void {
    try {
      fs.unlinkSync(configFilePath);
      fs.rmdirSync(path.dirname(configFilePath));
    } catch {
      // Ignore cleanup errors
    }
  }

  /**
   * Build command line arguments for data-gen.py.
   */
  private buildCommandArguments(
    config: DataGenerationConfig,
    configFilePath: string
  ): string[] {
    const commandArguments = [
      '--config', configFilePath,
      '--sample_size', config.sampleSize.toString(),
      '--model', config.model,
      '--max_new_tokens', config.maxTokens.toString(),
      '--batch_size', config.batchSize.toString(),
      '--output_dir', config.outputDir,
    ];

    if (config.saveReasoning) {
      commandArguments.push('--save_reasoning');
    }

    return commandArguments;
  }

  /**
   * Generate timestamp string matching Python script format.
   */
  private generateTimestamp(): string {
    const now = new Date();
    return (
      now.getFullYear().toString() +
      (now.getMonth() + 1).toString().padStart(2, '0') +
      now.getDate().toString().padStart(2, '0') +
      '_' +
      now.getHours().toString().padStart(2, '0') +
      now.getMinutes().toString().padStart(2, '0') +
      now.getSeconds().toString().padStart(2, '0')
    );
  }
}
