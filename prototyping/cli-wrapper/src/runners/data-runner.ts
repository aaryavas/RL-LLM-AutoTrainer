/**
 * Data generation runner.
 * Spawns the Python data-gen.py script with configured parameters.
 */

import { spawn } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import chalk from 'chalk';
import { DataGenerationConfig } from '../config/types';
import { printInfo, printSuccess, printError } from '../utils/display';

/**
 * Create a temporary configuration file for the Python script
 */
function createTempConfigFile(config: DataGenerationConfig): string {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'synth-data-'));
  const configPath = path.join(tempDir, 'auto_config.py');

  const labelDescriptions = config.labels
    .map((l) => `${l.name}: ${l.description}`)
    .join('\n');

  const categoriesTypes = config.categories.reduce(
    (acc, cat) => {
      acc[cat.name] = cat.types;
      return acc;
    },
    {} as Record<string, string[]>
  );

  const configContent = `# Auto-generated configuration file
labels = ${JSON.stringify(config.labels.map((l) => l.name))}

label_descriptions = """${labelDescriptions}"""

categories_types = ${JSON.stringify(categoriesTypes, null, 2).replace(/"/g, "'")}

use_case = "${config.useCase}"

prompt_examples = """${config.examples}"""
`;

  fs.writeFileSync(configPath, configContent);
  return configPath;
}

/**
 * Clean up temporary config file
 */
function cleanupTempConfig(configPath: string): void {
  try {
    fs.unlinkSync(configPath);
    fs.rmdirSync(path.dirname(configPath));
  } catch (error) {
    // Ignore cleanup errors
  }
}

/**
 * Run data generation
 */
export async function runDataGeneration(
  config: DataGenerationConfig,
  dataGenPath: string,
  prototypingDir: string
): Promise<string> {
  console.log(chalk.bold.green('\n🚀 STARTING DATA GENERATION'));
  console.log(chalk.gray('─'.repeat(50)));

  const configPath = createTempConfigFile(config);

  // Generate timestamp-based filename to match Python script output
  const now = new Date();
  const timestamp = now.getFullYear().toString() +
    (now.getMonth() + 1).toString().padStart(2, '0') +
    now.getDate().toString().padStart(2, '0') + '_' +
    now.getHours().toString().padStart(2, '0') +
    now.getMinutes().toString().padStart(2, '0') +
    now.getSeconds().toString().padStart(2, '0');
  const outputFile = path.join(config.outputDir, `${timestamp}.csv`);

  const args = [
    'run',
    'python',
    dataGenPath,
    '--config',
    configPath,
    '--sample_size',
    config.sampleSize.toString(),
    '--model',
    config.model,
    '--max_new_tokens',
    config.maxTokens.toString(),
    '--batch_size',
    config.batchSize.toString(),
    '--output_dir',
    config.outputDir,
  ];

  if (config.saveReasoning) {
    args.push('--save_reasoning');
  }

  printInfo('You may be prompted for your Hugging Face token if not already configured.\n');

  return new Promise<string>((resolve, reject) => {
    const dataGen = spawn('python', ['-m', 'uv', ...args], {
      cwd: prototypingDir,
      stdio: 'inherit',
      shell: false,
    });

    dataGen.on('close', (code) => {
      cleanupTempConfig(configPath);

      if (code === 0) {
        printSuccess('Data generation completed successfully!');
        console.log(chalk.cyan(`📁 Output file: ${outputFile}`));
        resolve(outputFile);
      } else {
        printError(`Data generation failed with code: ${code}`);
        reject(new Error(`Process exited with code ${code}`));
      }
    });

    dataGen.on('error', (error) => {
      cleanupTempConfig(configPath);
      printError(`Error running data generation: ${error.message}`);
      reject(error);
    });
  });
}
