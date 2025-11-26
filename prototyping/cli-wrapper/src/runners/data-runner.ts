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
 * Find the most recently created CSV file in a directory
 */
function findLatestCsvFile(dir: string): string | null {
  try {
    const files = fs.readdirSync(dir)
      .filter(f => f.endsWith('.csv'))
      .map(f => ({
        name: f,
        path: path.join(dir, f),
        mtime: fs.statSync(path.join(dir, f)).mtime.getTime()
      }))
      .sort((a, b) => b.mtime - a.mtime);
    
    return files.length > 0 ? files[0].path : null;
  } catch {
    return null;
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

  // Record files before generation to detect new files
  const outputDir = path.isAbsolute(config.outputDir) 
    ? config.outputDir 
    : path.join(prototypingDir, config.outputDir);
  
  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  const filesBefore = new Set(
    fs.existsSync(outputDir) 
      ? fs.readdirSync(outputDir).filter(f => f.endsWith('.csv'))
      : []
  );

  const args = [
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
    outputDir,
  ];

  if (config.saveReasoning) {
    args.push('--save_reasoning');
  }

  printInfo('You may be prompted for your Hugging Face token if not already configured.\n');

  return new Promise<string>((resolve, reject) => {
    const dataGen = spawn('uv', ['run', 'python', ...args], {
      cwd: prototypingDir,
      stdio: 'inherit',
      shell: false,
    });

    dataGen.on('close', (code) => {
      cleanupTempConfig(configPath);

      if (code === 0) {
        // Find the newly created file by comparing before/after
        const filesAfter = fs.readdirSync(outputDir).filter(f => f.endsWith('.csv'));
        const newFiles = filesAfter.filter(f => !filesBefore.has(f));
        
        let outputFile: string;
        if (newFiles.length > 0) {
          // Use the newest of the new files
          outputFile = path.join(outputDir, newFiles.sort().reverse()[0]);
        } else {
          // Fallback: find the most recently modified file
          const latestFile = findLatestCsvFile(outputDir);
          outputFile = latestFile || path.join(outputDir, 'output.csv');
        }

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
