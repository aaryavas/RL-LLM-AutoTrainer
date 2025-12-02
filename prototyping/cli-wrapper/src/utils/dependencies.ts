/**
 * Dependency checking utilities.
 */

import { spawn } from 'child_process';
import * as fs from 'fs';
import ora from 'ora';
import chalk from 'chalk';

/**
 * Check if a file exists
 */
export function fileExists(filePath: string): boolean {
  return fs.existsSync(filePath);
}

/**
 * Check if Python is available
 */
export async function checkPython(): Promise<boolean> {
  return new Promise((resolve) => {
    const pythonCheck = spawn('python3', ['--version']);

    pythonCheck.on('close', (code) => {
      resolve(code === 0);
    });

    pythonCheck.on('error', () => {
      resolve(false);
    });
  });
}

/**
 * Check all required dependencies for data generation
 */
export async function checkDataGenDependencies(pythonScriptPath: string): Promise<boolean> {
  const spinner = ora('Checking dependencies...').start();

  try {
    // Check if Python script exists
    if (!fileExists(pythonScriptPath)) {
      spinner.fail(chalk.red('Python script not found!'));
      console.log(chalk.yellow(`Expected location: ${pythonScriptPath}`));
      return false;
    }

    // Check if Python is available
    const pythonAvailable = await checkPython();
    if (!pythonAvailable) {
      spinner.fail(chalk.red('Python3 not found'));
      console.log(chalk.yellow('Please ensure Python 3 is installed'));
      return false;
    }

    spinner.succeed(chalk.green('All dependencies satisfied'));
    return true;
  } catch (error) {
    spinner.fail(chalk.red('Dependency check failed'));
    return false;
  }
}

/**
 * Check all required dependencies for fine-tuning
 */
export async function checkFineTuneDependencies(vbloraCliPath: string): Promise<boolean> {
  const spinner = ora('Checking fine-tuning dependencies...').start();

  try {
    // Check if VB-LoRA CLI exists
    if (!fileExists(vbloraCliPath)) {
      spinner.fail(chalk.red('VB-LoRA CLI not found!'));
      console.log(chalk.yellow(`Expected location: ${vbloraCliPath}`));
      return false;
    }

    // Check if Python is available
    const pythonAvailable = await checkPython();
    if (!pythonAvailable) {
      spinner.fail(chalk.red('Python3 not found'));
      console.log(chalk.yellow('Please ensure Python 3 is installed'));
      return false;
    }

    spinner.succeed(chalk.green('Fine-tuning dependencies satisfied'));
    return true;
  } catch (error) {
    spinner.fail(chalk.red('Fine-tuning dependency check failed'));
    return false;
  }
}
