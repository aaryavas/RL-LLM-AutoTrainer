/**
 * Data generation configuration steps.
 * Handles interactive prompts for configuring synthetic data generation.
 */

import * as inquirer from '@inquirer/prompts';
import chalk from 'chalk';
import { DataGenerationConfig, Label, Category } from '../config/types';
import { DATA_GEN_MODELS, DEFAULT_DATA_GEN } from '../config/defaults';
import { printSectionHeader, printSuccess } from '../utils/display';

/**
 * Configure use case
 */
export async function configureUseCase(): Promise<string> {
  printSectionHeader('📋', 'Use Case Configuration', 1);

  const useCase = await inquirer.input({
    message: 'What is the main purpose of your synthetic data?',
    default: 'text classification',
    validate: (input) => input.length > 0 || 'Use case is required',
  });

  printSuccess(`Use case set: ${useCase}`);
  return useCase;
}

/**
 * Configure labels
 */
export async function configureLabels(): Promise<Label[]> {
  printSectionHeader('🏷️', 'Labels Configuration', 2);

  const labels: Label[] = [];
  let addMore = true;

  while (addMore) {
    const labelName = await inquirer.input({
      message: 'Enter a label name:',
      validate: (input) => input.length > 0 || 'Label name is required',
    });

    const labelDesc = await inquirer.input({
      message: `Describe what '${labelName}' means:`,
      validate: (input) => input.length > 0 || 'Description is required',
    });

    labels.push({ name: labelName, description: labelDesc });
    printSuccess(`Added label: ${labelName}`);

    addMore = await inquirer.confirm({
      message: 'Add another label?',
      default: true,
    });
  }

  return labels;
}

/**
 * Configure categories
 */
export async function configureCategories(): Promise<Category[]> {
  printSectionHeader('🗂️', 'Categories Configuration', 3);

  const categories: Category[] = [];
  let addMore = true;

  while (addMore) {
    const categoryName = await inquirer.input({
      message: 'Enter a category name:',
      validate: (input) => input.length > 0 || 'Category name is required',
    });

    const typesInput = await inquirer.input({
      message: `Enter types for '${categoryName}' (comma-separated):`,
      validate: (input) => input.length > 0 || 'At least one type is required',
    });

    const types = typesInput.split(',').map((t) => t.trim());
    categories.push({ name: categoryName, types });
    printSuccess(`Added category: ${categoryName} with ${types.length} types`);

    addMore = await inquirer.confirm({
      message: 'Add another category?',
      default: true,
    });
  }

  return categories;
}

/**
 * Configure examples
 */
export async function configureExamples(): Promise<string> {
  printSectionHeader('💡', 'Example Configuration', 4);

  const numExamples = await inquirer.number({
    message: 'How many examples would you like to provide?',
    default: 2,
    validate: (input) => (input && input > 0) || 'Must provide at least 1 example',
  });

  const examples: string[] = [];

  for (let i = 0; i < numExamples!; i++) {
    console.log(chalk.yellow(`\n--- Example ${i + 1} ---`));
    console.log(chalk.gray('Enter your example (press Ctrl+D or type END on a new line to finish):'));

    const example = await inquirer.editor({
      message: `Example ${i + 1}:`,
      default: `LABEL: \nCATEGORY: \nTYPE: \nOUTPUT: \nREASONING: `,
    });

    examples.push(example);
    printSuccess(`Example ${i + 1} added`);
  }

  return examples.join('\n\n');
}

/**
 * Configure model and generation settings
 */
export async function configureModelSettings(): Promise<{
  model: string;
  sampleSize: number;
  maxTokens: number;
  batchSize: number;
  outputDir: string;
  saveReasoning: boolean;
}> {
  printSectionHeader('🤖', 'Model & Generation Settings', 5);

  const modelChoice = await inquirer.select({
    message: 'Select a model:',
    choices: DATA_GEN_MODELS,
  });

  let model: string;
  if (modelChoice === 'custom') {
    model = await inquirer.input({
      message: 'Enter custom model name (HuggingFace format):',
      validate: (input) => input.length > 0 || 'Model name is required',
    });
  } else {
    model = modelChoice;
  }

  const sampleSize = await inquirer.number({
    message: 'Number of samples to generate:',
    default: DEFAULT_DATA_GEN.sampleSize,
    validate: (input) => (input && input > 0) || 'Must be a positive number',
  });

  const maxTokens = await inquirer.number({
    message: 'Maximum tokens per sample:',
    default: DEFAULT_DATA_GEN.maxTokens,
    validate: (input) => (input && input > 0) || 'Must be a positive number',
  });

  const batchSize = await inquirer.number({
    message: 'Batch size for processing:',
    default: DEFAULT_DATA_GEN.batchSize,
    validate: (input) => (input && input > 0) || 'Must be a positive number',
  });

  const outputDir = await inquirer.input({
    message: 'Output directory:',
    default: DEFAULT_DATA_GEN.outputDir,
  });

  const saveReasoning = await inquirer.confirm({
    message: 'Save reasoning for each generated sample?',
    default: DEFAULT_DATA_GEN.saveReasoning,
  });

  printSuccess('Model and generation settings configured');

  return {
    model,
    sampleSize: sampleSize!,
    maxTokens: maxTokens!,
    batchSize: batchSize!,
    outputDir,
    saveReasoning,
  };
}

/**
 * Run all data generation configuration steps
 */
export async function configureDataGeneration(): Promise<DataGenerationConfig> {
  const useCase = await configureUseCase();
  const labels = await configureLabels();
  const categories = await configureCategories();
  const examples = await configureExamples();
  const modelSettings = await configureModelSettings();

  return {
    useCase,
    labels,
    categories,
    examples,
    ...modelSettings,
  };
}
