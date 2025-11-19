/**
 * Fine-tuning configuration steps.
 * Handles interactive prompts for configuring VB-LoRA fine-tuning.
 */

import * as inquirer from '@inquirer/prompts';
import chalk from 'chalk';
import {
  FineTuneConfig,
  TrainingConfig,
  VBLoRAConfig,
  HardwareConfig,
} from '../config/types';
import {
  DEFAULT_TRAINING,
  DEFAULT_VBLORA,
  DEFAULT_HARDWARE,
  getModelFamilyChoices,
  getVariantChoices,
  getPresetChoices,
  getVariantDefaults,
  PRESET_REGISTRY,
} from '../config/defaults';
import { printSectionHeader, printSuccess, printInfo } from '../utils/display';

/**
 * Configure model selection
 */
export async function configureModel(): Promise<{
  modelFamily: string;
  modelVariant: string;
  preset?: string;
}> {
  printSectionHeader('🤖', 'Model Selection', 1);

  const modelFamily = await inquirer.select({
    message: 'Select model family:',
    choices: getModelFamilyChoices(),
  });

  const variantChoices = getVariantChoices(modelFamily);
  const modelVariant = await inquirer.select({
    message: 'Select model variant:',
    choices: variantChoices,
  });

  const presetChoices = getPresetChoices();
  const preset = await inquirer.select({
    message: 'Use a preset configuration?',
    choices: presetChoices,
  });

  printSuccess(`Model: ${modelFamily}-${modelVariant}`);
  if (preset) {
    printSuccess(`Preset: ${preset}`);
  }

  return {
    modelFamily,
    modelVariant,
    preset: preset || undefined,
  };
}

/**
 * Configure training parameters
 */
export async function configureTraining(
  defaults: Partial<TrainingConfig> = {}
): Promise<TrainingConfig> {
  printSectionHeader('⚙️', 'Training Parameters', 2);

  const mergedDefaults = { ...DEFAULT_TRAINING, ...defaults };

  printInfo(`Showing defaults (press Enter to accept)`);

  const epochs = await inquirer.number({
    message: 'Number of epochs:',
    default: mergedDefaults.epochs,
    validate: (input) => (input && input > 0) || 'Must be positive',
  });

  const learningRate = await inquirer.input({
    message: 'Learning rate:',
    default: mergedDefaults.learningRate.toString(),
    validate: (input) => {
      const val = parseFloat(input);
      return (!isNaN(val) && val > 0) || 'Must be a positive number';
    },
  });

  const batchSize = await inquirer.number({
    message: 'Batch size per device:',
    default: mergedDefaults.batchSize,
    validate: (input) => (input && input > 0) || 'Must be positive',
  });

  const earlyStoppingPatience = await inquirer.number({
    message: 'Early stopping patience (epochs):',
    default: mergedDefaults.earlyStoppingPatience,
    validate: (input) => (input && input > 0) || 'Must be positive',
  });

  printSuccess('Training parameters configured');

  return {
    epochs: epochs!,
    learningRate: parseFloat(learningRate),
    batchSize: batchSize!,
    earlyStoppingPatience: earlyStoppingPatience!,
  };
}

/**
 * Configure VB-LoRA parameters
 */
export async function configureVBLoRA(
  defaults: Partial<VBLoRAConfig> = {}
): Promise<VBLoRAConfig> {
  printSectionHeader('🔧', 'VB-LoRA Parameters', 3);

  const mergedDefaults = { ...DEFAULT_VBLORA, ...defaults };

  printInfo(`Showing defaults (press Enter to accept)`);

  const numVectors = await inquirer.number({
    message: 'Number of vectors in vector bank:',
    default: mergedDefaults.numVectors,
    validate: (input) => (input && input > 0) || 'Must be positive',
  });

  const vectorLength = await inquirer.number({
    message: 'Vector length (must divide model dimensions evenly):',
    default: mergedDefaults.vectorLength,
    validate: (input) => (input && input > 0) || 'Must be positive',
  });

  const loraR = await inquirer.number({
    message: 'LoRA rank:',
    default: mergedDefaults.loraR,
    validate: (input) => (input && input > 0) || 'Must be positive',
  });

  const lrVectorBank = await inquirer.input({
    message: 'Learning rate for vector bank:',
    default: mergedDefaults.lrVectorBank.toString(),
    validate: (input) => {
      const val = parseFloat(input);
      return (!isNaN(val) && val > 0) || 'Must be a positive number';
    },
  });

  const lrLogits = await inquirer.input({
    message: 'Learning rate for logits:',
    default: mergedDefaults.lrLogits.toString(),
    validate: (input) => {
      const val = parseFloat(input);
      return (!isNaN(val) && val > 0) || 'Must be a positive number';
    },
  });

  printSuccess('VB-LoRA parameters configured');

  return {
    numVectors: numVectors!,
    vectorLength: vectorLength!,
    loraR: loraR!,
    lrVectorBank: parseFloat(lrVectorBank),
    lrLogits: parseFloat(lrLogits),
  };
}

/**
 * Configure hardware settings
 */
export async function configureHardware(): Promise<HardwareConfig> {
  printSectionHeader('💻', 'Hardware Settings', 4);

  const bits = await inquirer.select({
    message: 'Quantization bits:',
    choices: [
      { name: '4-bit (most memory efficient)', value: 4 },
      { name: '8-bit', value: 8 },
      { name: '16-bit', value: 16 },
      { name: '32-bit (full precision)', value: 32 },
    ],
    default: DEFAULT_HARDWARE.bits,
  });

  const bf16 = await inquirer.confirm({
    message: 'Use bfloat16 precision?',
    default: DEFAULT_HARDWARE.bf16,
  });

  let fp16 = DEFAULT_HARDWARE.fp16;
  if (!bf16) {
    fp16 = await inquirer.confirm({
      message: 'Use float16 precision?',
      default: DEFAULT_HARDWARE.fp16,
    });
  }

  printSuccess('Hardware settings configured');

  return {
    bits: bits as 4 | 8 | 16 | 32,
    bf16,
    fp16,
  };
}

/**
 * Configure output settings
 */
export async function configureOutput(defaultDataPath: string): Promise<{
  dataPath: string;
  outputDir: string;
  runName?: string;
  verbose: boolean;
  dryRun: boolean;
  showEpochMetrics: boolean;
}> {
  printSectionHeader('📁', 'Output Settings', 5);

  const dataPath = await inquirer.input({
    message: 'Path to training data (CSV):',
    default: defaultDataPath,
    validate: (input) => input.length > 0 || 'Data path is required',
  });

  const outputDir = await inquirer.input({
    message: 'Output directory for models:',
    default: './output/vblora_models',
  });

  const runName = await inquirer.input({
    message: 'Run name (leave empty for auto-generated):',
    default: '',
  });

  const verbose = await inquirer.confirm({
    message: 'Enable verbose logging?',
    default: false,
  });

  const dryRun = await inquirer.confirm({
    message: 'Dry run (show config without training)?',
    default: false,
  });

  const showEpochMetrics = await inquirer.confirm({
    message: 'Show detailed metrics at each epoch?',
    default: true,
  });

  printSuccess('Output settings configured');

  return {
    dataPath,
    outputDir,
    runName: runName || undefined,
    verbose,
    dryRun,
    showEpochMetrics,
  };
}

/**
 * Run all fine-tuning configuration steps
 */
export async function configureFineTuning(defaultDataPath: string): Promise<FineTuneConfig> {
  // Step 1: Model selection
  const { modelFamily, modelVariant, preset } = await configureModel();

  // Get variant-specific defaults
  const variantDefaults = getVariantDefaults(modelFamily, modelVariant);

  // Apply preset if selected
  let trainingDefaults: Partial<TrainingConfig> = variantDefaults;
  let vbloraDefaults: Partial<VBLoRAConfig> = variantDefaults;

  if (preset && PRESET_REGISTRY[preset]) {
    const presetConfig = PRESET_REGISTRY[preset];
    trainingDefaults = {
      ...trainingDefaults,
      epochs: presetConfig.epochs,
      batchSize: presetConfig.batchSize,
      learningRate: presetConfig.learningRate,
    };
    vbloraDefaults = {
      ...vbloraDefaults,
      numVectors: presetConfig.numVectors,
      loraR: presetConfig.loraR,
    };
  }

  // Step 2-5: Configure all parameters
  const training = await configureTraining(trainingDefaults);
  const vblora = await configureVBLoRA(vbloraDefaults);
  const hardware = await configureHardware();
  const outputSettings = await configureOutput(defaultDataPath);

  return {
    modelFamily,
    modelVariant,
    preset,
    training,
    vblora,
    hardware,
    dataPath: outputSettings.dataPath,
    testSize: 0.2,
    valSize: 0.1,
    textColumn: 'text',
    labelColumn: 'label',
    outputDir: outputSettings.outputDir,
    runName: outputSettings.runName,
    verbose: outputSettings.verbose,
    dryRun: outputSettings.dryRun,
    showEpochMetrics: outputSettings.showEpochMetrics,
  };
}
