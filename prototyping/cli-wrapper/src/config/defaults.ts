/**
 * Default configurations and model registry.
 * Extensible design for adding new model families and variants.
 */

import {
  ModelRegistry,
  PresetRegistry,
  TrainingConfig,
  VBLoRAConfig,
  HardwareConfig,
} from './types';

// =============================================================================
// Model Registry - Extensible for future models
// =============================================================================

export const MODEL_REGISTRY: ModelRegistry = {
  SmolLM2: {
    displayName: 'SmolLM2',
    huggingFacePrefix: 'HuggingFaceTB/SmolLM2',
    variants: {
      '135M': {
        batchSize: 8,
        numVectors: 60,
        learningRate: 3e-4,
      },
      '360M': {
        batchSize: 4,
        numVectors: 90,
        learningRate: 2e-4,
      },
      '1.7B': {
        batchSize: 2,
        numVectors: 120,
        learningRate: 1e-4,
      },
    },
  },
  // Future models can be added here:
  // Llama: { ... },
  // Mistral: { ... },
};

// =============================================================================
// Training Presets
// =============================================================================

export const PRESET_REGISTRY: PresetRegistry = {
  minimal: {
    description: 'Quick training for testing',
    epochs: 1,
    batchSize: 8,
    learningRate: 3e-4,
    numVectors: 60,
    loraR: 2,
  },
  standard: {
    description: 'Balanced training for most use cases',
    epochs: 3,
    batchSize: 4,
    learningRate: 2e-4,
    numVectors: 90,
    loraR: 4,
  },
  aggressive: {
    description: 'High capacity for complex tasks',
    epochs: 5,
    batchSize: 2,
    learningRate: 1e-4,
    numVectors: 120,
    loraR: 8,
  },
};

// =============================================================================
// Default Configurations
// =============================================================================

export const DEFAULT_TRAINING: TrainingConfig = {
  epochs: 3,
  learningRate: 2e-4,
  batchSize: 4,
  earlyStoppingPatience: 3,
};

export const DEFAULT_VBLORA: VBLoRAConfig = {
  numVectors: 90,
  vectorLength: 64,
  loraR: 4,
  lrVectorBank: 1e-3,
  lrLogits: 1e-2,
};

export const DEFAULT_HARDWARE: HardwareConfig = {
  bits: 4,
  bf16: false,
  fp16: false,
};

// =============================================================================
// Data Generation Defaults
// =============================================================================

export const DATA_GEN_MODELS = [
  { name: 'Llama 3.2 3B Instruct', value: 'meta-llama/Llama-3.2-3B-Instruct' },
  { name: 'Gemma 3 1B', value: 'google/gemma-3-1b-it' },
  { name: 'SmolLM2 1.7B Instruct', value: 'HuggingFaceTB/SmolLM2-1.7B-Instruct' },
  { name: 'Custom model', value: 'custom' },
];

export const DEFAULT_DATA_GEN = {
  sampleSize: 100,
  maxTokens: 256,
  batchSize: 20,
  outputDir: './generated_data',
  saveReasoning: true,
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get model families as choices for inquirer
 */
export function getModelFamilyChoices(): Array<{ name: string; value: string }> {
  return Object.entries(MODEL_REGISTRY).map(([key, config]) => ({
    name: config.displayName,
    value: key,
  }));
}

/**
 * Get variants for a specific model family
 */
export function getVariantChoices(family: string): Array<{ name: string; value: string }> {
  const familyConfig = MODEL_REGISTRY[family];
  if (!familyConfig) return [];

  return Object.keys(familyConfig.variants).map((variant) => ({
    name: `${familyConfig.displayName}-${variant}`,
    value: variant,
  }));
}

/**
 * Get full HuggingFace model name
 */
export function getHuggingFaceModelName(family: string, variant: string): string {
  const familyConfig = MODEL_REGISTRY[family];
  if (!familyConfig) return '';

  return `${familyConfig.huggingFacePrefix}-${variant}-Instruct`;
}

/**
 * Get variant-specific defaults
 */
export function getVariantDefaults(family: string, variant: string): Partial<TrainingConfig & VBLoRAConfig> {
  const familyConfig = MODEL_REGISTRY[family];
  if (!familyConfig || !familyConfig.variants[variant]) {
    return {};
  }

  const variantDefaults = familyConfig.variants[variant];
  return {
    batchSize: variantDefaults.batchSize,
    numVectors: variantDefaults.numVectors,
    learningRate: variantDefaults.learningRate,
  };
}

/**
 * Get preset choices for inquirer
 */
export function getPresetChoices(): Array<{ name: string; value: string }> {
  return [
    { name: 'None (use defaults)', value: '' },
    ...Object.entries(PRESET_REGISTRY).map(([key, config]) => ({
      name: `${key} - ${config.description}`,
      value: key,
    })),
  ];
}
