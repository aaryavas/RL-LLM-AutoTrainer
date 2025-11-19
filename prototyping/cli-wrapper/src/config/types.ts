/**
 * Type definitions for the synthetic data CLI.
 * Contains interfaces for data generation, fine-tuning, and model configuration.
 */

// =============================================================================
// Data Generation Types
// =============================================================================

export interface Label {
  name: string;
  description: string;
}

export interface Category {
  name: string;
  types: string[];
}

export interface DataGenerationConfig {
  useCase: string;
  labels: Label[];
  categories: Category[];
  examples: string;
  model: string;
  sampleSize: number;
  maxTokens: number;
  batchSize: number;
  outputDir: string;
  saveReasoning: boolean;
}

// =============================================================================
// Fine-Tuning Types
// =============================================================================

export interface TrainingConfig {
  epochs: number;
  learningRate: number;
  batchSize: number;
  earlyStoppingPatience: number;
}

export interface VBLoRAConfig {
  numVectors: number;
  vectorLength: number;
  loraR: number;
  lrVectorBank: number;
  lrLogits: number;
}

export interface HardwareConfig {
  bits: 4 | 8 | 16 | 32;
  bf16: boolean;
  fp16: boolean;
}

export interface FineTuneConfig {
  // Data
  dataPath: string;
  testSize: number;
  valSize: number;
  textColumn: string;
  labelColumn: string;

  // Model selection
  modelFamily: string;
  modelVariant: string;
  preset?: string;

  // Training parameters
  training: TrainingConfig;

  // VB-LoRA parameters
  vblora: VBLoRAConfig;

  // Hardware
  hardware: HardwareConfig;

  // Output
  outputDir: string;
  runName?: string;

  // Misc
  verbose: boolean;
  dryRun: boolean;
  showEpochMetrics: boolean;
}

// =============================================================================
// Model Registry Types
// =============================================================================

export interface ModelVariantDefaults {
  batchSize: number;
  numVectors: number;
  learningRate: number;
}

export interface ModelFamilyConfig {
  displayName: string;
  huggingFacePrefix: string;
  variants: Record<string, ModelVariantDefaults>;
}

export type ModelRegistry = Record<string, ModelFamilyConfig>;

// =============================================================================
// Preset Types
// =============================================================================

export interface PresetConfig {
  description: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  numVectors: number;
  loraR: number;
}

export type PresetRegistry = Record<string, PresetConfig>;
