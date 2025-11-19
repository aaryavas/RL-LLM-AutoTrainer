/**
 * ModelSettingsStep - Step for configuring model and generation settings.
 * Single Responsibility: Handle model selection and generation parameters.
 */

import {
  BoxRenderable,
  TextRenderable,
  type RenderContext,
} from '@opentui/core';
import { ModelSelector, type ModelOption } from '../../components/selection';
import { NumericInput } from '../../components/input';
import { KeyboardNavigator } from '../../components/navigation';

/**
 * Model settings data.
 */
export interface ModelSettings {
  modelId: string;
  sampleSize: number;
  maxTokens: number;
  batchSize: number;
}

/**
 * Callback for settings submission.
 */
export type ModelSettingsSubmitHandler = (settings: ModelSettings) => void;

export class ModelSettingsStep {
  private container: BoxRenderable;
  private modelSelector: ModelSelector;
  private sampleSizeInput: NumericInput;
  private maxTokensInput: NumericInput;
  private batchSizeInput: NumericInput;
  private keyboardNavigator: KeyboardNavigator;
  private submitHandlers: ModelSettingsSubmitHandler[] = [];

  constructor(renderer: RenderContext) {
    // Create container
    this.container = new BoxRenderable(renderer, {
      id: 'model-settings-step',
      flexDirection: 'column',
      width: '100%',
      gap: 1,
      padding: 1,
    });

    // Define available models
    const models: ModelOption[] = [
      {
        name: 'Llama 3.2 3B',
        modelId: 'meta-llama/Llama-3.2-3B-Instruct',
        description: 'Meta Llama 3.2 - Good quality',
        parameters: '3B params',
      },
      {
        name: 'Gemma 3 1B',
        modelId: 'google/gemma-3-1b-it',
        description: 'Google Gemma 3 - Fast',
        parameters: '1B params',
      },
      {
        name: 'SmolLM2 1.7B',
        modelId: 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        description: 'HuggingFace SmolLM2',
        parameters: '1.7B params',
      },
    ];

    // Create model selector
    this.modelSelector = new ModelSelector(renderer, {
      id: 'model-selector',
      label: 'Select Model',
      models,
      height: 5,
    });

    // Create sample size input
    this.sampleSizeInput = new NumericInput(renderer, {
      id: 'sample-size',
      label: 'Number of Samples',
      initialValue: 100,
      minimum: 1,
      maximum: 10000,
    });

    // Create max tokens input
    this.maxTokensInput = new NumericInput(renderer, {
      id: 'max-tokens',
      label: 'Max Tokens per Sample',
      initialValue: 256,
      minimum: 32,
      maximum: 2048,
    });

    // Create batch size input
    this.batchSizeInput = new NumericInput(renderer, {
      id: 'batch-size',
      label: 'Batch Size',
      initialValue: 20,
      minimum: 1,
      maximum: 100,
    });

    // Setup keyboard navigation
    this.keyboardNavigator = new KeyboardNavigator();
    this.keyboardNavigator.setElements([
      this.modelSelector,
      this.sampleSizeInput,
      this.maxTokensInput,
      this.batchSizeInput,
    ]);

    // Assemble step
    this.container.add(this.modelSelector.getRenderable());
    this.container.add(this.sampleSizeInput.getRenderable());
    this.container.add(this.maxTokensInput.getRenderable());
    this.container.add(this.batchSizeInput.getRenderable());
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Focus the first input.
   */
  focus(): void {
    this.keyboardNavigator.focusFirst();
  }

  /**
   * Get current settings.
   */
  getSettings(): ModelSettings {
    return {
      modelId: this.modelSelector.getSelectedModelId(),
      sampleSize: this.sampleSizeInput.getValue(),
      maxTokens: this.maxTokensInput.getValue(),
      batchSize: this.batchSizeInput.getValue(),
    };
  }

  /**
   * Validate the step.
   */
  validate(): boolean {
    const settings = this.getSettings();
    return (
      settings.modelId !== '' &&
      settings.sampleSize > 0 &&
      settings.maxTokens > 0 &&
      settings.batchSize > 0
    );
  }

  /**
   * Get keyboard navigator for parent screen.
   */
  getKeyboardNavigator(): KeyboardNavigator {
    return this.keyboardNavigator;
  }

  /**
   * Register handler for submission.
   */
  onSubmit(handler: ModelSettingsSubmitHandler): () => void {
    this.submitHandlers.push(handler);
    return () => {
      this.submitHandlers = this.submitHandlers.filter((h) => h !== handler);
    };
  }
}
