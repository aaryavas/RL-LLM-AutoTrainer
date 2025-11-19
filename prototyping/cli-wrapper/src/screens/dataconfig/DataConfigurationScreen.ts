/**
 * DataConfigurationScreen - Main screen for data generation configuration.
 * Single Responsibility: Orchestrate multi-step data generation configuration.
 */

import {
  BoxRenderable,
  type RenderContext,
  type KeyEvent,
} from '@opentui/core';
import { DataGenerationConfig, Label, Category } from '../../config/types';
import { StepProgressHeader } from '../../components/display';
import { FormStepController, type FormStep } from '../../components/navigation';
import { UseCaseInputStep } from './UseCaseInputStep';
import { ModelSettingsStep, type ModelSettings } from './ModelSettingsStep';

/**
 * Callback for configuration completion.
 */
export type ConfigurationCompleteHandler = (config: DataGenerationConfig) => void;

export class DataConfigurationScreen {
  private renderer: RenderContext;
  private container: BoxRenderable;
  private stepProgressHeader: StepProgressHeader;
  private stepContainer: BoxRenderable;
  private currentStepComponent: BoxRenderable | null = null;
  private useCaseStep: UseCaseInputStep;
  private modelSettingsStep: ModelSettingsStep;
  private formController: FormStepController;
  private completeHandlers: ConfigurationCompleteHandler[] = [];

  // Configuration data accumulated across steps
  private useCase: string = '';
  private labels: Label[] = [];
  private categories: Category[] = [];
  private examples: string = '';
  private modelSettings: ModelSettings | null = null;

  constructor(renderer: RenderContext) {
    this.renderer = renderer;

    // Create container
    this.container = new BoxRenderable(renderer, {
      id: 'data-config-screen',
      flexDirection: 'column',
      width: '100%',
      height: '100%',
      gap: 1,
    });

    // Create step progress header
    this.stepProgressHeader = new StepProgressHeader(renderer, {
      id: 'data-config-progress',
      totalSteps: 5,
      stepNames: [
        'Use Case',
        'Labels',
        'Categories',
        'Examples',
        'Model Settings',
      ],
    });

    // Create step container
    this.stepContainer = new BoxRenderable(renderer, {
      id: 'data-config-step-container',
      width: '100%',
      flexGrow: 1,
    });

    // Create step components
    this.useCaseStep = new UseCaseInputStep(renderer);
    this.modelSettingsStep = new ModelSettingsStep(renderer);

    // Wire up step events
    this.useCaseStep.onSubmit((value) => {
      this.useCase = value;
      this.formController.advanceToNextStep();
    });

    // Create form steps
    const formSteps: FormStep[] = [
      {
        id: 'use-case',
        name: 'Use Case',
        component: this.useCaseStep.getRenderable(),
        validate: () => this.useCaseStep.validate(),
        onEnter: () => this.useCaseStep.focus(),
      },
      {
        id: 'labels',
        name: 'Labels',
        component: this.createPlaceholderStep('Labels - Enter label names and descriptions'),
        validate: () => true, // Simplified for now
      },
      {
        id: 'categories',
        name: 'Categories',
        component: this.createPlaceholderStep('Categories - Enter category names and types'),
        validate: () => true,
      },
      {
        id: 'examples',
        name: 'Examples',
        component: this.createPlaceholderStep('Examples - Provide few-shot examples'),
        validate: () => true,
      },
      {
        id: 'model-settings',
        name: 'Model Settings',
        component: this.modelSettingsStep.getRenderable(),
        validate: () => this.modelSettingsStep.validate(),
        onEnter: () => this.modelSettingsStep.focus(),
      },
    ];

    // Create form controller
    this.formController = new FormStepController(formSteps);

    // Handle step changes
    this.formController.onStepChange((prev, curr, step) => {
      this.stepProgressHeader.setCurrentStep(curr + 1);
      this.updateStepContainer(step);
    });

    // Handle form completion
    this.formController.onComplete(() => {
      this.modelSettings = this.modelSettingsStep.getSettings();
      this.notifyCompleteHandlers();
    });

    // Initialize with first step
    this.updateStepContainer(formSteps[0]);

    // Assemble screen
    this.container.add(this.stepProgressHeader.getRenderable());
    this.container.add(this.stepContainer);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Handle keyboard event.
   */
  handleKeyEvent(keyEvent: KeyEvent): boolean {
    // Handle Escape to go back
    if (keyEvent.name === 'escape') {
      return this.formController.returnToPreviousStep();
    }

    // Handle Enter to advance (for steps without specific submit handling)
    if (keyEvent.name === 'return' || keyEvent.name === 'enter') {
      return this.formController.advanceToNextStep();
    }

    return false;
  }

  /**
   * Get the built configuration.
   */
  getConfiguration(): DataGenerationConfig {
    return {
      useCase: this.useCase,
      labels: this.labels.length > 0 ? this.labels : [
        { name: 'positive', description: 'Positive sentiment' },
        { name: 'negative', description: 'Negative sentiment' },
      ],
      categories: this.categories.length > 0 ? this.categories : [
        { name: 'review', types: ['product', 'service'] },
      ],
      examples: this.examples || 'Example placeholder',
      model: this.modelSettings?.modelId || 'meta-llama/Llama-3.2-3B-Instruct',
      sampleSize: this.modelSettings?.sampleSize || 100,
      maxTokens: this.modelSettings?.maxTokens || 256,
      batchSize: this.modelSettings?.batchSize || 20,
      outputDir: './generated_data',
      saveReasoning: true,
    };
  }

  /**
   * Register handler for completion.
   */
  onComplete(handler: ConfigurationCompleteHandler): () => void {
    this.completeHandlers.push(handler);
    return () => {
      this.completeHandlers = this.completeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Update step container with current step.
   */
  private updateStepContainer(step: FormStep): void {
    // Clear current content
    if (this.currentStepComponent) {
      this.stepContainer.remove(this.currentStepComponent.id);
    }

    // Add new step content
    this.currentStepComponent = step.component as BoxRenderable;
    this.stepContainer.add(step.component);
  }

  /**
   * Create placeholder step for development.
   */
  private createPlaceholderStep(message: string): BoxRenderable {
    const placeholder = new BoxRenderable(this.renderer, {
      id: `placeholder-${Date.now()}`,
      padding: 2,
    });

    const { TextRenderable } = require('@opentui/core');
    const text = new TextRenderable(this.renderer, {
      id: `placeholder-text-${Date.now()}`,
      content: message + '\n\nPress Enter to continue',
    });

    placeholder.add(text);
    return placeholder;
  }

  /**
   * Notify complete handlers.
   */
  private notifyCompleteHandlers(): void {
    const config = this.getConfiguration();
    for (const handler of this.completeHandlers) {
      handler(config);
    }
  }
}
