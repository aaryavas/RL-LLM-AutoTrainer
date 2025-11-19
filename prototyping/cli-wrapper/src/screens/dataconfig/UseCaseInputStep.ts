/**
 * UseCaseInputStep - Step for entering the use case description.
 * Single Responsibility: Handle use case text input for data generation.
 */

import {
  BoxRenderable,
  TextRenderable,
  type RenderContext,
} from '@opentui/core';
import { LabeledTextInput } from '../../components/input';

/**
 * Callback for use case submission.
 */
export type UseCaseSubmitHandler = (useCase: string) => void;

export class UseCaseInputStep {
  private container: BoxRenderable;
  private instructionText: TextRenderable;
  private useCaseInput: LabeledTextInput;
  private submitHandlers: UseCaseSubmitHandler[] = [];

  constructor(renderer: RenderContext) {
    // Create container
    this.container = new BoxRenderable(renderer, {
      id: 'use-case-step',
      flexDirection: 'column',
      width: '100%',
      gap: 1,
      padding: 1,
    });

    // Create instruction text
    this.instructionText = new TextRenderable(renderer, {
      id: 'use-case-instruction',
      content: [
        'What is the main purpose of your synthetic data?',
        '',
        'Examples:',
        '  - Customer service chatbot training',
        '  - Content moderation',
        '  - Sentiment analysis',
        '  - Text classification',
      ].join('\n'),
    });

    // Create input
    this.useCaseInput = new LabeledTextInput(renderer, {
      id: 'use-case-input',
      label: 'Use Case',
      placeholder: 'Enter your use case description...',
      initialValue: 'text classification',
      required: true,
    });

    // Wire up submission
    this.useCaseInput.onSubmit((value) => {
      if (value.trim()) {
        this.notifySubmitHandlers(value);
      }
    });

    // Assemble step
    this.container.add(this.instructionText);
    this.container.add(this.useCaseInput.getRenderable());
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Focus the input.
   */
  focus(): void {
    this.useCaseInput.focus();
  }

  /**
   * Get current value.
   */
  getValue(): string {
    return this.useCaseInput.getValue();
  }

  /**
   * Validate the step.
   */
  validate(): boolean {
    return this.getValue().trim().length > 0;
  }

  /**
   * Register handler for submission.
   */
  onSubmit(handler: UseCaseSubmitHandler): () => void {
    this.submitHandlers.push(handler);
    return () => {
      this.submitHandlers = this.submitHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Notify submit handlers.
   */
  private notifySubmitHandlers(value: string): void {
    for (const handler of this.submitHandlers) {
      handler(value);
    }
  }
}
