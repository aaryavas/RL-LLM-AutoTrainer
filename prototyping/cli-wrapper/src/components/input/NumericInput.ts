/**
 * NumericInput - Input field for numeric values with validation.
 * Single Responsibility: Handle numeric input with min/max constraints.
 */

import {
  BoxRenderable,
  TextRenderable,
  InputRenderable,
  InputRenderableEvents,
  type RenderContext,
} from '@opentui/core';

/**
 * Configuration for NumericInput.
 */
export interface NumericInputOptions {
  id: string;
  label: string;
  initialValue?: number;
  minimum?: number;
  maximum?: number;
  width?: number | string;
  required?: boolean;
}

/**
 * Callback for numeric value changes.
 */
export type NumericValueHandler = (value: number) => void;

export class NumericInput {
  private container: BoxRenderable;
  private labelText: TextRenderable;
  private inputField: InputRenderable;
  private errorText: TextRenderable;
  private changeHandlers: NumericValueHandler[] = [];
  private submitHandlers: NumericValueHandler[] = [];
  private minimum: number;
  private maximum: number;
  private currentValue: number;

  constructor(renderer: RenderContext, options: NumericInputOptions) {
    this.minimum = options.minimum ?? Number.MIN_SAFE_INTEGER;
    this.maximum = options.maximum ?? Number.MAX_SAFE_INTEGER;
    this.currentValue = options.initialValue ?? 0;

    // Create container
    this.container = new BoxRenderable(renderer, {
      id: `${options.id}-container`,
      flexDirection: 'column',
      width: (options.width || '100%') as number | `${number}%`,
      gap: 0,
    });

    // Create label
    const labelContent = options.required
      ? `${options.label} *`
      : options.label;

    this.labelText = new TextRenderable(renderer, {
      id: `${options.id}-label`,
      content: labelContent,
    });

    // Create input field
    this.inputField = new InputRenderable(renderer, {
      id: `${options.id}-input`,
      value: this.currentValue.toString(),
      width: '100%',
      height: 1,
    });

    // Create error text (hidden initially)
    this.errorText = new TextRenderable(renderer, {
      id: `${options.id}-error`,
      content: '',
    });

    // Wire up events
    this.inputField.on(InputRenderableEvents.INPUT, (value: string) => {
      this.handleInputChange(value);
    });

    this.inputField.on(InputRenderableEvents.CHANGE, (value: string) => {
      this.handleSubmit(value);
    });

    // Assemble component
    this.container.add(this.labelText);
    this.container.add(this.inputField);
    this.container.add(this.errorText);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Focus the input field.
   */
  focus(): void {
    this.inputField.focus();
  }

  /**
   * Get current numeric value.
   */
  getValue(): number {
    return this.currentValue;
  }

  /**
   * Set value programmatically.
   */
  setValue(value: number): void {
    this.currentValue = value;
    this.inputField.value = value.toString();
    this.clearError();
  }

  /**
   * Register handler for value changes.
   */
  onChange(handler: NumericValueHandler): () => void {
    this.changeHandlers.push(handler);
    return () => {
      this.changeHandlers = this.changeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Register handler for value submission.
   */
  onSubmit(handler: NumericValueHandler): () => void {
    this.submitHandlers.push(handler);
    return () => {
      this.submitHandlers = this.submitHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Handle input change and validate.
   */
  private handleInputChange(value: string): void {
    const parsed = parseInt(value, 10);

    if (isNaN(parsed)) {
      this.setError('Please enter a valid number');
      return;
    }

    if (parsed < this.minimum) {
      this.setError(`Value must be at least ${this.minimum}`);
      return;
    }

    if (parsed > this.maximum) {
      this.setError(`Value must be at most ${this.maximum}`);
      return;
    }

    this.clearError();
    this.currentValue = parsed;

    for (const handler of this.changeHandlers) {
      handler(parsed);
    }
  }

  /**
   * Handle submit and validate.
   */
  private handleSubmit(value: string): void {
    const parsed = parseInt(value, 10);

    if (isNaN(parsed) || parsed < this.minimum || parsed > this.maximum) {
      return; // Don't submit invalid values
    }

    this.currentValue = parsed;

    for (const handler of this.submitHandlers) {
      handler(parsed);
    }
  }

  /**
   * Set error message.
   */
  private setError(message: string): void {
    this.errorText.content = message;
  }

  /**
   * Clear error message.
   */
  private clearError(): void {
    this.errorText.content = '';
  }
}
