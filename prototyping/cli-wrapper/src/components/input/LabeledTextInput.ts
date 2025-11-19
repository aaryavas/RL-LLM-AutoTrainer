/**
 * LabeledTextInput - Text input field with associated label.
 * Single Responsibility: Combine label and input into a single reusable component.
 */

import {
  BoxRenderable,
  TextRenderable,
  InputRenderable,
  InputRenderableEvents,
  type RenderContext,
} from '@opentui/core';

/**
 * Configuration for LabeledTextInput.
 */
export interface LabeledTextInputOptions {
  id: string;
  label: string;
  placeholder?: string;
  initialValue?: string;
  width?: number | string;
  required?: boolean;
}

/**
 * Callback for value changes.
 */
export type ValueChangeHandler = (value: string) => void;

export class LabeledTextInput {
  private container: BoxRenderable;
  private labelText: TextRenderable;
  private inputField: InputRenderable;
  private changeHandlers: ValueChangeHandler[] = [];
  private submitHandlers: ValueChangeHandler[] = [];

  constructor(renderer: RenderContext, options: LabeledTextInputOptions) {
    // Create container with vertical layout
    this.container = new BoxRenderable(renderer, {
      id: `${options.id}-container`,
      flexDirection: 'column',
      width: (options.width || '100%') as number | `${number}%`,
      gap: 0,
    });

    // Create label with required indicator
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
      placeholder: options.placeholder || '',
      value: options.initialValue || '',
      width: '100%',
      height: 1,
    });

    // Wire up events
    this.inputField.on(InputRenderableEvents.INPUT, (value: string) => {
      this.notifyChangeHandlers(value);
    });

    this.inputField.on(InputRenderableEvents.CHANGE, (value: string) => {
      this.notifySubmitHandlers(value);
    });

    // Assemble component
    this.container.add(this.labelText);
    this.container.add(this.inputField);
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
   * Get current value.
   */
  getValue(): string {
    return this.inputField.value;
  }

  /**
   * Set value programmatically.
   */
  setValue(value: string): void {
    this.inputField.value = value;
  }

  /**
   * Clear the input field.
   */
  clear(): void {
    this.inputField.value = '';
  }

  /**
   * Register handler for value changes (on each keystroke).
   */
  onChange(handler: ValueChangeHandler): () => void {
    this.changeHandlers.push(handler);
    return () => {
      this.changeHandlers = this.changeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Register handler for value submission (on Enter).
   */
  onSubmit(handler: ValueChangeHandler): () => void {
    this.submitHandlers.push(handler);
    return () => {
      this.submitHandlers = this.submitHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Notify all change handlers.
   */
  private notifyChangeHandlers(value: string): void {
    for (const handler of this.changeHandlers) {
      handler(value);
    }
  }

  /**
   * Notify all submit handlers.
   */
  private notifySubmitHandlers(value: string): void {
    for (const handler of this.submitHandlers) {
      handler(value);
    }
  }
}
