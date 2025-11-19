/**
 * MultiLineTextInput - Multi-line text input using multiple input fields.
 * Single Responsibility: Handle multi-line text entry with line management.
 */

import {
  BoxRenderable,
  TextRenderable,
  InputRenderable,
  InputRenderableEvents,
  type RenderContext,
} from '@opentui/core';

/**
 * Configuration for MultiLineTextInput.
 */
export interface MultiLineTextInputOptions {
  id: string;
  label: string;
  placeholder?: string;
  initialValue?: string;
  width?: number | string;
  visibleLines?: number;
}

/**
 * Callback for value changes.
 */
export type MultiLineValueHandler = (value: string) => void;

export class MultiLineTextInput {
  private renderer: RenderContext;
  private container: BoxRenderable;
  private labelText: TextRenderable;
  private linesContainer: BoxRenderable;
  private inputLines: InputRenderable[] = [];
  private changeHandlers: MultiLineValueHandler[] = [];
  private submitHandlers: MultiLineValueHandler[] = [];
  private options: MultiLineTextInputOptions;
  private currentLineIndex: number = 0;

  constructor(renderer: RenderContext, options: MultiLineTextInputOptions) {
    this.renderer = renderer;
    this.options = options;

    // Create container
    this.container = new BoxRenderable(renderer, {
      id: `${options.id}-container`,
      flexDirection: 'column',
      width: (options.width || '100%') as number | `${number}%`,
      gap: 0,
    });

    // Create label
    this.labelText = new TextRenderable(renderer, {
      id: `${options.id}-label`,
      content: options.label,
    });

    // Create hint text
    const hintText = new TextRenderable(renderer, {
      id: `${options.id}-hint`,
      content: '(Enter for new line, Ctrl+Enter to submit)',
    });

    // Create lines container
    this.linesContainer = new BoxRenderable(renderer, {
      id: `${options.id}-lines`,
      flexDirection: 'column',
      width: '100%',
      gap: 0,
    });

    // Initialize with one line
    this.addInputLine();

    // Set initial value if provided
    if (options.initialValue) {
      this.setValue(options.initialValue);
    }

    // Assemble component
    this.container.add(this.labelText);
    this.container.add(hintText);
    this.container.add(this.linesContainer);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Focus the first input line.
   */
  focus(): void {
    if (this.inputLines.length > 0) {
      this.inputLines[this.currentLineIndex].focus();
    }
  }

  /**
   * Get combined value from all lines.
   */
  getValue(): string {
    return this.inputLines
      .map((input) => input.value)
      .join('\n')
      .trim();
  }

  /**
   * Set value, splitting by newlines.
   */
  setValue(value: string): void {
    const lines = value.split('\n');

    // Clear existing lines
    this.clearAllLines();

    // Add lines for content
    for (const line of lines) {
      const inputLine = this.addInputLine();
      inputLine.value = line;
    }

    this.notifyChangeHandlers();
  }

  /**
   * Clear all content.
   */
  clear(): void {
    this.clearAllLines();
    this.addInputLine();
    this.currentLineIndex = 0;
  }

  /**
   * Register handler for value changes.
   */
  onChange(handler: MultiLineValueHandler): () => void {
    this.changeHandlers.push(handler);
    return () => {
      this.changeHandlers = this.changeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Register handler for submission.
   */
  onSubmit(handler: MultiLineValueHandler): () => void {
    this.submitHandlers.push(handler);
    return () => {
      this.submitHandlers = this.submitHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Add a new input line.
   */
  private addInputLine(): InputRenderable {
    const lineIndex = this.inputLines.length;
    const inputLine = new InputRenderable(this.renderer, {
      id: `${this.options.id}-line-${lineIndex}`,
      placeholder: lineIndex === 0 ? (this.options.placeholder || '') : '',
      width: '100%',
      height: 1,
    });

    // Handle Enter to add new line
    inputLine.on(InputRenderableEvents.CHANGE, () => {
      this.handleLineSubmit(lineIndex);
    });

    // Handle input changes
    inputLine.on(InputRenderableEvents.INPUT, () => {
      this.notifyChangeHandlers();
    });

    this.inputLines.push(inputLine);
    this.linesContainer.add(inputLine);

    return inputLine;
  }

  /**
   * Handle line submission (Enter pressed).
   */
  private handleLineSubmit(lineIndex: number): void {
    // Add new line and focus it
    const newLine = this.addInputLine();
    this.currentLineIndex = lineIndex + 1;
    newLine.focus();
  }

  /**
   * Clear all input lines.
   */
  private clearAllLines(): void {
    for (const line of this.inputLines) {
      this.linesContainer.remove(line.id);
    }
    this.inputLines = [];
  }

  /**
   * Notify change handlers.
   */
  private notifyChangeHandlers(): void {
    const value = this.getValue();
    for (const handler of this.changeHandlers) {
      handler(value);
    }
  }
}
