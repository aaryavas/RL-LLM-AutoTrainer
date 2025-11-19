/**
 * ProgressIndicator - Display progress as text-based progress bar.
 * Single Responsibility: Show visual progress representation.
 */

import {
  BoxRenderable,
  TextRenderable,
  type RenderContext,
} from '@opentui/core';

/**
 * Configuration for ProgressIndicator.
 */
export interface ProgressIndicatorOptions {
  id: string;
  width?: number | string;
  barWidth?: number;
  showPercentage?: boolean;
  showFraction?: boolean;
}

export class ProgressIndicator {
  private container: BoxRenderable;
  private labelText: TextRenderable;
  private progressBarText: TextRenderable;
  private percentageText: TextRenderable;
  private barWidth: number;
  private currentValue: number = 0;
  private totalValue: number = 100;
  private showPercentage: boolean;
  private showFraction: boolean;

  constructor(renderer: RenderContext, options: ProgressIndicatorOptions) {
    this.barWidth = options.barWidth ?? 30;
    this.showPercentage = options.showPercentage ?? true;
    this.showFraction = options.showFraction ?? true;

    // Create container
    this.container = new BoxRenderable(renderer, {
      id: `${options.id}-container`,
      flexDirection: 'column',
      width: (options.width || '100%') as number | `${number}%`,
      gap: 0,
    });

    // Create label text
    this.labelText = new TextRenderable(renderer, {
      id: `${options.id}-label`,
      content: '',
    });

    // Create progress bar text
    this.progressBarText = new TextRenderable(renderer, {
      id: `${options.id}-bar`,
      content: this.generateProgressBar(0),
    });

    // Create percentage/fraction text
    this.percentageText = new TextRenderable(renderer, {
      id: `${options.id}-percentage`,
      content: '',
    });

    // Assemble component
    this.container.add(this.labelText);
    this.container.add(this.progressBarText);
    this.container.add(this.percentageText);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Set progress label text.
   */
  setLabel(label: string): void {
    this.labelText.content = label;
  }

  /**
   * Set progress value.
   */
  setProgress(current: number, total: number): void {
    this.currentValue = current;
    this.totalValue = total;
    this.updateDisplay();
  }

  /**
   * Set progress as percentage (0-100).
   */
  setPercentage(percentage: number): void {
    this.currentValue = percentage;
    this.totalValue = 100;
    this.updateDisplay();
  }

  /**
   * Get current progress percentage.
   */
  getPercentage(): number {
    if (this.totalValue === 0) return 0;
    return Math.round((this.currentValue / this.totalValue) * 100);
  }

  /**
   * Reset progress to zero.
   */
  reset(): void {
    this.currentValue = 0;
    this.updateDisplay();
  }

  /**
   * Update the progress display.
   */
  private updateDisplay(): void {
    const percentage = this.getPercentage();
    this.progressBarText.content = this.generateProgressBar(percentage);

    let statusText = '';
    if (this.showFraction) {
      statusText = `${this.currentValue}/${this.totalValue}`;
    }
    if (this.showPercentage) {
      if (statusText) {
        statusText += ` (${percentage}%)`;
      } else {
        statusText = `${percentage}%`;
      }
    }
    this.percentageText.content = statusText;
  }

  /**
   * Generate text-based progress bar.
   */
  private generateProgressBar(percentage: number): string {
    const filledWidth = Math.round((percentage / 100) * this.barWidth);
    const emptyWidth = this.barWidth - filledWidth;

    const filledChar = '=';
    const emptyChar = ' ';
    const pointerChar = '>';

    let bar = filledChar.repeat(Math.max(0, filledWidth - 1));
    if (filledWidth > 0) {
      bar += percentage < 100 ? pointerChar : filledChar;
    }
    bar += emptyChar.repeat(emptyWidth);

    return `[${bar}]`;
  }
}
