/**
 * StepProgressHeader - Display current step in multi-step form.
 * Single Responsibility: Show step indicator like "Step 2 of 5".
 */

import {
  BoxRenderable,
  TextRenderable,
  type RenderContext,
} from '@opentui/core';

/**
 * Configuration for StepProgressHeader.
 */
export interface StepProgressHeaderOptions {
  id: string;
  totalSteps: number;
  stepNames?: string[];
  width?: number | string;
}

export class StepProgressHeader {
  private container: BoxRenderable;
  private stepCountText: TextRenderable;
  private stepNameText: TextRenderable;
  private totalSteps: number;
  private stepNames: string[];
  private currentStep: number = 1;

  constructor(renderer: RenderContext, options: StepProgressHeaderOptions) {
    this.totalSteps = options.totalSteps;
    this.stepNames = options.stepNames || [];

    // Create container
    this.container = new BoxRenderable(renderer, {
      id: `${options.id}-container`,
      flexDirection: 'column',
      width: (options.width || '100%') as number | `${number}%`,
      gap: 0,
    });

    // Create step count text
    this.stepCountText = new TextRenderable(renderer, {
      id: `${options.id}-count`,
      content: this.generateStepCountText(),
    });

    // Create step name text
    this.stepNameText = new TextRenderable(renderer, {
      id: `${options.id}-name`,
      content: this.getCurrentStepName(),
    });

    // Assemble component
    this.container.add(this.stepCountText);
    this.container.add(this.stepNameText);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Set current step (1-indexed).
   */
  setCurrentStep(step: number): void {
    this.currentStep = Math.max(1, Math.min(step, this.totalSteps));
    this.updateDisplay();
  }

  /**
   * Advance to next step.
   */
  nextStep(): void {
    if (this.currentStep < this.totalSteps) {
      this.currentStep++;
      this.updateDisplay();
    }
  }

  /**
   * Go to previous step.
   */
  previousStep(): void {
    if (this.currentStep > 1) {
      this.currentStep--;
      this.updateDisplay();
    }
  }

  /**
   * Get current step number.
   */
  getCurrentStep(): number {
    return this.currentStep;
  }

  /**
   * Check if on last step.
   */
  isLastStep(): boolean {
    return this.currentStep === this.totalSteps;
  }

  /**
   * Check if on first step.
   */
  isFirstStep(): boolean {
    return this.currentStep === 1;
  }

  /**
   * Update step names.
   */
  setStepNames(names: string[]): void {
    this.stepNames = names;
    this.updateDisplay();
  }

  /**
   * Update total steps.
   */
  setTotalSteps(total: number): void {
    this.totalSteps = total;
    if (this.currentStep > total) {
      this.currentStep = total;
    }
    this.updateDisplay();
  }

  /**
   * Generate step count text.
   */
  private generateStepCountText(): string {
    return `Step ${this.currentStep} of ${this.totalSteps}`;
  }

  /**
   * Get current step name.
   */
  private getCurrentStepName(): string {
    const index = this.currentStep - 1;
    if (index >= 0 && index < this.stepNames.length) {
      return this.stepNames[index];
    }
    return '';
  }

  /**
   * Update the display.
   */
  private updateDisplay(): void {
    this.stepCountText.content = this.generateStepCountText();
    this.stepNameText.content = this.getCurrentStepName();
  }
}
