/**
 * FormStepController - Orchestrate multi-step form progression.
 * Single Responsibility: Manage step transitions and validation in forms.
 */

import { type Renderable } from '@opentui/core';

/**
 * Single form step definition.
 */
export interface FormStep {
  id: string;
  name: string;
  component: Renderable;
  validate?: () => boolean;
  onEnter?: () => void;
  onExit?: () => void;
}

/**
 * Callback for step changes.
 */
export type StepChangeHandler = (
  previousStep: number,
  currentStep: number,
  step: FormStep
) => void;

/**
 * Callback for form completion.
 */
export type FormCompleteHandler = () => void;

export class FormStepController {
  private steps: FormStep[];
  private currentStepIndex: number = 0;
  private stepChangeHandlers: StepChangeHandler[] = [];
  private completeHandlers: FormCompleteHandler[] = [];

  constructor(steps: FormStep[]) {
    this.steps = steps;
  }

  /**
   * Get all steps.
   */
  getSteps(): FormStep[] {
    return [...this.steps];
  }

  /**
   * Get current step index (0-based).
   */
  getCurrentStepIndex(): number {
    return this.currentStepIndex;
  }

  /**
   * Get current step.
   */
  getCurrentStep(): FormStep {
    return this.steps[this.currentStepIndex];
  }

  /**
   * Get total number of steps.
   */
  getTotalSteps(): number {
    return this.steps.length;
  }

  /**
   * Advance to next step if validation passes.
   */
  advanceToNextStep(): boolean {
    if (this.isLastStep()) {
      this.notifyCompleteHandlers();
      return false;
    }

    const currentStep = this.getCurrentStep();

    // Validate current step
    if (currentStep.validate && !currentStep.validate()) {
      return false;
    }

    // Exit current step
    if (currentStep.onExit) {
      currentStep.onExit();
    }

    // Move to next step
    const previousIndex = this.currentStepIndex;
    this.currentStepIndex++;

    // Enter new step
    const newStep = this.getCurrentStep();
    if (newStep.onEnter) {
      newStep.onEnter();
    }

    this.notifyStepChangeHandlers(previousIndex, this.currentStepIndex, newStep);
    return true;
  }

  /**
   * Return to previous step.
   */
  returnToPreviousStep(): boolean {
    if (this.isFirstStep()) {
      return false;
    }

    const currentStep = this.getCurrentStep();

    // Exit current step
    if (currentStep.onExit) {
      currentStep.onExit();
    }

    // Move to previous step
    const previousIndex = this.currentStepIndex;
    this.currentStepIndex--;

    // Enter previous step
    const newStep = this.getCurrentStep();
    if (newStep.onEnter) {
      newStep.onEnter();
    }

    this.notifyStepChangeHandlers(previousIndex, this.currentStepIndex, newStep);
    return true;
  }

  /**
   * Go to specific step by index.
   */
  goToStep(index: number): boolean {
    if (index < 0 || index >= this.steps.length) {
      return false;
    }

    if (index === this.currentStepIndex) {
      return true;
    }

    const currentStep = this.getCurrentStep();
    if (currentStep.onExit) {
      currentStep.onExit();
    }

    const previousIndex = this.currentStepIndex;
    this.currentStepIndex = index;

    const newStep = this.getCurrentStep();
    if (newStep.onEnter) {
      newStep.onEnter();
    }

    this.notifyStepChangeHandlers(previousIndex, this.currentStepIndex, newStep);
    return true;
  }

  /**
   * Check if on first step.
   */
  isFirstStep(): boolean {
    return this.currentStepIndex === 0;
  }

  /**
   * Check if on last step.
   */
  isLastStep(): boolean {
    return this.currentStepIndex === this.steps.length - 1;
  }

  /**
   * Check if form is complete (all steps passed).
   */
  isComplete(): boolean {
    return this.isLastStep() && this.validateCurrentStep();
  }

  /**
   * Validate current step.
   */
  validateCurrentStep(): boolean {
    const currentStep = this.getCurrentStep();
    if (currentStep.validate) {
      return currentStep.validate();
    }
    return true;
  }

  /**
   * Register handler for step changes.
   */
  onStepChange(handler: StepChangeHandler): () => void {
    this.stepChangeHandlers.push(handler);
    return () => {
      this.stepChangeHandlers = this.stepChangeHandlers.filter(
        (h) => h !== handler
      );
    };
  }

  /**
   * Register handler for form completion.
   */
  onComplete(handler: FormCompleteHandler): () => void {
    this.completeHandlers.push(handler);
    return () => {
      this.completeHandlers = this.completeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Reset to first step.
   */
  reset(): void {
    const previousIndex = this.currentStepIndex;
    this.currentStepIndex = 0;

    const newStep = this.getCurrentStep();
    if (newStep.onEnter) {
      newStep.onEnter();
    }

    this.notifyStepChangeHandlers(previousIndex, 0, newStep);
  }

  /**
   * Notify step change handlers.
   */
  private notifyStepChangeHandlers(
    previousStep: number,
    currentStep: number,
    step: FormStep
  ): void {
    for (const handler of this.stepChangeHandlers) {
      handler(previousStep, currentStep, step);
    }
  }

  /**
   * Notify complete handlers.
   */
  private notifyCompleteHandlers(): void {
    for (const handler of this.completeHandlers) {
      handler();
    }
  }
}
