/**
 * StateTransitionHandler - Manages valid state transitions between screens.
 * Single Responsibility: Validate and execute screen transitions.
 */

import { ApplicationState } from './ApplicationState';
import { ScreenIdentifier, TransitionResult } from './types';

/**
 * Defines which screens can transition to which other screens.
 */
const VALID_TRANSITIONS: Record<ScreenIdentifier, ScreenIdentifier[]> = {
  WELCOME: ['DATA_CONFIGURATION'],
  DATA_CONFIGURATION: ['DATA_EXECUTION', 'WELCOME'],
  DATA_EXECUTION: ['FINETUNE_CONFIGURATION', 'EXECUTION_COMPLETE', 'DATA_CONFIGURATION'],
  FINETUNE_CONFIGURATION: ['FINETUNE_EXECUTION', 'DATA_EXECUTION'],
  FINETUNE_EXECUTION: ['EXECUTION_COMPLETE', 'FINETUNE_CONFIGURATION'],
  EXECUTION_COMPLETE: ['WELCOME', 'DATA_CONFIGURATION'],
};

export class StateTransitionHandler {
  private applicationState: ApplicationState;

  constructor(applicationState: ApplicationState) {
    this.applicationState = applicationState;
  }

  /**
   * Check if a transition to the target screen is valid from current state.
   */
  canTransitionTo(targetScreen: ScreenIdentifier): boolean {
    const currentScreen = this.applicationState.getCurrentScreen();
    const validTargets = VALID_TRANSITIONS[currentScreen] || [];
    return validTargets.includes(targetScreen);
  }

  /**
   * Attempt to transition to the target screen.
   */
  transitionTo(targetScreen: ScreenIdentifier): TransitionResult {
    const previousScreen = this.applicationState.getCurrentScreen();

    if (!this.canTransitionTo(targetScreen)) {
      return {
        success: false,
        errorMessage: `Cannot transition from ${previousScreen} to ${targetScreen}`,
        previousScreen,
        newScreen: previousScreen,
      };
    }

    this.applicationState.setCurrentScreen(targetScreen);

    return {
      success: true,
      previousScreen,
      newScreen: targetScreen,
    };
  }

  /**
   * Get all valid transition targets from the current screen.
   */
  getValidTransitions(): ScreenIdentifier[] {
    const currentScreen = this.applicationState.getCurrentScreen();
    return VALID_TRANSITIONS[currentScreen] || [];
  }

  /**
   * Transition to welcome screen (restart flow).
   */
  transitionToWelcome(): TransitionResult {
    return this.transitionTo('WELCOME');
  }

  /**
   * Transition to data configuration screen.
   */
  transitionToDataConfiguration(): TransitionResult {
    return this.transitionTo('DATA_CONFIGURATION');
  }

  /**
   * Transition to data execution screen.
   */
  transitionToDataExecution(): TransitionResult {
    return this.transitionTo('DATA_EXECUTION');
  }

  /**
   * Transition to fine-tune configuration screen.
   */
  transitionToFineTuneConfiguration(): TransitionResult {
    return this.transitionTo('FINETUNE_CONFIGURATION');
  }

  /**
   * Transition to fine-tune execution screen.
   */
  transitionToFineTuneExecution(): TransitionResult {
    return this.transitionTo('FINETUNE_EXECUTION');
  }

  /**
   * Transition to execution complete screen.
   */
  transitionToExecutionComplete(): TransitionResult {
    return this.transitionTo('EXECUTION_COMPLETE');
  }
}
