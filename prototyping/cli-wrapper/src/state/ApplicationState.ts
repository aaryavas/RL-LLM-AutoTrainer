/**
 * ApplicationState - Central state container for the TUI application.
 * Single Responsibility: Hold and notify changes to application state.
 */

import { DataGenerationConfig, FineTuneConfig } from '../config/types';
import {
  ApplicationStateData,
  BatchProgress,
  ExecutionError,
  INITIAL_APPLICATION_STATE,
  ScreenIdentifier,
  StateChangeCallback,
} from './types';

export class ApplicationState {
  private stateData: ApplicationStateData;
  private changeCallbacks: StateChangeCallback[] = [];

  constructor(initialState?: Partial<ApplicationStateData>) {
    this.stateData = {
      ...INITIAL_APPLICATION_STATE,
      ...initialState,
    };
  }

  /**
   * Subscribe to state changes.
   */
  onStateChange(callback: StateChangeCallback): () => void {
    this.changeCallbacks.push(callback);
    return () => {
      this.changeCallbacks = this.changeCallbacks.filter((cb) => cb !== callback);
    };
  }

  /**
   * Get current state snapshot (immutable copy).
   */
  getState(): ApplicationStateData {
    return { ...this.stateData };
  }

  /**
   * Get current screen identifier.
   */
  getCurrentScreen(): ScreenIdentifier {
    return this.stateData.currentScreen;
  }

  /**
   * Set the current screen.
   */
  setCurrentScreen(screen: ScreenIdentifier): void {
    this.updateState({ currentScreen: screen });
  }

  /**
   * Update data generation configuration.
   */
  updateDataConfiguration(config: Partial<DataGenerationConfig>): void {
    this.updateState({
      dataConfiguration: {
        ...this.stateData.dataConfiguration,
        ...config,
      },
    });
  }

  /**
   * Update fine-tuning configuration.
   */
  updateFineTuneConfiguration(config: Partial<FineTuneConfig>): void {
    this.updateState({
      fineTuneConfiguration: {
        ...this.stateData.fineTuneConfiguration,
        ...config,
      },
    });
  }

  /**
   * Append a line to the execution log.
   */
  appendToExecutionLog(line: string): void {
    this.updateState({
      executionLog: [...this.stateData.executionLog, line],
    });
  }

  /**
   * Clear the execution log.
   */
  clearExecutionLog(): void {
    this.updateState({ executionLog: [] });
  }

  /**
   * Update execution progress.
   */
  setExecutionProgress(progress: BatchProgress): void {
    this.updateState({ executionProgress: progress });
  }

  /**
   * Set execution error.
   */
  setExecutionError(error: ExecutionError | null): void {
    this.updateState({ executionError: error });
  }

  /**
   * Set whether a process is currently running.
   */
  setProcessRunning(isRunning: boolean): void {
    this.updateState({ isProcessRunning: isRunning });
  }

  /**
   * Reset state to initial values.
   */
  reset(): void {
    this.updateState(INITIAL_APPLICATION_STATE);
  }

  /**
   * Internal method to update state and notify listeners.
   */
  private updateState(partialState: Partial<ApplicationStateData>): void {
    const previousState = { ...this.stateData };
    this.stateData = {
      ...this.stateData,
      ...partialState,
    };
    this.notifyStateChange(previousState, this.stateData);
  }

  /**
   * Notify all registered callbacks of state change.
   */
  private notifyStateChange(
    previousState: ApplicationStateData,
    newState: ApplicationStateData
  ): void {
    for (const callback of this.changeCallbacks) {
      callback(previousState, newState);
    }
  }
}
