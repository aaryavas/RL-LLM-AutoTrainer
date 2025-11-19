/**
 * State type definitions for the TUI application.
 * Single Responsibility: Define all state-related types and interfaces.
 */

import { DataGenerationConfig, FineTuneConfig } from '../config/types';

/**
 * Identifies which screen is currently active in the application.
 */
export type ScreenIdentifier =
  | 'WELCOME'
  | 'DATA_CONFIGURATION'
  | 'DATA_EXECUTION'
  | 'FINETUNE_CONFIGURATION'
  | 'FINETUNE_EXECUTION'
  | 'EXECUTION_COMPLETE';

/**
 * Represents progress through batch processing.
 */
export interface BatchProgress {
  currentBatch: number;
  totalBatches: number;
  percentComplete: number;
}

/**
 * Represents an error that occurred during execution.
 */
export interface ExecutionError {
  message: string;
  exitCode: number | null;
  timestamp: Date;
}

/**
 * Result of a state transition attempt.
 */
export interface TransitionResult {
  success: boolean;
  errorMessage?: string;
  previousScreen: ScreenIdentifier;
  newScreen: ScreenIdentifier;
}

/**
 * Callback signature for state change notifications.
 */
export type StateChangeCallback = (
  previousState: ApplicationStateData,
  newState: ApplicationStateData
) => void;

/**
 * The complete application state data structure.
 */
export interface ApplicationStateData {
  currentScreen: ScreenIdentifier;
  dataConfiguration: Partial<DataGenerationConfig>;
  fineTuneConfiguration: Partial<FineTuneConfig>;
  executionLog: string[];
  executionProgress: BatchProgress;
  executionError: ExecutionError | null;
  isProcessRunning: boolean;
}

/**
 * Default initial state for the application.
 */
export const INITIAL_APPLICATION_STATE: ApplicationStateData = {
  currentScreen: 'WELCOME',
  dataConfiguration: {},
  fineTuneConfiguration: {},
  executionLog: [],
  executionProgress: {
    currentBatch: 0,
    totalBatches: 0,
    percentComplete: 0,
  },
  executionError: null,
  isProcessRunning: false,
};
