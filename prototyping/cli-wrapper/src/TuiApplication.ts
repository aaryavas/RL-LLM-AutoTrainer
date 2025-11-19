/**
 * TuiApplication - Main TUI application orchestrator.
 * Single Responsibility: Initialize and coordinate all TUI components.
 */

import * as path from 'path';
import { type Renderable } from '@opentui/core';
import { ApplicationState, StateTransitionHandler, ScreenIdentifier } from './state';
import {
  TuiApplicationRoot,
  MainLayoutContainer,
  ScreenNavigator,
  TerminalSizeValidator,
} from './ui';
import { DataGenerationRunner, FineTuneRunner } from './process';
import {
  WelcomeScreen,
  DataConfigurationScreen,
  ProcessExecutionScreen,
  ExecutionCompleteScreen,
  FineTuneConfigScreen,
} from './screens';
import { DataGenerationConfig, FineTuneConfig } from './config/types';

export class TuiApplication {
  private tuiRoot: TuiApplicationRoot;
  private applicationState: ApplicationState;
  private stateTransitionHandler: StateTransitionHandler;
  private mainLayout: MainLayoutContainer | null = null;
  private screenNavigator: ScreenNavigator | null = null;

  // Screen instances
  private welcomeScreen: WelcomeScreen | null = null;
  private dataConfigScreen: DataConfigurationScreen | null = null;
  private processExecutionScreen: ProcessExecutionScreen | null = null;
  private executionCompleteScreen: ExecutionCompleteScreen | null = null;
  private fineTuneConfigScreen: FineTuneConfigScreen | null = null;

  // Runners
  private dataGenerationRunner: DataGenerationRunner;
  private fineTuneRunner: FineTuneRunner;

  // Configuration data
  private currentDataConfig: DataGenerationConfig | null = null;
  private currentFineTuneConfig: FineTuneConfig | null = null;
  private lastOutputPath: string = '';

  constructor() {
    this.tuiRoot = new TuiApplicationRoot();
    this.applicationState = new ApplicationState();
    this.stateTransitionHandler = new StateTransitionHandler(this.applicationState);

    // Setup runners with paths
    const prototypingDir = path.resolve(__dirname, '..');
    const dataGenPath = path.join(prototypingDir, 'data-gen.py');
    const vbloraDir = path.join(prototypingDir, 'vblora');
    const vbloraCliPath = path.join(vbloraDir, 'cli.py');

    this.dataGenerationRunner = new DataGenerationRunner(prototypingDir, dataGenPath);
    this.fineTuneRunner = new FineTuneRunner(vbloraDir, vbloraCliPath);
  }

  /**
   * Start the TUI application.
   */
  async start(): Promise<void> {
    // Validate terminal size
    const sizeValidator = new TerminalSizeValidator(80, 24);
    const validation = sizeValidator.validate();

    if (!validation.isValid) {
      console.error(validation.errorMessage);
      process.exit(1);
    }

    // Initialize TUI
    const renderer = await this.tuiRoot.initialize();

    // Create layout
    this.mainLayout = new MainLayoutContainer(renderer);

    // Create screens
    this.createScreens(renderer);

    // Setup screen navigator
    this.screenNavigator = new ScreenNavigator(this.mainLayout, this.applicationState);
    this.screenNavigator.setScreenFactory((screenId) => this.getScreenRenderable(screenId));

    // Navigate to initial screen
    this.screenNavigator.navigateToScreen('WELCOME');

    // Setup keyboard handling
    this.tuiRoot.onKeyboardEvent((keyEvent) => {
      this.handleKeyboardEvent(keyEvent);
    });
  }

  /**
   * Create all screen instances.
   */
  private createScreens(renderer: any): void {
    // Welcome screen
    this.welcomeScreen = new WelcomeScreen(renderer);
    this.welcomeScreen.onStart(() => {
      this.stateTransitionHandler.transitionToDataConfiguration();
    });

    // Data configuration screen
    this.dataConfigScreen = new DataConfigurationScreen(renderer);
    this.dataConfigScreen.onComplete((config) => {
      this.currentDataConfig = config;
      this.stateTransitionHandler.transitionToDataExecution();
      this.startDataGeneration(config);
    });

    // Process execution screen
    this.processExecutionScreen = new ProcessExecutionScreen(renderer);
    this.processExecutionScreen.onComplete((success, exitCode) => {
      this.handleExecutionComplete(success, exitCode);
    });

    // Execution complete screen
    this.executionCompleteScreen = new ExecutionCompleteScreen(renderer);
    this.executionCompleteScreen.onContinue(() => {
      this.stateTransitionHandler.transitionToFineTuneConfiguration();
    });
    this.executionCompleteScreen.onRestart(() => {
      this.stateTransitionHandler.transitionToWelcome();
    });
    this.executionCompleteScreen.onQuit(() => {
      this.shutdown();
    });

    // Fine-tune configuration screen
    this.fineTuneConfigScreen = new FineTuneConfigScreen(renderer);
    this.fineTuneConfigScreen.onComplete((config) => {
      this.currentFineTuneConfig = config;
      this.stateTransitionHandler.transitionToFineTuneExecution();
      this.startFineTuning(config);
    });
  }

  /**
   * Get renderable for a screen.
   */
  private getScreenRenderable(screenId: ScreenIdentifier): Renderable | null {
    switch (screenId) {
      case 'WELCOME':
        return this.welcomeScreen?.getRenderable() || null;
      case 'DATA_CONFIGURATION':
        return this.dataConfigScreen?.getRenderable() || null;
      case 'DATA_EXECUTION':
      case 'FINETUNE_EXECUTION':
        return this.processExecutionScreen?.getRenderable() || null;
      case 'EXECUTION_COMPLETE':
        return this.executionCompleteScreen?.getRenderable() || null;
      case 'FINETUNE_CONFIGURATION':
        return this.fineTuneConfigScreen?.getRenderable() || null;
      default:
        return null;
    }
  }

  /**
   * Handle keyboard events.
   */
  private handleKeyboardEvent(keyEvent: any): void {
    const currentScreen = this.applicationState.getCurrentScreen();

    // Route to appropriate screen handler
    switch (currentScreen) {
      case 'WELCOME':
        this.welcomeScreen?.handleKeyEvent(keyEvent);
        break;
      case 'DATA_CONFIGURATION':
        this.dataConfigScreen?.handleKeyEvent(keyEvent);
        break;
      case 'EXECUTION_COMPLETE':
        this.executionCompleteScreen?.handleKeyEvent(keyEvent);
        break;
      case 'FINETUNE_CONFIGURATION':
        this.fineTuneConfigScreen?.handleKeyEvent(keyEvent);
        break;
    }
  }

  /**
   * Start data generation process.
   */
  private startDataGeneration(config: DataGenerationConfig): void {
    const processEmitter = this.dataGenerationRunner.execute(config);
    this.processExecutionScreen?.attachToProcess(processEmitter, 'Data Generation');
    this.lastOutputPath = this.dataGenerationRunner.calculateOutputFilePath(config.outputDir);
  }

  /**
   * Start fine-tuning process.
   */
  private startFineTuning(config: FineTuneConfig): void {
    const processEmitter = this.fineTuneRunner.execute(config);
    this.processExecutionScreen?.attachToProcess(processEmitter, 'Fine-Tuning');
  }

  /**
   * Handle execution completion.
   */
  private handleExecutionComplete(success: boolean, exitCode: number | null): void {
    this.stateTransitionHandler.transitionToExecutionComplete();

    if (success) {
      this.executionCompleteScreen?.setSuccess(this.lastOutputPath);
    } else {
      this.executionCompleteScreen?.setFailure(`Process exited with code ${exitCode}`);
    }
  }

  /**
   * Shutdown the application.
   */
  shutdown(): void {
    this.tuiRoot.shutdown();
    process.exit(0);
  }
}

/**
 * Main entry point.
 */
export async function main(): Promise<void> {
  const app = new TuiApplication();
  await app.start();
}
