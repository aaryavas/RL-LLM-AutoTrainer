/**
 * ExecutionCompleteScreen - Display completion status and next steps.
 * Single Responsibility: Show success/failure and options after execution.
 */

import {
  BoxRenderable,
  TextRenderable,
  type RenderContext,
  type KeyEvent,
} from '@opentui/core';

/**
 * Callback for user actions.
 */
export type CompletionActionHandler = () => void;

export class ExecutionCompleteScreen {
  private container: BoxRenderable;
  private statusText: TextRenderable;
  private summaryText: TextRenderable;
  private optionsText: TextRenderable;
  private continueHandlers: CompletionActionHandler[] = [];
  private restartHandlers: CompletionActionHandler[] = [];
  private quitHandlers: CompletionActionHandler[] = [];
  private wasSuccessful: boolean = false;

  constructor(renderer: RenderContext) {
    // Create container
    this.container = new BoxRenderable(renderer, {
      id: 'execution-complete-screen',
      flexDirection: 'column',
      width: '100%',
      height: '100%',
      justifyContent: 'center',
      alignItems: 'center',
      gap: 2,
      padding: 2,
    });

    // Create status text
    this.statusText = new TextRenderable(renderer, {
      id: 'complete-status',
      content: '',
    });

    // Create summary text
    this.summaryText = new TextRenderable(renderer, {
      id: 'complete-summary',
      content: '',
    });

    // Create options text
    this.optionsText = new TextRenderable(renderer, {
      id: 'complete-options',
      content: '',
    });

    // Assemble screen
    this.container.add(this.statusText);
    this.container.add(this.summaryText);
    this.container.add(this.optionsText);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Set success state.
   */
  setSuccess(outputPath: string): void {
    this.wasSuccessful = true;
    this.statusText.content = '✅ Execution Completed Successfully';
    this.summaryText.content = `Output saved to: ${outputPath}`;
    this.optionsText.content = [
      'Options:',
      '',
      '  [Enter] Continue to fine-tuning',
      '  [R]     Restart from beginning',
      '  [Q]     Quit',
    ].join('\n');
  }

  /**
   * Set failure state.
   */
  setFailure(errorMessage: string): void {
    this.wasSuccessful = false;
    this.statusText.content = '❌ Execution Failed';
    this.summaryText.content = `Error: ${errorMessage}`;
    this.optionsText.content = [
      'Options:',
      '',
      '  [R] Retry / Restart',
      '  [Q] Quit',
    ].join('\n');
  }

  /**
   * Handle keyboard event.
   */
  handleKeyEvent(keyEvent: KeyEvent): boolean {
    const key = keyEvent.name?.toLowerCase();

    if (key === 'return' || key === 'enter') {
      if (this.wasSuccessful) {
        this.notifyContinueHandlers();
        return true;
      }
    }

    if (key === 'r') {
      this.notifyRestartHandlers();
      return true;
    }

    if (key === 'q') {
      this.notifyQuitHandlers();
      return true;
    }

    return false;
  }

  /**
   * Register handler for continue action.
   */
  onContinue(handler: CompletionActionHandler): () => void {
    this.continueHandlers.push(handler);
    return () => {
      this.continueHandlers = this.continueHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Register handler for restart action.
   */
  onRestart(handler: CompletionActionHandler): () => void {
    this.restartHandlers.push(handler);
    return () => {
      this.restartHandlers = this.restartHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Register handler for quit action.
   */
  onQuit(handler: CompletionActionHandler): () => void {
    this.quitHandlers.push(handler);
    return () => {
      this.quitHandlers = this.quitHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Notify continue handlers.
   */
  private notifyContinueHandlers(): void {
    for (const handler of this.continueHandlers) {
      handler();
    }
  }

  /**
   * Notify restart handlers.
   */
  private notifyRestartHandlers(): void {
    for (const handler of this.restartHandlers) {
      handler();
    }
  }

  /**
   * Notify quit handlers.
   */
  private notifyQuitHandlers(): void {
    for (const handler of this.quitHandlers) {
      handler();
    }
  }
}
