/**
 * TuiApplicationRoot - Initialize and coordinate TUI application lifecycle.
 * Single Responsibility: Manage CliRenderer lifecycle and global event handling.
 */

import {
  createCliRenderer,
  type CliRenderer,
  type KeyEvent,
} from '@opentui/core';

/**
 * Callback for keyboard events.
 */
export type KeyboardEventHandler = (keyEvent: KeyEvent) => void;

export class TuiApplicationRoot {
  private renderer: CliRenderer | null = null;
  private isInitialized: boolean = false;
  private keyboardHandlers: KeyboardEventHandler[] = [];

  /**
   * Initialize the TUI renderer.
   */
  async initialize(): Promise<CliRenderer> {
    if (this.isInitialized && this.renderer) {
      return this.renderer;
    }

    this.renderer = await createCliRenderer();
    this.isInitialized = true;

    this.setupProcessSignalHandlers();
    this.setupKeyboardEventRouting();

    return this.renderer;
  }

  /**
   * Get the renderer instance.
   */
  getRenderer(): CliRenderer {
    if (!this.renderer) {
      throw new Error('TUI application not initialized. Call initialize() first.');
    }
    return this.renderer;
  }

  /**
   * Check if application is initialized.
   */
  isApplicationInitialized(): boolean {
    return this.isInitialized;
  }

  /**
   * Register a keyboard event handler.
   */
  onKeyboardEvent(handler: KeyboardEventHandler): () => void {
    this.keyboardHandlers.push(handler);
    return () => {
      this.keyboardHandlers = this.keyboardHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Shutdown the application and restore terminal.
   */
  shutdown(): void {
    if (this.renderer) {
      this.renderer.stop();
      this.renderer = null;
      this.isInitialized = false;
    }
  }

  /**
   * Setup process signal handlers for graceful shutdown.
   */
  private setupProcessSignalHandlers(): void {
    // Handle Ctrl+C
    process.on('SIGINT', () => {
      this.shutdown();
      process.exit(0);
    });

    // Handle termination
    process.on('SIGTERM', () => {
      this.shutdown();
      process.exit(0);
    });

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      this.shutdown();
      console.error('Uncaught exception:', error);
      process.exit(1);
    });

    // Ensure terminal is stopped on exit
    process.on('exit', () => {
      if (this.renderer) {
        this.renderer.stop();
      }
    });
  }

  /**
   * Setup keyboard event routing to registered handlers.
   */
  private setupKeyboardEventRouting(): void {
    if (!this.renderer) return;

    this.renderer.keyInput.on('keypress', (keyEvent: KeyEvent) => {
      // Handle global quit shortcut
      if (keyEvent.ctrl && keyEvent.name === 'c') {
        this.shutdown();
        process.exit(0);
      }

      // Route to registered handlers
      for (const handler of this.keyboardHandlers) {
        handler(keyEvent);
      }
    });
  }
}
