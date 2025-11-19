/**
 * WelcomeScreen - Initial screen shown when application starts.
 * Single Responsibility: Display welcome message and start prompt.
 */

import {
  BoxRenderable,
  TextRenderable,
  type RenderContext,
  type KeyEvent,
} from '@opentui/core';

/**
 * Callback for start action.
 */
export type StartActionHandler = () => void;

export class WelcomeScreen {
  private container: BoxRenderable;
  private titleText: TextRenderable;
  private descriptionText: TextRenderable;
  private instructionsText: TextRenderable;
  private startHandlers: StartActionHandler[] = [];

  constructor(renderer: RenderContext) {
    // Create container
    this.container = new BoxRenderable(renderer, {
      id: 'welcome-screen',
      flexDirection: 'column',
      width: '100%',
      height: '100%',
      justifyContent: 'center',
      alignItems: 'center',
      gap: 2,
      padding: 2,
    });

    // Create title
    this.titleText = new TextRenderable(renderer, {
      id: 'welcome-title',
      content: 'Synthetic Data Generator & Fine-Tuning CLI',
    });

    // Create description
    this.descriptionText = new TextRenderable(renderer, {
      id: 'welcome-description',
      content: [
        'This tool helps you:',
        '',
        '  1. Generate synthetic training data using AI models',
        '  2. Fine-tune SmolLM2 models with VB-LoRA',
        '',
        'You will be guided through configuration steps for each phase.',
      ].join('\n'),
    });

    // Create instructions
    this.instructionsText = new TextRenderable(renderer, {
      id: 'welcome-instructions',
      content: 'Press Enter to start',
    });

    // Assemble screen
    this.container.add(this.titleText);
    this.container.add(this.descriptionText);
    this.container.add(this.instructionsText);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Handle keyboard event.
   */
  handleKeyEvent(keyEvent: KeyEvent): boolean {
    if (keyEvent.name === 'return' || keyEvent.name === 'enter') {
      this.notifyStartHandlers();
      return true;
    }
    return false;
  }

  /**
   * Register handler for start action.
   */
  onStart(handler: StartActionHandler): () => void {
    this.startHandlers.push(handler);
    return () => {
      this.startHandlers = this.startHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Notify start handlers.
   */
  private notifyStartHandlers(): void {
    for (const handler of this.startHandlers) {
      handler();
    }
  }
}
