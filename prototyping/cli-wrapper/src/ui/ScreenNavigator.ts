/**
 * ScreenNavigator - Handle screen transitions based on application state.
 * Single Responsibility: Switch between screens in the layout container.
 */

import { type Renderable } from '@opentui/core';
import { ApplicationState, ScreenIdentifier } from '../state';
import { MainLayoutContainer } from './MainLayoutContainer';

/**
 * Factory function to create screen renderables.
 */
export type ScreenFactory = (screenId: ScreenIdentifier) => Renderable | null;

/**
 * Configuration for a screen including header/footer text.
 */
export interface ScreenConfiguration {
  headerText: string;
  footerText: string;
}

/**
 * Default screen configurations.
 */
const SCREEN_CONFIGURATIONS: Record<ScreenIdentifier, ScreenConfiguration> = {
  WELCOME: {
    headerText: 'Synthetic Data Generator',
    footerText: 'Press Enter to start',
  },
  DATA_CONFIGURATION: {
    headerText: 'Data Generation Configuration',
    footerText: 'Tab: Next field | Enter: Confirm | Esc: Back',
  },
  DATA_EXECUTION: {
    headerText: 'Generating Synthetic Data',
    footerText: 'Ctrl+C: Cancel',
  },
  FINETUNE_CONFIGURATION: {
    headerText: 'Fine-Tuning Configuration',
    footerText: 'Tab: Next field | Enter: Confirm | Esc: Back',
  },
  FINETUNE_EXECUTION: {
    headerText: 'Fine-Tuning Model',
    footerText: 'Ctrl+C: Cancel',
  },
  EXECUTION_COMPLETE: {
    headerText: 'Execution Complete',
    footerText: 'Enter: Continue | Q: Quit',
  },
};

export class ScreenNavigator {
  private layoutContainer: MainLayoutContainer;
  private applicationState: ApplicationState;
  private screenFactory: ScreenFactory | null = null;
  private currentScreenId: ScreenIdentifier | null = null;

  constructor(
    layoutContainer: MainLayoutContainer,
    applicationState: ApplicationState
  ) {
    this.layoutContainer = layoutContainer;
    this.applicationState = applicationState;

    // Subscribe to state changes
    this.applicationState.onStateChange((previousState, newState) => {
      if (previousState.currentScreen !== newState.currentScreen) {
        this.navigateToScreen(newState.currentScreen);
      }
    });
  }

  /**
   * Set the screen factory for creating screen renderables.
   */
  setScreenFactory(factory: ScreenFactory): void {
    this.screenFactory = factory;
  }

  /**
   * Navigate to a specific screen.
   */
  navigateToScreen(screenId: ScreenIdentifier): void {
    if (this.currentScreenId === screenId) {
      return;
    }

    // Update header and footer
    const configuration = SCREEN_CONFIGURATIONS[screenId];
    this.layoutContainer.setHeaderText(configuration.headerText);
    this.layoutContainer.setFooterText(configuration.footerText);

    // Create and display screen content
    if (this.screenFactory) {
      const screenContent = this.screenFactory(screenId);
      if (screenContent) {
        this.layoutContainer.setContent(screenContent);
      } else {
        this.layoutContainer.clearContent();
      }
    }

    this.currentScreenId = screenId;
    this.layoutContainer.requestRender();
  }

  /**
   * Get the current screen identifier.
   */
  getCurrentScreenId(): ScreenIdentifier | null {
    return this.currentScreenId;
  }

  /**
   * Force refresh of the current screen.
   */
  refreshCurrentScreen(): void {
    if (this.currentScreenId) {
      const screenId = this.currentScreenId;
      this.currentScreenId = null;
      this.navigateToScreen(screenId);
    }
  }
}
