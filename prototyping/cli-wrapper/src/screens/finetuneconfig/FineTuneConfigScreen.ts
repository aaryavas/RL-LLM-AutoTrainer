/**
 * FineTuneConfigScreen - Screen for fine-tuning configuration.
 * Single Responsibility: Configure VB-LoRA fine-tuning parameters.
 */

import {
  BoxRenderable,
  TextRenderable,
  type RenderContext,
  type KeyEvent,
} from '@opentui/core';
import { FineTuneConfig } from '../../config/types';
import { getDefaultFineTuneConfig } from '../../config/defaults';
import { SingleSelectList } from '../../components/selection';
import { StepProgressHeader } from '../../components/display';

/**
 * Callback for configuration completion.
 */
export type FineTuneConfigCompleteHandler = (config: FineTuneConfig) => void;

export class FineTuneConfigScreen {
  private container: BoxRenderable;
  private stepProgressHeader: StepProgressHeader;
  private modelFamilySelect: SingleSelectList;
  private presetSelect: SingleSelectList;
  private dataPath: string = '';
  private completeHandlers: FineTuneConfigCompleteHandler[] = [];

  constructor(renderer: RenderContext) {
    // Create container
    this.container = new BoxRenderable(renderer, {
      id: 'finetune-config-screen',
      flexDirection: 'column',
      width: '100%',
      height: '100%',
      gap: 1,
    });

    // Create step progress header
    this.stepProgressHeader = new StepProgressHeader(renderer, {
      id: 'finetune-config-progress',
      totalSteps: 3,
      stepNames: ['Model Family', 'Preset', 'Confirm'],
    });

    // Create model family selector
    this.modelFamilySelect = new SingleSelectList(renderer, {
      id: 'model-family-select',
      label: 'Select Model Family',
      options: [
        { name: 'SmolLM2', description: 'HuggingFace SmolLM2 family', value: 'smollm2' },
      ],
      height: 4,
    });

    // Create preset selector
    this.presetSelect = new SingleSelectList(renderer, {
      id: 'preset-select',
      label: 'Select Training Preset',
      options: [
        { name: 'Quick Test', description: 'Fast testing - 1 epoch', value: 'quick_test' },
        { name: 'Standard', description: 'Balanced training - 3 epochs', value: 'standard' },
        { name: 'Thorough', description: 'Comprehensive - 5 epochs', value: 'thorough' },
      ],
      height: 5,
    });

    // Assemble screen
    this.container.add(this.stepProgressHeader.getRenderable());
    this.container.add(this.modelFamilySelect.getRenderable());
    this.container.add(this.presetSelect.getRenderable());
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Set the data path for fine-tuning.
   */
  setDataPath(path: string): void {
    this.dataPath = path;
  }

  /**
   * Focus the first element.
   */
  focus(): void {
    this.modelFamilySelect.focus();
  }

  /**
   * Handle keyboard event.
   */
  handleKeyEvent(keyEvent: KeyEvent): boolean {
    if (keyEvent.name === 'return' || keyEvent.name === 'enter') {
      this.notifyCompleteHandlers();
      return true;
    }
    return false;
  }

  /**
   * Get the built configuration.
   */
  getConfiguration(): FineTuneConfig {
    const defaultConfig = getDefaultFineTuneConfig();

    return {
      ...defaultConfig,
      dataPath: this.dataPath,
      modelFamily: 'smollm2',
      modelVariant: '360M',
    };
  }

  /**
   * Register handler for completion.
   */
  onComplete(handler: FineTuneConfigCompleteHandler): () => void {
    this.completeHandlers.push(handler);
    return () => {
      this.completeHandlers = this.completeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Notify complete handlers.
   */
  private notifyCompleteHandlers(): void {
    const config = this.getConfiguration();
    for (const handler of this.completeHandlers) {
      handler(config);
    }
  }
}
