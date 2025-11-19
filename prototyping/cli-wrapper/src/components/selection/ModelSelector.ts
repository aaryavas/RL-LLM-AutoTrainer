/**
 * ModelSelector - Specialized selection component for AI models.
 * Single Responsibility: Handle model selection with descriptions and metadata.
 */

import {
  BoxRenderable,
  TextRenderable,
  SelectRenderable,
  SelectRenderableEvents,
  type RenderContext,
  type SelectOption,
} from '@opentui/core';

/**
 * Model option with additional metadata.
 */
export interface ModelOption {
  name: string;
  modelId: string;
  description: string;
  parameters?: string;
}

/**
 * Configuration for ModelSelector.
 */
export interface ModelSelectorOptions {
  id: string;
  label: string;
  models: ModelOption[];
  initialSelectedIndex?: number;
  width?: number | string;
  height?: number;
}

/**
 * Callback for model selection.
 */
export type ModelSelectionHandler = (model: ModelOption) => void;

export class ModelSelector {
  private container: BoxRenderable;
  private labelText: TextRenderable;
  private selectList: SelectRenderable;
  private detailsText: TextRenderable;
  private models: ModelOption[];
  private selectHandlers: ModelSelectionHandler[] = [];

  constructor(renderer: RenderContext, options: ModelSelectorOptions) {
    this.models = options.models;

    // Create container
    this.container = new BoxRenderable(renderer, {
      id: `${options.id}-container`,
      flexDirection: 'column',
      width: (options.width || '100%') as number | `${number}%`,
      gap: 1,
    });

    // Create label
    this.labelText = new TextRenderable(renderer, {
      id: `${options.id}-label`,
      content: options.label,
    });

    // Convert models to select options
    const selectOptions: SelectOption[] = options.models.map((model) => ({
      name: model.name,
      description: model.description,
      value: model.modelId,
    }));

    // Create select list
    this.selectList = new SelectRenderable(renderer, {
      id: `${options.id}-select`,
      options: selectOptions,
      width: '100%',
      height: options.height || 6,
      showScrollIndicator: true,
    });

    // Create details text for selected model
    this.detailsText = new TextRenderable(renderer, {
      id: `${options.id}-details`,
      content: '',
    });

    // Set initial selection
    const initialIndex = options.initialSelectedIndex ?? 0;
    this.selectList.selectedIndex = initialIndex;
    this.updateDetailsText(initialIndex);

    // Wire up events - use ITEM_SELECTED for navigation changes
    this.selectList.on(
      SelectRenderableEvents.ITEM_SELECTED,
      (index: number) => {
        this.updateDetailsText(index);
        this.notifySelectHandlers(index);
      }
    );


    // Assemble component
    this.container.add(this.labelText);
    this.container.add(this.selectList);
    this.container.add(this.detailsText);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Focus the select list.
   */
  focus(): void {
    this.selectList.focus();
  }

  /**
   * Get currently selected model.
   */
  getSelectedModel(): ModelOption | null {
    const index = this.selectList.selectedIndex;
    return index >= 0 && index < this.models.length ? this.models[index] : null;
  }

  /**
   * Get selected model ID.
   */
  getSelectedModelId(): string {
    const model = this.getSelectedModel();
    return model ? model.modelId : '';
  }

  /**
   * Set selected index programmatically.
   */
  setSelectedIndex(index: number): void {
    this.selectList.selectedIndex = index;
    this.updateDetailsText(index);
  }

  /**
   * Register handler for model selection.
   */
  onSelect(handler: ModelSelectionHandler): () => void {
    this.selectHandlers.push(handler);
    return () => {
      this.selectHandlers = this.selectHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Update details text for selected model.
   */
  private updateDetailsText(index: number): void {
    if (index >= 0 && index < this.models.length) {
      const model = this.models[index];
      let details = `Model: ${model.modelId}`;
      if (model.parameters) {
        details += ` | ${model.parameters}`;
      }
      this.detailsText.content = details;
    } else {
      this.detailsText.content = '';
    }
  }

  /**
   * Notify select handlers.
   */
  private notifySelectHandlers(index: number): void {
    if (index >= 0 && index < this.models.length) {
      const model = this.models[index];
      for (const handler of this.selectHandlers) {
        handler(model);
      }
    }
  }
}
