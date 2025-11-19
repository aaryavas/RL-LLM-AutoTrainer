/**
 * SingleSelectList - Wrapper for SelectRenderable with label support.
 * Single Responsibility: Provide labeled single-selection list component.
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
 * Configuration for SingleSelectList.
 */
export interface SingleSelectListOptions {
  id: string;
  label: string;
  options: SelectOption[];
  initialSelectedIndex?: number;
  width?: number | string;
  height?: number;
}

/**
 * Callback for selection changes.
 */
export type SelectionChangeHandler = (
  index: number,
  option: SelectOption
) => void;

export class SingleSelectList {
  private container: BoxRenderable;
  private labelText: TextRenderable;
  private selectList: SelectRenderable;
  private changeHandlers: SelectionChangeHandler[] = [];
  private selectHandlers: SelectionChangeHandler[] = [];

  constructor(renderer: RenderContext, options: SingleSelectListOptions) {
    // Create container
    this.container = new BoxRenderable(renderer, {
      id: `${options.id}-container`,
      flexDirection: 'column',
      width: (options.width || '100%') as number | `${number}%`,
      gap: 0,
    });

    // Create label
    this.labelText = new TextRenderable(renderer, {
      id: `${options.id}-label`,
      content: options.label,
    });

    // Create select list
    this.selectList = new SelectRenderable(renderer, {
      id: `${options.id}-select`,
      options: options.options,
      width: '100%',
      height: options.height || 8,
      showScrollIndicator: true,
    });

    // Set initial selection
    if (options.initialSelectedIndex !== undefined) {
      this.selectList.selectedIndex = options.initialSelectedIndex;
    }

    // Wire up events - use ITEM_SELECTED for both navigation and selection
    this.selectList.on(
      SelectRenderableEvents.ITEM_SELECTED,
      (index: number, option: SelectOption) => {
        this.notifyChangeHandlers(index, option);
        this.notifySelectHandlers(index, option);
      }
    );

    // Assemble component
    this.container.add(this.labelText);
    this.container.add(this.selectList);
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
   * Get currently selected index.
   */
  getSelectedIndex(): number {
    return this.selectList.selectedIndex;
  }

  /**
   * Get currently selected option.
   */
  getSelectedOption(): SelectOption | null {
    const index = this.selectList.selectedIndex;
    const options = this.selectList.options;
    return index >= 0 && index < options.length ? options[index] : null;
  }

  /**
   * Set selected index programmatically.
   */
  setSelectedIndex(index: number): void {
    this.selectList.selectedIndex = index;
  }

  /**
   * Update options list.
   */
  setOptions(options: SelectOption[]): void {
    this.selectList.options = options;
  }

  /**
   * Register handler for selection changes (navigation).
   */
  onChange(handler: SelectionChangeHandler): () => void {
    this.changeHandlers.push(handler);
    return () => {
      this.changeHandlers = this.changeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Register handler for item selection (Enter pressed).
   */
  onSelect(handler: SelectionChangeHandler): () => void {
    this.selectHandlers.push(handler);
    return () => {
      this.selectHandlers = this.selectHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Notify change handlers.
   */
  private notifyChangeHandlers(index: number, option: SelectOption): void {
    for (const handler of this.changeHandlers) {
      handler(index, option);
    }
  }

  /**
   * Notify select handlers.
   */
  private notifySelectHandlers(index: number, option: SelectOption): void {
    for (const handler of this.selectHandlers) {
      handler(index, option);
    }
  }
}
