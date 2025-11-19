/**
 * EditableItemList - List with add/remove item functionality.
 * Single Responsibility: Manage a dynamic list of items with input field.
 */

import {
  BoxRenderable,
  TextRenderable,
  InputRenderable,
  InputRenderableEvents,
  type RenderContext,
} from '@opentui/core';
import { ItemListEntry } from './ItemListEntry';

/**
 * Configuration for EditableItemList.
 */
export interface EditableItemListOptions {
  id: string;
  label: string;
  placeholder?: string;
  initialItems?: string[];
  width?: number | string;
  minimumItems?: number;
}

/**
 * Callback for list changes.
 */
export type ItemListChangeHandler = (items: string[]) => void;

export class EditableItemList {
  private renderer: RenderContext;
  private container: BoxRenderable;
  private labelText: TextRenderable;
  private inputField: InputRenderable;
  private itemsContainer: BoxRenderable;
  private hintText: TextRenderable;
  private itemEntries: ItemListEntry[] = [];
  private items: string[] = [];
  private changeHandlers: ItemListChangeHandler[] = [];
  private options: EditableItemListOptions;

  constructor(renderer: RenderContext, options: EditableItemListOptions) {
    this.renderer = renderer;
    this.options = options;

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

    // Create hint text
    this.hintText = new TextRenderable(renderer, {
      id: `${options.id}-hint`,
      content: 'Enter to add | Type "done" to finish',
    });

    // Create input field for adding items
    this.inputField = new InputRenderable(renderer, {
      id: `${options.id}-input`,
      placeholder: options.placeholder || 'Enter item...',
      width: '100%',
      height: 1,
    });

    // Create container for item entries
    this.itemsContainer = new BoxRenderable(renderer, {
      id: `${options.id}-items`,
      flexDirection: 'column',
      width: '100%',
      gap: 0,
    });

    // Wire up input events
    this.inputField.on(InputRenderableEvents.CHANGE, (value: string) => {
      this.handleInputSubmit(value);
    });

    // Initialize with initial items
    if (options.initialItems) {
      for (const item of options.initialItems) {
        this.addItem(item);
      }
    }

    // Assemble component
    this.container.add(this.labelText);
    this.container.add(this.hintText);
    this.container.add(this.inputField);
    this.container.add(this.itemsContainer);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Focus the input field.
   */
  focus(): void {
    this.inputField.focus();
  }

  /**
   * Get all items.
   */
  getItems(): string[] {
    return [...this.items];
  }

  /**
   * Add an item to the list.
   */
  addItem(text: string): void {
    if (!text.trim()) return;

    const trimmedText = text.trim();
    this.items.push(trimmedText);

    const entry = new ItemListEntry(this.renderer, {
      id: `${this.options.id}-item-${this.items.length - 1}`,
      text: trimmedText,
      index: this.items.length - 1,
    });

    entry.onDelete((index) => {
      this.removeItemAtIndex(index);
    });

    this.itemEntries.push(entry);
    this.itemsContainer.add(entry.getRenderable());
    this.notifyChangeHandlers();
  }

  /**
   * Remove item at index.
   */
  removeItemAtIndex(index: number): void {
    const minimumItems = this.options.minimumItems ?? 0;
    if (this.items.length <= minimumItems) {
      return; // Don't allow removal below minimum
    }

    if (index >= 0 && index < this.items.length) {
      this.items.splice(index, 1);

      const entry = this.itemEntries[index];
      this.itemsContainer.remove(entry.getRenderable().id);
      this.itemEntries.splice(index, 1);

      // Update indices of remaining entries
      for (let i = index; i < this.itemEntries.length; i++) {
        this.itemEntries[i].setIndex(i);
      }

      this.notifyChangeHandlers();
    }
  }

  /**
   * Clear all items.
   */
  clearItems(): void {
    for (const entry of this.itemEntries) {
      this.itemsContainer.remove(entry.getRenderable().id);
    }
    this.items = [];
    this.itemEntries = [];
    this.notifyChangeHandlers();
  }

  /**
   * Check if input indicates completion ("done").
   */
  isCompletionInput(value: string): boolean {
    return value.toLowerCase().trim() === 'done';
  }

  /**
   * Register handler for list changes.
   */
  onChange(handler: ItemListChangeHandler): () => void {
    this.changeHandlers.push(handler);
    return () => {
      this.changeHandlers = this.changeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Handle input submission.
   */
  private handleInputSubmit(value: string): void {
    if (this.isCompletionInput(value)) {
      // Clear input, don't add "done" as item
      this.inputField.value = '';
      return;
    }

    if (value.trim()) {
      this.addItem(value);
      this.inputField.value = '';
    }
  }

  /**
   * Notify change handlers.
   */
  private notifyChangeHandlers(): void {
    const items = this.getItems();
    for (const handler of this.changeHandlers) {
      handler(items);
    }
  }
}
