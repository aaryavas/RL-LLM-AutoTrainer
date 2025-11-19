/**
 * ItemListEntry - Single item in an editable list.
 * Single Responsibility: Display and handle interactions for one list item.
 */

import {
  BoxRenderable,
  TextRenderable,
  type RenderContext,
} from '@opentui/core';

/**
 * Configuration for ItemListEntry.
 */
export interface ItemListEntryOptions {
  id: string;
  text: string;
  index: number;
  showDeleteHint?: boolean;
}

/**
 * Callback for delete action.
 */
export type ItemDeleteHandler = (index: number) => void;

export class ItemListEntry {
  private container: BoxRenderable;
  private indexText: TextRenderable;
  private contentText: TextRenderable;
  private deleteHintText: TextRenderable;
  private itemIndex: number;
  private deleteHandlers: ItemDeleteHandler[] = [];

  constructor(renderer: RenderContext, options: ItemListEntryOptions) {
    this.itemIndex = options.index;

    // Create container with horizontal layout
    this.container = new BoxRenderable(renderer, {
      id: `${options.id}-container`,
      flexDirection: 'row',
      width: '100%',
      gap: 1,
      padding: 0,
    });

    // Create index indicator
    this.indexText = new TextRenderable(renderer, {
      id: `${options.id}-index`,
      content: `${options.index + 1}.`,
    });

    // Create content text
    this.contentText = new TextRenderable(renderer, {
      id: `${options.id}-content`,
      content: options.text,
    });

    // Create delete hint (shown when focused)
    this.deleteHintText = new TextRenderable(renderer, {
      id: `${options.id}-delete-hint`,
      content: options.showDeleteHint ? '[Del]' : '',
    });

    // Assemble component
    this.container.add(this.indexText);
    this.container.add(this.contentText);
    this.container.add(this.deleteHintText);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Get the item text.
   */
  getText(): string {
    return String(this.contentText.content);
  }

  /**
   * Set the item text.
   */
  setText(text: string): void {
    this.contentText.content = text;
  }

  /**
   * Get the item index.
   */
  getIndex(): number {
    return this.itemIndex;
  }

  /**
   * Update the item index.
   */
  setIndex(index: number): void {
    this.itemIndex = index;
    this.indexText.content = `${index + 1}.`;
  }

  /**
   * Show delete hint.
   */
  showDeleteHint(): void {
    this.deleteHintText.content = '[Del]';
  }

  /**
   * Hide delete hint.
   */
  hideDeleteHint(): void {
    this.deleteHintText.content = '';
  }

  /**
   * Register handler for delete action.
   */
  onDelete(handler: ItemDeleteHandler): () => void {
    this.deleteHandlers.push(handler);
    return () => {
      this.deleteHandlers = this.deleteHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Trigger delete action.
   */
  triggerDelete(): void {
    for (const handler of this.deleteHandlers) {
      handler(this.itemIndex);
    }
  }
}
