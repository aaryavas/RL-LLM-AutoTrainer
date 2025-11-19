/**
 * KeyboardNavigator - Handle keyboard navigation between focusable elements.
 * Single Responsibility: Manage focus cycling through Tab/Shift+Tab.
 */

import { type Renderable, type KeyEvent } from '@opentui/core';

/**
 * Focusable element interface.
 */
export interface FocusableElement {
  focus: () => void;
  getRenderable?: () => Renderable;
}

/**
 * Callback for navigation events.
 */
export type NavigationHandler = (
  previousIndex: number,
  currentIndex: number
) => void;

export class KeyboardNavigator {
  private focusableElements: FocusableElement[] = [];
  private currentFocusIndex: number = 0;
  private navigationHandlers: NavigationHandler[] = [];
  private wrapAround: boolean;

  constructor(wrapAround: boolean = true) {
    this.wrapAround = wrapAround;
  }

  /**
   * Add a focusable element.
   */
  addElement(element: FocusableElement): void {
    this.focusableElements.push(element);
  }

  /**
   * Remove a focusable element.
   */
  removeElement(element: FocusableElement): void {
    const index = this.focusableElements.indexOf(element);
    if (index > -1) {
      this.focusableElements.splice(index, 1);
      if (this.currentFocusIndex >= this.focusableElements.length) {
        this.currentFocusIndex = Math.max(0, this.focusableElements.length - 1);
      }
    }
  }

  /**
   * Clear all elements.
   */
  clearElements(): void {
    this.focusableElements = [];
    this.currentFocusIndex = 0;
  }

  /**
   * Set elements from array.
   */
  setElements(elements: FocusableElement[]): void {
    this.focusableElements = [...elements];
    this.currentFocusIndex = 0;
  }

  /**
   * Handle keyboard event for navigation.
   */
  handleKeyEvent(keyEvent: KeyEvent): boolean {
    if (this.focusableElements.length === 0) {
      return false;
    }

    if (keyEvent.name === 'tab') {
      if (keyEvent.shift) {
        this.focusPrevious();
      } else {
        this.focusNext();
      }
      return true;
    }

    return false;
  }

  /**
   * Focus next element.
   */
  focusNext(): void {
    if (this.focusableElements.length === 0) return;

    const previousIndex = this.currentFocusIndex;

    if (this.currentFocusIndex < this.focusableElements.length - 1) {
      this.currentFocusIndex++;
    } else if (this.wrapAround) {
      this.currentFocusIndex = 0;
    }

    this.focusCurrentElement();
    this.notifyNavigationHandlers(previousIndex, this.currentFocusIndex);
  }

  /**
   * Focus previous element.
   */
  focusPrevious(): void {
    if (this.focusableElements.length === 0) return;

    const previousIndex = this.currentFocusIndex;

    if (this.currentFocusIndex > 0) {
      this.currentFocusIndex--;
    } else if (this.wrapAround) {
      this.currentFocusIndex = this.focusableElements.length - 1;
    }

    this.focusCurrentElement();
    this.notifyNavigationHandlers(previousIndex, this.currentFocusIndex);
  }

  /**
   * Focus element at specific index.
   */
  focusAtIndex(index: number): void {
    if (index < 0 || index >= this.focusableElements.length) return;

    const previousIndex = this.currentFocusIndex;
    this.currentFocusIndex = index;
    this.focusCurrentElement();
    this.notifyNavigationHandlers(previousIndex, this.currentFocusIndex);
  }

  /**
   * Focus the first element.
   */
  focusFirst(): void {
    this.focusAtIndex(0);
  }

  /**
   * Focus the last element.
   */
  focusLast(): void {
    this.focusAtIndex(this.focusableElements.length - 1);
  }

  /**
   * Get current focus index.
   */
  getCurrentFocusIndex(): number {
    return this.currentFocusIndex;
  }

  /**
   * Get current focused element.
   */
  getCurrentElement(): FocusableElement | null {
    if (this.currentFocusIndex < this.focusableElements.length) {
      return this.focusableElements[this.currentFocusIndex];
    }
    return null;
  }

  /**
   * Get total number of focusable elements.
   */
  getElementCount(): number {
    return this.focusableElements.length;
  }

  /**
   * Register handler for navigation events.
   */
  onNavigate(handler: NavigationHandler): () => void {
    this.navigationHandlers.push(handler);
    return () => {
      this.navigationHandlers = this.navigationHandlers.filter(
        (h) => h !== handler
      );
    };
  }

  /**
   * Focus the current element.
   */
  private focusCurrentElement(): void {
    const element = this.focusableElements[this.currentFocusIndex];
    if (element && element.focus) {
      element.focus();
    }
  }

  /**
   * Notify navigation handlers.
   */
  private notifyNavigationHandlers(
    previousIndex: number,
    currentIndex: number
  ): void {
    for (const handler of this.navigationHandlers) {
      handler(previousIndex, currentIndex);
    }
  }
}
