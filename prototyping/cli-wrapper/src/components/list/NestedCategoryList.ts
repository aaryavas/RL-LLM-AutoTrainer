/**
 * NestedCategoryList - Categories with nested types list.
 * Single Responsibility: Manage two-level category/type hierarchy.
 */

import {
  BoxRenderable,
  TextRenderable,
  InputRenderable,
  InputRenderableEvents,
  type RenderContext,
  type Renderable,
} from '@opentui/core';

/**
 * Category with types structure.
 */
export interface CategoryWithTypes {
  name: string;
  types: string[];
}

/**
 * Configuration for NestedCategoryList.
 */
export interface NestedCategoryListOptions {
  id: string;
  label: string;
  categoryPlaceholder?: string;
  typePlaceholder?: string;
  initialCategories?: CategoryWithTypes[];
  width?: number | string;
}

/**
 * Mode of operation.
 */
type InputMode = 'category' | 'type';

/**
 * Callback for category changes.
 */
export type CategoryListChangeHandler = (categories: CategoryWithTypes[]) => void;

export class NestedCategoryList {
  private renderer: RenderContext;
  private container: BoxRenderable;
  private labelText: TextRenderable;
  private inputField: InputRenderable;
  private modeText: TextRenderable;
  private categoriesContainer: BoxRenderable;
  private categoryDisplays: BoxRenderable[] = [];
  private categories: CategoryWithTypes[] = [];
  private currentMode: InputMode = 'category';
  private currentCategory: string = '';
  private changeHandlers: CategoryListChangeHandler[] = [];
  private options: NestedCategoryListOptions;

  constructor(renderer: RenderContext, options: NestedCategoryListOptions) {
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

    // Create mode indicator
    this.modeText = new TextRenderable(renderer, {
      id: `${options.id}-mode`,
      content: 'Enter category name (or "done" to finish):',
    });

    // Create input field
    this.inputField = new InputRenderable(renderer, {
      id: `${options.id}-input`,
      placeholder: options.categoryPlaceholder || 'Enter category...',
      width: '100%',
      height: 1,
    });

    // Create container for category display
    this.categoriesContainer = new BoxRenderable(renderer, {
      id: `${options.id}-categories`,
      flexDirection: 'column',
      width: '100%',
      gap: 0,
    });

    // Wire up input events
    this.inputField.on(InputRenderableEvents.CHANGE, (value: string) => {
      this.handleInputSubmit(value);
    });

    // Initialize with initial categories
    if (options.initialCategories) {
      for (const category of options.initialCategories) {
        this.categories.push({ ...category });
      }
      this.refreshCategoriesDisplay();
    }

    // Assemble component
    this.container.add(this.labelText);
    this.container.add(this.modeText);
    this.container.add(this.inputField);
    this.container.add(this.categoriesContainer);
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
   * Get all categories.
   */
  getCategories(): CategoryWithTypes[] {
    return this.categories.map((cat) => ({
      name: cat.name,
      types: [...cat.types],
    }));
  }

  /**
   * Check if done entering all categories.
   */
  isComplete(): boolean {
    return (
      this.currentMode === 'category' &&
      this.categories.length > 0 &&
      this.currentCategory === ''
    );
  }

  /**
   * Register handler for changes.
   */
  onChange(handler: CategoryListChangeHandler): () => void {
    this.changeHandlers.push(handler);
    return () => {
      this.changeHandlers = this.changeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Handle input submission.
   */
  private handleInputSubmit(value: string): void {
    const trimmedValue = value.trim();

    if (trimmedValue.toLowerCase() === 'done') {
      this.handleDoneInput();
      return;
    }

    if (!trimmedValue) {
      return;
    }

    if (this.currentMode === 'category') {
      this.handleCategoryInput(trimmedValue);
    } else {
      this.handleTypeInput(trimmedValue);
    }

    this.inputField.value = '';
  }

  /**
   * Handle "done" input.
   */
  private handleDoneInput(): void {
    if (this.currentMode === 'type') {
      // Finish current category, switch back to category mode
      this.switchToCategoryMode();
    }
    this.inputField.value = '';
  }

  /**
   * Handle category name input.
   */
  private handleCategoryInput(categoryName: string): void {
    this.currentCategory = categoryName;
    this.categories.push({
      name: categoryName,
      types: [],
    });

    this.switchToTypeMode(categoryName);
    this.refreshCategoriesDisplay();
    this.notifyChangeHandlers();
  }

  /**
   * Handle type input for current category.
   */
  private handleTypeInput(typeName: string): void {
    const currentCategoryIndex = this.categories.length - 1;
    if (currentCategoryIndex >= 0) {
      this.categories[currentCategoryIndex].types.push(typeName);
      this.refreshCategoriesDisplay();
      this.notifyChangeHandlers();
    }
  }

  /**
   * Switch to category input mode.
   */
  private switchToCategoryMode(): void {
    this.currentMode = 'category';
    this.currentCategory = '';
    this.modeText.content = 'Enter category name (or "done" to finish):';
    this.inputField.placeholder =
      this.options.categoryPlaceholder || 'Enter category...';
  }

  /**
   * Switch to type input mode.
   */
  private switchToTypeMode(categoryName: string): void {
    this.currentMode = 'type';
    this.modeText.content = `Enter types for "${categoryName}" (or "done" to finish category):`;
    this.inputField.placeholder =
      this.options.typePlaceholder || 'Enter type...';
  }

  /**
   * Refresh the display of all categories.
   */
  private refreshCategoriesDisplay(): void {
    // Clear existing display
    for (const display of this.categoryDisplays) {
      this.categoriesContainer.remove(display.id);
    }
    this.categoryDisplays = [];

    // Add category displays
    for (let i = 0; i < this.categories.length; i++) {
      const category = this.categories[i];
      const categoryBox = this.createCategoryDisplay(category, i);
      this.categoryDisplays.push(categoryBox);
      this.categoriesContainer.add(categoryBox);
    }
  }

  /**
   * Create display for a single category.
   */
  private createCategoryDisplay(
    category: CategoryWithTypes,
    index: number
  ): BoxRenderable {
    const categoryBox = new BoxRenderable(this.renderer, {
      id: `${this.options.id}-category-${index}`,
      flexDirection: 'column',
      width: '100%',
      padding: 0,
    });

    // Category name
    const nameText = new TextRenderable(this.renderer, {
      id: `${this.options.id}-category-${index}-name`,
      content: `${index + 1}. ${category.name}:`,
    });
    categoryBox.add(nameText);

    // Types
    if (category.types.length > 0) {
      const typesText = new TextRenderable(this.renderer, {
        id: `${this.options.id}-category-${index}-types`,
        content: `   Types: ${category.types.join(', ')}`,
      });
      categoryBox.add(typesText);
    }

    return categoryBox;
  }

  /**
   * Notify change handlers.
   */
  private notifyChangeHandlers(): void {
    const categories = this.getCategories();
    for (const handler of this.changeHandlers) {
      handler(categories);
    }
  }
}
