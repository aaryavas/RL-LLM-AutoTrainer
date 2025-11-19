/**
 * MainLayoutContainer - Provides header/content/footer layout structure.
 * Single Responsibility: Manage the main application layout with content swapping.
 */

import {
  BoxRenderable,
  TextRenderable,
  type CliRenderer,
  type Renderable,
} from '@opentui/core';

type DimensionValue = number | `${number}%`;

export class MainLayoutContainer {
  private renderer: CliRenderer;
  private rootContainer: BoxRenderable;
  private headerContainer: BoxRenderable;
  private contentContainer: BoxRenderable;
  private footerContainer: BoxRenderable;
  private headerText: TextRenderable;
  private footerText: TextRenderable;
  private contentChildren: Renderable[] = [];

  constructor(renderer: CliRenderer) {
    this.renderer = renderer;

    // Create root container (full screen)
    this.rootContainer = new BoxRenderable(renderer, {
      id: 'main-layout-root',
      width: '100%',
      height: '100%',
      flexDirection: 'column',
    });

    // Create header
    this.headerContainer = new BoxRenderable(renderer, {
      id: 'header-container',
      width: '100%',
      height: 3,
      borderStyle: 'single',
      justifyContent: 'center',
      alignItems: 'center',
    });

    this.headerText = new TextRenderable(renderer, {
      id: 'header-text',
      content: 'Synthetic Data Generator',
    });

    this.headerContainer.add(this.headerText);

    // Create content area (flexible)
    this.contentContainer = new BoxRenderable(renderer, {
      id: 'content-container',
      width: '100%',
      flexGrow: 1,
      flexDirection: 'column',
      padding: 1,
    });

    // Create footer
    this.footerContainer = new BoxRenderable(renderer, {
      id: 'footer-container',
      width: '100%',
      height: 2,
      borderStyle: 'single',
      justifyContent: 'center',
      alignItems: 'center',
    });

    this.footerText = new TextRenderable(renderer, {
      id: 'footer-text',
      content: 'Press Ctrl+C to exit',
    });

    this.footerContainer.add(this.footerText);

    // Assemble layout
    this.rootContainer.add(this.headerContainer);
    this.rootContainer.add(this.contentContainer);
    this.rootContainer.add(this.footerContainer);

    // Add to renderer root
    renderer.root.add(this.rootContainer);
  }

  /**
   * Set the header text.
   */
  setHeaderText(text: string): void {
    this.headerText.content = text;
  }

  /**
   * Set the footer text.
   */
  setFooterText(text: string): void {
    this.footerText.content = text;
  }

  /**
   * Clear the content area.
   */
  clearContent(): void {
    for (const child of this.contentChildren) {
      this.contentContainer.remove(child.id);
    }
    this.contentChildren = [];
  }

  /**
   * Set the content to display.
   */
  setContent(content: Renderable): void {
    this.clearContent();
    this.contentContainer.add(content);
    this.contentChildren.push(content);
  }

  /**
   * Add content to the content area.
   */
  addContent(content: Renderable): void {
    this.contentContainer.add(content);
    this.contentChildren.push(content);
  }

  /**
   * Get the content container for direct manipulation.
   */
  getContentContainer(): BoxRenderable {
    return this.contentContainer;
  }

  /**
   * Get the root container.
   */
  getRootContainer(): BoxRenderable {
    return this.rootContainer;
  }

  /**
   * Request a render update.
   */
  requestRender(): void {
    this.rootContainer.requestRender();
  }
}
