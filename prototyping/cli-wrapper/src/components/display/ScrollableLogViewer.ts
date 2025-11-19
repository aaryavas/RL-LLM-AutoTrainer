/**
 * ScrollableLogViewer - Scrollable container for process output logs.
 * Single Responsibility: Display and scroll through log lines.
 */

import {
  ScrollBoxRenderable,
  TextRenderable,
  type RenderContext,
} from '@opentui/core';

/**
 * Configuration for ScrollableLogViewer.
 */
export interface ScrollableLogViewerOptions {
  id: string;
  width?: number | string;
  height?: number | string;
  autoScroll?: boolean;
}

export class ScrollableLogViewer {
  private scrollBox: ScrollBoxRenderable;
  private logText: TextRenderable;
  private logLines: string[] = [];
  private autoScroll: boolean;
  private maxLogLines: number = 1000;

  constructor(renderer: RenderContext, options: ScrollableLogViewerOptions) {
    this.autoScroll = options.autoScroll ?? true;

    // Create scroll box
    this.scrollBox = new ScrollBoxRenderable(renderer, {
      id: `${options.id}-scrollbox`,
      rootOptions: {
        width: (options.width || '100%') as number | `${number}%`,
        height: (options.height || '100%') as number | `${number}%`,
      },
      contentOptions: {
        flexDirection: 'column',
        padding: 1,
      },
      scrollbarOptions: {
        showArrows: false,
      },
    });

    // Create text renderable for log content
    this.logText = new TextRenderable(renderer, {
      id: `${options.id}-text`,
      content: '',
    });

    this.scrollBox.content.add(this.logText);
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): ScrollBoxRenderable {
    return this.scrollBox;
  }

  /**
   * Focus for keyboard scrolling.
   */
  focus(): void {
    this.scrollBox.focus();
  }

  /**
   * Append a line to the log.
   */
  appendLine(line: string): void {
    this.logLines.push(line);

    // Trim if exceeds max lines
    if (this.logLines.length > this.maxLogLines) {
      this.logLines = this.logLines.slice(-this.maxLogLines);
    }

    this.updateLogContent();
  }

  /**
   * Append multiple lines.
   */
  appendLines(lines: string[]): void {
    for (const line of lines) {
      this.logLines.push(line);
    }

    // Trim if exceeds max lines
    if (this.logLines.length > this.maxLogLines) {
      this.logLines = this.logLines.slice(-this.maxLogLines);
    }

    this.updateLogContent();
  }

  /**
   * Clear all log content.
   */
  clearLog(): void {
    this.logLines = [];
    this.updateLogContent();
  }

  /**
   * Get all log lines.
   */
  getLogLines(): string[] {
    return [...this.logLines];
  }

  /**
   * Get log as single string.
   */
  getLogContent(): string {
    return this.logLines.join('\n');
  }

  /**
   * Set auto-scroll behavior.
   */
  setAutoScroll(enabled: boolean): void {
    this.autoScroll = enabled;
  }

  /**
   * Set maximum number of log lines to retain.
   */
  setMaxLogLines(max: number): void {
    this.maxLogLines = max;
    if (this.logLines.length > max) {
      this.logLines = this.logLines.slice(-max);
      this.updateLogContent();
    }
  }

  /**
   * Update the log text content.
   */
  private updateLogContent(): void {
    this.logText.content = this.logLines.join('\n');

    if (this.autoScroll) {
      // Scroll to bottom by setting y to a large value
      this.scrollBox.scrollTo({ x: 0, y: Number.MAX_SAFE_INTEGER });
    }
  }
}
