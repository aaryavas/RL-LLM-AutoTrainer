/**
 * TerminalSizeValidator - Validate terminal dimensions for TUI display.
 * Single Responsibility: Check and report on terminal size requirements.
 */

/**
 * Terminal dimensions.
 */
export interface TerminalDimensions {
  columns: number;
  rows: number;
}

/**
 * Validation result.
 */
export interface SizeValidationResult {
  isValid: boolean;
  currentDimensions: TerminalDimensions;
  minimumDimensions: TerminalDimensions;
  errorMessage?: string;
}

export class TerminalSizeValidator {
  private minimumColumns: number;
  private minimumRows: number;

  constructor(minimumColumns: number = 80, minimumRows: number = 24) {
    this.minimumColumns = minimumColumns;
    this.minimumRows = minimumRows;
  }

  /**
   * Get current terminal dimensions.
   */
  getCurrentDimensions(): TerminalDimensions {
    return {
      columns: process.stdout.columns || 80,
      rows: process.stdout.rows || 24,
    };
  }

  /**
   * Validate that terminal meets minimum size requirements.
   */
  validate(): SizeValidationResult {
    const currentDimensions = this.getCurrentDimensions();
    const minimumDimensions = {
      columns: this.minimumColumns,
      rows: this.minimumRows,
    };

    const isColumnsValid = currentDimensions.columns >= this.minimumColumns;
    const isRowsValid = currentDimensions.rows >= this.minimumRows;
    const isValid = isColumnsValid && isRowsValid;

    let errorMessage: string | undefined;

    if (!isValid) {
      const issues: string[] = [];
      if (!isColumnsValid) {
        issues.push(`width ${currentDimensions.columns} < ${this.minimumColumns}`);
      }
      if (!isRowsValid) {
        issues.push(`height ${currentDimensions.rows} < ${this.minimumRows}`);
      }
      errorMessage = `Terminal too small: ${issues.join(', ')}. Minimum required: ${this.minimumColumns}x${this.minimumRows}`;
    }

    return {
      isValid,
      currentDimensions,
      minimumDimensions,
      errorMessage,
    };
  }

  /**
   * Check if terminal is a TTY (interactive terminal).
   */
  isTTY(): boolean {
    return process.stdout.isTTY === true;
  }

  /**
   * Get formatted size string.
   */
  getFormattedSize(): string {
    const dimensions = this.getCurrentDimensions();
    return `${dimensions.columns}x${dimensions.rows}`;
  }
}
