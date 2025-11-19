/**
 * ProcessOutputParser - Parses progress and metrics from Python output.
 * Single Responsibility: Parse and interpret process output patterns.
 */

import { BatchProgress } from '../state/types';

/**
 * Parsed training metrics from vblora output.
 */
export interface TrainingMetrics {
  epoch: number;
  totalEpochs: number;
  loss: number;
  learningRate: number;
}

/**
 * Classification of a log line.
 */
export type LineClassification = 'progress' | 'error' | 'warning' | 'info';

export class ProcessOutputParser {
  // Pattern for data-gen.py batch progress: "⚡ Saved batch number 3/5"
  private readonly BATCH_PROGRESS_PATTERN = /[⚡]\s*Saved batch number (\d+)\/(\d+)/;

  // Pattern for rocket emoji progress: "🚀 Synthetic data will be appended..."
  private readonly SYNTHETIC_DATA_START = /🚀\s*Synthetic data will be appended.*in (\d+) batch/;

  // Pattern for training epoch progress
  private readonly EPOCH_PROGRESS_PATTERN = /Epoch\s+(\d+)\/(\d+)/i;

  // Pattern for loss values
  private readonly LOSS_PATTERN = /loss[:\s]+([0-9.]+)/i;

  // Pattern for learning rate
  private readonly LR_PATTERN = /lr[:\s]+([0-9.e-]+)/i;

  // Error indicators
  private readonly ERROR_INDICATORS = [
    'error',
    'exception',
    'traceback',
    'failed',
    '❌',
  ];

  // Warning indicators
  private readonly WARNING_INDICATORS = ['warning', '⚠️', 'warn'];

  /**
   * Parse batch progress from a log line.
   */
  parseBatchProgress(line: string): BatchProgress | null {
    const match = line.match(this.BATCH_PROGRESS_PATTERN);
    if (match) {
      const currentBatch = parseInt(match[1], 10);
      const totalBatches = parseInt(match[2], 10);
      return {
        currentBatch,
        totalBatches,
        percentComplete: Math.round((currentBatch / totalBatches) * 100),
      };
    }
    return null;
  }

  /**
   * Parse total batches from initial synthetic data message.
   */
  parseTotalBatches(line: string): number | null {
    const match = line.match(this.SYNTHETIC_DATA_START);
    if (match) {
      return parseInt(match[1], 10);
    }
    return null;
  }

  /**
   * Parse training metrics from a log line.
   */
  parseTrainingMetrics(line: string): Partial<TrainingMetrics> | null {
    const epochMatch = line.match(this.EPOCH_PROGRESS_PATTERN);
    const lossMatch = line.match(this.LOSS_PATTERN);
    const lrMatch = line.match(this.LR_PATTERN);

    if (!epochMatch && !lossMatch && !lrMatch) {
      return null;
    }

    const metrics: Partial<TrainingMetrics> = {};

    if (epochMatch) {
      metrics.epoch = parseInt(epochMatch[1], 10);
      metrics.totalEpochs = parseInt(epochMatch[2], 10);
    }

    if (lossMatch) {
      metrics.loss = parseFloat(lossMatch[1]);
    }

    if (lrMatch) {
      metrics.learningRate = parseFloat(lrMatch[1]);
    }

    return metrics;
  }

  /**
   * Classify a log line by its content.
   */
  classifyLine(line: string): LineClassification {
    const lowerLine = line.toLowerCase();

    for (const indicator of this.ERROR_INDICATORS) {
      if (lowerLine.includes(indicator)) {
        return 'error';
      }
    }

    for (const indicator of this.WARNING_INDICATORS) {
      if (lowerLine.includes(indicator)) {
        return 'warning';
      }
    }

    if (this.parseBatchProgress(line) || this.parseTrainingMetrics(line)) {
      return 'progress';
    }

    return 'info';
  }

  /**
   * Check if a line indicates an error.
   */
  isErrorLine(line: string): boolean {
    return this.classifyLine(line) === 'error';
  }

  /**
   * Check if a line indicates a warning.
   */
  isWarningLine(line: string): boolean {
    return this.classifyLine(line) === 'warning';
  }

  /**
   * Extract error message from a line.
   */
  extractErrorMessage(line: string): string {
    // Remove common prefixes
    let message = line
      .replace(/^(error|exception|failed)[:\s]*/i, '')
      .replace(/^❌\s*/, '')
      .trim();

    return message || line;
  }
}
