/**
 * ProcessExecutionScreen - Display running process with live output.
 * Single Responsibility: Show process logs and progress during execution.
 */

import {
  BoxRenderable,
  TextRenderable,
  type RenderContext,
} from '@opentui/core';
import { ScrollableLogViewer } from '../../components/display/ScrollableLogViewer';
import { ProgressIndicator } from '../../components/display/ProgressIndicator';
import { ProcessEventEmitter } from '../../process/BufferedProcessSpawner';
import { ProcessOutputParser } from '../../process/ProcessOutputParser';

/**
 * Callback for execution completion.
 */
export type ExecutionCompleteHandler = (success: boolean, exitCode: number | null) => void;

export class ProcessExecutionScreen {
  private container: BoxRenderable;
  private statusText: TextRenderable;
  private progressIndicator: ProgressIndicator;
  private logViewer: ScrollableLogViewer;
  private outputParser: ProcessOutputParser;
  private completeHandlers: ExecutionCompleteHandler[] = [];
  private isRunning: boolean = false;

  constructor(renderer: RenderContext) {
    this.outputParser = new ProcessOutputParser();

    // Create container
    this.container = new BoxRenderable(renderer, {
      id: 'process-execution-screen',
      flexDirection: 'column',
      width: '100%',
      height: '100%',
      gap: 1,
    });

    // Create status text
    this.statusText = new TextRenderable(renderer, {
      id: 'execution-status',
      content: 'Waiting to start...',
    });

    // Create progress indicator
    this.progressIndicator = new ProgressIndicator(renderer, {
      id: 'execution-progress',
      barWidth: 40,
    });

    // Create log viewer
    this.logViewer = new ScrollableLogViewer(renderer, {
      id: 'execution-log',
      height: '80%',
      autoScroll: true,
    });

    // Assemble screen
    this.container.add(this.statusText);
    this.container.add(this.progressIndicator.getRenderable());
    this.container.add(this.logViewer.getRenderable());
  }

  /**
   * Get the renderable container.
   */
  getRenderable(): BoxRenderable {
    return this.container;
  }

  /**
   * Focus the log viewer.
   */
  focus(): void {
    this.logViewer.focus();
  }

  /**
   * Attach to a process event emitter.
   */
  attachToProcess(processEmitter: ProcessEventEmitter, processName: string): void {
    this.isRunning = true;
    this.statusText.content = `Running ${processName}...`;
    this.logViewer.clearLog();
    this.progressIndicator.reset();

    // Handle log lines
    processEmitter.on('line', (line: string, stream: 'stdout' | 'stderr') => {
      this.handleLogLine(line, stream);
    });

    // Handle process exit
    processEmitter.on('exit', (code: number | null, signal: string | null) => {
      this.handleProcessExit(code, signal);
    });

    // Handle errors
    processEmitter.on('error', (error: Error) => {
      this.logViewer.appendLine(`Error: ${error.message}`);
      this.statusText.content = `Error: ${error.message}`;
    });
  }

  /**
   * Check if process is running.
   */
  getIsRunning(): boolean {
    return this.isRunning;
  }

  /**
   * Register handler for completion.
   */
  onComplete(handler: ExecutionCompleteHandler): () => void {
    this.completeHandlers.push(handler);
    return () => {
      this.completeHandlers = this.completeHandlers.filter((h) => h !== handler);
    };
  }

  /**
   * Handle a log line.
   */
  private handleLogLine(line: string, stream: 'stdout' | 'stderr'): void {
    // Append to log viewer
    const prefix = stream === 'stderr' ? '[ERR] ' : '';
    this.logViewer.appendLine(prefix + line);

    // Check for progress updates
    const batchProgress = this.outputParser.parseBatchProgress(line);
    if (batchProgress) {
      this.progressIndicator.setLabel(`Batch ${batchProgress.currentBatch}/${batchProgress.totalBatches}`);
      this.progressIndicator.setProgress(
        batchProgress.currentBatch,
        batchProgress.totalBatches
      );
    }

    // Check for total batches initialization
    const totalBatches = this.outputParser.parseTotalBatches(line);
    if (totalBatches) {
      this.progressIndicator.setProgress(0, totalBatches);
    }
  }

  /**
   * Handle process exit.
   */
  private handleProcessExit(code: number | null, signal: string | null): void {
    this.isRunning = false;

    const success = code === 0;

    if (success) {
      this.statusText.content = 'Process completed successfully';
      this.progressIndicator.setPercentage(100);
    } else {
      const reason = signal ? `signal ${signal}` : `code ${code}`;
      this.statusText.content = `Process failed with ${reason}`;
    }

    this.logViewer.appendLine('');
    this.logViewer.appendLine(success ? '✅ Complete' : `❌ Failed`);

    // Notify handlers
    for (const handler of this.completeHandlers) {
      handler(success, code);
    }
  }
}
