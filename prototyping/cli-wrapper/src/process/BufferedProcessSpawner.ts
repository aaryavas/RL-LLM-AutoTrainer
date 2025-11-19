/**
 * BufferedProcessSpawner - Spawns child processes with buffered I/O.
 * Single Responsibility: Spawn processes and emit line-by-line output events.
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';

/**
 * Events emitted by ProcessEventEmitter.
 */
export interface ProcessEvents {
  line: (line: string, stream: 'stdout' | 'stderr') => void;
  error: (error: Error) => void;
  exit: (code: number | null, signal: string | null) => void;
}

/**
 * Event emitter for process output.
 */
export class ProcessEventEmitter extends EventEmitter {
  private childProcess: ChildProcess | null = null;

  /**
   * Set the underlying child process.
   */
  setChildProcess(process: ChildProcess): void {
    this.childProcess = process;
  }

  /**
   * Kill the underlying process.
   */
  kill(signal: NodeJS.Signals = 'SIGTERM'): boolean {
    if (this.childProcess) {
      return this.childProcess.kill(signal);
    }
    return false;
  }

  /**
   * Check if process is still running.
   */
  isRunning(): boolean {
    return this.childProcess !== null && this.childProcess.exitCode === null;
  }
}

export class BufferedProcessSpawner {
  private workingDirectory: string;

  constructor(workingDirectory: string) {
    this.workingDirectory = workingDirectory;
  }

  /**
   * Spawn a process with buffered stdio that emits line events.
   */
  spawnProcess(command: string, args: string[]): ProcessEventEmitter {
    const emitter = new ProcessEventEmitter();

    const childProcess = spawn(command, args, {
      cwd: this.workingDirectory,
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: true,
    });

    emitter.setChildProcess(childProcess);

    this.attachStreamHandlers(childProcess, emitter);
    this.attachProcessHandlers(childProcess, emitter);

    return emitter;
  }

  /**
   * Spawn a Python script with unbuffered output (-u flag).
   */
  spawnPythonScript(scriptPath: string, args: string[]): ProcessEventEmitter {
    return this.spawnProcess('python', ['-u', scriptPath, ...args]);
  }

  /**
   * Spawn a Python script using uv run with unbuffered output.
   */
  spawnPythonWithUv(scriptPath: string, args: string[]): ProcessEventEmitter {
    return this.spawnProcess('uv', ['run', 'python', '-u', scriptPath, ...args]);
  }

  /**
   * Attach handlers for stdout and stderr streams.
   */
  private attachStreamHandlers(
    childProcess: ChildProcess,
    emitter: ProcessEventEmitter
  ): void {
    let stdoutBuffer = '';
    let stderrBuffer = '';

    if (childProcess.stdout) {
      childProcess.stdout.on('data', (data: Buffer) => {
        stdoutBuffer = this.processBuffer(
          stdoutBuffer + data.toString(),
          emitter,
          'stdout'
        );
      });
    }

    if (childProcess.stderr) {
      childProcess.stderr.on('data', (data: Buffer) => {
        stderrBuffer = this.processBuffer(
          stderrBuffer + data.toString(),
          emitter,
          'stderr'
        );
      });
    }
  }

  /**
   * Process buffer and emit complete lines.
   */
  private processBuffer(
    buffer: string,
    emitter: ProcessEventEmitter,
    stream: 'stdout' | 'stderr'
  ): string {
    const lines = buffer.split('\n');

    // Emit all complete lines
    for (let i = 0; i < lines.length - 1; i++) {
      const line = lines[i].trimEnd();
      if (line) {
        emitter.emit('line', line, stream);
      }
    }

    // Return incomplete line (or empty string)
    return lines[lines.length - 1];
  }

  /**
   * Attach handlers for process events.
   */
  private attachProcessHandlers(
    childProcess: ChildProcess,
    emitter: ProcessEventEmitter
  ): void {
    childProcess.on('error', (error: Error) => {
      emitter.emit('error', error);
    });

    childProcess.on('exit', (code: number | null, signal: string | null) => {
      emitter.emit('exit', code, signal);
    });
  }
}
