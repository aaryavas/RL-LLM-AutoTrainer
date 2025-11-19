#!/usr/bin/env node

/**
 * CLI entry point.
 * Initializes commander and launches the main CLI application.
 */

import { Command } from 'commander';
import { SyntheticDataCLI } from './cli';

// CLI Setup
const program = new Command();

program
  .name('synth-data')
  .description('Interactive CLI for synthetic data generation and model fine-tuning')
  .version('1.1.0')
  .action(async () => {
    const cli = new SyntheticDataCLI();
    await cli.run();
  });

program.parse();
