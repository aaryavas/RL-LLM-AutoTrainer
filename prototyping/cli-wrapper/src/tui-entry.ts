#!/usr/bin/env node
/**
 * TUI Entry Point - Launch the TUI-based CLI application.
 * Single Responsibility: Bootstrap and run the TUI application.
 */

import { main } from './TuiApplication';

// Run the TUI application
main().catch((error) => {
  console.error('Failed to start TUI application:', error);
  process.exit(1);
});
