#!/usr/bin/env node
"use strict";
/**
 * CLI entry point.
 * Initializes commander and launches the main CLI application.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const commander_1 = require("commander");
const cli_1 = require("./cli");
// CLI Setup
const program = new commander_1.Command();
program
    .name('synth-data')
    .description('Interactive CLI for synthetic data generation and model fine-tuning')
    .version('1.1.0')
    .action(async () => {
    const cli = new cli_1.SyntheticDataCLI();
    await cli.run();
});
program.parse();
//# sourceMappingURL=index.js.map