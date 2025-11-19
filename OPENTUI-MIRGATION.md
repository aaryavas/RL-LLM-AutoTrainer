# Repository Flow Analysis (current-cli)

### 1\. Runtime Flow

The current CLI (`current-cli.txt`) operates as a **linear, synchronous, imperative script**. It relies heavily on blocking I/O operations (`await inquirer...`) and hands over complete terminal control to child processes during execution phases.

**Execution Pipeline:**

1.  **Entrypoint (`index.ts`)**: Instantiates `SyntheticDataCLI` and calls `.run()`.
2.  **Boot (`cli.ts`)**: Checks dependencies (Python/VBLoRA) via `checkDataGenDependencies`.
3.  **Phase 1: Data Generation (`runDataGenerationPhase`)**:
      * **Config (Blocking)**: Calls `configureDataGeneration()` -\> Uses `inquirer` to gather `useCase`, `labels`, `categories` (looping prompt), `examples`, and `modelSettings`.
      * **Execution (Blocking)**: Calls `runDataGeneration()` -\> Generates temp config -\> Spawns `python data-gen.py` with **`stdio: 'inherit'`**.
      * **Output**: Node process pauses; Python process owns stdout/stdin.
4.  **Phase 2: Fine-Tuning (`runFineTuningPhase`)**:
      * **Prompt**: Asks to proceed.
      * **Config (Blocking)**: Calls `configureFineTuning()` -\> Selects model family/variant/presets via `inquirer`.
      * **Execution (Blocking)**: Calls `runFineTuning()` -\> Spawns `python vblora.py` with **`stdio: 'inherit'`**.
5.  **Exit**: Prints summary and exits.

### 2\. Component Mapping & Responsibilities

| Module | Responsibility | Interaction Style |
| :--- | :--- | :--- |
| `cli.ts` | Main Orchestrator | Imperative / Linear |
| `steps/*.ts` | Configuration Logic | Blocking Prompts (`inquirer`) |
| `runners/*.ts` | Process Execution | Blocking Spawn (`stdio: inherit`) |
| `utils/display.ts` | Output Formatting | Direct `console.log` / `chalk` |

### 3\. Anti-Patterns Blocking TUI Migration

1.  **IO Monopoly (`stdio: 'inherit'`)**: The runners (`data-runner.ts`, `finetune-runner.ts`) grant child processes direct access to the TTY. This destroys any TUI layout active at the time.
2.  **Implicit State**: The application state (e.g., "Configuring Labels") is held in the call stack (execution pointer inside `configureLabels`), not in a state object. A TUI requires a reified state machine (e.g., `state.currentView = 'LABEL_CONFIG'`).
3.  **Blocking Input**: `await inquirer.prompt()` halts the event loop logic for that "step". OpenTUI is event-driven; it doesn't "wait" at a line of code, it reacts to events.
4.  **Direct Console Writes**: `utils/display.ts` writes directly to `stdout`. In OpenTUI, text must be rendered to a specific `TextRenderable` component buffer, never `console.log`.

-----

# Relevant sst-opentui Capabilities

### 1\. Rendering Model (Retained Mode)

  * **`CliRenderer`**: The engine. It owns the screen buffer and diffs changes to the terminal. It manages the event loop (`start()`, `requestRender()`).
  * **`Renderable`**: The base class for UI nodes. Unlike React, these are mutable objects. You modify properties (`box.width = 10`) and the renderer updates the view.

### 2\. Layout Primitives (Yoga / Flexbox)

  * **`BoxRenderable`**: The fundamental building block (div). Supports `flexDirection`, `justifyContent`, `alignItems`, `padding`, `border`.
      * *Relevance:* We will use a root `BoxRenderable` as our "App Shell" and swap child Boxes for different screens (Config vs Running).

### 3\. Input & Event Architecture

  * **Event Bubbling**: Components emit events (e.g., `SelectRenderable` emits `onSelect`).
  * **`InputRenderable`**: Handles text entry, cursor position, and focus.
  * **`SelectRenderable` / `TabSelectRenderable`**: Handles list navigation.
      * *Relevance:* Replaces `inquirer`. Instead of `const ans = await prompt()`, we render a `SelectRenderable` and listen for its `onSubmit` event.

### 4\. Output & Scrolling

  * **`ScrollBoxRenderable`**: A scrollable container for content exceeding the viewport.
  * **`FrameBuffer` / `OptimizedBuffer`**: Low-level drawing surface.
      * *Relevance:* Essential for the "Runner" phase. We cannot simply print logs. We must append strings to a `TextRenderable` inside a `ScrollBoxRenderable` with `stickyScroll: true`.

-----

# Target Architecture

We will shift from **Linear Script** to **Event-Driven State Machine**.

### 1\. Folder Structure

```
src/
├── core/
│   ├── state.ts            # Global Store (Signals or Proxy state)
│   ├── process-manager.ts  # Wraps spawn, emits 'data' events instead of inheriting stdio
│   └── theme.ts            # Centralized TUI colors/styles
├── ui/
│   ├── root.ts             # TUI Entrypoint (Renderer setup)
│   ├── components/
│   │   ├── StepWizard.ts   # Orchestrates multi-step forms
│   │   ├── LogViewer.ts    # ScrollBox wrapper for process output
│   │   └── StatusBar.ts    # Footer info
│   └── screens/
│       ├── WelcomeScreen.ts
│       ├── DataConfigScreen.ts
│       ├── FineTuneConfigScreen.ts
│       └── ProcessRunnerScreen.ts
└── index.ts
```

### 2\. UI Rendering Layers

  * **Layer 0 (Renderer)**: `CliRenderer` (Singleton).
  * **Layer 1 (Layout)**: `MainLayout` (BoxRenderable).
      * **Header**: Static `TextRenderable`.
      * **ContentArea**: Dynamic `BoxRenderable` (Swapped based on State).
      * **Footer**: Status/Key-hints.

### 3\. Data Flow & Event Loop

1.  **State Change**: `AppState` changes `currentView` to `DATA_CONFIG`.
2.  **Render**: `MainLayout` clears `ContentArea` and appends `DataConfigScreen`.
3.  **Interaction**: User interacts with `InputRenderable`.
4.  **Transition**: User presses Enter -\> `StepWizard` advances.
5.  **Execution**: `ProcessManager` starts Python -\> Emits `log` event -\> `AppState` updates `logs[]` -\> `LogViewer` appends text.

-----

# Step-by-Step Migration Plan

## Phase 1: The TUI Foundation (Safe)

**Goal:** Establish the rendering engine without breaking existing logic.

1.  **Install OpenTUI**: Add `@opentui/core` to dependencies.
2.  **Create `src/ui/root.ts`**: Initialize `createCliRenderer`.
3.  **Create `MainLayout`**: A full-screen `BoxRenderable` with a Header and an empty Content area.
4.  **Verification**: Run the CLI with a flag (e.g., `--tui`). It should show a static TUI frame.

## Phase 2: The Process Manager (Crucial Refactor)

**Goal:** Decouple process execution from the console.

1.  **Create `core/process-manager.ts`**:
      * Create a function `spawnBuffered(command, args)` that returns an `EventEmitter`.
      * Spawn Python with `stdio: ['pipe', 'pipe', 'pipe']`.
      * Listen to `stdout` and `stderr`. Emit `line` events.
2.  **Refactor Runners**:
      * Modify `runDataGeneration` and `runFineTuning` to accept an optional `onLog` callback.
      * If `onLog` is present, use `spawnBuffered` (TUI mode).
      * If absent, keep using `inherit` (Legacy mode).

## Phase 3: Configuration Screens (High Effort)

**Goal:** Replace Inquirer with TUI Components.

1.  **Create `DataConfigScreen`**:
      * Map `configureUseCase` -\> `InputRenderable`.
      * Map `configureModel` -\> `SelectRenderable`.
      * Map "Categories" loop -\> A custom list builder component (Input + Add Button + List View).
2.  **State Wiring**: Create a `configState` object that updates as the user interacts with these components.

## Phase 4: The Runner Screen (Integration)

**Goal:** Visualize the running process.

1.  **Create `ProcessRunnerScreen`**:
      * Contains a `LogViewer` (`ScrollBoxRenderable`).
      * Contains a `ProgressBar` (Parse "Step X/Y" from Python logs).
2.  **Wire Up**:
      * When config finishes, switch view to `ProcessRunnerScreen`.
      * Invoke `runDataGeneration` passing a callback that appends text to `LogViewer`.

## Phase 5: Cutover & Cleanup

1.  **Default to TUI**: Remove the `--tui` flag requirement.
2.  **Legacy Fallback**: Detect `!process.stdout.isTTY` (CI environments) and fallback to the old `inquirer` flow automatically.

-----

# Production Readiness Considerations

### 1\. Error Recovery

  * **Terminal Restore**: Ensure `renderer.restoreTerminal()` is wrapped in a `try/finally` block or `process.on('exit')`. If the TUI crashes without restoring, the user's terminal will be broken (invisible cursor, no echo).
  * **Python Failures**: If the Python script errors, the TUI must not close. It should show a "Failed" red state in the `LogViewer` and offer a "Retry" or "Back" button.

### 2\. Theming & Polish

  * **Theme Object**: Create `core/theme.ts` exporting standard RGBA colors (Primary, Secondary, Error, Background). Do not use hardcoded hex strings in components.
  * **Loading States**: Use spinners (animated TextRenderables) when waiting for Python initialization (which can be slow).

### 3\. Asynchronous Workloads

  * **Non-Blocking UI**: While Python runs, the UI must remain responsive (e.g., allowing the user to scroll through logs or cancel the process). Using `spawn` with events handles this naturally in Node.js.

-----

# Risks and Trade-offs

### 1\. Python Buffering (High Risk)

  * **Risk**: Python buffers `stdout` by default when not attached to a TTY. Logs might appear in huge chunks after long delays.
  * **Mitigation**: Always spawn Python with the `-u` (unbuffered) flag: `spawn('python', ['-u', script...])`.

### 2\. Input Complexity

  * **Risk**: The "Categories" step allows adding multiple items in a loop. Replicating this `inquirer` flow in a TUI is complex (requires a mini CRUD UI inside the screen).
  * **Trade-off**: We might simplify the initial TUI version to accept comma-separated strings for categories instead of a multi-step add loop.

### 3\. Terminal Size

  * **Risk**: TUI layouts break on very small terminals.
  * **Mitigation**: Add a check on startup. If `rows < 10` or `columns < 40`, abort TUI and fall back to legacy CLI mode or print a warning.