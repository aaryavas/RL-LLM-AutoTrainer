/**
 * UI module exports.
 */

export {
  TuiApplicationRoot,
  type KeyboardEventHandler,
} from './TuiApplicationRoot';

export { MainLayoutContainer } from './MainLayoutContainer';

export {
  ScreenNavigator,
  type ScreenFactory,
  type ScreenConfiguration,
} from './ScreenNavigator';

export {
  TerminalSizeValidator,
  type TerminalDimensions,
  type SizeValidationResult,
} from './TerminalSizeValidator';
