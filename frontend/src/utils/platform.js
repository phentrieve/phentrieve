/**
 * True on Apple platforms (macOS, iOS, iPadOS), where the submit shortcut is
 * Cmd+Enter instead of Ctrl+Enter (#572). Both keys always work; this only
 * decides which one the UI shows.
 */
export function isApplePlatform(nav = typeof navigator !== 'undefined' ? navigator : null) {
  if (!nav) return false;
  const platform = nav.userAgentData?.platform || nav.platform || nav.userAgent || '';
  return /mac|iphone|ipad|ipod/i.test(platform);
}

// U+2318 PLACE OF INTEREST SIGN, the Command key symbol.
export const COMMAND_KEY_SYMBOL = '⌘';
