import { describe, it, expect } from 'vitest';
import { isApplePlatform } from '../../utils/platform';

describe('isApplePlatform', () => {
  it.each([
    [{ platform: 'MacIntel' }, true],
    [{ platform: 'iPhone' }, true],
    [{ userAgentData: { platform: 'macOS' } }, true],
    [{ platform: 'Win32' }, false],
    [{ platform: 'Linux x86_64' }, false],
    [{ userAgentData: { platform: 'Windows' }, platform: 'MacIntel' }, false],
  ])('detects %j as %s', (nav, expected) => {
    expect(isApplePlatform(nav)).toBe(expected);
  });

  it('returns false without a navigator', () => {
    expect(isApplePlatform(null)).toBe(false);
  });
});
