import { findPhoneNumbersInText } from 'libphonenumber-js';
import { ADDRESS_RULES, GLOBAL_RULES, LOCALE_RULES, SUPPORTED_PII_LOCALES } from './ruleConfig';
import { VALIDATORS } from './validators';

function createSummary() {
  return { high: {}, review: {} };
}

function addSummary(summary, finding) {
  summary[finding.confidence][finding.category] =
    (summary[finding.confidence][finding.category] ?? 0) + 1;
}

function buildFinding({ rule, start, end, index }) {
  return {
    id: `${rule.id}-${index}`,
    ruleId: rule.id,
    category: rule.category,
    confidence: rule.confidence,
    start,
    end,
    redactionToken: rule.redactionToken,
  };
}

function isRuleEnabledForLocale(rule, locale) {
  return rule.enabled && (rule.locales.includes('*') || rule.locales.includes(locale));
}

function candidateForValidation(matchText) {
  const normalizedText = String(matchText);
  const identifierMatch = /[A-Z]?\d[\d .-]{6,20}[A-Z]?/iu.exec(normalizedText);
  return identifierMatch?.[0] ?? normalizedText;
}

function applyRules(text, rules, locale) {
  const findings = [];
  for (const rule of rules) {
    if (!isRuleEnabledForLocale(rule, locale)) continue;
    for (const pattern of rule.patterns) {
      pattern.lastIndex = 0;
      let match;
      while ((match = pattern.exec(text)) !== null) {
        const matchText = match[0];
        const validator = rule.validator ? VALIDATORS[rule.validator] : null;
        if (validator && !validator(candidateForValidation(matchText))) continue;
        findings.push(
          buildFinding({
            rule,
            start: match.index,
            end: match.index + matchText.length,
            index: findings.length,
          })
        );
      }
    }
  }
  return findings;
}

function regionForLocale(locale) {
  switch (locale) {
    case 'de':
      return 'DE';
    case 'fr':
      return 'FR';
    case 'es':
      return 'ES';
    case 'nl':
      return 'NL';
    default:
      return 'US';
  }
}

// Left boundary keeps "Hotel", "Mittel", "Beruf" or "typically" from acting as a label;
// an optional "number"/"Nr." may sit between the label and the digits.
const PHONE_LABEL_BEFORE =
  // eslint-disable-next-line security/detect-unsafe-regex -- only ever run on a 24-character window.
  /(?<!\p{L})(?:tel|phone|fax|mobil|handy|cell|call|tél|teléfono|telefoon|ruf)\p{L}*(?:[\s.-]*(?:number|nummer|numéro|número|no|nr))?[.:]*\s*$/iu;
// Formatted North American numbers have no trunk zero: "(202) 555-0199", "202-555-0199".
const NANP_FORMATTED = /^\(?\d{3}\)?[ -]\d{3}-\d{4}$/u;
// The only dotted form accepted: French "01.23.45.67.89". Other dots are dates or decimals.
const FRENCH_DOTTED = /^0\d\.\d{2}\.\d{2}\.\d{2}\.\d{2}$/u;

// libphonenumber accepts bare digit runs, so lab values, blood pressure,
// dates and genomic coordinates parse as German numbers. Keep a match only
// when it looks dialled: international prefix, trunk zero, or a phone label.
function looksLikePhoneNumber(text, match) {
  const value = text.slice(match.startsAt, match.endsAt);
  if (value.startsWith('+') || NANP_FORMATTED.test(value)) return true;
  if (PHONE_LABEL_BEFORE.test(text.slice(Math.max(0, match.startsAt - 24), match.startsAt))) {
    return true;
  }
  if (value.includes('.')) return FRENCH_DOTTED.test(value);
  return /^\(?0/u.test(value) && value.replace(/\D/gu, '').length >= 9;
}

function applyPhoneRule(text, locale) {
  const matches = findPhoneNumbersInText(text, regionForLocale(locale)).filter((match) =>
    looksLikePhoneNumber(text, match)
  );
  return matches.map((match, index) => ({
    id: `global.phone-${locale}-${index}-${match.startsAt}-${match.endsAt}`,
    ruleId: 'global.phone',
    category: 'phone',
    confidence: 'high',
    start: match.startsAt,
    end: match.endsAt,
    redactionToken: '[REDACTED_PHONE]',
  }));
}

function applyAddressRule(text, locale) {
  let pattern;
  switch (locale) {
    case 'de':
      pattern = ADDRESS_RULES.de;
      break;
    case 'fr':
      pattern = ADDRESS_RULES.fr;
      break;
    case 'es':
      pattern = ADDRESS_RULES.es;
      break;
    case 'nl':
      pattern = ADDRESS_RULES.nl;
      break;
    default:
      pattern = ADDRESS_RULES.en;
  }
  if (!pattern) return [];

  const findings = [];
  pattern.lastIndex = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    findings.push({
      id: `address-${locale}-${findings.length}`,
      ruleId: `${locale}.address`,
      category: 'address',
      confidence: 'high',
      start: match.index,
      end: match.index + match[0].length,
      redactionToken: '[REDACTED_ADDRESS]',
    });
  }
  return findings;
}

function mergeOverlappingFindings(findings) {
  const sorted = [...findings].sort((a, b) => a.start - b.start || a.end - b.end);
  const merged = [];
  for (const finding of sorted) {
    const previous = merged[merged.length - 1];
    if (previous && finding.start < previous.end) {
      const previousLength = previous.end - previous.start;
      const findingLength = finding.end - finding.start;
      previous.end = Math.max(previous.end, finding.end);
      if (
        previous.confidence !== 'high' &&
        (finding.confidence === 'high' || findingLength > previousLength)
      ) {
        previous.ruleId = finding.ruleId;
        previous.category = finding.category;
        previous.confidence = finding.confidence;
        previous.redactionToken = finding.redactionToken;
      }
      continue;
    }
    merged.push({ ...finding });
  }
  return merged;
}

function rulesForLocale(locale) {
  switch (locale) {
    case 'de':
      return LOCALE_RULES.de;
    case 'fr':
      return LOCALE_RULES.fr;
    case 'es':
      return LOCALE_RULES.es;
    case 'nl':
      return LOCALE_RULES.nl;
    default:
      return LOCALE_RULES.en;
  }
}

function localesForScan(locale) {
  return [locale, ...SUPPORTED_PII_LOCALES.filter((supportedLocale) => supportedLocale !== locale)];
}

export function scanPii(text, { locale = 'en', includeGlobalRules = true } = {}) {
  const normalizedLocale = SUPPORTED_PII_LOCALES.includes(locale) ? locale : 'en';
  const source = String(text ?? '');
  const scanLocales = localesForScan(normalizedLocale);
  const findings = mergeOverlappingFindings([
    ...(includeGlobalRules ? applyRules(source, GLOBAL_RULES, normalizedLocale) : []),
    ...scanLocales.flatMap((scanLocale) =>
      applyRules(source, rulesForLocale(scanLocale), scanLocale)
    ),
    ...(includeGlobalRules
      ? scanLocales.flatMap((scanLocale) => applyPhoneRule(source, scanLocale))
      : []),
    ...scanLocales.flatMap((scanLocale) => applyAddressRule(source, scanLocale)),
  ]);

  const summary = createSummary();
  findings.forEach((finding) => addSummary(summary, finding));
  return {
    hasFindings: findings.length > 0,
    findings,
    summary,
  };
}
