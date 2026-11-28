/**
 * Per-term colours for full-text note marks (#562), modelled on ontotator's
 * `assignMarkColours`.
 *
 * A finding's colour slot starts from a stable hash of its HPO ID. When a term
 * is first seen:
 * - while the palette still has an unused colour, it takes the first unused
 *   slot from its hash onwards, so the first MARK_COLOUR_COUNT terms all differ;
 * - after that, colours must repeat. It never takes the colour of a different
 *   term in the same sentence or in the same (merged) mark, and among the rest
 *   it takes the colour whose last mark lies furthest back, so repeats sit as
 *   far apart as possible (also across sentence boundaries).
 * Once assigned, a term keeps its slot for the whole note.
 *
 * Colour never encodes state: manual, negated and active stay line styles or
 * patterns in CSS. The palette itself lives in `styles/markColours.css`.
 */

export const MARK_COLOUR_COUNT = 6;

// A plain-text run that ends a sentence: terminal punctuation followed by
// whitespace or end of run, or any line break.
const SENTENCE_BOUNDARY = /[.!?;](\s|$)|\n/;

/** Stable slot for an id (32-bit FNV-1a, modulo the palette size). */
export function hashSlot(id) {
  const text = typeof id === 'string' ? id : '';
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash % MARK_COLOUR_COUNT;
}

function firstFreeSlot(termId, taken) {
  const preferred = hashSlot(termId);
  for (let step = 0; step < MARK_COLOUR_COUNT; step += 1) {
    const slot = (preferred + step) % MARK_COLOUR_COUNT;
    if (!taken.has(slot)) return slot;
  }
  return preferred;
}

// Least recently used slot outside `avoid`; ties go to the hash order.
function leastRecentSlot(termId, avoid, lastUse) {
  const preferred = hashSlot(termId);
  let best = null;
  for (let step = 0; step < MARK_COLOUR_COUNT; step += 1) {
    const slot = (preferred + step) % MARK_COLOUR_COUNT;
    if (avoid.has(slot)) continue;
    if (best === null || lastUse.get(slot) < lastUse.get(best)) best = slot;
  }
  // More distinct neighbours than colours: fall back to the hashed slot.
  return best === null ? preferred : best;
}

/**
 * Assign a colour slot to every term that has a finding mark in `segments`
 * (the output of `buildSegmentsFromAnnotations`). Returns Map<hpoId, slot>.
 */
export function assignMarkColours(segments) {
  const slots = new Map();
  const lastUse = new Map(Array.from({ length: MARK_COLOUR_COUNT }, (_, slot) => [slot, -1]));
  let sentenceTerms = new Set();
  let markIndex = 0;

  (Array.isArray(segments) ? segments : []).forEach((segment) => {
    if (!segment) return;
    if (!segment.highlighted) {
      if (!segment.modifier && SENTENCE_BOUNDARY.test(segment.text || '')) {
        sentenceTerms = new Set();
      }
      return;
    }
    const termIds = (Array.isArray(segment.termIds) ? segment.termIds : []).filter(
      (id) => typeof id === 'string'
    );
    termIds.forEach((termId) => {
      if (slots.has(termId)) return;
      const used = new Set(slots.values());
      if (used.size < MARK_COLOUR_COUNT) {
        slots.set(termId, firstFreeSlot(termId, used));
        return;
      }
      const avoid = new Set();
      [...sentenceTerms, ...termIds].forEach((other) => {
        if (other !== termId && slots.has(other)) avoid.add(slots.get(other));
      });
      slots.set(termId, leastRecentSlot(termId, avoid, lastUse));
    });
    termIds.forEach((termId) => {
      sentenceTerms.add(termId);
      lastUse.set(slots.get(termId), markIndex);
    });
    markIndex += 1;
  });

  return slots;
}

/**
 * Return `segments` with a `colourSlot` field on finding and modifier segments.
 * A mark linked to several terms (merged overlap, or a modifier shared by
 * several findings) takes the first term's colour; its tooltip lists them all.
 */
export function applyMarkColours(segments) {
  const slots = assignMarkColours(segments);
  return (Array.isArray(segments) ? segments : []).map((segment) => {
    if (!segment || (!segment.highlighted && !segment.modifier)) return segment;
    const first = Array.isArray(segment.termIds) ? segment.termIds[0] : null;
    if (typeof first !== 'string') return segment;
    const colourSlot = slots.has(first) ? slots.get(first) : hashSlot(first);
    return { ...segment, colourSlot };
  });
}

/** CSS class for a slot, or an empty string when there is none. */
export function markColourClass(slot) {
  return Number.isInteger(slot) && slot >= 0 && slot < MARK_COLOUR_COUNT
    ? `ph-mark-slot-${slot}`
    : '';
}
