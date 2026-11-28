import { describe, it, expect } from 'vitest';
import {
  MARK_COLOUR_COUNT,
  applyMarkColours,
  assignMarkColours,
  hashSlot,
  markColourClass,
} from '../../utils/markColours';
import { buildSegmentsFromAnnotations } from '../../composables/useUserNoteAnnotations';

function mark(termIds, text = 'x') {
  return { highlighted: true, termIds, text };
}

function plain(text) {
  return { highlighted: false, text };
}

// Find two ids whose hashed slots collide, to exercise the neighbour rule.
function collidingPair() {
  const seen = new Map();
  for (let i = 0; i < 1000; i += 1) {
    const id = `HP:${String(i).padStart(7, '0')}`;
    const slot = hashSlot(id);
    if (seen.has(slot)) return [seen.get(slot), id];
    seen.set(slot, id);
  }
  throw new Error('no collision found');
}

describe('hashSlot', () => {
  it('is stable and within the palette', () => {
    const slot = hashSlot('HP:0001250');
    expect(hashSlot('HP:0001250')).toBe(slot);
    expect(slot).toBeGreaterThanOrEqual(0);
    expect(slot).toBeLessThan(MARK_COLOUR_COUNT);
  });
});

describe('assignMarkColours (#562)', () => {
  it('moves a neighbour in the same sentence off a colliding slot', () => {
    const [a, b] = collidingPair();
    const slots = assignMarkColours([mark([a]), plain(' and '), mark([b]), plain('.')]);
    expect(slots.get(a)).toBe(hashSlot(a));
    expect(slots.get(b)).not.toBe(slots.get(a));
  });

  it('gives different terms different colours while the palette has room', () => {
    const ids = Array.from({ length: MARK_COLOUR_COUNT }, (_, i) => `HP:000010${i}`);
    // Each term in its own sentence: distinct colours are still required.
    const segments = ids.flatMap((id) => [mark([id]), plain('. ')]);
    const slots = assignMarkColours(segments);
    expect(new Set(ids.map((id) => slots.get(id))).size).toBe(MARK_COLOUR_COUNT);
  });

  it('repeats colours only away from sentence neighbours once the palette is full', () => {
    const ids = Array.from({ length: MARK_COLOUR_COUNT }, (_, i) => `HP:000010${i}`);
    const extra = ['HP:0009990', 'HP:0009991'];
    const segments = [
      ...ids.flatMap((id) => [mark([id]), plain('. ')]),
      mark([ids[0]]),
      plain(' with '),
      mark([extra[0]]),
      plain(' and '),
      mark([extra[1]]),
      plain('.'),
    ];
    const slots = assignMarkColours(segments);
    expect(slots.get(extra[0])).not.toBe(slots.get(ids[0]));
    expect(slots.get(extra[1])).not.toBe(slots.get(ids[0]));
    expect(slots.get(extra[1])).not.toBe(slots.get(extra[0]));
  });

  it('gives the same term the same slot everywhere in the note', () => {
    const [a, b] = collidingPair();
    const segments = applyMarkColours([
      mark([a]),
      plain(' with '),
      mark([b]),
      plain('. Again '),
      mark([b]),
      plain(' alone.'),
    ]);
    const slotsOfB = segments.filter((s) => s.termIds?.[0] === b).map((s) => s.colourSlot);
    expect(new Set(slotsOfB).size).toBe(1);
  });

  it('separates terms that share one merged mark', () => {
    const [a, b] = collidingPair();
    const slots = assignMarkColours([mark([a, b])]);
    expect(slots.get(a)).not.toBe(slots.get(b));
  });

  it('falls back to the hashed slot when there are more neighbours than colours', () => {
    const ids = Array.from({ length: MARK_COLOUR_COUNT + 1 }, (_, i) => `HP:900000${i}`);
    const segments = ids.flatMap((id) => [mark([id]), plain(', ')]);
    const slots = assignMarkColours(segments);
    expect(slots.size).toBe(ids.length);
    ids.forEach((id) => expect(slots.get(id)).toBeLessThan(MARK_COLOUR_COUNT));
  });
});

describe('applyMarkColours segment fields', () => {
  it('adds colourSlot to finding and modifier segments, not to plain text', () => {
    const note = 'Severe hearing loss and ataxia.';
    const annotations = [
      {
        id: 'a1',
        hpoId: 'HP:0000365',
        label: 'Hearing impairment',
        status: 'affirmed',
        origin: 'auto',
        spans: [{ start: 7, end: 19, text: 'hearing loss' }],
        modifiers: [
          {
            term_id: 'HP:0012828',
            label: 'Severe',
            segments: [{ start: 0, end: 6, text: 'Severe' }],
          },
        ],
      },
      {
        id: 'a2',
        hpoId: 'HP:0001251',
        label: 'Ataxia',
        status: 'affirmed',
        origin: 'auto',
        spans: [{ start: 24, end: 30, text: 'ataxia' }],
      },
    ];

    const segments = applyMarkColours(buildSegmentsFromAnnotations(note, annotations));
    const modifier = segments.find((s) => s.modifier);
    const findings = segments.filter((s) => s.highlighted);
    const plainSegments = segments.filter((s) => !s.highlighted && !s.modifier);

    expect(findings).toHaveLength(2);
    expect(findings[0].colourSlot).not.toBe(findings[1].colourSlot);
    // A modifier takes its finding's colour.
    expect(modifier.colourSlot).toBe(findings[0].colourSlot);
    plainSegments.forEach((s) => expect(s.colourSlot).toBeUndefined());
  });
});

describe('markColourClass', () => {
  it('maps valid slots to classes and rejects others', () => {
    expect(markColourClass(0)).toBe('ph-mark-slot-0');
    expect(markColourClass(MARK_COLOUR_COUNT)).toBe('');
    expect(markColourClass(null)).toBe('');
  });
});
