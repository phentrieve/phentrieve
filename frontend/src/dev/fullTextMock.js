/**
 * Mock full-text turn for the dev-only UI preview (/dev/full-text).
 *
 * A fictional German letter (no real patient) with findings that cover every
 * mark state: affirmed, negated, manual, modifiers, more terms than palette
 * colours, a long label and a family-history finding. Offsets are computed from
 * the note so the fixture stays valid when the text is edited. Umlauts are
 * written as escapes to keep the file ASCII.
 */

export const MOCK_TURN_ID = 'dev-full-text-mock';

export const MOCK_NOTE =
  'Vorstellung eines 7-jährigen Jungen mit schwerer sensorineuraler ' +
  'Schwerhörigkeit beidseits und progredienter Ataxie seit dem Kleinkindalter. ' +
  'Krampfanfälle wurden bisher nicht beobachtet. Klinisch zeigen sich eine ' +
  'Mikrozephalie, ein Kleinwuchs und eine muskuläre Hypotonie. Sonographisch ' +
  'keine Hepatomegalie, keine Splenomegalie. Die Sprachentwicklung ist deutlich ' +
  'verzögert, zudem besteht eine Skoliose.\n\n' +
  'Familienanamnese: Die Mutter leidet an arterieller Hypertonie.';

function span(text) {
  const start = MOCK_NOTE.indexOf(text);
  if (start < 0) throw new Error(`mock text not in note: ${text}`);
  return { start, end: start + text.length, text };
}

function term(hpoId, name, evidence, { status = 'affirmed', confidence = 0.9, modifiers } = {}) {
  return {
    hpo_id: hpoId,
    name,
    status,
    ...(status === 'negated' ? { excluded: true } : {}),
    confidence,
    source_chunk_ids: [1],
    top_evidence_chunk_id: 1,
    text_attributions: [{ chunk_id: 1, matched_text_in_chunk: evidence }],
    ...(modifiers ? { modifiers } : {}),
  };
}

function modifier(termId, label, text) {
  return { term_id: termId, label, segments: [span(text)], source: 'llm' };
}

export function buildMockItem() {
  return {
    id: MOCK_TURN_ID,
    type: 'textProcess',
    query: MOCK_NOTE,
    response: {
      meta: {},
      processed_chunks: [
        { chunk_id: 1, text: MOCK_NOTE, start_char: 0, end_char: MOCK_NOTE.length },
      ],
      aggregated_hpo_terms: [
        term(
          'HP:0008619',
          'Bilateral sensorineural hearing impairment',
          'sensorineuraler Schwerhörigkeit beidseits',
          { confidence: 0.94, modifiers: [modifier('HP:0012828', 'Severe', 'schwerer')] }
        ),
        term('HP:0002073', 'Progressive cerebellar ataxia', 'progredienter Ataxie', {
          confidence: 0.88,
          modifiers: [modifier('HP:0011463', 'Childhood onset', 'seit dem Kleinkindalter')],
        }),
        term('HP:0001250', 'Seizure', 'Krampfanfälle', { status: 'negated', confidence: 0.91 }),
        term('HP:0000252', 'Microcephaly', 'Mikrozephalie', { confidence: 0.97 }),
        term('HP:0004322', 'Short stature', 'Kleinwuchs', { confidence: 0.93 }),
        term('HP:0001252', 'Hypotonia', 'muskuläre Hypotonie', { confidence: 0.86 }),
        term('HP:0002240', 'Hepatomegaly', 'Hepatomegalie', {
          status: 'negated',
          confidence: 0.95,
        }),
        term('HP:0001744', 'Splenomegaly', 'Splenomegalie', {
          status: 'negated',
          confidence: 0.95,
        }),
        term(
          'HP:0000750',
          'Delayed speech and language development',
          'Sprachentwicklung ist deutlich verzögert',
          { confidence: 0.82 }
        ),
      ],
      family_history_findings: [
        {
          hpo_id: 'HP:0000822',
          name: 'Hypertension',
          status: 'affirmed',
          text_attributions: [
            { chunk_id: 1, matched_text_in_chunk: 'Die Mutter leidet an arterieller Hypertonie' },
          ],
        },
      ],
      other_person_findings: [],
    },
  };
}

/** A curator-added finding, as the store keeps it after a manual annotation. */
export function buildMockManualAnnotation() {
  return {
    id: 'manual-HP:0002650-dev',
    hpoId: 'HP:0002650',
    label: 'Scoliosis',
    status: 'affirmed',
    origin: 'manual',
    confidence: null,
    spans: [span('Skoliose')],
  };
}
