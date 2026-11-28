import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import ResultItem from '../../components/ResultItem.vue';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en },
  silentTranslationWarn: true,
  silentFallbackWarn: true,
  missing: () => '',
});

const MODIFIERS = [
  {
    term_id: 'HP:0012828',
    label: 'Severe',
    segments: [{ start: 0, end: 6, text: 'severe' }],
    source: 'llm',
  },
];

function mountItem(result, props = {}, slots = {}) {
  return mount(ResultItem, {
    props: { rank: 1, result, ...props },
    slots,
    global: { plugins: [vuetify, i18n] },
  });
}

describe('ResultItem compact row (#569)', () => {
  it('renders label, HPO ID and actions on one line without a card border', () => {
    const wrapper = mountItem(
      { hpo_id: 'HP:0001250', label: 'Seizure', confidence: 0.91 },
      { compact: true }
    );

    const row = wrapper.get('.phenotype-card-row');
    expect(row.classes()).toContain('phenotype-card-row--compact');
    expect(row.classes()).not.toContain('v-list-item--border');
    const line = wrapper.get('.phenotype-card-row__compact-line');
    expect(line.text()).toContain('Seizure');
    expect(line.text()).toContain('HP:0001250');
    expect(line.find('.add-btn').exists()).toBe(true);
    expect(wrapper.get('.phenotype-card-row__compact-label').attributes('title')).toBe('Seizure');
  });

  it('moves modifiers into the expandable details', async () => {
    const wrapper = mountItem(
      { hpo_id: 'HP:0001250', label: 'Seizure', confidence: 0.91, modifiers: MODIFIERS },
      { compact: true }
    );

    expect(wrapper.find('[data-testid="modifier-chip"]').exists()).toBe(false);
    const toggle = wrapper.get('[data-testid="result-item-details-toggle"]');
    expect(toggle.attributes('aria-expanded')).toBe('false');

    await toggle.trigger('click');

    expect(toggle.attributes('aria-expanded')).toBe('true');
    expect(wrapper.get('[data-testid="result-item-details"]').text()).toContain('Severe');
  });

  it('keeps modifiers inline in the default (non-compact) row', () => {
    const wrapper = mountItem({
      hpo_id: 'HP:0001250',
      label: 'Seizure',
      confidence: 0.91,
      modifiers: MODIFIERS,
    });

    expect(wrapper.find('.phenotype-card-row--compact').exists()).toBe(false);
    expect(wrapper.findAll('[data-testid="modifier-chip"]')).toHaveLength(1);
  });

  it('renders the badges slot inside the row', () => {
    const wrapper = mountItem(
      { hpo_id: 'HP:0001250', label: 'Seizure', confidence: 0.91 },
      { compact: true },
      { badges: '<span data-testid="extra-badge">Manual</span>' }
    );

    expect(
      wrapper.get('.phenotype-card-row__compact-line').find('[data-testid="extra-badge"]').exists()
    ).toBe(true);
  });
});

describe('ResultItem negated badge (#570)', () => {
  it('shows absent findings as a neutral outlined chip, not a warning', () => {
    const wrapper = mountItem(
      { hpo_id: 'HP:0001250', label: 'Seizure', confidence: 0.91, assertion_status: 'negated' },
      { compact: true }
    );

    const chip = wrapper.get('.assertion-chip');
    expect(chip.classes()).toContain('v-chip--variant-outlined');
    expect(chip.classes().some((c) => c.includes('warning'))).toBe(false);
  });
});

describe('ResultItem rank badge colour (#562)', () => {
  it('uses the finding mark colour slot when given', () => {
    const wrapper = mountItem(
      { hpo_id: 'HP:0001250', label: 'Seizure', confidence: 0.91 },
      { compact: true, markSlot: 3 }
    );
    const badge = wrapper.get('[data-testid="result-item-rank"]');
    expect(badge.classes()).toContain('ph-mark-slot-3');
    expect(badge.classes()).toContain('result-item__mark-badge');
  });

  it('keeps the primary badge without a slot', () => {
    const wrapper = mountItem({ hpo_id: 'HP:0001250', label: 'Seizure', confidence: 0.91 });
    expect(wrapper.get('[data-testid="result-item-rank"]').classes()).not.toContain(
      'result-item__mark-badge'
    );
  });
});

describe('ResultItem without a score', () => {
  it('hides the score chip instead of showing 0.00 (manual findings)', () => {
    const wrapper = mountItem(
      { hpo_id: 'HP:0002650', label: 'Scoliosis', confidence: null, similarity: null },
      { compact: true }
    );
    expect(wrapper.text()).not.toContain('0.00');
  });
});
