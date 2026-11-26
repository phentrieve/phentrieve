import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import ModifierChips from '../../components/ModifierChips.vue';
import AggregatedTermsView from '../../components/AggregatedTermsView.vue';
import PhenotypeCardRow from '../../components/PhenotypeCardRow.vue';
import ResultItem from '../../components/ResultItem.vue';
import PhenotypeCollectionPanel from '../../components/PhenotypeCollectionPanel.vue';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en },
  silentTranslationWarn: true,
  silentFallbackWarn: true,
  missing: () => '',
});

const SAMPLE_MODIFIERS = [
  {
    term_id: 'HP:0012828',
    label: 'Severe',
    segments: [{ start: 12, end: 18, text: 'severe' }],
    source: 'llm',
  },
  {
    term_id: 'HP:0003593',
    label: 'Infantile onset',
    segments: [{ start: 30, end: 43, text: 'since infancy' }],
    source: 'rule',
  },
];

const SAMPLE_WARNINGS = [
  {
    code: 'modifier_ungrounded',
    message: 'Modifier span could not be verified verbatim',
    term_id: 'HP:0012828',
  },
];

describe('ModifierChips', () => {
  it('renders chips with labels, HPO links, and tooltip content including segments', () => {
    const wrapper = mount(ModifierChips, {
      props: { modifiers: SAMPLE_MODIFIERS },
      global: { plugins: [vuetify, i18n] },
    });

    const chips = wrapper.findAll('[data-testid="modifier-chip"]');
    expect(chips).toHaveLength(2);
    expect(chips[0].text()).toContain('Severe');
    expect(chips[0].attributes('href')).toContain('HP:0012828');
    expect(chips[0].attributes('aria-label')).toBe('Severe (HP:0012828)');

    expect(chips[1].text()).toContain('Infantile onset');
    expect(chips[1].attributes('href')).toContain('HP:0003593');

    const tooltips = wrapper.findAllComponents({ name: 'VTooltip' });
    expect(tooltips).toHaveLength(2);
    expect(tooltips[0].props('text')).toContain('HP:0012828');
    expect(tooltips[0].props('text')).toContain('"severe"');
    expect(tooltips[1].props('text')).toContain('HP:0003593');
    expect(tooltips[1].props('text')).toContain('"since infancy"');
  });

  it('renders nothing when modifiers are empty or undefined', () => {
    const emptyWrapper = mount(ModifierChips, {
      props: { modifiers: [] },
      global: { plugins: [vuetify, i18n] },
    });
    expect(emptyWrapper.find('[data-testid="modifier-chips"]').exists()).toBe(false);

    const undefWrapper = mount(ModifierChips, {
      global: { plugins: [vuetify, i18n] },
    });
    expect(undefWrapper.find('[data-testid="modifier-chips"]').exists()).toBe(false);
  });
});

describe('AggregatedTermsView clinical modifiers', () => {
  it('displays modifiers and warnings when present and preserves modifiers on collection add', async () => {
    const wrapper = mount(AggregatedTermsView, {
      props: {
        terms: [
          {
            hpo_id: 'HP:0001250',
            name: 'Seizure',
            confidence: 0.92,
            status: 'affirmed',
            modifiers: SAMPLE_MODIFIERS,
            warnings: SAMPLE_WARNINGS,
          },
        ],
      },
      global: { plugins: [vuetify, i18n] },
    });

    expect(wrapper.findAll('[data-testid="modifier-chip"]')).toHaveLength(2);
    expect(wrapper.text()).toContain('Severe');
    expect(wrapper.text()).toContain('Infantile onset');
    expect(wrapper.find('[data-testid="finding-warning-indicator"]').exists()).toBe(true);

    await wrapper.get('.add-btn').trigger('click');
    expect(wrapper.emitted('add-to-collection')).toHaveLength(1);
    expect(wrapper.emitted('add-to-collection')[0][0]).toEqual({
      hpo_id: 'HP:0001250',
      name: 'Seizure',
      label: 'Seizure',
      modifiers: SAMPLE_MODIFIERS,
    });
  });

  it('renders no modifier chips or warning indicator when absent', async () => {
    const wrapper = mount(AggregatedTermsView, {
      props: {
        terms: [
          {
            hpo_id: 'HP:0001251',
            name: 'Ataxia',
            confidence: 0.85,
            status: 'affirmed',
          },
        ],
      },
      global: { plugins: [vuetify, i18n] },
    });

    expect(wrapper.find('[data-testid="modifier-chips"]').exists()).toBe(false);
    expect(wrapper.find('[data-testid="finding-warning-indicator"]').exists()).toBe(false);

    await wrapper.get('.add-btn').trigger('click');
    expect(wrapper.emitted('add-to-collection')[0][0]).toEqual({
      hpo_id: 'HP:0001251',
      name: 'Ataxia',
      label: 'Ataxia',
    });
  });
});

describe('PhenotypeCardRow and ResultItem clinical modifiers', () => {
  it('renders modifiers and warnings on PhenotypeCardRow when supplied', () => {
    const wrapper = mount(PhenotypeCardRow, {
      props: {
        hpoId: 'HP:0001250',
        label: 'Seizure',
        modifiers: SAMPLE_MODIFIERS,
        warnings: SAMPLE_WARNINGS,
      },
      global: { plugins: [vuetify, i18n] },
    });

    expect(wrapper.findAll('[data-testid="modifier-chip"]')).toHaveLength(2);
    expect(wrapper.find('[data-testid="finding-warning-indicator"]').exists()).toBe(true);
  });

  it('renders nothing extra on PhenotypeCardRow when modifiers and warnings are absent', () => {
    const wrapper = mount(PhenotypeCardRow, {
      props: {
        hpoId: 'HP:0001250',
        label: 'Seizure',
      },
      global: { plugins: [vuetify, i18n] },
    });

    expect(wrapper.find('[data-testid="modifier-chips"]').exists()).toBe(false);
    expect(wrapper.find('[data-testid="finding-warning-indicator"]').exists()).toBe(false);
  });

  it('passes modifiers and warnings through ResultItem to PhenotypeCardRow', () => {
    const wrapper = mount(ResultItem, {
      props: {
        rank: 1,
        result: {
          hpo_id: 'HP:0001250',
          label: 'Seizure',
          confidence: 0.91,
          modifiers: SAMPLE_MODIFIERS,
          warnings: SAMPLE_WARNINGS,
        },
      },
      global: { plugins: [vuetify, i18n] },
    });

    expect(wrapper.findAll('[data-testid="modifier-chip"]')).toHaveLength(2);
    expect(wrapper.find('[data-testid="finding-warning-indicator"]').exists()).toBe(true);
  });
});

describe('PhenotypeCollectionPanel clinical modifiers', () => {
  it('renders modifier chips for collected phenotypes that have modifiers', () => {
    const wrapper = mount(PhenotypeCollectionPanel, {
      props: {
        panelOpen: true,
        phenotypes: [
          {
            hpo_id: 'HP:0001250',
            label: 'Seizure',
            assertion_status: 'affirmed',
            modifiers: SAMPLE_MODIFIERS,
          },
          {
            hpo_id: 'HP:0001251',
            label: 'Ataxia',
            assertion_status: 'affirmed',
          },
        ],
      },
      global: {
        plugins: [vuetify, i18n],
        stubs: {
          'v-navigation-drawer': {
            template: '<div><slot /><slot name="append" /></div>',
          },
        },
      },
    });

    const chips = wrapper.findAll('[data-testid="modifier-chip"]');
    expect(chips).toHaveLength(2);
    expect(chips[0].text()).toContain('Severe');
    expect(chips[1].text()).toContain('Infantile onset');
  });
});
