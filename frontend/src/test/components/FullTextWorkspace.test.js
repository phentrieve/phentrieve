import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import FullTextWorkspace from '../../components/query/FullTextWorkspace.vue';

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });

const vuetify = createVuetify({ components, directives });

const baseSegments = [
  { key: 'plain-0', text: 'Patient had ', highlighted: false },
  {
    key: 'mark-0',
    text: 'seizures',
    highlighted: true,
    termIds: ['HP:0001250'],
    annotationIds: ['auto-HP:0001250-0'],
    tooltip: 'Seizure (HP:0001250)',
  },
  { key: 'plain-1', text: ' and ', highlighted: false },
  {
    key: 'mark-1',
    text: 'developmental delay',
    highlighted: true,
    termIds: ['HP:0001263'],
    annotationIds: ['auto-HP:0001263-1'],
    tooltip: 'Developmental delay (HP:0001263)',
  },
];

function mountWorkspace(props = {}) {
  return mount(FullTextWorkspace, {
    props: {
      summary: 'Patient had seizures and developmental delay',
      meta: '6 words',
      expanded: true,
      segments: baseSegments,
      activePhenotypeId: null,
      ...props,
    },
    global: { plugins: [vuetify, i18n] },
  });
}

describe('FullTextWorkspace annotated note', () => {
  it('renders every highlighted segment as a mark (no hover collapse)', () => {
    const wrapper = mountWorkspace();
    expect(wrapper.findAll('[data-testid="annotated-note-span"]')).toHaveLength(2);
  });

  it('exposes each annotated span to keyboard and assistive technology', () => {
    const wrapper = mountWorkspace();
    const marks = wrapper.findAll('[data-testid="annotated-note-span"]');
    marks.forEach((mark) => {
      expect(mark.attributes('tabindex')).toBe('0');
      expect(mark.attributes('aria-label')).toBeTruthy();
    });
    expect(marks[0].attributes('aria-label')).toContain('Seizure (HP:0001250)');
  });

  it('keeps all spans highlighted while emphasising only the active phenotype', () => {
    const wrapper = mountWorkspace({ activePhenotypeId: 'HP:0001250' });
    const marks = wrapper.findAll('[data-testid="annotated-note-span"]');
    expect(marks).toHaveLength(2);
    expect(marks[0].classes()).toContain('annotated-note-span--active');
    expect(marks[1].classes()).not.toContain('annotated-note-span--active');
  });

  it('gives the expand/collapse toggle an accessible name', () => {
    const collapsed = mountWorkspace({ expanded: false });
    expect(collapsed.get('[data-testid="user-note-summary-toggle"]').attributes('aria-label')).toBe(
      'Expand note'
    );

    const expanded = mountWorkspace({ expanded: true });
    expect(expanded.get('[data-testid="user-note-summary-toggle"]').attributes('aria-label')).toBe(
      'Collapse note'
    );
  });

  it('emits hover/clear for keyboard focus parity with mouse', async () => {
    const wrapper = mountWorkspace();
    const firstMark = wrapper.findAll('[data-testid="annotated-note-span"]')[0];

    await firstMark.trigger('focus');
    expect(wrapper.emitted('hover')?.[0]?.[0]).toEqual(['HP:0001250']);

    await firstMark.trigger('blur');
    expect(wrapper.emitted('clear-hover')).toBeTruthy();
  });

  it('applies a distinct class to manually curated spans', () => {
    const wrapper = mountWorkspace({
      segments: [
        { key: 'plain-0', text: 'A ', highlighted: false },
        {
          key: 'mark-auto',
          text: 'auto term',
          highlighted: true,
          termIds: ['HP:1'],
          annotationIds: ['auto-HP:1-0'],
          manual: false,
          tooltip: 'Auto (HP:1)',
        },
        { key: 'plain-1', text: ' and ', highlighted: false },
        {
          key: 'mark-manual',
          text: 'manual term',
          highlighted: true,
          termIds: ['HP:2'],
          annotationIds: ['manual-HP:2-1'],
          manual: true,
          tooltip: 'Manual (HP:2)',
        },
      ],
    });
    const marks = wrapper.findAll('[data-testid="annotated-note-span"]');
    expect(marks[0].classes()).not.toContain('annotated-note-span--manual');
    expect(marks[1].classes()).toContain('annotated-note-span--manual');
  });

  it('marks expose button semantics for the curation menu', () => {
    const wrapper = mountWorkspace();
    const firstMark = wrapper.findAll('[data-testid="annotated-note-span"]')[0];
    expect(firstMark.attributes('role')).toBe('button');
    expect(firstMark.attributes('aria-haspopup')).toBe('menu');
    expect(firstMark.attributes('aria-label')).toContain('Edit annotation');
  });

  it('emits span-activate with annotation + term ids on click', async () => {
    const wrapper = mountWorkspace();
    const firstMark = wrapper.findAll('[data-testid="annotated-note-span"]')[0];

    await firstMark.trigger('click');

    const payload = wrapper.emitted('span-activate')?.[0]?.[0];
    expect(payload).toBeTruthy();
    expect(payload.annotationIds).toEqual(['auto-HP:0001250-0']);
    expect(payload.termIds).toEqual(['HP:0001250']);
    expect(payload).toHaveProperty('rect');
  });

  it('emits span-activate on Enter and on right-click', async () => {
    const wrapper = mountWorkspace();
    const firstMark = wrapper.findAll('[data-testid="annotated-note-span"]')[0];

    await firstMark.trigger('keydown', { key: 'Enter' });
    expect(wrapper.emitted('span-activate')).toHaveLength(1);

    await firstMark.trigger('contextmenu');
    expect(wrapper.emitted('span-activate')).toHaveLength(2);
  });
});

describe('FullTextWorkspace modifier evidence', () => {
  const modifierSegments = [
    {
      key: 'modifier-0-0',
      text: 'Severe',
      highlighted: false,
      modifier: true,
      termIds: ['HP:0000365'],
      modifierIds: ['HP:0012828'],
      tooltip: 'Severe (HP:0012828)',
      findingTooltip: 'Hearing impairment (HP:0000365)',
    },
    { key: 'plain-1', text: ' ', highlighted: false },
    {
      key: 'mark-1',
      text: 'hearing loss',
      highlighted: true,
      termIds: ['HP:0000365'],
      annotationIds: ['auto-HP:0000365-0'],
      tooltip: 'Hearing impairment (HP:0000365)',
    },
  ];

  it('renders modifier evidence as a non-editable, labelled marker', () => {
    const wrapper = mountWorkspace({ segments: modifierSegments });
    const modifier = wrapper.get('[data-testid="annotated-note-modifier"]');
    expect(modifier.text()).toBe('Severe');
    expect(modifier.element.tagName).toBe('SPAN');
    expect(modifier.attributes('role')).toBeUndefined();
    expect(modifier.attributes('tabindex')).toBe('0');
    expect(modifier.attributes('aria-description')).toBe(
      'Clinical modifier Severe (HP:0012828) of Hearing impairment (HP:0000365)'
    );
    expect(wrapper.findAll('[data-testid="annotated-note-span"]')).toHaveLength(1);
  });

  it('emits hover for the linked finding and does not open the annotation menu', async () => {
    const wrapper = mountWorkspace({ segments: modifierSegments });
    const modifier = wrapper.get('[data-testid="annotated-note-modifier"]');

    await modifier.trigger('mouseenter');
    expect(wrapper.emitted('hover')?.[0]).toEqual([['HP:0000365']]);
    await modifier.trigger('mouseleave');
    expect(wrapper.emitted('clear-hover')).toHaveLength(1);

    await modifier.trigger('click');
    expect(wrapper.emitted('span-activate')).toBeUndefined();
  });

  it('emphasises modifier evidence when its finding is active', () => {
    const wrapper = mountWorkspace({
      segments: modifierSegments,
      activePhenotypeId: 'HP:0000365',
    });
    expect(wrapper.get('[data-testid="annotated-note-modifier"]').classes()).toContain(
      'annotated-note-modifier--active'
    );
  });
});
