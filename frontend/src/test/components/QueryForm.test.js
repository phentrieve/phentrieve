import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import QueryForm from '../../components/query/QueryForm.vue';

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });
const vuetify = createVuetify({ components, directives });

function mountForm(props) {
  return mount(QueryForm, {
    props: {
      queryInputLabel: 'Query',
      queryInputDescription: 'Query',
      textProcessInputLabel: 'Note',
      textProcessInputDescription: 'Note',
      searchButtonLabel: 'Search HPO terms',
      ...props,
    },
    global: { plugins: [vuetify, i18n] },
  });
}

describe('QueryForm search button', () => {
  it('stays enabled while loading so its spinner is not dimmed, but cannot resubmit', async () => {
    const wrapper = mountForm({ modelValue: '', isLoading: true });
    const button = wrapper.get('.search-submit-button');

    expect(button.attributes('disabled')).toBeUndefined();
    await button.trigger('click');
    expect(wrapper.emitted('submit')).toBeUndefined();
  });

  it('is disabled with empty input when idle and submits with text', async () => {
    expect(
      mountForm({ modelValue: '' }).get('.search-submit-button').attributes('disabled')
    ).toBeDefined();

    const wrapper = mountForm({ modelValue: 'short stature' });
    await wrapper.get('.search-submit-button').trigger('click');
    expect(wrapper.emitted('submit')).toHaveLength(1);
  });
});

describe('QueryForm compact full-text field', () => {
  it('shrinks an empty full-text field to one row once results exist', () => {
    const wrapper = mountForm({ modelValue: '', isTextProcessModeActive: true, compact: true });
    expect(wrapper.get('textarea').attributes('rows')).toBe('1');
    expect(wrapper.find('.search-input--compact').exists()).toBe(true);
  });

  it('grows back to three rows as soon as the user types, and is full size without results', () => {
    const typing = mountForm({
      modelValue: 'Ataxia',
      isTextProcessModeActive: true,
      compact: true,
    });
    expect(typing.get('textarea').attributes('rows')).toBe('3');
    expect(typing.find('.search-input--compact').exists()).toBe(false);

    const initial = mountForm({ modelValue: '', isTextProcessModeActive: true });
    expect(initial.get('textarea').attributes('rows')).toBe('3');
    expect(initial.find('.search-input--compact').exists()).toBe(false);
  });
});
