import { afterEach, describe, it, expect, vi } from 'vitest';
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

describe('QueryForm submit shortcut (#572)', () => {
  afterEach(() => vi.restoreAllMocks());

  function onPlatform(platform) {
    vi.spyOn(window.navigator, 'platform', 'get').mockReturnValue(platform);
  }

  it('shows Ctrl+Enter in the note hint and tooltip on non-Apple platforms', () => {
    onPlatform('Linux x86_64');
    const wrapper = mountForm({ modelValue: 'Ataxia', isTextProcessModeActive: true });
    const button = wrapper.get('.search-submit-button');

    expect(wrapper.text()).toContain('Ctrl+Enter to analyze');
    expect(wrapper.vm.submitTooltip).toBe('Analyze (Ctrl+Enter)');
    expect(button.attributes('aria-keyshortcuts')).toBe('Control+Enter Meta+Enter');
  });

  it('shows Cmd+Enter on macOS', () => {
    onPlatform('MacIntel');
    const wrapper = mountForm({ modelValue: 'Ataxia', isTextProcessModeActive: true });

    expect(wrapper.text()).toContain('\u2318+Enter to analyze');
    expect(wrapper.get('.search-submit-button').attributes('aria-keyshortcuts')).toBe(
      'Meta+Enter Control+Enter'
    );
  });

  it('names plain Enter in query mode', () => {
    onPlatform('Win32');
    const wrapper = mountForm({ modelValue: 'short stature' });

    expect(wrapper.vm.submitTooltip).toBe('Search (Enter)');
    expect(wrapper.get('.search-submit-button').attributes('aria-keyshortcuts')).toBe('Enter');
  });

  it('submits a note on Ctrl+Enter and Cmd+Enter but not on plain Enter', async () => {
    const wrapper = mountForm({ modelValue: 'Ataxia', isTextProcessModeActive: true });
    const textarea = wrapper.get('textarea');

    await textarea.trigger('keydown', { key: 'Enter' });
    expect(wrapper.emitted('submit')).toBeUndefined();

    await textarea.trigger('keydown', { key: 'Enter', ctrlKey: true });
    await textarea.trigger('keydown', { key: 'Enter', metaKey: true });
    expect(wrapper.emitted('submit')).toHaveLength(2);
  });

  it('emphasises the hint only while the note has text', () => {
    const empty = mountForm({ modelValue: '', isTextProcessModeActive: true });
    expect(empty.find('.search-input--has-text').exists()).toBe(false);

    const filled = mountForm({ modelValue: 'Ataxia', isTextProcessModeActive: true });
    expect(filled.find('.search-input--has-text').exists()).toBe(true);
  });
});
