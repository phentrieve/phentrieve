import { afterEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { nextTick } from 'vue';
import { createI18n } from 'vue-i18n';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import en from '../../locales/en.json';
import PiiReviewDialog from '../../components/PiiReviewDialog.vue';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });

Object.defineProperty(window, 'visualViewport', {
  configurable: true,
  value: {
    width: 1024,
    height: 768,
    offsetLeft: 0,
    offsetTop: 0,
    pageLeft: 0,
    pageTop: 0,
    scale: 1,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  },
});

const TEXT = 'Mail jane@example.org on 12/03/2024';
const FINDINGS = [
  { id: 'email-0', category: 'email', confidence: 'high', start: 5, end: 21 },
  { id: 'date-0', category: 'date', confidence: 'review', start: 25, end: 35 },
];

function mountDialog(props = {}) {
  return mount(PiiReviewDialog, {
    props: {
      modelValue: true,
      text: TEXT,
      findings: FINDINGS,
      ...props,
    },
    global: { plugins: [vuetify, i18n] },
    attachTo: document.body,
  });
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('PiiReviewDialog', () => {
  it('lists each finding with its category and matched text', async () => {
    const wrapper = mountDialog();
    await nextTick();

    expect(document.body.textContent).toContain('Email');
    expect(document.body.textContent).toContain('jane@example.org');
    expect(document.body.textContent).toContain('Date');
    expect(document.body.textContent).toContain('12/03/2024');

    wrapper.unmount();
  });

  it('emits continue with high-confidence findings selected by default', async () => {
    const wrapper = mountDialog();
    await nextTick();

    document.querySelector('[data-testid="pii-cancel"]').click();
    document.querySelector('[data-testid="pii-continue"]').click();
    await nextTick();

    expect(wrapper.emitted('cancel')).toHaveLength(1);
    expect(wrapper.emitted('continue')[0]).toEqual([['email-0']]);

    wrapper.unmount();
  });

  it('lets the user deselect a false positive', async () => {
    const wrapper = mountDialog();
    await nextTick();

    document.querySelector('[data-testid="pii-finding-email-0"] input').click();
    await nextTick();
    document.querySelector('[data-testid="pii-continue"]').click();
    await nextTick();

    expect(wrapper.emitted('continue')[0]).toEqual([[]]);

    wrapper.unmount();
  });
});
