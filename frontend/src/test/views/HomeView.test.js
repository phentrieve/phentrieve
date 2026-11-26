import { describe, it, expect, beforeEach, vi } from 'vitest';
import { mount, flushPromises } from '@vue/test-utils';
import { setActivePinia, createPinia } from 'pinia';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import HomeView from '../../views/HomeView.vue';
import { useConversationStore } from '../../stores/conversation';

// The template imports the public logo, which has no file path under vitest.
vi.mock('/favicon.svg', () => ({ default: '/favicon.svg' }));

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });

beforeEach(() => setActivePinia(createPinia()));

describe('HomeView reset dialog', () => {
  it('closes on Escape (#466)', async () => {
    const wrapper = mount(HomeView, {
      attachTo: document.body,
      global: { plugins: [vuetify, i18n], stubs: { QueryInterface: true } },
    });

    wrapper.vm.showResetDialog = true;
    await flushPromises();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    await flushPromises();

    expect(wrapper.vm.showResetDialog).toBe(false);
    wrapper.unmount();
  });
});

describe('HomeView compact header', () => {
  function mountHome() {
    return mount(HomeView, {
      global: { plugins: [vuetify, i18n], stubs: { QueryInterface: true } },
    });
  }

  it('shows the full header with the welcome text before any query', () => {
    const wrapper = mountHome();
    expect(wrapper.find('[data-testid="home-welcome-text"]').exists()).toBe(true);
    expect(wrapper.find('.content-wrapper--compact').exists()).toBe(false);
    expect(wrapper.get('img.logo-icon').attributes('width')).toBe('65');
  });

  it('shrinks to a small logo row without the welcome text once results exist', async () => {
    const wrapper = mountHome();
    useConversationStore().addQuery({ query: 'Ataxia', loading: false });
    await flushPromises();

    expect(wrapper.find('[data-testid="home-welcome-text"]').exists()).toBe(false);
    expect(wrapper.find('.content-wrapper--compact').exists()).toBe(true);
    expect(wrapper.get('img.logo-icon').attributes('width')).toBe('32');
  });
});
