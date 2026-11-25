import { describe, it, expect } from 'vitest';
import { mount, shallowMount } from '@vue/test-utils';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';
import AdvancedOptionsPanel from '../../components/AdvancedOptionsPanel.vue';
import {
  DEFAULT_AGGREGATED_TERM_CONFIDENCE,
  DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
  DEFAULT_MIN_SEGMENT_LENGTH,
  DEFAULT_NUM_RESULTS_PER_CHUNK,
  DEFAULT_SIMILARITY_THRESHOLD,
  DEFAULT_SPLIT_THRESHOLD,
  DEFAULT_STEP_SIZE,
  DEFAULT_WINDOW_SIZE,
} from '../../constants/defaults';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });

describe('AdvancedOptionsPanel', () => {
  it('takes its prop defaults from constants/defaults.js (#426)', () => {
    const wrapper = shallowMount(AdvancedOptionsPanel, { global: { plugins: [i18n] } });

    expect(wrapper.props()).toMatchObject({
      similarityThreshold: DEFAULT_SIMILARITY_THRESHOLD,
      windowSize: DEFAULT_WINDOW_SIZE,
      stepSize: DEFAULT_STEP_SIZE,
      chunkRetrievalThreshold: DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
      aggregatedTermConfidence: DEFAULT_AGGREGATED_TERM_CONFIDENCE,
      splitThreshold: DEFAULT_SPLIT_THRESHOLD,
      minSegmentLength: DEFAULT_MIN_SEGMENT_LENGTH,
      numResultsPerChunk: DEFAULT_NUM_RESULTS_PER_CHUNK,
    });
  });

  it('renders extractModifiers switch off by default in LLM mode and emits updates when toggled', async () => {
    const wrapper = mount(AdvancedOptionsPanel, {
      props: {
        visible: true,
        isTextProcessModeActive: true,
        textProcessOptions: {
          extractionBackend: 'llm',
          llmMode: 'two_phase',
        },
      },
      global: { plugins: [vuetify, i18n] },
    });

    const toggle = wrapper.findComponent('[data-testid="extract-modifiers-switch"]');
    expect(toggle.exists()).toBe(true);
    expect(toggle.props('modelValue')).toBe(false);

    await toggle.vm.$emit('update:modelValue', true);
    expect(wrapper.emitted('update:textProcessOptions')).toHaveLength(1);
    expect(wrapper.emitted('update:textProcessOptions')[0][0]).toEqual({
      extractionBackend: 'llm',
      llmMode: 'two_phase',
      extractModifiers: true,
    });
  });

  it('hides extractModifiers switch in standard extraction backend mode', () => {
    const wrapper = mount(AdvancedOptionsPanel, {
      props: {
        visible: true,
        isTextProcessModeActive: true,
        textProcessOptions: {
          extractionBackend: 'standard',
        },
      },
      global: { plugins: [vuetify, i18n] },
    });

    expect(wrapper.find('[data-testid="extract-modifiers-switch"]').exists()).toBe(false);
  });
});
