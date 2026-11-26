<template>
  <span
    v-if="validWarnings.length > 0"
    class="finding-warning-indicator d-inline-flex align-center"
    tabindex="0"
    role="img"
    :aria-label="accessibleLabel"
    data-testid="finding-warning-indicator"
    @click.stop
  >
    <v-icon size="small" color="warning">mdi-alert-circle-outline</v-icon>
    <v-tooltip
      activator="parent"
      location="top"
      open-on-focus
      :text="warningSummary"
      :aria-label="accessibleLabel"
    />
  </span>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps({
  warnings: {
    type: Array,
    default: () => [],
  },
});

let i18n = null;
try {
  i18n = useI18n();
} catch {
  i18n = null;
}

const warningsLabel = computed(() =>
  i18n ? i18n.t('resultsDisplay.warningsLabel') : 'Extraction warnings'
);

const validWarnings = computed(() =>
  (Array.isArray(props.warnings) ? props.warnings : []).filter(
    (warning) =>
      warning &&
      ((typeof warning.message === 'string' && warning.message.trim() !== '') ||
        (typeof warning.code === 'string' && warning.code.trim() !== ''))
  )
);

const warningSummary = computed(() =>
  validWarnings.value
    .map((warning) => {
      const text = (warning.message || warning.code || '').trim();
      return warning.term_id ? `${warning.term_id}: ${text}` : text;
    })
    .join('; ')
);

const accessibleLabel = computed(() =>
  warningSummary.value ? `${warningsLabel.value}: ${warningSummary.value}` : warningsLabel.value
);
</script>

<style scoped>
.finding-warning-indicator {
  cursor: help;
  border-radius: 4px;
}

.finding-warning-indicator:focus-visible {
  outline: 2px solid rgb(var(--v-theme-primary));
  outline-offset: 2px;
}
</style>
