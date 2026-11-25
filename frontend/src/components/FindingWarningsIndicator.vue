<template>
  <span
    v-if="validWarnings.length > 0"
    class="finding-warning-indicator d-inline-flex align-center"
    tabindex="0"
    role="img"
    :aria-label="warningSummary"
    data-testid="finding-warning-indicator"
    @click.stop
  >
    <v-icon size="small" color="warning">mdi-alert-circle-outline</v-icon>
    <v-tooltip
      activator="parent"
      location="top"
      :text="warningSummary"
      :aria-label="warningSummary"
    />
  </span>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  warnings: {
    type: Array,
    default: () => [],
  },
});

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
</script>

<style scoped>
.finding-warning-indicator {
  cursor: help;
}
</style>
