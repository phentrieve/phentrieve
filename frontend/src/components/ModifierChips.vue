<template>
  <div
    v-if="validModifiers.length > 0"
    class="modifier-chips d-flex flex-wrap align-center ga-1"
    data-testid="modifier-chips"
    role="list"
    :aria-label="labelText"
  >
    <v-chip
      v-for="(modifier, idx) in validModifiers"
      :key="`${modifier.term_id}-${idx}`"
      :size="size"
      variant="tonal"
      color="info"
      label
      class="modifier-chip"
      data-testid="modifier-chip"
      role="listitem"
      :href="hpoTermUrl(modifier.term_id)"
      target="_blank"
      rel="noopener noreferrer"
      :aria-label="modifierAriaLabel(modifier)"
      @click.stop
    >
      {{ modifier.label }}
      <v-tooltip
        activator="parent"
        location="top"
        :text="modifierTooltip(modifier)"
        :aria-label="modifierTooltip(modifier)"
      />
    </v-chip>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { HPO_TERM_URL } from '../constants/urls';

const props = defineProps({
  modifiers: {
    type: Array,
    default: () => [],
  },
  size: {
    type: String,
    default: 'x-small',
  },
});

let i18n = null;
try {
  i18n = useI18n();
} catch {
  i18n = null;
}

const labelText = computed(() => (i18n ? i18n.t('resultsDisplay.modifiersLabel') : 'Modifiers'));

const validModifiers = computed(() =>
  (Array.isArray(props.modifiers) ? props.modifiers : []).filter(
    (modifier) =>
      modifier &&
      typeof modifier.term_id === 'string' &&
      modifier.term_id.trim() !== '' &&
      typeof modifier.label === 'string' &&
      modifier.label.trim() !== ''
  )
);

function hpoTermUrl(termId) {
  return HPO_TERM_URL(termId);
}

function modifierTooltip(modifier) {
  const quotes = (Array.isArray(modifier.segments) ? modifier.segments : [])
    .map((segment) => (typeof segment?.text === 'string' ? segment.text.trim() : ''))
    .filter(Boolean)
    .map((text) => `"${text}"`);

  if (quotes.length > 0) {
    return `${modifier.term_id} · ${quotes.join(', ')}`;
  }

  return modifier.term_id;
}

function modifierAriaLabel(modifier) {
  return `${modifier.label} (${modifier.term_id})`;
}
</script>

<style scoped>
.modifier-chip {
  text-decoration: none;
}
</style>
