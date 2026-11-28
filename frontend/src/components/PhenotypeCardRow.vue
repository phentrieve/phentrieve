<template>
  <v-list-item
    class="phenotype-card-row"
    :class="{
      'phenotype-card-row--clickable': clickable,
      'phenotype-card-row--compact': compact,
      'mb-1 rounded-lg': !compact,
    }"
    :color="color"
    :border="!compact"
    density="compact"
    @click="clickable ? $emit('click') : undefined"
  >
    <template v-if="$slots.prepend" #prepend>
      <slot name="prepend" />
    </template>
    <v-list-item-title v-if="compact" class="phenotype-card-row__compact-title">
      <div class="phenotype-card-row__compact-line">
        <span
          class="text-body-2 font-weight-medium text-high-emphasis phenotype-card-row__compact-label"
          :title="label"
          >{{ label }}</span
        >
        <a
          :href="hpoTermUrl(hpoId)"
          target="_blank"
          rel="noopener noreferrer"
          class="phenotype-card-row__hpo-link phenotype-card-row__hpo-link--compact text-caption"
          :title="`View ${hpoId} in HPO Browser`"
          @click.stop
        >
          <span class="phenotype-card-row__hpo-id">{{ hpoId }}</span>
        </a>
        <slot name="inline-tools" />
        <FindingWarningsIndicator v-if="warnings && warnings.length > 0" :warnings="warnings" />
        <div v-if="$slots.actions" class="d-flex align-center ml-auto phenotype-card-row__actions">
          <slot name="actions" />
        </div>
      </div>
      <ModifierChips v-if="modifiers && modifiers.length > 0" :modifiers="modifiers" class="mt-1" />
      <slot name="details" />
    </v-list-item-title>
    <v-list-item-title v-else class="pb-2">
      <div class="d-flex flex-wrap align-center">
        <div class="flex-grow-1">
          <div class="d-flex align-center flex-wrap ga-1 mb-1">
            <span
              class="text-body-1 font-weight-medium text-high-emphasis phenotype-card-row__label"
              >{{ label }}</span
            >
            <slot name="inline-tools" />
            <FindingWarningsIndicator
              v-if="warnings && warnings.length > 0"
              :warnings="warnings"
              class="ml-1"
            />
          </div>
          <a
            :href="hpoTermUrl(hpoId)"
            target="_blank"
            rel="noopener noreferrer"
            class="phenotype-card-row__hpo-link text-caption"
            :title="`View ${hpoId} in HPO Browser`"
            @click.stop
          >
            <span class="phenotype-card-row__hpo-id">{{ hpoId }}</span>
            <v-icon size="x-small" class="ml-1">mdi-open-in-new</v-icon>
          </a>
          <ModifierChips
            v-if="modifiers && modifiers.length > 0"
            :modifiers="modifiers"
            class="mt-1"
          />
        </div>
        <div v-if="$slots.actions" class="d-flex align-center ml-auto phenotype-card-row__actions">
          <slot name="actions" />
        </div>
      </div>
      <div v-if="$slots.metadata" class="mt-2">
        <slot name="metadata" />
      </div>
      <slot name="details" />
    </v-list-item-title>
  </v-list-item>
</template>

<script setup>
import { HPO_TERM_URL } from '../constants/urls';
import ModifierChips from './ModifierChips.vue';
import FindingWarningsIndicator from './FindingWarningsIndicator.vue';

defineProps({
  hpoId: { type: String, required: true },
  label: { type: String, required: true },
  color: { type: String, default: 'grey-lighten-5' },
  clickable: { type: Boolean, default: false },
  modifiers: { type: Array, default: () => [] },
  warnings: { type: Array, default: () => [] },
  // One-line row for dense lists (full-text findings, #569); modifiers are
  // passed only when the caller wants them inline.
  compact: { type: Boolean, default: false },
});

defineEmits(['click']);

function hpoTermUrl(hpoId) {
  return HPO_TERM_URL(hpoId);
}
</script>

<style scoped>
.phenotype-card-row__hpo-id {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.phenotype-card-row__hpo-link {
  text-decoration: none;
  color: rgba(var(--v-theme-on-surface), var(--v-medium-emphasis-opacity));
  display: inline-flex;
  align-items: center;
  transition: color 0.2s;
}

.phenotype-card-row__hpo-link:hover {
  color: rgb(var(--v-theme-primary));
  text-decoration: underline;
  text-underline-offset: 2px;
}

.phenotype-card-row__label {
  max-width: 100%;
  display: inline-block;
}

.phenotype-card-row__actions {
  gap: 4px;
}

/* Vuetify's default prepend spacer leaves a wide gap after the rank badge */
.phenotype-card-row :deep(.v-list-item__prepend .v-list-item__spacer) {
  width: 8px;
}

.phenotype-card-row--clickable {
  cursor: pointer;
}

/* Compact variant (#569): no card border, a hairline separator, tight padding. */
.phenotype-card-row--compact {
  min-height: 0 !important;
  padding-top: 2px !important;
  padding-bottom: 2px !important;
  border-bottom: 1px solid rgba(var(--v-theme-on-surface), 0.08);
}

.phenotype-card-row__compact-title {
  white-space: normal;
  overflow: visible;
}

.phenotype-card-row__compact-line {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  min-height: 32px;
}

.phenotype-card-row__compact-label {
  min-width: 0;
  flex: 0 1 auto;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.phenotype-card-row__hpo-link--compact {
  flex-shrink: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.72rem !important;
}

.phenotype-card-row--compact .phenotype-card-row__actions {
  flex-shrink: 0;
}
</style>
