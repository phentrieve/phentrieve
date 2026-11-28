<template>
  <PhenotypeCardRow
    :hpo-id="result.hpo_id"
    :label="result.label"
    :color="isCollected ? 'primary-lighten-5' : 'grey-lighten-5'"
    :modifiers="compact ? [] : result.modifiers"
    :warnings="result.warnings"
    :compact="compact"
  >
    <template #prepend>
      <v-badge
        :content="rank"
        :color="markClass ? undefined : 'primary'"
        inline
        class="mr-2"
        :class="markClass ? ['result-item__mark-badge', markClass] : []"
        data-testid="result-item-rank"
      />
    </template>

    <template #inline-tools>
      <v-chip
        v-if="assertionBadge"
        size="x-small"
        :color="assertionBadge.color"
        :variant="assertionBadge.variant"
        :prepend-icon="assertionBadge.icon"
        label
        class="assertion-chip"
        :class="{ 'ml-1': !compact }"
      >
        {{ assertionBadge.label }}
      </v-chip>
      <slot name="badges" />
      <v-btn
        v-if="hasDetails"
        variant="text"
        size="x-small"
        icon
        density="compact"
        data-testid="result-item-details-toggle"
        :class="{ 'ml-1': !compact }"
        :aria-expanded="detailsExpanded ? 'true' : 'false'"
        @click.stop="detailsExpanded = !detailsExpanded"
      >
        <v-icon size="small">
          {{
            detailsExpanded ? 'mdi-chevron-up' : compact ? 'mdi-chevron-down' : 'mdi-information'
          }}
        </v-icon>
        <v-tooltip
          activator="parent"
          location="top"
          :aria-label="
            detailsExpanded
              ? $t('resultsDisplay.hideDetails', 'Hide details')
              : $t('resultsDisplay.showDetails', 'Show details')
          "
        >
          {{
            detailsExpanded
              ? $t('resultsDisplay.hideDetails', 'Hide details')
              : $t('resultsDisplay.showDetails', 'Show details')
          }}
        </v-tooltip>
      </v-btn>
    </template>

    <template #actions>
      <div
        class="d-flex flex-row align-center gap-1 score-chips-container"
        :class="compact ? 'mr-1 score-chips-container--compact' : 'mr-2'"
      >
        <SimilarityScore
          v-if="displayScore != null"
          :score="displayScore"
          :type="scoreType"
          :decimals="2"
          :show-animation="false"
          :subtle="compact"
        />
      </div>

      <v-btn
        :icon="isCollected ? 'mdi-check-circle' : 'mdi-plus-circle'"
        size="small"
        :color="isCollected ? 'success' : 'primary'"
        variant="text"
        class="flex-shrink-0 add-btn"
        :class="{ 'add-btn--compact': compact }"
        :density="compact ? 'compact' : 'default'"
        :disabled="isCollected"
        :title="
          isCollected
            ? $t('resultsDisplay.alreadyInCollectionTooltip', { id: result.hpo_id })
            : $t('resultsDisplay.addToCollectionTooltip', { id: result.hpo_id })
        "
        :aria-label="
          isCollected
            ? $t('resultsDisplay.alreadyInCollectionAriaLabel', {
                id: result.hpo_id,
                label: result.label,
              })
            : $t('resultsDisplay.addToCollectionAriaLabel', {
                id: result.hpo_id,
                label: result.label,
              })
        "
        @click.stop="$emit('add-to-collection', result)"
      />
    </template>

    <template #details>
      <v-expand-transition>
        <div
          v-if="detailsExpanded && hasDetails"
          data-testid="result-item-details"
          class="details-section"
          :class="compact ? 'mt-1 mb-1' : 'mt-3 pt-3'"
        >
          <div v-if="compact && result.modifiers && result.modifiers.length > 0" class="mb-2">
            <ModifierChips :modifiers="result.modifiers" />
          </div>
          <div v-if="result.definition && result.definition.trim()" class="mb-3">
            <div class="text-caption text-medium-emphasis mb-1 font-weight-bold">
              {{ $t('resultsDisplay.definitionLabel', 'Definition') }}:
            </div>
            <div class="text-body-2 definition-text">
              {{ result.definition }}
            </div>
          </div>

          <div v-if="result.synonyms && result.synonyms.length > 0" class="mb-2">
            <div class="text-caption text-medium-emphasis mb-1 font-weight-bold">
              {{ $t('resultsDisplay.synonymsLabel', 'Synonyms') }}:
            </div>
            <div class="d-flex flex-wrap ga-1">
              <v-chip
                v-for="(synonym, synIdx) in result.synonyms"
                :key="synIdx"
                size="small"
                variant="tonal"
                color="blue-grey"
                class="synonym-chip"
              >
                {{ synonym }}
              </v-chip>
            </div>
          </div>
        </div>
      </v-expand-transition>
    </template>
  </PhenotypeCardRow>
</template>

<script setup>
import { ref, computed } from 'vue';
import { useI18n } from 'vue-i18n';
import SimilarityScore from './SimilarityScore.vue';
import PhenotypeCardRow from './PhenotypeCardRow.vue';
import ModifierChips from './ModifierChips.vue';
import { markColourClass } from '../utils/markColours';

const props = defineProps({
  result: { type: Object, required: true },
  rank: { type: Number, required: true },
  isCollected: { type: Boolean, default: false },
  // One-line row (#569): modifiers move into the expandable details.
  compact: { type: Boolean, default: false },
  // Colour slot of the finding's marks in the note (#562); null = default.
  markSlot: { type: Number, default: null },
});

defineEmits(['add-to-collection']);

const { t } = useI18n();

const detailsExpanded = ref(false);

const markClass = computed(() => markColourClass(props.markSlot));

const hasDetails = computed(() => {
  return (
    (props.result.definition && props.result.definition.trim()) ||
    (props.result.synonyms && props.result.synonyms.length > 0) ||
    (props.compact && Array.isArray(props.result.modifiers) && props.result.modifiers.length > 0)
  );
});

const assertionBadge = computed(() => {
  if (props.result.assertion_status === 'negated') {
    // Absent findings read as a neutral state, not as an error or warning (#570).
    return {
      label: t('resultsDisplay.negated'),
      color: undefined,
      variant: 'outlined',
      icon: 'mdi-minus-circle-outline',
    };
  }

  if (props.result.assertion_status === 'uncertain') {
    return { label: t('resultsDisplay.uncertain'), color: 'blue-grey', variant: 'tonal' };
  }

  return null;
});

function parseScoreValue(value) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return null;
}

const displayScore = computed(() => {
  const explicitScore = parseScoreValue(props.result.score);
  if (explicitScore != null) {
    return explicitScore;
  }

  const confidenceScore = parseScoreValue(props.result.confidence);
  if (confidenceScore != null) {
    return confidenceScore;
  }

  // No score at all (e.g. a manual full-text finding): null hides the chip.
  return parseScoreValue(props.result.similarity);
});

const scoreType = computed(() => {
  if (props.result.scoreType === 'confidence' || props.result.scoreType === 'similarity') {
    return props.result.scoreType;
  }

  if (
    parseScoreValue(props.result.confidence) != null &&
    parseScoreValue(props.result.similarity) == null
  ) {
    return 'confidence';
  }

  return 'similarity';
});
</script>

<style scoped>
.add-btn {
  transform: scale(1.2);
  margin-left: 8px;
}

.score-chips-container {
  margin-top: 4px;
}

.score-chips-container--compact {
  margin-top: 0;
}

.add-btn.add-btn--compact {
  transform: none;
  margin-left: 0;
}

/* Rank badge in the finding's mark colour (#562); the solid shades reach
   >= 4.5:1 against white text. */
/* Fixed badge width keeps labels aligned past rank 9. */
.phenotype-card-row--compact :deep(.v-badge__badge) {
  min-width: 26px;
}

.result-item__mark-badge :deep(.v-badge__badge) {
  background: rgb(var(--ph-mark-solid));
  color: #fff;
}

.assertion-chip {
  font-weight: 600;
}

.details-section {
  border-top: 1px solid rgba(var(--v-theme-on-surface), 0.12);
  background-color: rgba(var(--v-theme-on-surface), 0.02);
  padding: 8px 12px;
  border-radius: 4px;
}

.definition-text {
  line-height: 1.6;
  color: rgba(var(--v-theme-on-surface), 0.87);
}

.synonym-chip {
  font-size: 0.75rem;
}
</style>
