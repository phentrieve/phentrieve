<template>
  <div class="full-text-receipt">
    <div class="full-text-receipt__title">{{ $t('queryInterface.fullTextReceipt.title') }}</div>
    <div class="text-caption text-medium-emphasis">
      {{ receiptMeta }}
    </div>
    <v-alert
      v-if="item?.response?.meta?.incomplete"
      data-testid="full-text-incomplete-warning"
      type="warning"
      variant="tonal"
      density="compact"
      class="mt-2"
    >
      {{ $t('queryInterface.fullTextReceipt.incomplete') }}
    </v-alert>
    <div v-if="phenotypes.length > 0" class="d-flex justify-end mt-3">
      <v-btn
        data-testid="full-text-response-add-all"
        size="small"
        rounded="pill"
        color="primary"
        :variant="confirmingAddAll ? 'flat' : 'tonal'"
        @click="emitAddAll"
      >
        <v-icon start size="small">mdi-plus-circle</v-icon>
        {{
          $t(
            confirmingAddAll
              ? 'queryInterface.fullTextReceipt.addAllConfirm'
              : 'queryInterface.fullTextReceipt.addAll',
            { count: phenotypes.length }
          )
        }}
      </v-btn>
    </div>
    <v-list
      v-if="phenotypes.length > 0"
      density="compact"
      class="rounded-lg mt-2 py-0 full-text-response-list"
    >
      <div
        v-for="(phenotype, index) in phenotypes"
        :key="`${item.id}-${phenotype.hpo_id}`"
        data-testid="full-text-response-phenotype"
        class="full-text-response-phenotype"
        :class="[
          markColourClass(markSlotFor(phenotype.hpo_id)),
          { 'full-text-response-phenotype--active': hoveredPhenotypeId === phenotype.hpo_id },
        ]"
        @mouseenter="emit('hover-phenotype', phenotype.hpo_id)"
        @mouseleave="emit('clear-hover')"
      >
        <ResultItem
          :result="mapTextProcessPhenotypeToResult(phenotype)"
          :rank="index + 1"
          :is-collected="isCollected(phenotype.hpo_id)"
          :mark-slot="markSlotFor(phenotype.hpo_id)"
          compact
          @add-to-collection="emit('add-to-collection', normalizeTextProcessPhenotype(phenotype))"
        >
          <template #badges>
            <v-chip
              v-if="phenotype.origin === 'manual'"
              data-testid="finding-origin-badge"
              size="x-small"
              color="primary"
              variant="tonal"
              label
            >
              {{ translate('annotatedDocumentPane.origin.manual', 'Manual') }}
            </v-chip>
          </template>
        </ResultItem>
      </div>
    </v-list>
    <section
      v-for="group in relatedFindings"
      :key="group.key"
      :data-testid="`full-text-related-findings-${group.key}`"
      class="mt-3"
    >
      <div class="full-text-receipt__title">
        {{ $t(`queryInterface.fullTextReceipt.${group.key}Findings`) }}
      </div>
      <div class="text-caption text-medium-emphasis">
        {{ $t('queryInterface.fullTextReceipt.relatedFindingsHint') }}
      </div>
      <v-list density="compact" class="full-text-response-list">
        <v-list-item
          v-for="term in group.terms"
          :key="`${group.key}-${term.hpo_id}`"
          :title="term.name"
          :subtitle="relatedFindingDetail(term)"
        >
          <ModifierChips
            v-if="term.modifiers && term.modifiers.length > 0"
            :modifiers="term.modifiers"
            class="mt-1"
          />
          <template v-if="isExcludedFinding(term)" #append>
            <v-chip size="x-small" variant="outlined" prepend-icon="mdi-minus-circle-outline" label>
              {{ $t('queryInterface.fullTextReceipt.relatedFindingAbsent') }}
            </v-chip>
          </template>
        </v-list-item>
      </v-list>
    </section>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import ResultItem from './ResultItem.vue';
import ModifierChips from './ModifierChips.vue';
import { useFullTextCuration } from '../composables/useFullTextCuration';
import { hashSlot, markColourClass } from '../utils/markColours';

defineOptions({
  name: 'FullTextResponseReceipt',
});

const props = defineProps({
  item: {
    type: Object,
    required: true,
  },
  noteText: {
    type: String,
    default: '',
  },
  collectedPhenotypes: {
    type: Array,
    default: () => [],
  },
  hoveredPhenotypeId: {
    type: String,
    default: null,
  },
});

const emit = defineEmits([
  'add-to-collection',
  'add-all-to-collection',
  'hover-phenotype',
  'clear-hover',
]);

let i18n = null;
try {
  i18n = useI18n();
} catch {
  i18n = null;
}
function translate(key, fallback, params) {
  if (!i18n) return fallback;
  // A numeric `count` selects the plural form ("1 word | {count} words").
  return typeof params?.count === 'number'
    ? i18n.t(key, params, params.count)
    : i18n.t(key, params ?? {});
}

const curation = useFullTextCuration(props.item.id);

// Seed once the response is available (idempotent; shared with the note curator).
watch(
  () => props.item?.response,
  () => curation.ensureSeeded(props.item, props.noteText || props.item?.query || ''),
  { immediate: true }
);

// Curated, term-level findings (auto terms minus removals, plus manual additions,
// with replaced terms reflecting their new HPO id). Origin drives the badge.
const phenotypes = computed(() =>
  curation.findings.value.filter(
    (term) => term && typeof term.hpo_id === 'string' && typeof term.name === 'string'
  )
);

// HPO term details (definition/synonyms) are only on the original response, not
// in the lightweight annotation model — look them up by id for display.
const detailsByHpoId = computed(() => {
  const map = new Map();
  const terms = Array.isArray(props.item?.response?.aggregated_hpo_terms)
    ? props.item.response.aggregated_hpo_terms
    : [];
  terms.forEach((term) => {
    if (term && typeof term.hpo_id === 'string') map.set(term.hpo_id, term);
  });
  return map;
});

const receiptMeta = computed(() => {
  const terms = phenotypes.value.length;
  const chunks = props.item?.response?.processed_chunks?.length ?? 0;

  if (chunks === 0 && typeof props.item?.query === 'string' && props.item.query.trim().length > 0) {
    return translate(
      'queryInterface.fullTextReceipt.metaNoteReview',
      `${terms} findings in submitted note review`,
      { count: terms }
    );
  }

  return translate(
    'queryInterface.fullTextReceipt.metaSections',
    `${terms} findings across ${chunks} document sections`,
    { count: terms, sections: chunks }
  );
});

// Findings about relatives and other people stay out of the patient's list and "Add all".
const relatedFindings = computed(() => {
  const response = props.item?.response;
  return [
    { key: 'family', terms: response?.family_history_findings },
    { key: 'other', terms: response?.other_person_findings },
  ].filter((group) => Array.isArray(group.terms) && group.terms.length > 0);
});

function relatedFindingDetail(term) {
  const evidence = term.text_attributions?.[0]?.matched_text_in_chunk;
  return evidence ? `${term.hpo_id} · "${evidence}"` : term.hpo_id;
}

function isExcludedFinding(term) {
  return normalizeAssertionStatus(term.status, term.excluded) === 'negated';
}

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

function matchingResponseDetails(term) {
  if (term?.origin === 'manual') {
    return null;
  }
  const details = detailsByHpoId.value.get(term.hpo_id);
  if (!details) {
    return null;
  }
  const termStatus = normalizeAssertionStatus(term.status, term.excluded);
  const detailStatus = normalizeAssertionStatus(details.status, details.excluded);
  return termStatus === detailStatus ? details : null;
}

function resolveTermModifiers(term) {
  if (Array.isArray(term.modifiers) && term.modifiers.length > 0) {
    return term.modifiers;
  }
  const details = matchingResponseDetails(term);
  return Array.isArray(details?.modifiers) ? details.modifiers : [];
}

function resolveTermWarnings(term) {
  if (Array.isArray(term.warnings) && term.warnings.length > 0) {
    return term.warnings;
  }
  const details = matchingResponseDetails(term);
  return Array.isArray(details?.warnings) ? details.warnings : [];
}

function mapTextProcessPhenotypeToResult(term) {
  const confidence = parseScoreValue(term.confidence);
  const similarity = parseScoreValue(term.similarity);
  const details = detailsByHpoId.value.get(term.hpo_id) || {};
  const modifiers = resolveTermModifiers(term);
  const warnings = resolveTermWarnings(term);

  return {
    hpo_id: term.hpo_id,
    label: term.name,
    confidence,
    score: confidence,
    scoreType: 'confidence',
    // Manual findings have no score; null hides the chip instead of a red 0.00.
    similarity: confidence ?? similarity ?? null,
    definition: details.definition || '',
    synonyms: Array.isArray(details.synonyms) ? details.synonyms : [],
    assertion_status: normalizeAssertionStatus(term.status, term.excluded),
    ...(modifiers.length > 0 ? { modifiers } : {}),
    ...(warnings.length > 0 ? { warnings } : {}),
  };
}

function normalizeAssertionStatus(status, excluded = false) {
  // Ruled-out findings ('absent', 'normal', or flagged excluded by the API) must
  // never reach the collection as present phenotypes (#404).
  if (excluded === true || status === 'absent' || status === 'normal') {
    return 'negated';
  }

  if (status === 'present') {
    return 'affirmed';
  }

  if (
    status === 'affirmed' ||
    status === 'negated' ||
    status === 'uncertain' ||
    status === 'unknown'
  ) {
    return status;
  }

  return 'unknown';
}

function normalizeTextProcessPhenotype(term) {
  const modifiers = resolveTermModifiers(term);
  return {
    hpo_id: term.hpo_id,
    label: term.name,
    assertion_status: normalizeAssertionStatus(term.status, term.excluded),
    ...(modifiers.length > 0 ? { modifiers } : {}),
  };
}

// Same colour slot as the finding's marks in the note (#562). A term with no
// visible mark (e.g. redacted note) still gets its stable hashed slot.
function markSlotFor(hpoId) {
  const slots = curation.colourSlots.value;
  return slots.has(hpoId) ? slots.get(hpoId) : hashSlot(hpoId);
}

function isCollected(hpoId) {
  return props.collectedPhenotypes.some((entry) => entry.hpo_id === hpoId);
}

// Two-step add (mirrors the collection's clear): first click arms, a second within 4s adds
const confirmingAddAll = ref(false);
let confirmTimer = null;
onBeforeUnmount(() => clearTimeout(confirmTimer));

function emitAddAll() {
  clearTimeout(confirmTimer);
  if (!confirmingAddAll.value) {
    confirmingAddAll.value = true;
    confirmTimer = setTimeout(() => (confirmingAddAll.value = false), 4000);
    return;
  }
  confirmingAddAll.value = false;
  emit(
    'add-all-to-collection',
    phenotypes.value.map((phenotype) => normalizeTextProcessPhenotype(phenotype))
  );
}
</script>

<style scoped>
.full-text-receipt {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.full-text-receipt__title {
  font-weight: 600;
}

.full-text-response-list {
  background: transparent;
}

.full-text-response-phenotype {
  border-radius: 6px;
  transition: background-color 0.16s ease;
}

/* Hovering a mark in the note highlights its row; a background tint keeps the
   compact list from jumping (#569). */
.full-text-response-phenotype--active {
  background: rgba(var(--ph-mark-rgb, 37, 99, 235), 0.1);
  box-shadow: inset 3px 0 0 rgb(var(--ph-mark-solid, 29, 78, 216));
}
</style>
