<template>
  <v-container class="dev-preview py-6">
    <v-alert type="info" variant="tonal" density="compact" class="mb-4">
      Dev preview with mock data: fictional note, no API call, nothing is redacted. Only available
      in the Vite dev server.
      <template #append>
        <v-btn size="small" variant="text" data-testid="dev-preview-reset" @click="reset">
          Reset
        </v-btn>
      </template>
    </v-alert>

    <v-sheet :key="renderKey" class="dev-preview__note pa-4 mb-4" rounded="lg">
      <FullTextNoteCurator
        :item="item"
        :note-text="item.query"
        :expanded="expanded"
        :active-phenotype-id="hoveredId"
        @toggle="expanded = !expanded"
        @hover="onNoteHover"
        @clear-hover="hoveredId = null"
        @add-to-collection="onAdd"
      />
    </v-sheet>

    <v-sheet :key="`receipt-${renderKey}`" class="pa-4" rounded="lg" border>
      <FullTextResponseReceipt
        :item="item"
        :note-text="item.query"
        :collected-phenotypes="collected"
        :hovered-phenotype-id="hoveredId"
        @add-to-collection="onAdd"
        @add-all-to-collection="(list) => list.forEach(onAdd)"
        @hover-phenotype="hoveredId = $event"
        @clear-hover="hoveredId = null"
      />
    </v-sheet>
  </v-container>
</template>

<script setup>
import { ref } from 'vue';
import FullTextNoteCurator from '../components/FullTextNoteCurator.vue';
import FullTextResponseReceipt from '../components/FullTextResponseReceipt.vue';
import { useFullTextCurationStore } from '../stores/fullTextCuration';
import { seedAnnotationsFromResponse } from '../composables/useUserNoteAnnotations';
import { MOCK_TURN_ID, buildMockItem, buildMockManualAnnotation } from '../dev/fullTextMock';

defineOptions({ name: 'DevFullTextPreview' });

const store = useFullTextCurationStore();
const item = buildMockItem();
const expanded = ref(true);
const hoveredId = ref(null);
const collected = ref([]);
const renderKey = ref(0);

function seed() {
  store.dropTurn(MOCK_TURN_ID);
  store.seedTurn(MOCK_TURN_ID, [
    ...seedAnnotationsFromResponse({ note: item.query, response: item.response }),
    buildMockManualAnnotation(),
  ]);
}

if (!store.isSeeded(MOCK_TURN_ID)) seed();

function reset() {
  seed();
  collected.value = [];
  hoveredId.value = null;
  renderKey.value += 1;
}

function onNoteHover(termIds) {
  if (Array.isArray(termIds) && termIds.length > 0) hoveredId.value = termIds[0];
}

function onAdd(phenotype) {
  if (!collected.value.some((entry) => entry.hpo_id === phenotype.hpo_id)) {
    collected.value = [...collected.value, phenotype];
  }
}
</script>

<style scoped>
.dev-preview {
  max-width: 920px;
}

.dev-preview__note {
  background: rgba(var(--v-theme-primary), 0.06);
}
</style>
