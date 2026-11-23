<template>
  <v-navigation-drawer
    :model-value="store.panelOpen"
    location="right"
    :width="drawerWidth"
    mobile-breakpoint="md"
    disable-resize-watcher
    style="z-index: 1510"
    :aria-label="t('diseaseRanking.title')"
    data-testid="disease-ranking-drawer"
    @update:model-value="onDrawerToggle"
    @keydown.esc="store.close()"
  >
    <div class="d-flex align-start pa-4 pb-3">
      <div class="flex-grow-1" style="min-width: 0">
        <h2 class="text-h6" data-testid="disease-ranking-title">
          {{ t('diseaseRanking.title') }}
        </h2>
        <div class="text-body-2 text-medium-emphasis">
          {{ t('diseaseRanking.subtitle') }}
        </div>
      </div>
      <v-btn
        icon="mdi-close"
        variant="text"
        :aria-label="t('diseaseRanking.close')"
        data-testid="disease-ranking-close"
        @click="store.close()"
      />
    </div>
    <v-divider />

    <div class="pa-4 pb-0">
      <v-alert
        type="info"
        variant="tonal"
        density="compact"
        icon="mdi-information-outline"
        class="text-body-2"
        data-testid="disease-ranking-notice"
      >
        <strong>{{ t('diseaseRanking.notice.title') }}</strong>
        {{ t('diseaseRanking.notice.body') }}
      </v-alert>
    </div>

    <div class="d-flex flex-wrap align-center ga-4 px-4 pt-3">
      <v-btn-toggle
        :model-value="store.level"
        mandatory
        density="comfortable"
        color="primary"
        variant="outlined"
        divided
        :aria-label="t('diseaseRanking.levelLabel')"
        @update:model-value="store.setLevel($event)"
      >
        <v-btn value="disease" data-testid="disease-ranking-level-disease">
          {{ t('diseaseRanking.level.disease') }}
        </v-btn>
        <v-btn value="gene" data-testid="disease-ranking-level-gene">
          {{ t('diseaseRanking.level.gene') }}
        </v-btn>
      </v-btn-toggle>
      <v-switch
        :model-value="store.includeUncertain"
        color="primary"
        density="compact"
        hide-details
        inset
        data-testid="disease-ranking-uncertain"
        :label="t('diseaseRanking.includeUncertain', { count: store.uncertainCount })"
        @update:model-value="store.setIncludeUncertain($event)"
      />
    </div>

    <div
      class="d-flex flex-wrap px-4 pt-1 pb-2 text-caption text-medium-emphasis status-line"
      data-testid="disease-ranking-status"
    >
      <template v-if="result">
        <span>{{ t('diseaseRanking.status.scored', { count: result.scored.length }) }}</span>
        <span v-if="result.excluded.length">
          {{ t('diseaseRanking.status.excluded', { count: result.excluded.length }) }}
        </span>
        <span v-if="result.not_scored.length">
          {{ t('diseaseRanking.status.notScored', { count: result.not_scored.length }) }}
        </span>
      </template>
      <span>{{ t('diseaseRanking.status.auto') }}</span>
    </div>
    <v-progress-linear
      :active="store.loading"
      indeterminate
      color="primary"
      height="2"
      :aria-label="t('diseaseRanking.loading')"
    />

    <div v-if="store.error" class="pa-4">
      <v-alert type="error" variant="tonal" density="compact" data-testid="disease-ranking-error">
        {{ errorMessage }}
      </v-alert>
    </div>

    <div
      v-else-if="!store.loading && !result"
      class="pa-6 text-center text-body-2 text-medium-emphasis"
      data-testid="disease-ranking-empty"
    >
      {{ t('diseaseRanking.empty') }}
    </div>

    <v-expansion-panels
      v-else-if="result"
      v-model="openRow"
      variant="accordion"
      flat
      class="disease-ranking-list"
      data-testid="disease-ranking-list"
    >
      <v-expansion-panel
        v-for="row in rows"
        :key="row.key"
        :value="row.key"
        :data-testid="`disease-ranking-row-${row.key}`"
      >
        <v-expansion-panel-title class="py-2">
          <div class="ranking-row">
            <span class="ranking-rank text-medium-emphasis">{{ row.rank }}</span>
            <div class="ranking-main">
              <div class="font-weight-medium" :class="{ 'font-italic': row.isGene }">
                {{ row.title }}
              </div>
              <div class="d-flex flex-wrap align-center ga-1 mt-1 text-caption">
                <v-chip v-for="id in row.ids" :key="id" size="x-small" variant="tonal" label>
                  {{ id }}
                </v-chip>
                <span v-if="row.subtitle" class="text-medium-emphasis ml-1">{{
                  row.subtitle
                }}</span>
                <v-chip
                  v-if="row.explanation.conflicts.length"
                  size="x-small"
                  color="warning"
                  variant="tonal"
                  prepend-icon="mdi-alert-circle-outline"
                >
                  {{
                    t('diseaseRanking.conflictChip', { count: row.explanation.conflicts.length })
                  }}
                </v-chip>
              </div>
            </div>
            <div class="ranking-score">
              <v-progress-linear
                :model-value="row.score * 100"
                color="primary"
                bg-color="surface-variant"
                height="6"
                rounded
                :aria-label="t('diseaseRanking.scoreLabel', { score: row.scoreText })"
              />
              <span class="ranking-score-text">{{ row.scoreText }}</span>
            </div>
          </div>
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <div class="text-overline">{{ t('diseaseRanking.why') }}</div>
          <div class="ranking-table-wrap">
            <table class="ranking-table text-body-2">
              <thead>
                <tr>
                  <th scope="col">{{ t('diseaseRanking.table.profile') }}</th>
                  <th scope="col">{{ t('diseaseRanking.table.disease') }}</th>
                  <th scope="col">{{ t('diseaseRanking.table.frequency') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="m in row.explanation.matches" :key="m.profile_id">
                  <td>{{ m.profile_label }}</td>
                  <td>
                    {{ m.disease_term_label }}
                    <span class="text-medium-emphasis">
                      ({{ t(`diseaseRanking.relation.${m.relation}`) }})
                    </span>
                  </td>
                  <td>{{ formatPercent(m.frequency) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <v-alert
            v-for="c in row.explanation.conflicts"
            :key="`c-${c.profile_id}`"
            type="warning"
            variant="tonal"
            density="compact"
            class="mt-2 text-body-2"
          >
            {{
              t('diseaseRanking.conflict', {
                profile: c.profile_label,
                disease: c.disease_term_label,
              })
            }}
          </v-alert>
          <div v-if="row.explanation.unexplained.length" class="mt-2 text-body-2">
            {{ t('diseaseRanking.unexplained') }}
            {{ row.explanation.unexplained.map((m) => m.profile_label).join(', ') }}
          </div>
          <div class="mt-2 text-caption text-medium-emphasis d-flex flex-wrap ga-2">
            <span>
              {{ t('diseaseRanking.sources') }}
              {{ row.explanation.annotation_sources.map(sourceName).join(' + ') }}
            </span>
            <a
              v-for="link in row.links"
              :key="link.href"
              :href="link.href"
              target="_blank"
              rel="noopener noreferrer"
            >
              {{ link.label }}
              <v-icon size="x-small" icon="mdi-open-in-new" />
            </a>
          </div>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <template #append>
      <v-divider />
      <div class="pa-3 text-caption text-medium-emphasis" data-testid="disease-ranking-footer">
        <div v-if="result">
          {{
            t('diseaseRanking.footer.summary', {
              shown: rows.length,
              total: result.total_candidates,
              hpo: result.index_info?.hpo_version ?? '-',
              mondo: result.index_info?.mondo_version ?? '-',
            })
          }}
        </div>
        <div>{{ t('diseaseRanking.footer.attribution') }}</div>
      </div>
    </template>
  </v-navigation-drawer>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { useDisplay } from 'vuetify';
import { useDiseaseRankingStore } from '../stores/diseaseRanking';

const props = defineProps({
  phenotypes: { type: Array, default: () => [] },
});

const { t } = useI18n();
const display = useDisplay();
// Beside the case workspace from md up; full width over it on small screens
const drawerWidth = computed(() => (display.mdAndUp.value ? 560 : display.width.value));
const store = useDiseaseRankingStore();
const openRow = ref(null);

const result = computed(() => store.result);

function idLinks(id) {
  const [prefix, num] = String(id).split(':');
  if (prefix === 'OMIM') return { label: 'OMIM', href: `https://omim.org/entry/${num}` };
  if (prefix === 'ORPHA') {
    return { label: 'Orphanet', href: `https://www.orpha.net/en/disease/detail/${num}` };
  }
  return null;
}

const rows = computed(() => {
  const data = result.value;
  if (!data) return [];
  if (data.level === 'gene') {
    return data.genes.map((g) => ({
      key: `gene-${g.gene}`,
      rank: g.rank,
      title: g.gene,
      isGene: true,
      ids: [g.best_disease_id],
      subtitle: t('diseaseRanking.bestDisease', { label: g.best_disease_label }),
      score: g.score,
      scoreText: g.score.toFixed(2),
      explanation: g.explanation,
      links: [idLinks(g.best_disease_id)].filter(Boolean),
    }));
  }
  return data.diseases.map((d) => ({
    key: d.disease_id,
    rank: d.rank,
    title: d.label,
    isGene: false,
    ids: [d.disease_id, ...d.twins],
    subtitle: d.genes.length ? t('diseaseRanking.genes', { genes: d.genes.join(', ') }) : '',
    score: d.score,
    scoreText: d.score.toFixed(2),
    explanation: d.explanation,
    links: [d.disease_id, ...d.twins].map(idLinks).filter(Boolean),
  }));
});

const errorMessage = computed(() => {
  const status = store.error?.status;
  if (status === 503) return t('diseaseRanking.errors.unavailable');
  if (status === 404) return t('diseaseRanking.errors.disabled');
  return t('diseaseRanking.errors.generic');
});

function formatPercent(value) {
  return `${Math.round(Number(value) * 100)} %`;
}

function sourceName(source) {
  if (source === 'orpha') return 'Orphanet';
  if (source === 'omim') return 'OMIM';
  return source;
}

function onDrawerToggle(value) {
  if (!value) store.close();
}

watch(
  () => props.phenotypes,
  (value) => store.scheduleRank(value),
  { deep: true }
);
watch(
  () => store.result,
  () => {
    openRow.value = null;
  }
);
</script>

<style scoped>
.status-line {
  column-gap: 12px;
  row-gap: 2px;
}
.status-line > span {
  white-space: nowrap;
}
.ranking-row {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) 120px;
  gap: 12px;
  align-items: center;
  width: 100%;
}
.ranking-rank {
  text-align: right;
  font-weight: 500;
}
.ranking-main {
  min-width: 0;
}
.ranking-score {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ranking-score-text {
  font-variant-numeric: tabular-nums;
  width: 34px;
  text-align: right;
}
.ranking-table-wrap {
  overflow-x: auto;
}
.ranking-table {
  width: 100%;
  border-collapse: collapse;
}
.ranking-table th {
  text-align: left;
  font-weight: 500;
  padding: 4px 8px 4px 0;
  color: rgba(var(--v-theme-on-surface), var(--v-medium-emphasis-opacity));
}
.ranking-table td {
  padding: 6px 8px 6px 0;
  border-top: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
  vertical-align: top;
}
</style>
