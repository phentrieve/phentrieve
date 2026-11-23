/**
 * Pinia store for the disease ranking panel (research use, flagged on the server).
 *
 * Nothing here is persisted: profiles and results stay in memory (compare #547).
 * While the panel is open, the ranking reruns (debounced) when the collection,
 * the level or the uncertain switch changes; a superseded request is aborted and
 * its response dropped.
 *
 * @module stores/diseaseRanking
 */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import PhentrieveService from '../services/PhentrieveService';
import { logService } from '../services/logService';
import { isExcluded, isUncertain } from '../composables/usePhenotypeCollection';

export const RANK_DEBOUNCE_MS = 400;
export const RANK_TOP_K = 20;

// Collection and B0 export vocabulary (phentrieve/assertion_vocab.py) for a present finding.
const AFFIRMED_ASSERTIONS = new Set(['', 'affirmed', 'present', 'yes', 'positive', 'confirmed']);

/**
 * Maps a collected phenotype to the API profile term. Excluded and uncertain use the
 * same sets as the Phenopacket export; 'unknown' and any unrecognized status fail
 * closed to 'uncertain', so they are never scored as present by default.
 */
export function toProfileTerm(phenotype) {
  const status = phenotype?.assertion_status;
  let assertion = 'uncertain';
  if (isExcluded(status)) assertion = 'negated';
  else if (isUncertain(status)) assertion = 'uncertain';
  else if (
    AFFIRMED_ASSERTIONS.has(
      String(status ?? '')
        .trim()
        .toLowerCase()
    )
  )
    assertion = 'affirmed';
  return { hpo_id: phenotype.hpo_id, assertion };
}

export const useDiseaseRankingStore = defineStore('diseaseRanking', () => {
  const service = PhentrieveService;

  const status = ref(null); // config-info `disease_ranking`, null until checked
  const panelOpen = ref(false);
  const level = ref('disease');
  const includeUncertain = ref(false);
  const loading = ref(false);
  const error = ref(null);
  const result = ref(null);
  const lastPhenotypes = ref([]);

  let timer = null;
  let controller = null;
  let seq = 0;
  let statusPromise = null;

  const available = computed(() => Boolean(status.value?.enabled && status.value?.index_ready));
  const uncertainCount = computed(
    () => lastPhenotypes.value.filter((p) => toProfileTerm(p).assertion === 'uncertain').length
  );
  const presentCount = computed(
    () => lastPhenotypes.value.filter((p) => toProfileTerm(p).assertion === 'affirmed').length
  );

  async function checkAvailability() {
    if (status.value) return status.value;
    if (!statusPromise) {
      statusPromise = service
        .getConfigInfo()
        .then((config) => {
          status.value = config?.disease_ranking ?? { enabled: false, index_ready: false };
          return status.value;
        })
        .catch((err) => {
          // Leave status unset so the next workspace open asks again (cold start, blip)
          logService.warn('Disease ranking status unavailable', {
            message: err?.originalErrorDetails?.message ?? String(err),
          });
          return { enabled: false, index_ready: false };
        })
        .finally(() => {
          statusPromise = null;
        });
    }
    return statusPromise;
  }

  function cancelPending() {
    clearTimeout(timer);
    timer = null;
    if (controller) {
      controller.abort();
      controller = null;
    }
  }

  async function rankNow(phenotypes = lastPhenotypes.value) {
    cancelPending();
    lastPhenotypes.value = Array.isArray(phenotypes) ? [...phenotypes] : [];
    const terms = lastPhenotypes.value.filter((p) => p?.hpo_id).map(toProfileTerm);
    const mine = ++seq;
    const scorable = (t) =>
      t.assertion === 'affirmed' || (includeUncertain.value && t.assertion === 'uncertain');
    if (!terms.some(scorable)) {
      result.value = null;
      error.value = null;
      loading.value = false;
      return;
    }
    controller = new AbortController();
    loading.value = true;
    error.value = null;
    try {
      const data = await service.rankDiseases(
        {
          terms,
          level: level.value,
          top_k: RANK_TOP_K,
          include_uncertain: includeUncertain.value,
        },
        { signal: controller.signal }
      );
      if (mine !== seq) return; // superseded
      result.value = data;
    } catch (err) {
      if (mine !== seq || err?.type === 'CANCELED') return;
      error.value = err;
      result.value = null;
    } finally {
      if (mine === seq) {
        loading.value = false;
        controller = null;
      }
    }
  }

  function scheduleRank(phenotypes) {
    lastPhenotypes.value = Array.isArray(phenotypes) ? [...phenotypes] : [];
    if (!panelOpen.value) return;
    clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      rankNow(lastPhenotypes.value);
    }, RANK_DEBOUNCE_MS);
  }

  function open(phenotypes) {
    panelOpen.value = true;
    return rankNow(phenotypes);
  }

  function close() {
    panelOpen.value = false;
    cancelPending();
    seq++;
    loading.value = false;
  }

  function setLevel(value) {
    if (value !== 'disease' && value !== 'gene') return;
    level.value = value;
    if (panelOpen.value) scheduleRank(lastPhenotypes.value);
  }

  function setIncludeUncertain(value) {
    includeUncertain.value = Boolean(value);
    if (panelOpen.value) scheduleRank(lastPhenotypes.value);
  }

  function $reset() {
    close();
    status.value = null;
    level.value = 'disease';
    includeUncertain.value = false;
    error.value = null;
    result.value = null;
    lastPhenotypes.value = [];
  }

  return {
    status,
    panelOpen,
    level,
    includeUncertain,
    loading,
    error,
    result,
    available,
    uncertainCount,
    presentCount,
    checkAvailability,
    rankNow,
    scheduleRank,
    open,
    close,
    setLevel,
    setIncludeUncertain,
    $reset,
  };
});
