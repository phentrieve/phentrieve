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
// Must match DiseaseRankingRequestAPI.terms max_length on the server
export const MAX_PROFILE_TERMS = 200;
const ASSERTION_PRIORITY = { affirmed: 0, uncertain: 1, negated: 2 };
const HPO_ID_RE = /^HP:\d{7}$/;

/**
 * Profile terms for the request: one entry per valid HPO ID (first wins), present terms
 * first, then uncertain, then excluded, capped at MAX_PROFILE_TERMS.
 * @returns {{terms: Array<{hpo_id: string, assertion: string}>, dropped: number}}
 */
export function buildProfile(phenotypes) {
  const seen = new Set();
  const unique = [];
  for (const p of Array.isArray(phenotypes) ? phenotypes : []) {
    const hpoId = typeof p?.hpo_id === 'string' ? p.hpo_id.trim() : '';
    if (!hpoId || !HPO_ID_RE.test(hpoId) || seen.has(hpoId)) continue;
    seen.add(hpoId);
    unique.push(toProfileTerm({ ...p, hpo_id: hpoId }));
  }
  const ordered = unique
    .map((t, i) => ({ t, i }))
    .sort(
      (a, b) => ASSERTION_PRIORITY[a.t.assertion] - ASSERTION_PRIORITY[b.t.assertion] || a.i - b.i
    )
    .map(({ t }) => t);
  return {
    terms: ordered.slice(0, MAX_PROFILE_TERMS),
    dropped: Math.max(0, ordered.length - MAX_PROFILE_TERMS),
  };
}

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
  const droppedTerms = ref(0);

  let timer = null;
  let controller = null;
  let seq = 0;
  let statusPromise = null;

  const available = computed(() => Boolean(status.value?.enabled && status.value?.index_ready));
  const dedupedProfileTerms = computed(() => buildProfile(lastPhenotypes.value).terms);
  const uncertainCount = computed(
    () => dedupedProfileTerms.value.filter((t) => t.assertion === 'uncertain').length
  );
  const presentCount = computed(
    () => dedupedProfileTerms.value.filter((t) => t.assertion === 'affirmed').length
  );

  async function checkAvailability() {
    // Re-check if ranking is enabled but the index was still building on the last probe.
    if (status.value && (!status.value.enabled || status.value.index_ready)) {
      return status.value;
    }
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
    const { terms, dropped } = buildProfile(lastPhenotypes.value);
    droppedTerms.value = dropped;
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
    // Cancel any in-flight request immediately so it cannot resolve with stale
    // results while the debounce timer is waiting.
    cancelPending();
    seq++;
    loading.value = false;
    timer = setTimeout(() => {
      timer = null;
      rankNow(lastPhenotypes.value);
    }, RANK_DEBOUNCE_MS);
  }

  function open(phenotypes) {
    panelOpen.value = true;
    result.value = null;
    error.value = null;
    return rankNow(phenotypes);
  }

  function close() {
    panelOpen.value = false;
    cancelPending();
    seq++;
    loading.value = false;
  }

  function clearCase() {
    close();
    error.value = null;
    result.value = null;
    lastPhenotypes.value = [];
    droppedTerms.value = 0;
  }

  function setLevel(value) {
    if ((value !== 'disease' && value !== 'gene') || value === level.value) return;
    level.value = value;
    if (panelOpen.value) scheduleRank(lastPhenotypes.value);
  }

  function setIncludeUncertain(value) {
    const next = Boolean(value);
    if (next === includeUncertain.value) return;
    includeUncertain.value = next;
    if (panelOpen.value) scheduleRank(lastPhenotypes.value);
  }

  function $reset() {
    clearCase();
    status.value = null;
    level.value = 'disease';
    includeUncertain.value = false;
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
    droppedTerms,
    checkAvailability,
    rankNow,
    scheduleRank,
    open,
    close,
    clearCase,
    setLevel,
    setIncludeUncertain,
    $reset,
  };
});
