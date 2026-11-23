import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';

vi.mock('../../services/PhentrieveService', () => ({
  default: {
    rankDiseases: vi.fn(),
    getConfigInfo: vi.fn(),
  },
}));
vi.mock('../../services/logService', () => ({
  logService: { info: vi.fn(), debug: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import PhentrieveService from '../../services/PhentrieveService';
import {
  useDiseaseRankingStore,
  toProfileTerm,
  RANK_DEBOUNCE_MS,
} from '../../stores/diseaseRanking';

const PHENOTYPES = [
  { hpo_id: 'HP:0001166', label: 'Arachnodactyly', assertion_status: 'affirmed' },
  { hpo_id: 'HP:0001249', label: 'Intellectual disability', assertion_status: 'negated' },
  { hpo_id: 'HP:0002650', label: 'Scoliosis', assertion_status: 'unknown' },
];

function response(level = 'disease') {
  return { level, diseases: [], genes: [], scored: [], excluded: [], not_scored: [] };
}

describe('diseaseRanking store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('maps collection statuses to API assertions', () => {
    expect(toProfileTerm({ hpo_id: 'HP:1', assertion_status: 'affirmed' }).assertion).toBe(
      'affirmed'
    );
    expect(toProfileTerm({ hpo_id: 'HP:1', assertion_status: 'present' }).assertion).toBe(
      'affirmed'
    );
    expect(toProfileTerm({ hpo_id: 'HP:1', assertion_status: 'absent' }).assertion).toBe('negated');
    expect(toProfileTerm({ hpo_id: 'HP:1', assertion_status: 'negated' }).assertion).toBe(
      'negated'
    );
    expect(toProfileTerm({ hpo_id: 'HP:1', assertion_status: 'unknown' }).assertion).toBe(
      'uncertain'
    );
    expect(toProfileTerm({ hpo_id: 'HP:1' }).assertion).toBe('affirmed');
  });

  it('is available only when enabled and the index is ready', async () => {
    PhentrieveService.getConfigInfo.mockResolvedValueOnce({
      disease_ranking: { enabled: true, index_ready: false },
    });
    const store = useDiseaseRankingStore();
    await store.checkAvailability();
    expect(store.available).toBe(false);

    setActivePinia(createPinia());
    PhentrieveService.getConfigInfo.mockResolvedValueOnce({
      disease_ranking: { enabled: true, index_ready: true },
    });
    const ready = useDiseaseRankingStore();
    await ready.checkAvailability();
    expect(ready.available).toBe(true);
    await ready.checkAvailability();
    expect(PhentrieveService.getConfigInfo).toHaveBeenCalledTimes(2); // cached per store
  });

  it('treats a failing or old config-info as unavailable', async () => {
    PhentrieveService.getConfigInfo.mockRejectedValueOnce(new Error('offline'));
    const store = useDiseaseRankingStore();
    await store.checkAvailability();
    expect(store.available).toBe(false);

    setActivePinia(createPinia());
    PhentrieveService.getConfigInfo.mockResolvedValueOnce({});
    const old = useDiseaseRankingStore();
    await old.checkAvailability();
    expect(old.available).toBe(false);
  });

  it('opens and ranks immediately with the mapped profile', async () => {
    PhentrieveService.rankDiseases.mockResolvedValueOnce(response());
    const store = useDiseaseRankingStore();
    await store.open(PHENOTYPES);
    expect(store.panelOpen).toBe(true);
    const [payload, options] = PhentrieveService.rankDiseases.mock.calls[0];
    expect(payload).toEqual({
      terms: [
        { hpo_id: 'HP:0001166', assertion: 'affirmed' },
        { hpo_id: 'HP:0001249', assertion: 'negated' },
        { hpo_id: 'HP:0002650', assertion: 'uncertain' },
      ],
      level: 'disease',
      top_k: 20,
      include_uncertain: false,
    });
    expect(options.signal).toBeInstanceOf(AbortSignal);
    expect(store.result).toEqual(response());
    expect(store.uncertainCount).toBe(1);
    expect(store.presentCount).toBe(1);
  });

  it('debounces reruns while open and does nothing while closed', async () => {
    PhentrieveService.rankDiseases.mockResolvedValue(response());
    const store = useDiseaseRankingStore();
    store.scheduleRank(PHENOTYPES);
    vi.advanceTimersByTime(RANK_DEBOUNCE_MS * 2);
    expect(PhentrieveService.rankDiseases).not.toHaveBeenCalled();

    await store.open(PHENOTYPES);
    expect(PhentrieveService.rankDiseases).toHaveBeenCalledTimes(1);
    store.scheduleRank(PHENOTYPES.slice(0, 1));
    store.scheduleRank(PHENOTYPES.slice(0, 2));
    vi.advanceTimersByTime(RANK_DEBOUNCE_MS - 1);
    expect(PhentrieveService.rankDiseases).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(1);
    expect(PhentrieveService.rankDiseases).toHaveBeenCalledTimes(2);
    expect(PhentrieveService.rankDiseases.mock.calls[1][0].terms).toHaveLength(2);
  });

  it('reruns when the level or the uncertain switch changes', async () => {
    PhentrieveService.rankDiseases.mockResolvedValue(response());
    const store = useDiseaseRankingStore();
    await store.open(PHENOTYPES);
    store.setLevel('gene');
    vi.advanceTimersByTime(RANK_DEBOUNCE_MS);
    store.setIncludeUncertain(true);
    vi.advanceTimersByTime(RANK_DEBOUNCE_MS);
    const last = PhentrieveService.rankDiseases.mock.calls.at(-1)[0];
    expect(last.level).toBe('gene');
    expect(last.include_uncertain).toBe(true);
    store.setLevel('pathway');
    expect(store.level).toBe('gene');
  });

  it('drops a superseded response', async () => {
    let resolveFirst;
    PhentrieveService.rankDiseases
      .mockImplementationOnce(() => new Promise((resolve) => (resolveFirst = resolve)))
      .mockResolvedValueOnce(response('gene'));
    const store = useDiseaseRankingStore();
    const first = store.open(PHENOTYPES);
    const firstSignal = PhentrieveService.rankDiseases.mock.calls[0][1].signal;
    await store.rankNow(PHENOTYPES);
    expect(firstSignal.aborted).toBe(true);
    resolveFirst(response('disease'));
    await first;
    expect(store.result.level).toBe('gene');
    expect(store.loading).toBe(false);
  });

  it('skips the request without a scorable term and clears the result', async () => {
    PhentrieveService.rankDiseases.mockResolvedValueOnce(response());
    const store = useDiseaseRankingStore();
    await store.open(PHENOTYPES);
    await store.rankNow([PHENOTYPES[1], PHENOTYPES[2]]);
    expect(PhentrieveService.rankDiseases).toHaveBeenCalledTimes(1);
    expect(store.result).toBeNull();
  });

  it('keeps the error and ignores canceled requests', async () => {
    PhentrieveService.rankDiseases.mockRejectedValueOnce({ status: 503, type: 'SERVER_ERROR' });
    const store = useDiseaseRankingStore();
    await store.open(PHENOTYPES);
    expect(store.error.status).toBe(503);
    expect(store.result).toBeNull();

    PhentrieveService.rankDiseases.mockRejectedValueOnce({ type: 'CANCELED' });
    await store.rankNow(PHENOTYPES);
    expect(store.error).toBeNull();
  });

  it('close cancels pending work', async () => {
    PhentrieveService.rankDiseases.mockResolvedValue(response());
    const store = useDiseaseRankingStore();
    await store.open(PHENOTYPES);
    store.scheduleRank(PHENOTYPES);
    store.close();
    vi.advanceTimersByTime(RANK_DEBOUNCE_MS * 2);
    expect(PhentrieveService.rankDiseases).toHaveBeenCalledTimes(1);
    expect(store.panelOpen).toBe(false);
  });
});
