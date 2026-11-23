import { describe, it, expect, vi, beforeEach } from 'vitest';
import { mount, flushPromises } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import { createI18n } from 'vue-i18n';
import en from '../../locales/en.json';

vi.mock('../../services/PhentrieveService', () => ({
  default: { rankDiseases: vi.fn(), getConfigInfo: vi.fn() },
}));
vi.mock('../../services/logService', () => ({
  logService: { info: vi.fn(), debug: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import PhentrieveService from '../../services/PhentrieveService';
import DiseaseRankingPanel from '../../components/DiseaseRankingPanel.vue';
import PhenotypeCollectionPanel from '../../components/PhenotypeCollectionPanel.vue';
import { useDiseaseRankingStore } from '../../stores/diseaseRanking';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });
const drawerStub = {
  'v-navigation-drawer': {
    template:
      '<div class="drawer-stub" :data-testid="$attrs[\'data-testid\']"><slot /><slot name="append" /></div>',
  },
};

const explanation = {
  matches: [
    {
      profile_id: 'HP:0001166',
      profile_label: 'Arachnodactyly',
      disease_term_id: 'HP:0001166',
      disease_term_label: 'Arachnodactyly',
      relation: 'exact',
      shared_id: 'HP:0001166',
      shared_label: 'Arachnodactyly',
      match: 0.97,
      frequency: 0.895,
      source: 'both',
    },
  ],
  unexplained: [
    {
      profile_id: 'HP:0000767',
      profile_label: 'Pectus excavatum',
      disease_term_id: 'HP:0000765',
      disease_term_label: 'Abnormal thorax morphology',
      relation: 'broader',
      shared_id: 'HP:0000765',
      shared_label: 'Abnormal thorax morphology',
      match: 0.2,
      frequency: 0.35,
      source: 'omim',
    },
  ],
  conflicts: [
    {
      profile_id: 'HP:0001249',
      profile_label: 'Intellectual disability',
      disease_term_id: 'HP:0001249',
      disease_term_label: 'Intellectual disability',
      frequency: 0.5,
    },
  ],
  annotation_sources: ['omim', 'orpha'],
};

const DISEASE_RESULT = {
  level: 'disease',
  diseases: [
    {
      rank: 1,
      disease_id: 'OMIM:154700',
      label: 'Marfan syndrome',
      score: 0.66,
      genes: ['FBN1'],
      twins: ['ORPHA:558'],
      explanation,
    },
  ],
  genes: [],
  scored: ['HP:0001166', 'HP:0000767'],
  excluded: ['HP:0001249'],
  not_scored: [{ hpo_id: 'HP:0002650', reason: 'uncertain' }],
  total_candidates: 7650,
  scorer_version: 'x',
  index_info: { hpo_version: 'v2026-09-01', mondo_version: 'v2026-09-01' },
  intended_use: 'Research use only.',
};

function mountPanel(phenotypes = []) {
  return mount(DiseaseRankingPanel, {
    props: { phenotypes },
    global: { plugins: [vuetify, i18n], stubs: drawerStub },
  });
}

describe('DiseaseRankingPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
  });

  it('shows the research-use notice and the empty state', () => {
    const store = useDiseaseRankingStore();
    store.panelOpen = true;
    const wrapper = mountPanel();
    expect(wrapper.find('[data-testid="disease-ranking-notice"]').text()).toContain(
      'Research use only, not a diagnosis.'
    );
    expect(wrapper.find('[data-testid="disease-ranking-empty"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="disease-ranking-footer"]').text()).toContain('Orphanet');
  });

  it('renders ranked diseases with twins, genes, score and conflicts', async () => {
    const store = useDiseaseRankingStore();
    store.panelOpen = true;
    store.result = DISEASE_RESULT;
    const wrapper = mountPanel();
    await flushPromises();
    const row = wrapper.find('[data-testid="disease-ranking-row-OMIM:154700"]');
    expect(row.exists()).toBe(true);
    const text = row.text();
    expect(text).toContain('Marfan syndrome');
    expect(text).toContain('OMIM:154700');
    expect(text).toContain('ORPHA:558');
    expect(text).toContain('Gene FBN1');
    expect(text).toContain('0.66');
    expect(text).toContain('1 conflict(s)');
    const status = wrapper.find('[data-testid="disease-ranking-status"]').text();
    expect(status).toContain('Scored: 2 term(s)');
    expect(status).toContain('1 excluded');
    expect(status).toContain('1 not scored');
    expect(wrapper.find('[data-testid="disease-ranking-footer"]').text()).toContain(
      'Top 1 of 7650'
    );
  });

  it('shows the explanation when a row is expanded', async () => {
    const store = useDiseaseRankingStore();
    store.panelOpen = true;
    store.result = DISEASE_RESULT;
    const wrapper = mountPanel();
    await wrapper.find('[data-testid="disease-ranking-row-OMIM:154700"] button').trigger('click');
    await flushPromises();
    const text = wrapper.text();
    expect(text).toContain('Why this match');
    expect(text).toContain('(same)');
    expect(text).toContain('90 %');
    expect(text).toContain('Not explained: Pectus excavatum');
    expect(text).toContain('Intellectual disability is excluded in the profile');
    const links = wrapper.findAll('a').map((a) => a.attributes('href'));
    expect(links).toContain('https://omim.org/entry/154700');
    expect(links).toContain('https://www.orpha.net/en/disease/detail/558');
  });

  it('renders the gene level', async () => {
    const store = useDiseaseRankingStore();
    store.panelOpen = true;
    store.result = {
      ...DISEASE_RESULT,
      level: 'gene',
      diseases: [],
      genes: [
        {
          rank: 1,
          gene: 'FBN1',
          score: 0.66,
          best_disease_id: 'OMIM:154700',
          best_disease_label: 'Marfan syndrome',
          explanation,
        },
      ],
    };
    const wrapper = mountPanel();
    const text = wrapper.find('[data-testid="disease-ranking-row-gene-FBN1"]').text();
    expect(text).toContain('FBN1');
    expect(text).toContain('best disease: Marfan syndrome');
  });

  it('maps a 503 to the unavailable message', () => {
    const store = useDiseaseRankingStore();
    store.panelOpen = true;
    store.error = { status: 503 };
    const wrapper = mountPanel();
    expect(wrapper.find('[data-testid="disease-ranking-error"]').text()).toContain(
      'not available on this server'
    );
  });

  it('level toggle and close call the store', async () => {
    PhentrieveService.rankDiseases.mockResolvedValue(DISEASE_RESULT);
    const store = useDiseaseRankingStore();
    store.panelOpen = true;
    store.result = DISEASE_RESULT;
    const setLevel = vi.spyOn(store, 'setLevel');
    const close = vi.spyOn(store, 'close');
    const wrapper = mountPanel();
    await wrapper.find('[data-testid="disease-ranking-level-gene"]').trigger('click');
    expect(setLevel).toHaveBeenCalledWith('gene');
    await wrapper.find('[data-testid="disease-ranking-close"]').trigger('click');
    expect(close).toHaveBeenCalled();
  });

  it('reschedules the ranking when the collection changes', async () => {
    const store = useDiseaseRankingStore();
    const schedule = vi.spyOn(store, 'scheduleRank');
    const wrapper = mountPanel([{ hpo_id: 'HP:0001166', assertion_status: 'affirmed' }]);
    await wrapper.setProps({
      phenotypes: [
        { hpo_id: 'HP:0001166', assertion_status: 'affirmed' },
        { hpo_id: 'HP:0001083', assertion_status: 'affirmed' },
      ],
    });
    expect(schedule).toHaveBeenCalledTimes(1);
    expect(schedule.mock.calls[0][0]).toHaveLength(2);
  });
});

describe('PhenotypeCollectionPanel disease ranking entry', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
  });

  async function mountCollection(status, phenotypes) {
    PhentrieveService.getConfigInfo.mockResolvedValue({ disease_ranking: status });
    const wrapper = mount(PhenotypeCollectionPanel, {
      props: { phenotypes, panelOpen: true, sexOptions: [] },
      global: { plugins: [vuetify, i18n], stubs: drawerStub },
    });
    await flushPromises();
    return wrapper;
  }

  const phenotypes = [
    { hpo_id: 'HP:0001166', label: 'Arachnodactyly', assertion_status: 'affirmed' },
  ];

  it('is hidden when the server has the feature off or no index', async () => {
    const off = await mountCollection({ enabled: false, index_ready: false }, phenotypes);
    expect(off.find('[data-testid="collection-rank-diseases"]').exists()).toBe(false);
    expect(off.find('[data-testid="disease-ranking-drawer"]').exists()).toBe(false);

    setActivePinia(createPinia());
    const noIndex = await mountCollection({ enabled: true, index_ready: false }, phenotypes);
    expect(noIndex.find('[data-testid="collection-rank-diseases"]').exists()).toBe(false);
  });

  it('opens the ranking with the collected phenotypes', async () => {
    PhentrieveService.rankDiseases.mockResolvedValue(DISEASE_RESULT);
    const wrapper = await mountCollection({ enabled: true, index_ready: true }, phenotypes);
    const button = wrapper.find('[data-testid="collection-rank-diseases"]');
    expect(button.exists()).toBe(true);
    expect(button.text()).toContain('Similar diseases');
    await button.trigger('click');
    await flushPromises();
    expect(PhentrieveService.rankDiseases).toHaveBeenCalledTimes(1);
    expect(PhentrieveService.rankDiseases.mock.calls[0][0].terms).toEqual([
      { hpo_id: 'HP:0001166', assertion: 'affirmed' },
    ]);
    expect(useDiseaseRankingStore().panelOpen).toBe(true);
  });

  it('is disabled without a present phenotype', async () => {
    const wrapper = await mountCollection({ enabled: true, index_ready: true }, [
      { hpo_id: 'HP:0001249', assertion_status: 'negated' },
    ]);
    expect(
      wrapper.find('[data-testid="collection-rank-diseases"]').attributes('disabled')
    ).toBeDefined();
  });
});

describe('DiseaseRankingPanel with the app Vuetify registration', () => {
  it('uses only registered Vuetify components', async () => {
    setActivePinia(createPinia());
    const appVuetify = (await import('../../plugins/vuetify')).default;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const store = useDiseaseRankingStore();
    store.panelOpen = true;
    store.result = DISEASE_RESULT;
    mount(DiseaseRankingPanel, {
      props: { phenotypes: [] },
      global: { plugins: [appVuetify, i18n], stubs: drawerStub },
    });
    await flushPromises();
    const unresolved = warn.mock.calls
      .map((args) => String(args[0]))
      .filter((msg) => msg.includes('Failed to resolve component'));
    warn.mockRestore();
    expect(unresolved).toEqual([]);
  });
});
