import { ref } from 'vue';
import { useConversationStore } from '../stores/conversation';
import { useFileDownload } from './useFileDownload';
import { logService } from '../services/logService';

// Mirrors the Python `is_excluded` set (B0 contract, phentrieve/assertion_vocab.py)
// so an assertion_status of 'absent' -- and 'normal', a normalcy verdict that is a
// ruled-out abnormality -- exports as excluded: true in the Phenopacket, not only
// literal 'negated'.
const EXCLUDED_ASSERTIONS = new Set([
  'negated',
  'negative',
  'absent',
  'excluded',
  'no',
  'denied',
  'normal',
]);
export const isExcluded = (a) =>
  EXCLUDED_ASSERTIONS.has(
    String(a ?? '')
      .trim()
      .toLowerCase()
  );
// Mirrors the Python uncertain set: a hedged finding is neither observed nor
// excluded, so it is left out of the Phenopacket (#399).
const UNCERTAIN_ASSERTIONS = new Set(['uncertain', 'possible', 'suspected', 'probable']);
export const isUncertain = (a) =>
  UNCERTAIN_ASSERTIONS.has(
    String(a ?? '')
      .trim()
      .toLowerCase()
  );

/**
 * Normalize and deduplicate clinical modifiers attached to a phenotype.
 * Preserves the API contract shape ({ term_id, label, segments?, source? }).
 */
export function normalizeModifiers(modifiers) {
  if (!Array.isArray(modifiers) || modifiers.length === 0) {
    return [];
  }

  const seenIds = new Set();
  const normalized = [];

  modifiers.forEach((modifier) => {
    if (!modifier || typeof modifier !== 'object') return;
    const termId = typeof modifier.term_id === 'string' ? modifier.term_id.trim() : '';
    const label = typeof modifier.label === 'string' ? modifier.label.trim() : '';
    if (!termId || !label || seenIds.has(termId)) return;

    seenIds.add(termId);
    const entry = { term_id: termId, label };
    if (Array.isArray(modifier.segments) && modifier.segments.length > 0) {
      const validSegments = modifier.segments
        .filter(
          (seg) =>
            seg &&
            typeof seg === 'object' &&
            Number.isInteger(seg.start) &&
            Number.isInteger(seg.end) &&
            seg.start >= 0 &&
            seg.end > seg.start &&
            typeof seg.text === 'string' &&
            seg.text.trim().length > 0
        )
        .map((seg) => ({
          start: seg.start,
          end: seg.end,
          text: seg.text,
        }));
      if (validSegments.length > 0) {
        entry.segments = validSegments;
      }
    }
    if (modifier.source !== undefined && modifier.source !== null) {
      entry.source = modifier.source;
    }
    normalized.push(entry);
  });

  return normalized;
}

function normalizeExportAssertionStatus(status) {
  if (isExcluded(status)) return 'negated';
  if (isUncertain(status)) return 'uncertain';
  return 'affirmed';
}

/**
 * Serialize collected phenotypes for POST /api/v1/phenopackets/export.
 * Includes `modifiers` on each phenotype item only when non-empty.
 */
export function buildPhenopacketExportPayload({ caseId = null, phenotypes = [] } = {}) {
  const serializedPhenotypes = (Array.isArray(phenotypes) ? phenotypes : []).map((cp) => {
    const modifiers = normalizeModifiers(cp.modifiers);
    return {
      hpo_id: cp.hpo_id,
      label: cp.label,
      assertion_status: normalizeExportAssertionStatus(cp.assertion_status),
      ...(modifiers.length > 0 ? { modifiers } : {}),
    };
  });

  return {
    ...(caseId ? { case_id: caseId } : {}),
    phenotypes: serializedPhenotypes,
  };
}

/**
 * Composable for managing phenotype collection state and actions.
 * Extracted from QueryInterface.vue to reduce component size.
 */
export function usePhenotypeCollection() {
  const conversationStore = useConversationStore();
  const { downloadText, downloadJson } = useFileDownload();

  // Subject information for Phenopacket export
  const phenopacketSubjectId = ref('');
  const phenopacketSex = ref(null);
  const phenopacketDateOfBirth = ref(null);

  function addPhenotype(phenotype, queryAssertionStatus = null) {
    const modifiers = normalizeModifiers(phenotype?.modifiers);
    const normalizedPhenotype =
      modifiers.length > 0 ? { ...phenotype, modifiers } : { ...phenotype };
    if (modifiers.length === 0 && 'modifiers' in normalizedPhenotype) {
      delete normalizedPhenotype.modifiers;
    }

    logService.info('Adding phenotype to collection', {
      phenotype: normalizedPhenotype,
      queryAssertionStatus,
      phenotypeHasAssertionStatus: normalizedPhenotype.assertion_status ? true : false,
      phenotypeAssertionStatus: normalizedPhenotype.assertion_status,
    });

    const added = conversationStore.addPhenotype(normalizedPhenotype, queryAssertionStatus);
    if (added) {
      logService.debug('Phenotype added to collection via store', {
        hpo_id: phenotype.hpo_id,
      });
    } else {
      // If the phenotype is already collected with the same polarity and the
      // new hit carries modifiers, merge any unseen modifiers onto the entry.
      if (modifiers.length > 0) {
        const existing = conversationStore.collectedPhenotypes.find(
          (item) => item.hpo_id === phenotype.hpo_id
        );
        const incomingStatus = normalizedPhenotype.assertion_status || queryAssertionStatus;
        const polarityMatches =
          !incomingStatus || isExcluded(existing?.assertion_status) === isExcluded(incomingStatus);
        if (existing && polarityMatches) {
          const merged = normalizeModifiers([...(existing.modifiers || []), ...modifiers]);
          if (merged.length > 0) {
            existing.modifiers = merged;
          }
        }
      }
      logService.debug('Phenotype was duplicate, not added', {
        hpo_id: phenotype.hpo_id,
      });
    }
  }

  function removePhenotype(index) {
    let phenotype = null;
    if (index >= 0 && index < conversationStore.collectedPhenotypes.length) {
      // Safe array index access after explicit bounds check.
      // eslint-disable-next-line security/detect-object-injection
      phenotype = conversationStore.collectedPhenotypes[index];
    }
    logService.info('Removing phenotype from collection', {
      index,
      phenotype,
    });
    conversationStore.removePhenotype(index);
  }

  function toggleAssertionStatus(index) {
    conversationStore.toggleAssertionStatus(index);
    let phenotype = null;
    if (index >= 0 && index < conversationStore.collectedPhenotypes.length) {
      // Safe array index access after explicit bounds check.
      // eslint-disable-next-line security/detect-object-injection
      phenotype = conversationStore.collectedPhenotypes[index];
    }
    if (phenotype) {
      logService.info('Toggled phenotype assertion status', {
        hpo_id: phenotype.hpo_id,
        newStatus: phenotype.assertion_status,
      });
    }
  }

  function clearCollection() {
    logService.info('Clearing phenotype collection and subject information');
    conversationStore.clearPhenotypes();
    phenopacketSubjectId.value = '';
    phenopacketSex.value = null;
    phenopacketDateOfBirth.value = null;
  }

  function toggleCollectionPanel() {
    conversationStore.toggleCollectionPanel();
  }

  function exportCollectionAsText() {
    const phenotypes = conversationStore.collectedPhenotypes;
    logService.info('Exporting phenotypes as text', { count: phenotypes.length });
    let exportText = 'HPO Phenotypes Collection\n';
    exportText += 'Exported on: ' + new Date().toLocaleString() + '\n\n';
    exportText += 'ID\tLabel\tAssertion Status\n';
    phenotypes.forEach((p) => {
      exportText += `${p.hpo_id}\t${p.label}\t${p.assertion_status || 'affirmed'}\n`;
    });
    downloadText(exportText, 'hpo_phenotypes.txt');
  }

  function exportAsPhenopacket() {
    const phenotypes = conversationStore.collectedPhenotypes;
    if (phenotypes.length === 0) {
      logService.warn('Attempted to export empty phenopacket collection');
      return;
    }
    logService.info('Starting Phenopacket export process', {
      count: phenotypes.length,
    });
    try {
      const timestamp = new Date().toISOString();
      const phenopacketId = `phentrieve-export-${Date.now()}`;
      const phenopacket = {
        id: phenopacketId,
        metaData: {
          created: timestamp,
          createdBy: 'Phentrieve Frontend',
          phenopacketSchemaVersion: '2.0.0',
          resources: [
            {
              id: 'phentrieve',
              name: 'Phentrieve',
              namespacePrefix: 'Phentrieve',
              url: 'https://phentrieve.org/',
              version: import.meta.env.VITE_APP_VERSION || '1.0.0',
              iriPrefix: 'phentrieve',
            },
          ],
        },
        phenotypicFeatures: [],
      };
      if (
        phenopacketSubjectId.value ||
        phenopacketSex.value !== null ||
        phenopacketDateOfBirth.value
      ) {
        phenopacket.subject = {};
        if (phenopacketSubjectId.value) phenopacket.subject.id = phenopacketSubjectId.value.trim();
        if (phenopacketSex.value !== null) {
          const sexMap = { 0: 'UNKNOWN_SEX', 1: 'FEMALE', 2: 'MALE', 3: 'OTHER_SEX' };
          phenopacket.subject.sex = sexMap[phenopacketSex.value];
        }
        if (phenopacketDateOfBirth.value) {
          const dob = new Date(phenopacketDateOfBirth.value + 'T00:00:00Z');
          if (!isNaN(dob.getTime())) {
            // Phenopacket v2: subject.dateOfBirth (ISO8601).
            // Do NOT use timeAtLastEncounter — that field has different semantics.
            phenopacket.subject.dateOfBirth = dob.toISOString();
          }
        }
        if (Object.keys(phenopacket.subject).length === 0) delete phenopacket.subject;
      }
      phenotypes.forEach((cp) => {
        if (isUncertain(cp.assertion_status)) return;
        const feature = {
          type: { id: cp.hpo_id, label: cp.label },
          excluded: isExcluded(cp.assertion_status),
        };
        const modifiers = normalizeModifiers(cp.modifiers);
        if (modifiers.length > 0) {
          feature.modifiers = modifiers.map((modifier) => ({
            id: modifier.term_id,
            label: modifier.label,
          }));
        }
        phenopacket.phenotypicFeatures.push(feature);
      });
      const filename = `phentrieve_phenopacket_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
      downloadJson(phenopacket, filename);
      logService.info('Phenopacket successfully exported', { filename });
    } catch (error) {
      logService.error('Error during Phenopacket export', { error });
      // Re-throw so the caller can surface this via an accessible notification
      // (snackbar/toast) rather than a blocking, untestable alert() dialog.
      throw error instanceof Error ? error : new Error('Error exporting Phenopacket.');
    }
  }

  return {
    phenopacketSubjectId,
    phenopacketSex,
    phenopacketDateOfBirth,
    addPhenotype,
    removePhenotype,
    toggleAssertionStatus,
    clearCollection,
    toggleCollectionPanel,
    exportCollectionAsText,
    exportAsPhenopacket,
    buildPhenopacketExportPayload: (caseId = null) =>
      buildPhenopacketExportPayload({
        caseId: caseId || phenopacketSubjectId.value?.trim() || null,
        phenotypes: conversationStore.collectedPhenotypes,
      }),
  };
}
