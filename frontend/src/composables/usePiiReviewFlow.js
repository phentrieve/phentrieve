import { ref } from 'vue';
import { redactPiiFindings, scanPii } from '../pii';

export function usePiiReviewFlow({ logService = null } = {}) {
  const piiReviewDialogVisible = ref(false);
  const pendingPiiSubmission = ref(null);

  function openPiiReview(pendingSubmission) {
    pendingPiiSubmission.value = pendingSubmission;
    piiReviewDialogVisible.value = true;
  }

  function cancelPiiReview() {
    piiReviewDialogVisible.value = false;
    pendingPiiSubmission.value = null;
  }

  function scanSubmissionForPii({
    text,
    locale = 'en',
    useTextProcessMode = false,
    isAutoSubmit = false,
  }) {
    const scanResult = scanPii(text, { locale });

    // Only high-confidence findings interrupt the submission; review-level ones
    // (titled names, bare dates) are offered in the dialog but never block.
    if (!scanResult.findings.some((finding) => finding.confidence === 'high')) {
      return { needsReview: false, scanResult };
    }

    const pendingSubmission = {
      text,
      useTextProcessMode,
      isAutoSubmit,
      scanResult,
    };

    openPiiReview(pendingSubmission);
    logService?.info?.('PII review required before submission', {
      mode: useTextProcessMode ? 'textProcess' : 'query',
      textLength: String(text ?? '').length,
      piiSummary: scanResult.summary,
    });

    return { needsReview: true, scanResult, pendingSubmission };
  }

  // selectedFindingIds: the findings the user left ticked in the dialog.
  // Without it, every high-confidence finding is redacted.
  async function continueWithPiiRedaction({ submitQueryText, selectedFindingIds } = {}) {
    const pending = pendingPiiSubmission.value;

    if (!pending) {
      return null;
    }

    const selectedFindings = pending.scanResult.findings.filter((finding) =>
      selectedFindingIds ? selectedFindingIds.includes(finding.id) : finding.confidence === 'high'
    );
    const redaction = redactPiiFindings(pending.text, selectedFindings, {
      includeReviewFindings: true,
    });

    cancelPiiReview();

    if (typeof submitQueryText === 'function') {
      await submitQueryText({
        currentQuery: redaction.text,
        rawQueryForHistory: pending.text,
        redactedQueryForHistory: redaction.text,
        useTextProcessMode: pending.useTextProcessMode,
        isAutoSubmit: pending.isAutoSubmit,
        piiScanResult: pending.scanResult,
        redactionApplied: redaction.changed,
      });
    }

    return redaction;
  }

  return {
    piiReviewDialogVisible,
    pendingPiiSubmission,
    openPiiReview,
    cancelPiiReview,
    scanSubmissionForPii,
    continueWithPiiRedaction,
  };
}
