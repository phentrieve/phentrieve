import { describe, expect, it, vi } from 'vitest';
import { usePiiReviewFlow } from '../../composables/usePiiReviewFlow';

describe('usePiiReviewFlow', () => {
  it('opens a review dialog and stores pending submission metadata when PII is detected', () => {
    const logService = { info: vi.fn() };
    const flow = usePiiReviewFlow({ logService });

    const result = flow.scanSubmissionForPii({
      text: 'MRN: AB-123456 with seizures',
      locale: 'en',
      useTextProcessMode: false,
      isAutoSubmit: false,
    });

    expect(result.needsReview).toBe(true);
    expect(flow.piiReviewDialogVisible.value).toBe(true);
    expect(flow.pendingPiiSubmission.value).toMatchObject({
      text: 'MRN: AB-123456 with seizures',
      useTextProcessMode: false,
      isAutoSubmit: false,
    });
    expect(flow.pendingPiiSubmission.value.scanResult.summary.high.medical_record).toBe(1);
  });

  it('returns a clean scan result without opening review when no PII is detected', () => {
    const flow = usePiiReviewFlow();

    const result = flow.scanSubmissionForPii({
      text: 'short syndrome query',
      locale: 'en',
      useTextProcessMode: false,
      isAutoSubmit: false,
    });

    expect(result.needsReview).toBe(false);
    expect(result.scanResult.hasFindings).toBe(false);
    expect(flow.piiReviewDialogVisible.value).toBe(false);
    expect(flow.pendingPiiSubmission.value).toBeNull();
  });

  it('continues with local redaction and submits redacted text while preserving raw history metadata', async () => {
    const flow = usePiiReviewFlow();
    const submitQueryText = vi.fn().mockResolvedValue(undefined);

    flow.scanSubmissionForPii({
      text: 'MRN: AB-123456 with seizures',
      locale: 'en',
      useTextProcessMode: true,
      isAutoSubmit: true,
    });

    await flow.continueWithPiiRedaction({ submitQueryText });

    expect(submitQueryText).toHaveBeenCalledWith(
      expect.objectContaining({
        currentQuery: expect.stringContaining('[REDACTED_MRN]'),
        rawQueryForHistory: 'MRN: AB-123456 with seizures',
        redactedQueryForHistory: expect.stringContaining('[REDACTED_MRN]'),
        useTextProcessMode: true,
        isAutoSubmit: true,
        redactionApplied: true,
      })
    );
    expect(flow.piiReviewDialogVisible.value).toBe(false);
    expect(flow.pendingPiiSubmission.value).toBeNull();
  });

  it('does not open review for review-confidence findings alone', () => {
    const flow = usePiiReviewFlow();

    const result = flow.scanSubmissionForPii({
      text: 'Herr Popp, seen on 12/03/2024, has seizures',
      locale: 'de',
    });

    expect(result.needsReview).toBe(false);
    expect(result.scanResult.hasFindings).toBe(true);
    expect(flow.piiReviewDialogVisible.value).toBe(false);
  });

  it('redacts only the findings the user left selected', async () => {
    const flow = usePiiReviewFlow();
    const submitQueryText = vi.fn().mockResolvedValue(undefined);
    const text = 'Email jane@example.org, MRN: AB-123456, seen on 12/03/2024';

    const { scanResult } = flow.scanSubmissionForPii({ text, locale: 'en' });
    const idOf = (category) => scanResult.findings.find((f) => f.category === category).id;

    await flow.continueWithPiiRedaction({
      submitQueryText,
      selectedFindingIds: [idOf('email'), idOf('date')],
    });

    expect(submitQueryText.mock.calls[0][0].currentQuery).toBe(
      'Email [REDACTED_EMAIL], MRN: AB-123456, seen on [REDACTED_DATE]'
    );
  });

  it('submits the original text when every finding is deselected', async () => {
    const flow = usePiiReviewFlow();
    const submitQueryText = vi.fn().mockResolvedValue(undefined);
    const text = 'MRN: AB-123456 with seizures';

    flow.scanSubmissionForPii({ text, locale: 'en' });
    await flow.continueWithPiiRedaction({ submitQueryText, selectedFindingIds: [] });

    expect(submitQueryText).toHaveBeenCalledWith(
      expect.objectContaining({ currentQuery: text, redactionApplied: false })
    );
  });

  it('cancels a pending review without mutating input or submitting', () => {
    const flow = usePiiReviewFlow();

    flow.openPiiReview({
      text: 'MRN: AB-123456 with seizures',
      useTextProcessMode: false,
      isAutoSubmit: false,
      scanResult: { hasFindings: true, findings: [], summary: { high: { mrn: 1 }, review: {} } },
    });

    flow.cancelPiiReview();

    expect(flow.piiReviewDialogVisible.value).toBe(false);
    expect(flow.pendingPiiSubmission.value).toBeNull();
  });
});
