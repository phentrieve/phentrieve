<template>
  <v-dialog
    v-model="dialogVisible"
    max-width="560"
    persistent
    :content-props="{ role: 'alertdialog', 'aria-labelledby': 'pii-review-title' }"
  >
    <v-card>
      <v-card-title id="pii-review-title" class="d-flex align-center ga-2">
        <v-icon icon="mdi-shield-alert-outline" color="warning" />
        <span>{{ $t('queryInterface.piiReview.title') }}</span>
      </v-card-title>

      <v-card-text>
        <p class="mb-2">{{ $t('queryInterface.piiReview.description') }}</p>

        <v-checkbox
          v-for="finding in findings"
          :key="finding.id"
          v-model="selectedIds"
          :value="finding.id"
          :data-testid="`pii-finding-${finding.id}`"
          density="compact"
          hide-details
        >
          <template #label>
            <span>
              {{ getCategoryLabel(finding.category) }}:
              <code>{{ text.slice(finding.start, finding.end) }}</code>
            </span>
          </template>
        </v-checkbox>

        <p class="text-caption mt-4 mb-0">
          {{ $t('queryInterface.piiReview.safetyAid') }}
        </p>
      </v-card-text>

      <v-card-actions class="flex-wrap ga-2">
        <v-btn data-testid="pii-cancel" variant="text" @click="$emit('cancel')">
          {{ $t('queryInterface.piiReview.cancel') }}
        </v-btn>
        <v-spacer />
        <v-btn data-testid="pii-continue" color="primary" @click="$emit('continue', selectedIds)">
          {{ $t('queryInterface.piiReview.continue') }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script>
export default {
  name: 'PiiReviewDialog',
  props: {
    modelValue: {
      type: Boolean,
      default: false,
    },
    text: {
      type: String,
      default: '',
    },
    findings: {
      type: Array,
      default: () => [],
    },
  },
  emits: ['update:modelValue', 'cancel', 'continue'],
  data() {
    return { selectedIds: [] };
  },
  computed: {
    dialogVisible: {
      get() {
        return this.modelValue;
      },
      set(value) {
        this.$emit('update:modelValue', value);
      },
    },
  },
  watch: {
    findings: {
      immediate: true,
      // High-confidence findings start ticked; review-level ones are opt-in.
      handler(findings) {
        this.selectedIds = findings
          .filter((finding) => finding.confidence === 'high')
          .map((finding) => finding.id);
      },
    },
  },
  methods: {
    getCategoryLabel(category) {
      return this.$t(`queryInterface.piiReview.categories.${category}`);
    },
  },
};
</script>
