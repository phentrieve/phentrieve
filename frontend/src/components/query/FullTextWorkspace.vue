<template>
  <div data-testid="user-note-summary" class="user-note-summary">
    <div class="user-note-summary__header">
      <v-icon size="small" color="primary">mdi-file-document-outline</v-icon>
      <span class="text-body-2 font-weight-medium">{{ $t('queryInterface.form.noteLabel') }}</span>
      <span class="text-caption text-medium-emphasis">
        {{ meta }}
      </span>
      <v-btn
        data-testid="user-note-summary-toggle"
        size="x-small"
        variant="text"
        icon
        :aria-label="
          $t(expanded ? 'queryInterface.form.collapseNote' : 'queryInterface.form.expandNote')
        "
        @click="$emit('toggle')"
      >
        <v-icon size="small">
          {{ expanded ? 'mdi-chevron-up' : 'mdi-chevron-down' }}
        </v-icon>
      </v-btn>
    </div>
    <p v-if="!expanded" class="mb-0 text-body-2 text-medium-emphasis user-note-summary__preview">
      {{ summary }}
    </p>
    <div
      v-if="expanded"
      ref="expandedContainer"
      data-testid="user-note-expanded"
      class="user-note-summary__expanded"
      @mouseup="onNoteMouseUp"
    >
      <span v-for="segment in segments" :key="segment.key">
        <v-tooltip
          v-if="segment.highlighted"
          location="top"
          :open-delay="180"
          open-on-focus
          max-width="280"
          content-class="annotated-note-tooltip"
          :aria-label="segment.tooltip"
        >
          <template #activator="{ props }">
            <mark
              v-bind="props"
              data-testid="annotated-note-span"
              class="annotated-note-span"
              :class="[
                markColourClass(segment.colourSlot),
                {
                  'annotated-note-span--active':
                    activePhenotypeId && segment.termIds.includes(activePhenotypeId),
                  'annotated-note-span--manual': segment.manual,
                  'annotated-note-span--negated': segment.negated,
                },
              ]"
              role="button"
              aria-haspopup="menu"
              tabindex="0"
              :aria-label="`Edit annotation: ${segment.tooltip}`"
              @mouseenter="$emit('hover', segment.termIds)"
              @mouseleave="$emit('clear-hover')"
              @focus="$emit('hover', segment.termIds)"
              @blur="$emit('clear-hover')"
              @click="activateSpan(segment, $event)"
              @contextmenu.prevent="activateSpan(segment, $event)"
              @keydown.enter.prevent="activateSpan(segment, $event)"
              @keydown.space.prevent="activateSpan(segment, $event)"
            >
              {{ segment.text }}
            </mark>
          </template>
          <div class="annotated-note-tooltip__content">
            <div class="annotated-note-tooltip__eyebrow">Linked phenotype</div>
            <div class="annotated-note-tooltip__label">{{ segment.tooltip }}</div>
            <div v-if="segment.negated" class="annotated-note-tooltip__negated">
              {{ $t('queryInterface.phenotypeCollection.assertionStatus.negated') }}
            </div>
          </div>
        </v-tooltip>
        <v-tooltip
          v-else-if="segment.modifier"
          location="top"
          :open-delay="180"
          open-on-focus
          max-width="280"
          content-class="annotated-note-tooltip"
          :aria-label="segment.tooltip"
        >
          <template #activator="{ props }">
            <span
              v-bind="props"
              data-testid="annotated-note-modifier"
              class="annotated-note-modifier"
              :class="[
                markColourClass(segment.colourSlot),
                {
                  'annotated-note-modifier--active':
                    activePhenotypeId && segment.termIds.includes(activePhenotypeId),
                },
              ]"
              tabindex="0"
              :aria-description="
                $t('annotatedDocumentPane.modifierAriaLabel', {
                  label: segment.tooltip,
                  finding: segment.findingTooltip,
                })
              "
              @mouseenter="$emit('hover', segment.termIds)"
              @mouseleave="$emit('clear-hover')"
              @focus="$emit('hover', segment.termIds)"
              @blur="$emit('clear-hover')"
              >{{ segment.text }}</span
            >
          </template>
          <div class="annotated-note-tooltip__content">
            <div class="annotated-note-tooltip__eyebrow">
              {{ $t('annotatedDocumentPane.modifierEyebrow') }}
            </div>
            <div class="annotated-note-tooltip__label">{{ segment.tooltip }}</div>
            <div class="annotated-note-tooltip__finding">
              {{
                $t('annotatedDocumentPane.modifierOf', {
                  finding: segment.findingTooltip,
                })
              }}
            </div>
          </div>
        </v-tooltip>
        <span v-else>{{ segment.text }}</span>
      </span>
    </div>
  </div>
</template>

<script>
import { computeSelectionOffsets, trimSelection } from '../../composables/useUserNoteAnnotations';
import { markColourClass } from '../../utils/markColours';

export default {
  name: 'FullTextWorkspace',
  props: {
    summary: {
      type: String,
      required: true,
    },
    meta: {
      type: String,
      required: true,
    },
    expanded: {
      type: Boolean,
      default: false,
    },
    segments: {
      type: Array,
      default: () => [],
    },
    activePhenotypeId: {
      type: String,
      default: null,
    },
  },
  emits: ['toggle', 'hover', 'clear-hover', 'span-activate', 'text-select'],
  methods: {
    markColourClass,
    activateSpan(segment, event) {
      const rect = event?.currentTarget?.getBoundingClientRect?.() || null;
      this.$emit('span-activate', {
        annotationIds: Array.isArray(segment.annotationIds) ? segment.annotationIds : [],
        termIds: Array.isArray(segment.termIds) ? segment.termIds : [],
        rect,
        text: segment.text,
      });
    },
    onNoteMouseUp() {
      const selection =
        typeof window !== 'undefined' && window.getSelection ? window.getSelection() : null;
      if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
        return;
      }
      const rawText = selection.toString();
      if (!rawText.trim()) {
        return;
      }
      const container = this.$refs.expandedContainer;
      if (!container) {
        return;
      }
      const range = selection.getRangeAt(0);
      if (!container.contains(range.startContainer) || !container.contains(range.endContainer)) {
        return;
      }
      // Selections fully inside a single existing mark are edited via click, not
      // turned into a new manual annotation.
      const startEl =
        range.startContainer.nodeType === 1
          ? range.startContainer
          : range.startContainer.parentElement;
      const endEl =
        range.endContainer.nodeType === 1 ? range.endContainer : range.endContainer.parentElement;
      const startMark = startEl?.closest?.('.annotated-note-span') || null;
      const endMark = endEl?.closest?.('.annotated-note-span') || null;
      if (startMark && startMark === endMark) {
        return;
      }
      const offsets = computeSelectionOffsets(container, range);
      if (!offsets) {
        return;
      }
      // Trim whitespace from the selection AND keep the offsets aligned, so the
      // stored span text matches noteText.slice(start, end) and renders.
      const trimmed = trimSelection(rawText, offsets.start, offsets.end);
      if (!trimmed || trimmed.end <= trimmed.start) {
        return;
      }
      const rect = range.getBoundingClientRect?.() || null;
      this.$emit('text-select', {
        text: trimmed.text,
        start: trimmed.start,
        end: trimmed.end,
        rect,
      });
    },
  },
};
</script>

<style scoped>
.user-note-summary {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.user-note-summary__header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.user-note-summary__header .text-caption {
  margin-left: auto;
}

.user-note-summary__preview {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.user-note-summary__expanded {
  margin-top: 8px;
  font-size: 0.95rem;
  line-height: 1.6;
  color: rgba(var(--v-theme-on-surface), 0.9);
}

/* Finding marks take their term's colour slot (#562, styles/markColours.css);
   the fallback is the blue of slot 0. State never changes the hue:
   - manual: solid 2px underline
   - negated (absent): hatch in the term colour plus a dashed frame (#570)
   - active (hovered in the findings list): solid frame */
.annotated-note-span,
.user-note-summary__expanded mark {
  background: rgba(var(--ph-mark-rgb, 37, 99, 235), var(--ph-mark-alpha, 0.2));
  color: inherit;
  border-radius: 4px;
  cursor: pointer;
  transition:
    background-color 0.18s ease,
    outline-color 0.18s ease;
}

.annotated-note-span:hover,
.user-note-summary__expanded mark:hover {
  background: rgba(var(--ph-mark-rgb, 37, 99, 235), var(--ph-mark-alpha-hover, 0.3));
}

/* The doubled class outranks the base ".user-note-summary__expanded mark"
   selector (0,2,1). */
.annotated-note-span.annotated-note-span--manual {
  text-decoration-line: underline;
  text-decoration-style: solid;
  text-decoration-thickness: 2px;
  text-decoration-color: rgba(var(--v-theme-on-surface), 0.75);
  text-underline-offset: 3px;
}

/* Absent findings: no wavy or coloured underline, which reads as a spell-check
   error (#570). The dashed frame keeps the state visible without colour. */
.annotated-note-span.annotated-note-span--negated {
  background: repeating-linear-gradient(
    135deg,
    rgba(var(--ph-mark-rgb, 37, 99, 235), calc(var(--ph-mark-alpha, 0.2) * 1.4)) 0 2px,
    rgba(var(--ph-mark-rgb, 37, 99, 235), calc(var(--ph-mark-alpha, 0.2) * 0.25)) 2px 6px
  );
  outline: 1px dashed rgba(var(--v-theme-on-surface), 0.5);
  outline-offset: -1px;
}

.annotated-note-span.annotated-note-span--negated:hover {
  background: repeating-linear-gradient(
    135deg,
    rgba(var(--ph-mark-rgb, 37, 99, 235), calc(var(--ph-mark-alpha-hover, 0.3) * 1.4)) 0 2px,
    rgba(var(--ph-mark-rgb, 37, 99, 235), calc(var(--ph-mark-alpha-hover, 0.3) * 0.4)) 2px 6px
  );
}

.annotated-note-span.annotated-note-span--active {
  background-color: rgba(var(--ph-mark-rgb, 37, 99, 235), var(--ph-mark-alpha-hover, 0.3));
  outline: 2px solid rgb(var(--v-theme-on-surface));
  outline-offset: 1px;
}

.annotated-note-span.annotated-note-span--negated.annotated-note-span--active {
  outline-style: dashed;
}

/* Modifier evidence (severity, onset, laterality, ...) is read-only context for
   a finding, styled as in ontotator: a lighter tint of the finding fill and a
   1 px double underline, never the solid fill of an annotation. While its
   finding is hovered or active the underline turns into a dotted accent line. */
.annotated-note-modifier {
  background: rgba(var(--ph-mark-rgb, 37, 99, 235), calc(var(--ph-mark-alpha, 0.2) * 0.5));
  border-radius: 3px;
  text-decoration-line: underline;
  text-decoration-style: double;
  text-decoration-thickness: 1px;
  text-decoration-color: rgba(var(--v-theme-on-surface), 0.7);
  text-underline-offset: 3px;
  cursor: help;
  transition: background-color 0.18s ease;
}

.annotated-note-modifier:hover,
.annotated-note-modifier:focus-visible {
  background: rgba(var(--ph-mark-rgb, 37, 99, 235), var(--ph-mark-alpha, 0.2));
}

.annotated-note-modifier--active {
  background: rgba(var(--ph-mark-rgb, 37, 99, 235), var(--ph-mark-alpha, 0.2));
  text-decoration-style: dotted;
  text-decoration-thickness: 2px;
  text-decoration-color: rgb(var(--v-theme-primary));
}

@media (forced-colors: active) {
  /* Fills are dropped in forced-colors mode; system colours keep marks visible. */
  .annotated-note-span,
  .user-note-summary__expanded mark {
    background: Mark;
    color: MarkText;
  }

  .annotated-note-span.annotated-note-span--negated {
    background: Canvas;
    color: CanvasText;
    outline-color: CanvasText;
  }

  .annotated-note-span.annotated-note-span--active {
    outline-color: Highlight;
  }

  .annotated-note-modifier {
    text-decoration-color: CanvasText;
  }

  .annotated-note-modifier--active {
    text-decoration-color: Highlight;
  }
}

.annotated-note-modifier:focus-visible {
  outline: 2px solid rgb(var(--v-theme-primary));
  outline-offset: 2px;
}

/* Keyboard focus parity: a focused span gets the same emphasis as hover, plus a
   visible focus ring for non-pointer users (WCAG 2.4.7 / 1.4.13). */
.annotated-note-span:focus-visible {
  outline: 2px solid rgb(var(--v-theme-primary));
  outline-offset: 2px;
  background-color: rgba(var(--ph-mark-rgb, 37, 99, 235), var(--ph-mark-alpha-hover, 0.3));
}
</style>

<!--
  The v-tooltip content is teleported to <body>, outside this component's
  scoped styles, so a scoped :deep() rule can never reach it. Style it from a
  GLOBAL block instead. Vuetify's default tooltip surface is dark
  (surface-variant); we override it with the theme surface/on-surface tokens so
  the text stays readable (WCAG AA contrast) in BOTH light and dark themes.
  The selector includes `.v-tooltip >` to out-specify Vuetify's own
  `.v-tooltip > .v-overlay__content` rule deterministically.
-->
<style>
.v-tooltip > .v-overlay__content.annotated-note-tooltip {
  border-radius: 14px;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.12);
  background: rgb(var(--v-theme-surface));
  color: rgb(var(--v-theme-on-surface));
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.18);
  padding: 10px 12px;
}

.annotated-note-tooltip .annotated-note-tooltip__content {
  display: grid;
  gap: 4px;
}

.annotated-note-tooltip .annotated-note-tooltip__eyebrow {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: rgb(var(--v-theme-primary));
}

/* Teleported tooltip: must live in this global block to apply. */
.annotated-note-tooltip .annotated-note-tooltip__negated {
  margin-top: 4px;
  font-weight: 600;
  color: rgba(var(--v-theme-on-surface), 0.7);
}

.annotated-note-tooltip .annotated-note-tooltip__finding {
  font-size: 0.75rem;
  color: rgba(var(--v-theme-on-surface), 0.7);
}

.annotated-note-tooltip .annotated-note-tooltip__label {
  font-size: 0.82rem;
  line-height: 1.4;
  color: rgb(var(--v-theme-on-surface));
}
</style>
