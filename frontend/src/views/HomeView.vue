<template>
  <div class="fill-height">
    <div class="content-wrapper" :class="{ 'content-wrapper--compact': hasConversation }">
      <div class="d-flex justify-center align-center pb-1 home-header">
        <v-tooltip location="bottom" :aria-label="t('home.resetTooltip')">
          <template #activator="{ props: tooltipProps }">
            <img
              v-bind="tooltipProps"
              src="/favicon.svg"
              :alt="t('home.logoAlt', 'Phentrieve Logo - Click to reset')"
              :width="hasConversation ? 32 : 65"
              :height="hasConversation ? 32 : 65"
              class="mr-2 logo-icon"
              loading="lazy"
              role="button"
              tabindex="0"
              :aria-label="t('home.resetAriaLabel', 'Reset conversation')"
              @click="showResetDialog = true"
              @keydown.enter="showResetDialog = true"
              @keydown.space.prevent="showResetDialog = true"
            />
          </template>
          {{ t('home.resetTooltip') }}
        </v-tooltip>
        <h1 class="text-h5 font-weight-light logo-text" aria-label="Phentrieve">
          <span class="logo-visible">Phen</span><span class="logo-bracket">[</span
          ><span class="logo-hidden">otype re</span><span class="logo-bracket">]</span
          ><span class="logo-visible">trieve</span>
        </h1>
      </div>
      <div
        v-if="!hasConversation"
        data-testid="home-welcome-text"
        class="text-center px-4 py-0"
        style="max-width: 600px"
      >
        <p class="text-body-2 text-medium-emphasis mb-1">
          {{ t('queryInterface.welcomeText') }}
        </p>
      </div>
      <QueryInterface />
    </div>

    <!-- Reset Confirmation Dialog -->
    <v-dialog v-model="showResetDialog" max-width="400">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon color="warning" class="mr-2">mdi-restart</v-icon>
          <span class="text-h6">{{ t('home.resetTitle', 'Reset Application?') }}</span>
        </v-card-title>

        <v-card-text>
          {{
            t(
              'home.resetMessage',
              'This will clear all conversation history and collected phenotypes. Your disclaimer acknowledgment will be preserved.'
            )
          }}
        </v-card-text>

        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showResetDialog = false">
            {{ t('common.cancel', 'Cancel') }}
          </v-btn>
          <v-btn color="warning" variant="flat" @click="handleReset">
            {{ t('home.resetConfirm', 'Reset') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script>
import { computed, ref } from 'vue';
import QueryInterface from '@/components/QueryInterface.vue';
import { useLogStore } from '@/stores/log';
import { useConversationStore } from '@/stores/conversation';
import { useDiseaseRankingStore } from '@/stores/diseaseRanking';
import { logService } from '@/services/logService';
import { useI18n } from 'vue-i18n';

export default {
  name: 'HomeView',
  components: {
    QueryInterface,
  },
  setup() {
    const logStore = useLogStore();
    const conversationStore = useConversationStore();
    const diseaseRankingStore = useDiseaseRankingStore();
    const { t } = useI18n();

    // Once there are results the header shrinks to a small logo row, so the
    // sticky search bar and the results get the screen.
    const hasConversation = computed(() => conversationStore.hasConversation);

    // Dialog state
    const showResetDialog = ref(false);

    /**
     * Handle reset action - clears conversation history and phenotypes
     * Preserves disclaimer acknowledgment (per UX best practice)
     */
    function handleReset() {
      logService.info('User initiated app reset via logo click');
      conversationStore.resetAll();
      diseaseRankingStore.clearCase();
      showResetDialog.value = false;
      logService.info('App state reset complete');
    }

    return {
      logStore,
      conversationStore,
      hasConversation,
      t,
      showResetDialog,
      handleReset,
    };
  },
};
</script>

<style scoped>
/* Logo icon with subtle pulse animation on hover */
.logo-icon {
  cursor: pointer;
  transition: transform 0.2s ease-out;
  border-radius: 8px;
}

.logo-icon:hover {
  /* will-change creates compositor layer - only apply when needed for animation */
  will-change: transform;
  animation: subtle-pulse 1.5s ease-in-out infinite;
}

.logo-icon:focus {
  outline: 2px solid rgb(var(--v-theme-primary));
  outline-offset: 2px;
}

.logo-icon:active {
  transform: scale(0.95);
  animation: none;
}

@keyframes subtle-pulse {
  0%,
  100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
}

.content-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  padding-top: 10vh; /* Replace absolute positioning with padding */
}

.content-wrapper--compact {
  padding-top: 8px;
}

.content-wrapper--compact .logo-text {
  font-size: 1.125rem !important;
}

.fill-height {
  min-height: 100vh;
  position: relative;
  display: flex;
  flex-direction: column;
}

.logo-text {
  position: relative;
  cursor: default;
  display: inline-flex;
  overflow: hidden;
  white-space: nowrap;
}

.logo-hidden {
  max-width: 0;
  opacity: 0;
  transition:
    max-width 0.5s ease,
    opacity 0.3s ease;
  display: inline-block;
  overflow: hidden;
  color: rgb(var(--v-theme-primary));
}

.logo-visible {
  display: inline-block;
}

.logo-bracket {
  display: inline-block;
  max-width: 0;
  opacity: 0;
  overflow: hidden;
  transition:
    max-width 0.5s ease,
    opacity 0.3s ease;
  color: #42a5f5;
  font-weight: bold;
}

.logo-text:hover .logo-hidden,
.logo-text:hover .logo-bracket {
  max-width: 150px;
  opacity: 1;
}
</style>
