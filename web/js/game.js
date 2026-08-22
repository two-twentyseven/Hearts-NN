import * as api from "./api.js";
import { animateAction, wait } from "./animations.js";
import {
  renderActivePlayer,
  renderGameChrome,
  renderOpponentHands,
  renderPlayerLabels,
  renderStatusBanner,
  renderTrick,
  renderTrickPiles,
} from "./layout.js";
import {
  bindPassButton,
  getSelectedCards,
  renderHumanHand,
  resetPassingSelection,
  setPassingSelectionCallback,
  updatePassButton,
} from "./passing.js";
import {
  hideAllModals,
  renderEmptyScoreboard,
  renderGameOverModal,
  renderRoundModal,
  renderScoreboard,
  showModal,
} from "./scoring.js";
import { clearError, setChromeVisibility, setLoading, setStatusText, showError, showWelcome } from "./ui.js";

let gameId = null;
let currentState = null;
let busy = false;

function setButtonDisabled(id, disabled) {
  const button = document.getElementById(id);
  if (button) button.disabled = disabled;
}

function setBusy(value, showOverlay = true) {
  busy = value;
  setLoading(value && showOverlay);
  setButtonDisabled("btn-new-game", value);
  setButtonDisabled("btn-pass", value || getSelectedCards().length !== 3);
  setButtonDisabled("btn-next-round", value);
  setButtonDisabled("btn-play-again", value);
  setButtonDisabled("btn-start", value);
}

function render(state) {
  currentState = state;
  const modalOpen = state.phase === "ROUND_OVER" || state.phase === "GAME_OVER";
  setChromeVisibility({ gameActive: true, hideNewGame: modalOpen });
  renderPlayerLabels(state);
  renderOpponentHands(state);
  renderTrick(state);
  renderTrickPiles(state);
  renderActivePlayer(state);
  renderStatusBanner(state);
  renderGameChrome(state);
  renderScoreboard(state);
  renderHumanHand(state, handlePlayCard);
  updatePassButton(state);

  if (state.phase === "ROUND_OVER") {
    renderRoundModal(state);
    showModal("modal-round");
  } else if (state.phase === "GAME_OVER") {
    renderGameOverModal(state);
    showModal("modal-game-over");
  } else {
    hideAllModals();
  }
}

async function applyState(state) {
  if (!state?.game_id) {
    throw new Error("Server returned an invalid game state");
  }

  gameId = state.game_id;
  showWelcome(false);
  clearError();

  try {
    render(state);
  } catch (error) {
    showWelcome(true);
    setChromeVisibility({ gameActive: false });
    throw error;
  }
}

async function advanceBotTurns() {
  while (
    currentState?.phase === "PLAYING"
    && currentState.current_player !== currentState.viewer
  ) {
    await wait(320);
    const state = await api.tickGame(gameId);
    if (!state.action) break;
    await animateAction(state.action);
    await applyState(state);
  }
}

async function runAction(
  action,
  { animate = false, continueBots = false, showOverlay = true } = {},
) {
  if (busy) return;
  setBusy(true, showOverlay);
  try {
    const state = await action();
    if (animate) {
      await animateAction(state.action);
    }
    await applyState(state);
    if (continueBots) {
      await advanceBotTurns();
    }
  } catch (error) {
    showError(error.message || "Something went wrong");
    console.error(error);
  } finally {
    setBusy(false);
    if (currentState) updatePassButton(currentState);
  }
}

export async function startNewGame() {
  await runAction(async () => {
    resetPassingSelection();
    hideAllModals();
    return api.createGame();
  });
}

async function handlePass() {
  if (!gameId) return;
  const cards = getSelectedCards();
  if (cards.length !== 3) return;

  await runAction(
    async () => {
      const state = await api.submitPass(gameId, cards);
      resetPassingSelection();
      return state;
    },
    { animate: true, continueBots: true, showOverlay: false },
  );
}

async function handlePlayCard(cardId) {
  if (!gameId || !currentState) return;
  if (currentState.phase !== "PLAYING" || currentState.current_player !== 0) return;

  await runAction(
    () => api.playCard(gameId, cardId),
    { animate: true, continueBots: true, showOverlay: false },
  );
}

async function handleNextRound() {
  if (!gameId) return;

  await runAction(
    async () => {
      hideAllModals();
      resetPassingSelection();
      return api.nextRound(gameId);
    },
    { continueBots: true, showOverlay: false },
  );
}

async function connectAndStart() {
  if (busy) return;

  setStatusText("Connecting to server…");
  setLoading(true);
  setButtonDisabled("btn-start", true);

  try {
    await api.checkConnection();
    clearError();
  } catch (error) {
    showWelcome(true);
    setChromeVisibility({ gameActive: false });
    setStatusText("Server offline — start uvicorn to play");
    showError(error.message);
    console.error(error);
    return;
  } finally {
    setLoading(false);
    setButtonDisabled("btn-start", false);
  }

  await startNewGame();
}

export function initGame() {
  setPassingSelectionCallback(() => {
    if (currentState) updatePassButton(currentState);
  });

  bindPassButton(handlePass);
  document.getElementById("btn-new-game").addEventListener("click", startNewGame);
  document.getElementById("btn-play-again").addEventListener("click", startNewGame);
  document.getElementById("btn-next-round").addEventListener("click", handleNextRound);
  document.getElementById("btn-start").addEventListener("click", connectAndStart);

  renderEmptyScoreboard();
  setChromeVisibility({ gameActive: false });
  setLoading(false);
  showWelcome(true);
  setStatusText("Click Start Game to begin");
}
