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

const DIFFICULTY_HINTS = {
  random: "Opponents pass and play at random.",
  human: "Opponents use human passing and play heuristics.",
  omniscient:
    "Opponents see every card in every hand and search ahead with the self-play trained network.",
};
const DIFFICULTIES = Object.keys(DIFFICULTY_HINTS);

const SUIT_SYMBOL = { C: "♣", D: "♦", H: "♥", S: "♠" };

let gameId = null;
let currentState = null;
let busy = false;
let revealHands = false;
let reviewCursor = 0;
let atLive = true;
let seenRound = null;

function selectedDifficulty() {
  const checked = document.querySelector('input[name="difficulty"]:checked');
  return DIFFICULTIES.includes(checked?.value) ? checked.value : "random";
}

function syncDifficultyHint() {
  const hint = document.getElementById("difficulty-hint");
  if (hint) hint.textContent = DIFFICULTY_HINTS[selectedDifficulty()];
}

function setButtonDisabled(id, disabled) {
  const button = document.getElementById(id);
  if (button) button.disabled = disabled;
}

function prettyCard(cardId) {
  if (!cardId || cardId.length < 2) return cardId || "";
  const rank = cardId[0] === "T" ? "10" : cardId[0];
  return `${rank}${SUIT_SYMBOL[cardId[1]] || cardId[1]}`;
}

function timeline(state) {
  const replay = state?.replay;
  if (!replay) return [];
  const plays = replay.plays || [];
  if (plays.length === 0) return [];
  return replay.opening ? [replay.opening, ...plays] : [...plays];
}

function resetReviewCursor() {
  reviewCursor = 0;
  atLive = true;
  seenRound = null;
}

function syncReviewCursor(state) {
  const steps = timeline(state);
  if (seenRound !== state.round_number) {
    seenRound = state.round_number;
    reviewCursor = steps.length;
    atLive = true;
    return;
  }
  if (atLive) {
    reviewCursor = steps.length;
    return;
  }
  reviewCursor = Math.min(reviewCursor, steps.length);
  atLive = reviewCursor >= steps.length;
}

function viewFromFrame(live, frame) {
  const hands = frame.hands;
  const actor = frame.actor;
  const viewer = live.viewer;
  const displayHands = hands.map((cards, seat) => (
    seat === actor && frame.decision_hand ? frame.decision_hand : cards
  ));
  const hand = actor === viewer && frame.decision_hand
    ? frame.decision_hand
    : hands[viewer];
  let messages;
  if (frame.kind === "opening") {
    messages = ["Opening hands"];
  } else {
    const name = live.players?.[actor] ?? "Player";
    const legalCount = frame.legal_moves?.length ?? 0;
    messages = [`${name} played ${prettyCard(frame.card)}`, `${legalCount} legal`];
  }
  return {
    ...live,
    phase: "PLAYING",
    hand,
    hands: displayHands,
    legal_moves: [],
    hand_counts: hands.map((cards) => cards.length),
    current_player: frame.current_player,
    trick_number: frame.trick_number,
    current_trick: frame.current_trick,
    tricks_won_counts: frame.tricks_won_counts,
    hearts_broken: frame.hearts_broken,
    pass_submitted: [true, true, true, true],
    messages,
    reveal_hands: revealHands,
    reviewing: true,
    review_actor: actor,
    review_card: frame.card,
    review_legal: frame.legal_moves || [],
  };
}

function presentedState(live) {
  if (atLive) {
    return { ...live, reveal_hands: revealHands, reviewing: false };
  }
  const steps = timeline(live);
  const frame = steps[reviewCursor];
  if (!frame) return { ...live, reveal_hands: revealHands, reviewing: false };
  return viewFromFrame(live, frame);
}

function updateReviewControls() {
  const steps = currentState ? timeline(currentState) : [];
  const back = document.getElementById("btn-review-back");
  const forward = document.getElementById("btn-review-forward");
  const reveal = document.getElementById("btn-reveal-hands");
  if (back) back.disabled = busy || !currentState || reviewCursor <= 0;
  if (forward) forward.disabled = busy || !currentState || reviewCursor >= steps.length;
  if (reveal) {
    reveal.setAttribute("aria-pressed", revealHands ? "true" : "false");
    reveal.textContent = revealHands ? "Hide hands" : "Reveal hands";
  }
}

function setBusy(value, showOverlay = true) {
  busy = value;
  setLoading(value && showOverlay);
  setButtonDisabled("btn-new-game", value);
  setButtonDisabled("btn-pass", value || getSelectedCards().length !== 3);
  setButtonDisabled("btn-next-round", value);
  setButtonDisabled("btn-play-again", value);
  setButtonDisabled("btn-start", value);
  updateReviewControls();
}

function render(state) {
  currentState = state;
  syncReviewCursor(state);
  const table = presentedState(state);
  const modalOpen = atLive && (state.phase === "ROUND_OVER" || state.phase === "GAME_OVER");
  setChromeVisibility({ gameActive: true, hideNewGame: modalOpen });
  renderPlayerLabels(table);
  renderOpponentHands(table);
  renderTrick(table);
  renderTrickPiles(table);
  renderActivePlayer(table);
  renderStatusBanner(table);
  renderGameChrome(table);
  renderScoreboard(state);
  renderHumanHand(table, handlePlayCard);
  updatePassButton(atLive ? state : table);
  updateReviewControls();

  if (!atLive) {
    hideAllModals();
  } else if (state.phase === "ROUND_OVER") {
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

function stepReview(delta) {
  if (busy || !currentState) return;
  const steps = timeline(currentState);
  const next = reviewCursor + delta;
  if (next < 0 || next > steps.length) return;
  reviewCursor = next;
  atLive = reviewCursor >= steps.length;
  render(currentState);
}

function toggleRevealHands() {
  revealHands = !revealHands;
  if (currentState) render(currentState);
  else updateReviewControls();
}

export async function startNewGame() {
  resetReviewCursor();
  await runAction(async () => {
    resetPassingSelection();
    hideAllModals();
    return api.createGame(selectedDifficulty());
  });
}

function returnToWelcome() {
  if (busy) return;
  gameId = null;
  currentState = null;
  resetReviewCursor();
  resetPassingSelection();
  hideAllModals();
  renderEmptyScoreboard();
  showWelcome(true);
  setChromeVisibility({ gameActive: false });
  setStatusText("Click Start Game to begin");
  syncDifficultyHint();
}

async function handlePass() {
  if (!atLive || !gameId) return;
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
  if (!atLive || !gameId || !currentState) return;
  if (currentState.phase !== "PLAYING" || currentState.current_player !== 0) return;

  await runAction(
    () => api.playCard(gameId, cardId),
    { animate: true, continueBots: true, showOverlay: false },
  );
}

async function handleNextRound() {
  if (!atLive || !gameId) return;

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
  document.getElementById("btn-review-back").addEventListener("click", () => stepReview(-1));
  document.getElementById("btn-review-forward").addEventListener("click", () => stepReview(1));
  document.getElementById("btn-reveal-hands").addEventListener("click", toggleRevealHands);
  document.getElementById("btn-new-game").addEventListener("click", returnToWelcome);
  document.getElementById("btn-play-again").addEventListener("click", returnToWelcome);
  document.getElementById("btn-next-round").addEventListener("click", handleNextRound);
  document.getElementById("btn-start").addEventListener("click", connectAndStart);
  document.querySelectorAll('input[name="difficulty"]').forEach((input) => {
    input.addEventListener("change", syncDifficultyHint);
  });

  renderEmptyScoreboard();
  setChromeVisibility({ gameActive: false });
  setLoading(false);
  showWelcome(true);
  setStatusText("Click Start Game to begin");
  syncDifficultyHint();
}
