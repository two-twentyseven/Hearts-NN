import { createHandStripElement, sortCardIds } from "./cards.js";

let selectedCards = new Set();
let onSelectionChange = null;

export function resetPassingSelection() {
  selectedCards = new Set();
  if (onSelectionChange) onSelectionChange(selectedCards);
}

export function setPassingSelectionCallback(callback) {
  onSelectionChange = callback;
}

export function getSelectedCards() {
  return [...selectedCards];
}

function toggleCard(cardId) {
  if (selectedCards.has(cardId)) {
    selectedCards.delete(cardId);
  } else if (selectedCards.size < 3) {
    selectedCards.add(cardId);
  }
  if (onSelectionChange) onSelectionChange(selectedCards);
}

function syncPassingSelection(container) {
  for (const cardEl of container.querySelectorAll(".hand-card")) {
    cardEl.classList.toggle("hand-card--selected", selectedCards.has(cardEl.dataset.cardId));
  }
}

export function renderHumanHand(state, onCardClick) {
  const container = document.getElementById("hand-south");
  container.replaceChildren();

  const sorted = sortCardIds(state.hand);
  const legalSet = new Set(state.legal_moves);
  const isPassing = state.phase === "PASSING" && !state.pass_submitted[0];
  const isPlaying = state.phase === "PLAYING" && state.current_player === 0;

  const reviewing = Boolean(state.reviewing);
  const reviewMine = reviewing && state.review_actor === 0;
  const reviewLegal = new Set(reviewMine ? state.review_legal || [] : []);

  sorted.forEach((cardId, index) => {
    let options = { zIndex: index + 1 };

    if (reviewing) {
      options = {
        ...options,
        choice: reviewMine && cardId === state.review_card,
        option: reviewMine && reviewLegal.has(cardId) && cardId !== state.review_card,
        illegal: reviewMine && !reviewLegal.has(cardId),
      };
    } else if (isPassing) {
      options = {
        ...options,
        selectable: true,
        selected: selectedCards.has(cardId),
        onClick: () => {
          toggleCard(cardId);
          syncPassingSelection(container);
        },
      };
    } else if (isPlaying) {
      const legal = legalSet.has(cardId);
      options = {
        ...options,
        legal,
        illegal: !legal,
        onClick: legal ? () => onCardClick(cardId) : null,
      };
    } else {
      options = { ...options, illegal: true };
    }

    container.appendChild(createHandStripElement(cardId, options));
  });
}

export function updatePassButton(state) {
  const btn = document.getElementById("btn-pass");
  const actionBar = document.getElementById("action-bar");
  const isPassing = state.phase === "PASSING" && !state.pass_submitted[0];

  if (isPassing && state.pass_direction !== "HOLD") {
    actionBar.classList.remove("hidden");
    btn.disabled = selectedCards.size !== 3;

    const dirLabels = {
      LEFT: "left",
      RIGHT: "right",
      ACROSS: "across",
    };
    const dir = dirLabels[state.pass_direction] || "";
    btn.textContent = `Pass cards to the ${dir}`;
  } else {
    actionBar.classList.add("hidden");
    btn.disabled = true;
  }
}

export function bindPassButton(handler) {
  document.getElementById("btn-pass").addEventListener("click", handler);
}
