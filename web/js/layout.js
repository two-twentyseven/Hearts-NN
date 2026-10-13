import { createCardBack, createCardElement, createHandStripElement, sortCardIds } from "./cards.js";
import { renderPassPattern } from "./pass-pattern.js";

const SEAT_TO_TRICK = {
  0: "trick-south",
  1: "trick-west",
  2: "trick-north",
  3: "trick-east",
};

const SEAT_TO_HAND = {
  0: "hand-south",
  1: "hand-west",
  2: "hand-north",
  3: "hand-east",
};

const SEAT_TO_LABEL = {
  0: "label-south",
  1: "label-west",
  2: "label-north",
  3: "label-east",
};

const SEAT_TO_ZONE_EL = {
  0: "zone-south",
  1: "zone-west",
  2: "zone-north",
  3: "zone-east",
};

const SEAT_TO_PILE = {
  0: "pile-south",
  1: "pile-west",
  2: "pile-north",
  3: "pile-east",
};

export function renderOpponentHands(state) {
  for (const seat of [1, 2, 3]) {
    const container = document.getElementById(SEAT_TO_HAND[seat]);
    container.replaceChildren();
    const cards = state.hands?.[seat];
    const open = Boolean(state.reveal_hands && Array.isArray(cards));
    container.classList.toggle("opponent-hand--open", open);
    if (!open) {
      container.appendChild(createCardBack(state.hand_counts[seat]));
      continue;
    }
    const reviewing = Boolean(state.reviewing && state.review_actor === seat);
    const legal = new Set(reviewing ? state.review_legal || [] : []);
    sortCardIds(cards).forEach((cardId, index) => {
      container.appendChild(createHandStripElement(cardId, {
        zIndex: index + 1,
        choice: reviewing && cardId === state.review_card,
        option: reviewing && legal.has(cardId) && cardId !== state.review_card,
        illegal: reviewing && !legal.has(cardId),
      }));
    });
  }
}

export function renderTrickPiles(state) {
  for (const seat of [0, 1, 2, 3]) {
    const pile = document.getElementById(SEAT_TO_PILE[seat]);
    const count = state.tricks_won_counts?.[seat] ?? 0;
    pile.replaceChildren();
    if (count === 0) continue;

    const card = document.createElement("div");
    card.className = "trick-pile__card";
    const badge = document.createElement("span");
    badge.className = "trick-pile__count";
    badge.textContent = String(count);
    pile.append(card, badge);
  }
}

export function renderPlayerLabels(state) {
  for (const seat of [0, 1, 2, 3]) {
    const label = document.getElementById(SEAT_TO_LABEL[seat]);
    const passDone = state.pass_submitted[seat];
    const indicator = state.phase === "PASSING"
      ? `<span class="pass-indicator ${passDone ? "pass-indicator--done" : ""}" title="${passDone ? "Passed" : "Passing"}"></span>`
      : "";
    label.innerHTML = `${state.players[seat]}${indicator}`;
  }
}

export function renderActivePlayer(state) {
  for (const seat of [0, 1, 2, 3]) {
    const zone = document.getElementById(SEAT_TO_ZONE_EL[seat]);
    const isActive = state.current_player === seat && state.phase === "PLAYING";
    zone.classList.toggle("player-zone--active", isActive);
  }
}

export function renderTrick(state) {
  for (const seat of [0, 1, 2, 3]) {
    const slot = document.getElementById(SEAT_TO_TRICK[seat]);
    slot.replaceChildren();
  }

  for (const play of state.current_trick) {
    const slot = document.getElementById(SEAT_TO_TRICK[play.player]);
    const card = createCardElement(play.card, { played: true });
    if (
      state.reviewing
      && play.player === state.review_actor
      && play.card === state.review_card
    ) {
      card.classList.add("card--just-played");
    }
    slot.appendChild(card);
  }
}

export function renderStatusBanner(state) {
  const banner = document.getElementById("status-banner");
  const parts = [...state.messages];

  if (state.phase === "PASSING" && state.pass_direction !== "HOLD") {
    parts.unshift(`Round ${state.round_number}`);
  } else if (state.phase === "PLAYING") {
    parts.unshift(`Trick ${state.trick_number} of 13`);
  }

  banner.textContent = parts.join(" · ");
}

export function renderGameChrome(state) {
  const roundBadge = document.getElementById("round-badge");
  roundBadge.textContent = state.round_number
    ? `Round ${state.round_number}`
    : "Round —";

  document.getElementById("target-score").textContent = state.target_score ?? 100;

  const heartsBadge = document.getElementById("hearts-broken-badge");
  heartsBadge.classList.toggle("hidden", !state.hearts_broken);

  renderPassPattern(state);
}

export function fanRotation(index, total) {
  if (total <= 1) return 0;
  const spread = Math.min(40, total * 3);
  const start = -spread / 2;
  const step = spread / (total - 1);
  return start + index * step;
}
