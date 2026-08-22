const SUIT_ORDER = { C: 0, D: 1, H: 2, S: 3 };

const RANK_ORDER = {
  "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
  "9": 9, T: 10, J: 11, Q: 12, K: 13, A: 14,
};

const CARD_ASSET_BASE = "/assets/cards";
const CARD_ASSET_VERSION = "8";

const RANK_DISPLAY = {
  T: "10",
  J: "J",
  Q: "Q",
  K: "K",
  A: "A",
  "2": "2",
  "3": "3",
  "4": "4",
  "5": "5",
  "6": "6",
  "7": "7",
  "8": "8",
  "9": "9",
};

const SUIT_INFO = {
  H: { symbol: "♥", color: "red" },
  D: { symbol: "♦", color: "red" },
  S: { symbol: "♠", color: "black" },
  C: { symbol: "♣", color: "black" },
};

export function parseCardId(cardId) {
  return {
    rank: cardId[0],
    suit: cardId[1],
    id: cardId,
  };
}

export function sortCardIds(cardIds) {
  return [...cardIds].sort((a, b) => {
    const ca = parseCardId(a);
    const cb = parseCardId(b);
    const suitDiff = SUIT_ORDER[ca.suit] - SUIT_ORDER[cb.suit];
    if (suitDiff !== 0) return suitDiff;
    return RANK_ORDER[ca.rank] - RANK_ORDER[cb.rank];
  });
}

function cardImageSrc(cardId, faceDown = false) {
  const file = faceDown ? "back.png" : `${cardId}.png`;
  return `${CARD_ASSET_BASE}/${file}?v=${CARD_ASSET_VERSION}`;
}

export function createCardElement(cardId, options = {}) {
  const {
    faceDown = false,
    selectable = false,
    selected = false,
    legal = false,
    illegal = false,
    played = false,
    onClick = null,
  } = options;

  const card = document.createElement("div");
  card.className = "card";
  if (cardId) card.dataset.cardId = cardId;

  if (faceDown) card.classList.add("card--back");
  else card.classList.add("card--face");

  if (selectable) card.classList.add("card--selectable");
  if (selected) card.classList.add("card--selected");
  if (legal) card.classList.add("card--legal");
  if (illegal) card.classList.add("card--illegal");
  if (played) card.classList.add("card--played");

  const img = document.createElement("img");
  img.className = "card__img";
  img.src = cardImageSrc(cardId, faceDown);
  img.alt = faceDown ? "Card back" : cardId;
  img.draggable = false;
  img.loading = "eager";
  card.appendChild(img);

  if (onClick && (selectable || legal || faceDown)) {
    card.addEventListener("click", onClick);
  }

  return card;
}

export function createCardBack(count = 1) {
  const fragment = document.createDocumentFragment();
  const displayCount = Math.min(count, 7);
  for (let i = 0; i < displayCount; i += 1) {
    fragment.appendChild(createCardElement("", { faceDown: true }));
  }
  return fragment;
}

export function createHandStripElement(cardId, options = {}) {
  const {
    selectable = false,
    selected = false,
    legal = false,
    illegal = false,
    onClick = null,
    zIndex = 1,
  } = options;

  const { rank, suit } = parseCardId(cardId);
  const suitInfo = SUIT_INFO[suit];
  const rankLabel = RANK_DISPLAY[rank] ?? rank;

  const card = document.createElement("div");
  card.className = "hand-card";
  card.dataset.cardId = cardId;
  card.style.zIndex = String(zIndex);

  card.classList.add(suitInfo.color === "red" ? "hand-card--red" : "hand-card--black");
  if (rankLabel.length > 1) card.classList.add("hand-card--wide-rank");
  if (selectable) card.classList.add("hand-card--selectable");
  if (selected) card.classList.add("hand-card--selected");
  if (legal) card.classList.add("hand-card--legal");
  if (illegal) card.classList.add("hand-card--illegal");

  const index = document.createElement("div");
  index.className = "hand-card__index";

  const rankEl = document.createElement("span");
  rankEl.className = "hand-card__rank";
  rankEl.textContent = rankLabel;

  const suitEl = document.createElement("span");
  suitEl.className = "hand-card__suit";
  suitEl.textContent = suitInfo.symbol;

  index.append(rankEl, suitEl);
  card.appendChild(index);

  if (onClick && (selectable || legal)) {
    card.addEventListener("click", onClick);
  }

  return card;
}
