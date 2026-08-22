import { createCardElement } from "./cards.js";

const SEAT_HAND_IDS = ["hand-south", "hand-west", "hand-north", "hand-east"];
const SEAT_TRICK_IDS = ["trick-south", "trick-west", "trick-north", "trick-east"];
const SEAT_PILE_IDS = ["pile-south", "pile-west", "pile-north", "pile-east"];

export function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function centeredRect(element, width, height) {
  const rect = element.getBoundingClientRect();
  return {
    left: rect.left + (rect.width - width) / 2,
    top: rect.top + (rect.height - height) / 2,
    width,
    height,
  };
}

async function flyCard(cardId, fromRect, toRect, { faceDown = false, duration = 460 } = {}) {
  const layer = document.getElementById("animation-layer");
  const card = createCardElement(cardId, { faceDown });
  card.classList.add("flying-card");
  Object.assign(card.style, {
    left: `${fromRect.left}px`,
    top: `${fromRect.top}px`,
    width: `${fromRect.width}px`,
    height: `${fromRect.height}px`,
  });
  layer.appendChild(card);

  const animation = card.animate(
    [
      {
        left: `${fromRect.left}px`,
        top: `${fromRect.top}px`,
        width: `${fromRect.width}px`,
        height: `${fromRect.height}px`,
        transform: "scale(1)",
      },
      {
        left: `${toRect.left}px`,
        top: `${toRect.top}px`,
        width: `${toRect.width}px`,
        height: `${toRect.height}px`,
        transform: "scale(1)",
      },
    ],
    { duration, easing: "cubic-bezier(0.22, 1, 0.36, 1)", fill: "forwards" },
  );

  await animation.finished;
  card.remove();
}

function playSource(action) {
  const hand = document.getElementById(SEAT_HAND_IDS[action.player]);
  if (action.player === 0) {
    return hand.querySelector(`[data-card-id="${action.card}"]`) ?? hand;
  }
  return hand.lastElementChild ?? hand;
}

async function collectTrick(winner) {
  await wait(260);
  const pile = document.getElementById(SEAT_PILE_IDS[winner]);
  const destination = centeredRect(pile, 36, 53);
  const flights = [];

  for (let seat = 0; seat < SEAT_TRICK_IDS.length; seat += 1) {
    const slot = document.getElementById(SEAT_TRICK_IDS[seat]);
    const card = slot.querySelector(".card");
    if (!card) continue;
    const cardId = card.dataset.cardId;
    const source = card.getBoundingClientRect();
    card.style.opacity = "0";
    flights.push(
      wait(flights.length * 70).then(() =>
        flyCard(cardId, source, destination, { duration: 380 }),
      ),
    );
  }

  await Promise.all(flights);
  for (const slotId of SEAT_TRICK_IDS) {
    document.getElementById(slotId).replaceChildren();
  }
  await wait(180);
}

export async function animatePlay(action) {
  const sourceEl = playSource(action);
  const source = sourceEl.getBoundingClientRect();
  const slot = document.getElementById(SEAT_TRICK_IDS[action.player]);
  const destination = centeredRect(slot, 68, 100);

  await flyCard(action.card, source, destination);
  slot.replaceChildren(createCardElement(action.card));

  if (action.trick_winner !== null && action.trick_winner !== undefined) {
    await collectTrick(action.trick_winner);
  } else {
    await wait(180);
  }
}

export async function animatePass(action) {
  const humanHand = document.getElementById(SEAT_HAND_IDS[0]);
  const targetHand = document.getElementById(SEAT_HAND_IDS[action.target]);
  const targetRect = centeredRect(targetHand, 40, 59);

  const outgoing = action.cards.map((cardId, index) => {
    const sourceEl = humanHand.querySelector(`[data-card-id="${cardId}"]`);
    if (!sourceEl) return Promise.resolve();
    const source = sourceEl.getBoundingClientRect();
    sourceEl.style.opacity = "0";
    return wait(index * 80).then(() =>
      flyCard(cardId, source, targetRect, { faceDown: true, duration: 520 }),
    );
  });

  const opponentTransfers = action.routes
    .filter((route) => route.source !== 0 && route.target !== 0)
    .flatMap((route) => {
      const sourceHand = document.getElementById(SEAT_HAND_IDS[route.source]);
      const destinationHand = document.getElementById(SEAT_HAND_IDS[route.target]);
      const source = centeredRect(sourceHand, 40, 59);
      const destination = centeredRect(destinationHand, 40, 59);
      return [0, 1, 2].map((index) =>
        wait(index * 80).then(() =>
          flyCard("", source, destination, { faceDown: true, duration: 520 }),
        ),
      );
    });

  await Promise.all([...outgoing, ...opponentTransfers]);
  await wait(180);

  const sourceHand = document.getElementById(SEAT_HAND_IDS[action.source]);
  const sourceRect = centeredRect(sourceHand, 40, 59);
  const destination = centeredRect(humanHand, 54, 80);
  await Promise.all(
    action.incoming.map((cardId, index) =>
      wait(index * 90).then(() =>
        flyCard(cardId, sourceRect, destination, { duration: 540 }),
      ),
    ),
  );
  await wait(120);
}

export async function animateAction(action) {
  if (!action) return;
  if (action.type === "play") {
    await animatePlay(action);
  } else if (action.type === "pass") {
    await animatePass(action);
  }
}
