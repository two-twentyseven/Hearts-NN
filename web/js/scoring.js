export function renderEmptyScoreboard() {
  const container = document.getElementById("score-table");
  container.replaceChildren();

  const header = document.createElement("div");
  header.className = "score-row score-row--header";
  header.innerHTML = `
    <span>Player</span>
    <span class="score-row__round">Round</span>
    <span class="score-row__total">Total</span>
  `;
  container.appendChild(header);

  ["You", "West", "North", "East"].forEach((name) => {
    const row = document.createElement("div");
    row.className = "score-row";
    if (name === "You") row.classList.add("score-row--viewer");
    row.innerHTML = `
      <span class="score-row__name">${name}</span>
      <span class="score-row__round">—</span>
      <span class="score-row__total">0</span>
    `;
    container.appendChild(row);
  });
}

export function renderScoreboard(state) {
  const container = document.getElementById("score-table");
  container.replaceChildren();

  const header = document.createElement("div");
  header.className = "score-row score-row--header";
  header.innerHTML = `
    <span>Player</span>
    <span class="score-row__round">Round</span>
    <span class="score-row__total">Total</span>
  `;
  container.appendChild(header);

  const leaderScore = Math.min(...state.scores);

  state.players.forEach((name, index) => {
    const row = document.createElement("div");
    row.className = "score-row";
    if (index === 0) row.classList.add("score-row--viewer");
    if (state.scores[index] === leaderScore) row.classList.add("score-row--leader");

    const roundScore = state.last_round_scores
      ? state.last_round_scores[index]
      : "—";

    row.innerHTML = `
      <span class="score-row__name">${name}</span>
      <span class="score-row__round">${roundScore}</span>
      <span class="score-row__total">${state.scores[index]}</span>
    `;
    container.appendChild(row);
  });
}

export function renderRoundModal(state) {
  const body = document.getElementById("modal-round-body");
  body.replaceChildren();

  if (!state.last_round_scores) {
    body.textContent = "No round scores available.";
    return;
  }

  state.players.forEach((name, index) => {
    const row = document.createElement("div");
    row.className = "modal-score-row";
    row.innerHTML = `
      <span>${name}</span>
      <span>${state.last_round_scores[index]} pts</span>
    `;
    body.appendChild(row);
  });
}

export function renderGameOverModal(state) {
  const body = document.getElementById("modal-game-over-body");
  body.replaceChildren();

  const winnerIndex = state.scores.indexOf(Math.min(...state.scores));
  const title = document.createElement("p");
  title.textContent = `${state.players[winnerIndex]} wins with ${state.scores[winnerIndex]} points!`;
  body.appendChild(title);

  state.players.forEach((name, index) => {
    const row = document.createElement("div");
    row.className = "modal-score-row";
    row.innerHTML = `
      <span>${name}</span>
      <span>${state.scores[index]} pts</span>
    `;
    body.appendChild(row);
  });
}

export function showModal(id) {
  const modal = document.getElementById(id);
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
}

export function hideModal(id) {
  const modal = document.getElementById(id);
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
}

export function hideAllModals() {
  hideModal("modal-round");
  hideModal("modal-game-over");
}
