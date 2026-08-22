const PASS_CYCLE = ["LEFT", "RIGHT", "ACROSS", "HOLD"];

const PASS_LABELS = {
  LEFT: "Left",
  RIGHT: "Right",
  ACROSS: "Across",
  HOLD: "Hold",
};

const PASS_ICONS = {
  LEFT: "←",
  RIGHT: "→",
  ACROSS: "↕",
  HOLD: "✋",
};

/** Seat indices: 0=South, 1=West, 2=North, 3=East */
const SEAT_POSITIONS = {
  0: { x: 50, y: 88 },
  1: { x: 12, y: 50 },
  2: { x: 50, y: 12 },
  3: { x: 88, y: 50 },
};

function passTarget(seat, direction) {
  if (direction === "LEFT") return (seat + 1) % 4;
  if (direction === "RIGHT") return (seat - 1 + 4) % 4;
  if (direction === "ACROSS") return (seat + 2) % 4;
  return null;
}

function cycleIndexForRound(roundNumber) {
  if (!roundNumber) return 0;
  return (roundNumber - 1) % PASS_CYCLE.length;
}

export function renderPassPatternCycle(state) {
  const container = document.getElementById("pass-pattern-cycle");
  if (!container) return;

  container.replaceChildren();
  const activeIndex = cycleIndexForRound(state.round_number);

  PASS_CYCLE.forEach((direction, index) => {
    const item = document.createElement("div");
    item.className = "pass-cycle__item";
    if (index === activeIndex) item.classList.add("pass-cycle__item--active");
    if (direction === "HOLD") item.classList.add("pass-cycle__item--hold");

    item.innerHTML = `
      <span class="pass-cycle__icon" aria-hidden="true">${PASS_ICONS[direction]}</span>
      <span class="pass-cycle__label">${PASS_LABELS[direction]}</span>
    `;
    container.appendChild(item);
  });
}

export function renderPassArrows(state) {
  const overlay = document.getElementById("pass-arrows");
  if (!overlay) return;

  const show =
    state.phase === "PASSING" &&
    state.pass_direction !== "HOLD" &&
    !state.pass_submitted[0];

  overlay.classList.toggle("hidden", !show);
  if (!show) {
    overlay.replaceChildren();
    return;
  }

  const from = SEAT_POSITIONS[0];
  const targetSeat = passTarget(0, state.pass_direction);
  const to = SEAT_POSITIONS[targetSeat];

  overlay.replaceChildren();

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "pass-arrows__svg");
  svg.setAttribute("viewBox", "0 0 100 100");
  svg.setAttribute("preserveAspectRatio", "none");

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  marker.setAttribute("id", "pass-arrowhead");
  marker.setAttribute("markerWidth", "6");
  marker.setAttribute("markerHeight", "6");
  marker.setAttribute("refX", "5");
  marker.setAttribute("refY", "3");
  marker.setAttribute("orient", "auto");

  const arrowPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  arrowPath.setAttribute("d", "M0,0 L6,3 L0,6 Z");
  arrowPath.setAttribute("fill", "var(--pass-arrow-color)");
  marker.appendChild(arrowPath);
  defs.appendChild(marker);
  svg.appendChild(defs);

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", String(from.x));
  line.setAttribute("y1", String(from.y));
  line.setAttribute("x2", String(to.x));
  line.setAttribute("y2", String(to.y));
  line.setAttribute("class", "pass-arrows__line");
  line.setAttribute("marker-end", "url(#pass-arrowhead)");
  svg.appendChild(line);

  const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
  label.setAttribute("x", String((from.x + to.x) / 2));
  label.setAttribute("y", String((from.y + to.y) / 2));
  label.setAttribute("class", "pass-arrows__label");
  label.textContent = `Pass ${PASS_LABELS[state.pass_direction].toLowerCase()}`;
  svg.appendChild(label);

  overlay.appendChild(svg);
}

export function renderPassPattern(state) {
  renderPassPatternCycle(state);
  renderPassArrows(state);
}
