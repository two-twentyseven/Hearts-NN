export function showError(message) {
  const banner = document.getElementById("error-banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
}

export function clearError() {
  const banner = document.getElementById("error-banner");
  banner.textContent = "";
  banner.classList.add("hidden");
}

export function setLoading(active) {
  const overlay = document.getElementById("loading-overlay");
  overlay.classList.toggle("hidden", !active);
  overlay.setAttribute("aria-hidden", active ? "false" : "true");
}

export function showWelcome(show) {
  const screen = document.getElementById("welcome-screen");
  screen.classList.toggle("hidden", !show);
  screen.setAttribute("aria-hidden", show ? "false" : "true");
  document.body.classList.toggle("game-active", !show);
}

export function setChromeVisibility({ gameActive, hideNewGame = false }) {
  const showNewGame = gameActive && !hideNewGame;
  document.getElementById("btn-new-game").classList.toggle("hidden", !showNewGame);
  document.getElementById("round-badge").classList.toggle("hidden", !gameActive);
  document.getElementById("pass-pattern").classList.toggle("hidden", !gameActive);
  document.getElementById("review-controls").classList.toggle("hidden", !gameActive);
}

export function setStatusText(text) {
  document.getElementById("status-banner").textContent = text;
}
