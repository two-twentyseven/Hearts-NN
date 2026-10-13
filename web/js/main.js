import { initGame } from "./game.js?v=8";
import { showError } from "./ui.js";

document.addEventListener("DOMContentLoaded", () => {
  try {
    initGame();
  } catch (error) {
    showError(`Failed to start UI: ${error.message}`);
    console.error(error);
  }
});

window.addEventListener("unhandledrejection", (event) => {
  showError(event.reason?.message || "Unexpected error");
  console.error(event.reason);
});
