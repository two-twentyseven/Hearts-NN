const API_CANDIDATES = [
  "/api",
  "http://127.0.0.1:8000/api",
  "http://localhost:8000/api",
];

let resolvedApiBase = null;

async function probeApiBase() {
  if (resolvedApiBase) return resolvedApiBase;

  for (const base of API_CANDIDATES) {
    try {
      const response = await fetch(`${base}/health`, { method: "GET" });
      if (response.ok) {
        resolvedApiBase = base;
        return base;
      }
    } catch {
      // try next candidate
    }
  }

  throw new Error(
    "Cannot reach the Hearts server. Run: uvicorn hearts.api.app:app --reload"
  );
}

export async function checkConnection() {
  await probeApiBase();
  return true;
}

async function request(path, options = {}) {
  const base = await probeApiBase();
  let response;

  try {
    response = await fetch(`${base}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (error) {
    resolvedApiBase = null;
    throw new Error(
      "Lost connection to the server. Make sure uvicorn is running on port 8000."
    );
  }

  if (!response.ok) {
    let detail = { message: response.statusText };
    try {
      const body = await response.json();
      detail = body.detail || body;
    } catch {
      // ignore parse errors
    }
    const error = new Error(detail.message || "Request failed");
    error.reason = detail.reason;
    throw error;
  }

  return response.json();
}

export function createGame(difficulty = "random") {
  return request("/games", {
    method: "POST",
    body: JSON.stringify({ difficulty }),
  });
}

export function getGame(gameId) {
  return request(`/games/${gameId}`);
}

export function submitPass(gameId, cards) {
  return request(`/games/${gameId}/pass`, {
    method: "POST",
    body: JSON.stringify({ cards }),
  });
}

export function playCard(gameId, cardId) {
  return request(`/games/${gameId}/play`, {
    method: "POST",
    body: JSON.stringify({ card: cardId }),
  });
}

export function tickGame(gameId) {
  return request(`/games/${gameId}/tick`, { method: "POST" });
}

export function nextRound(gameId) {
  return request(`/games/${gameId}/next-round`, { method: "POST" });
}
