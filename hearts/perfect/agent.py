from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch

from hearts.cards import Card
from hearts.perfect.game import card_from_id
from hearts.perfect.mcts import MCTSConfig, run_mcts
from hearts.perfect.model import PerfectNet, load_checkpoint
from hearts.state import GameState, Phase

CHECKPOINT_ENV = "HEARTS_PERFECT_CKPT"
SIMS_ENV = "HEARTS_PERFECT_SIMS"
DEFAULT_CHECKPOINT = Path("artifacts/perfect/latest.pt")
DEFAULT_SIMS = 400


def default_checkpoint_path() -> Path:
    return Path(os.environ.get(CHECKPOINT_ENV, str(DEFAULT_CHECKPOINT)))


def default_simulations() -> int:
    return int(os.environ.get(SIMS_ENV, DEFAULT_SIMS))


class OmniscientBot:
    """Perfect-information player for live games: trained PerfectNet + paranoid MCTS.

    Works on any ``GameState`` in the PLAYING phase, including passing rounds and
    later rounds of a match; the encoder reads hands, the current trick, won
    tricks and match scores directly and ignores the event history.
    """

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        *,
        n_simulations: int | None = None,
        net: PerfectNet | None = None,
        seed: int | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint is not None else default_checkpoint_path()
        self.mcts_cfg = MCTSConfig(
            n_simulations=n_simulations if n_simulations is not None else default_simulations(),
            dirichlet_eps=0.0,
            opponent_mode="paranoid",
        )
        self._net = net
        self._net_iteration: int | None = None
        self._loaded_mtime: float | None = None
        self._lock = threading.RLock()  # play() holds it while net() may lazily load
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------- loading
    def available(self) -> bool:
        return self._net is not None or self.checkpoint.exists()

    def _checkpoint_mtime(self) -> float | None:
        try:
            return self.checkpoint.stat().st_mtime
        except OSError:
            return None

    @property
    def net(self) -> PerfectNet:
        """Cached network; hot-reloads when the checkpoint file changes (e.g. training is still running)."""
        with self._lock:
            mtime = self._checkpoint_mtime()
            needs_load = self._net is None or (mtime is not None and mtime != self._loaded_mtime)
            if needs_load:
                if mtime is None:
                    if self._net is not None:
                        return self._net
                    raise FileNotFoundError(
                        f"No omniscient checkpoint at {self.checkpoint}. "
                        f"Train one with scripts/perfect_self_play.py or set ${CHECKPOINT_ENV}."
                    )
                try:
                    net, payload = load_checkpoint(self.checkpoint)
                except Exception:
                    # Likely a checkpoint mid-write by the trainer; keep the current net if we have one.
                    if self._net is not None:
                        return self._net
                    raise
                net.eval()
                self._net = net
                self._net_iteration = payload.get("iteration")
                self._loaded_mtime = mtime
            return self._net

    @property
    def iteration(self) -> int | None:
        _ = self.net
        return self._net_iteration

    def reload(self) -> None:
        """Drop the cached network so the next move picks up a newer checkpoint."""
        with self._lock:
            self._net = None
            self._net_iteration = None
            self._loaded_mtime = None

    # ------------------------------------------------------------------ acting
    @torch.inference_mode()
    def play(self, state: GameState) -> Card | None:
        if state.phase is not Phase.PLAYING:
            return None
        legal = state.legal_moves_for(state.current_player)
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]
        with self._lock:
            result = run_mcts(state, self.net, self.mcts_cfg, self._rng, add_noise=False)
        return card_from_id(int(np.argmax(result.visit_counts)))

    def pass_cards(self, hand: list[Card]) -> list[Card]:
        """Passing is outside what the network was trained on; use the hand heuristic."""
        from hearts.human_bot import select_pass

        return select_pass(list(hand))


_shared: OmniscientBot | None = None
_shared_lock = threading.Lock()


def shared_bot() -> OmniscientBot:
    """Process-wide bot instance (one loaded network per server)."""
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = OmniscientBot()
    return _shared


def set_shared_bot(bot: OmniscientBot | None) -> None:
    global _shared
    with _shared_lock:
        _shared = bot
