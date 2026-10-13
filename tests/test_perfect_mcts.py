from __future__ import annotations

import numpy as np
import pytest
import torch

from hearts.cards import Card, Rank, Suit
from hearts.config import GameConfig
from hearts.perfect.encode import normalize_points, normalize_scores
from hearts.perfect.game import card_from_id, new_hold_round
from hearts.perfect.mcts import MCTSConfig, Node, _select_child, run_mcts, select_action
from hearts.perfect.model import NetConfig, PerfectNet
from hearts.rules import NUM_PLAYERS, PassDirection
from hearts.state import DECK_SIZE, GameState, Phase


def c(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def _small_net(seed: int = 0) -> PerfectNet:
    torch.manual_seed(seed)
    return PerfectNet(NetConfig(d_model=64, n_blocks=1)).eval()


def _advance(state: GameState, n: int) -> GameState:
    for _ in range(n):
        legal = state.legal_moves_for(state.current_player)
        state = state.play_card(state.current_player, legal[0])
    return state


def test_visits_sum_and_only_legal() -> None:
    state = _advance(new_hold_round(seed=5), 4)  # trick 2 leader, several legal cards
    legal_ids = {card.card_id for card in state.legal_moves_for(state.current_player)}
    assert len(legal_ids) > 1
    cfg = MCTSConfig(n_simulations=40)
    result = run_mcts(state, _small_net(), cfg, np.random.default_rng(0))
    assert result.visit_counts.sum() == cfg.n_simulations
    assert set(np.flatnonzero(result.visit_counts)) <= legal_ids
    assert np.isclose(result.pi.sum(), 1.0)
    assert set(result.action_ids) == legal_ids
    assert result.root is not None and result.root.N == cfg.n_simulations + 1


def test_single_legal_move_is_short_circuited() -> None:
    state = new_hold_round(seed=1)  # 2 of clubs is forced
    result = run_mcts(state, _small_net(), MCTSConfig(n_simulations=30), np.random.default_rng(0))
    two_clubs = c(Rank.TWO, Suit.CLUBS).card_id
    assert result.pi[two_clubs] == 1.0
    assert result.visit_counts.sum() == 1


def test_terminal_node_uses_true_scores() -> None:
    state = _advance(new_hold_round(seed=2), 51)
    assert state.phase is Phase.PLAYING
    last_card = state.hands[state.current_player][0]
    final = state.play_card(state.current_player, last_card)
    node = Node(final)
    assert node.is_terminal
    assert np.allclose(node.terminal_value, normalize_scores(final.last_round_scores))


def _node_with_children(to_move: int, child_values: dict[int, np.ndarray]) -> Node:
    """Build a fake expanded node with visited children (no engine state needed)."""
    node = Node.__new__(Node)
    node.state = None
    node.to_move = to_move
    node.legal_ids = sorted(child_values)
    node.priors = np.zeros(DECK_SIZE)
    for a in node.legal_ids:
        node.priors[a] = 1.0 / len(node.legal_ids)
    node.children = {}
    node.N = 0
    node.W = np.zeros(NUM_PLAYERS)
    node.terminal_value = None
    for a, v in child_values.items():
        child = Node.__new__(Node)
        child.state = None
        child.to_move = (to_move + 1) % NUM_PLAYERS
        child.legal_ids = []
        child.priors = None
        child.children = {}
        child.terminal_value = None
        child.N = 10
        child.W = np.asarray(v, dtype=np.float64) * 10
        node.children[a] = child
        node.N += 10
        node.W += child.W
    return node


def test_selection_sign_root_maximizes_opponent_minimizes() -> None:
    root_player = 0
    # Action 0 is great for seat 0 and bad for seat 1; action 1 is the reverse.
    values = {
        0: np.array([0.8, -0.8, 0.0, 0.0]),
        1: np.array([-0.8, 0.8, 0.0, 0.0]),
    }
    cfg = MCTSConfig(n_simulations=1, c_puct=0.0)
    root_node = _node_with_children(to_move=0, child_values=values)
    assert _select_child(root_node, root_player, cfg) == 0
    opp_node = _node_with_children(to_move=1, child_values=values)
    # Paranoid: opponent minimizes seat 0's value.
    assert _select_child(opp_node, root_player, cfg) == 1
    # Self-interest: opponent maximizes its own value (also 1 here); flip the payoff to separate them.
    values_selfish = {
        0: np.array([0.8, 0.9, 0.0, 0.0]),  # good for both 0 and 1
        1: np.array([-0.8, -0.9, 0.0, 0.0]),
    }
    opp_node2 = _node_with_children(to_move=1, child_values=values_selfish)
    assert _select_child(opp_node2, root_player, cfg) == 1  # paranoid still hurts seat 0
    cfg_self = MCTSConfig(n_simulations=1, c_puct=0.0, opponent_mode="self_interest")
    assert _select_child(opp_node2, root_player, cfg_self) == 0


def _endgame_two_cards_each() -> GameState:
    """Trick 12, seat 0 to lead, hearts broken.

    Seat 0: 2C, AH   Seat 1: 7H, QS   Seat 2: 4C, 2H   Seat 3: 5C, 3H

    Lead AH: everyone must follow hearts (seat 1's only heart is 7H); seat 0 takes 4.
      Trick 13: seat 0 leads 2C, seat 1 dumps QS, seat 3 wins with 5C. Seat 0 total = 4.
    Lead 2C: seat 1 is void in clubs and chooses. Seat 3 wins with 5C and leads 3H;
      seat 0's AH wins trick 13 and takes whatever seat 1 still holds.
        seat 1 dumps QS now (on seat 3): seat 0 ends with 3H+AH+2H+7H = 4.
        seat 1 keeps QS for trick 13:      seat 0 ends with 3H+AH+2H+QS = 16.
    Seat 1 scores 0 either way, so only a paranoid seat 1 keeps the queen, and only
    a paranoid seat 0 therefore strictly prefers leading AH.
    """
    hands = [
        [c(Rank.TWO, Suit.CLUBS), c(Rank.ACE, Suit.HEARTS)],
        [c(Rank.SEVEN, Suit.HEARTS), c(Rank.QUEEN, Suit.SPADES)],
        [c(Rank.FOUR, Suit.CLUBS), c(Rank.TWO, Suit.HEARTS)],
        [c(Rank.FIVE, Suit.CLUBS), c(Rank.THREE, Suit.HEARTS)],
    ]
    in_hand = {card for hand in hands for card in hand}
    rest = [card_from_id(i) for i in range(DECK_SIZE) if card_from_id(i) not in in_hand]
    hearts_rest = [x for x in rest if x.is_heart]
    others = [x for x in rest if not x.is_heart]
    # Spread the remaining hearts so nobody can shoot the moon; non-point cards to seat 2/3.
    tricks_won = [[], [], [], []]
    for i, h in enumerate(hearts_rest):
        tricks_won[1 + (i % 3)].append(h)
    for i, x in enumerate(others):
        tricks_won[2 + (i % 2)].append(x)
    return GameState(
        config=GameConfig(),
        phase=Phase.PLAYING,
        round_number=1,
        hands=hands,
        pass_direction=PassDirection.HOLD,
        trick_number=12,
        trick_leader=0,
        current_player=0,
        current_trick=[],
        hearts_broken=True,
        tricks_won=tricks_won,
        scores=[0, 0, 0, 0],
    )


def _finish_with(state: GameState, moves: list[Card]) -> list[int]:
    for card in moves:
        state = state.play_card(state.current_player, card)
    assert state.phase in (Phase.ROUND_OVER, Phase.GAME_OVER)
    return state.last_round_scores


def test_endgame_construction_matches_docstring() -> None:
    state = _endgame_two_cards_each()
    base = [sum(x.points(state.config) for x in won) for won in state.tricks_won]
    lead_ah = _finish_with(
        state,
        [c(Rank.ACE, Suit.HEARTS), c(Rank.SEVEN, Suit.HEARTS), c(Rank.TWO, Suit.HEARTS), c(Rank.THREE, Suit.HEARTS),
         c(Rank.TWO, Suit.CLUBS), c(Rank.QUEEN, Suit.SPADES), c(Rank.FOUR, Suit.CLUBS), c(Rank.FIVE, Suit.CLUBS)],
    )
    assert lead_ah[0] - base[0] == 4
    keep_queen = _finish_with(
        state,
        [c(Rank.TWO, Suit.CLUBS), c(Rank.SEVEN, Suit.HEARTS), c(Rank.FOUR, Suit.CLUBS), c(Rank.FIVE, Suit.CLUBS),
         c(Rank.THREE, Suit.HEARTS), c(Rank.ACE, Suit.HEARTS), c(Rank.QUEEN, Suit.SPADES), c(Rank.TWO, Suit.HEARTS)],
    )
    assert keep_queen[0] - base[0] == 16
    assert keep_queen[1] - base[1] == 0
    dump_queen = _finish_with(
        state,
        [c(Rank.TWO, Suit.CLUBS), c(Rank.QUEEN, Suit.SPADES), c(Rank.FOUR, Suit.CLUBS), c(Rank.FIVE, Suit.CLUBS),
         c(Rank.THREE, Suit.HEARTS), c(Rank.ACE, Suit.HEARTS), c(Rank.SEVEN, Suit.HEARTS), c(Rank.TWO, Suit.HEARTS)],
    )
    assert dump_queen[0] - base[0] == 4
    assert dump_queen[1] - base[1] == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_paranoid_opponent_keeps_queen_and_root_avoids_it(seed: int) -> None:
    state = _endgame_two_cards_each()
    cfg = MCTSConfig(n_simulations=100, c_puct=1.0, dirichlet_eps=0.0)
    result = run_mcts(state, _small_net(seed), cfg, np.random.default_rng(seed), add_noise=False)
    ah = c(Rank.ACE, Suit.HEARTS).card_id
    two_c = c(Rank.TWO, Suit.CLUBS).card_id
    qs = c(Rank.QUEEN, Suit.SPADES).card_id
    seven_h = c(Rank.SEVEN, Suit.HEARTS).card_id
    base = sum(x.points(state.config) for x in state.tricks_won[0])

    # Forced follow-ups are collapsed, so every child here is a terminal node with exact values.
    two_c_node = result.root.children[two_c]
    keep, dump = two_c_node.children[seven_h], two_c_node.children[qs]
    assert keep.is_terminal and dump.is_terminal
    assert np.isclose(keep.W[0] / keep.N, normalize_points(base + 16))
    assert np.isclose(dump.W[0] / dump.N, normalize_points(base + 4))
    # Paranoid seat 1 keeps the queen (plays 7H): pure exploitation at that node selects it,
    # and it received the majority of that node's visits.
    assert _select_child(two_c_node, 0, MCTSConfig(n_simulations=1, c_puct=0.0)) == seven_h
    assert keep.N > dump.N
    # Root therefore prefers the AH lead (4 points, forced) over 2C (16 under paranoia).
    assert result.root.children[ah].is_terminal
    assert np.isclose(result.q_values[ah], normalize_points(base + 4))
    assert result.q_values[two_c] < result.q_values[ah]
    assert result.visit_counts[ah] > result.visit_counts[two_c]


def test_forced_moves_are_collapsed_in_tree() -> None:
    state = _endgame_two_cards_each()
    cfg = MCTSConfig(n_simulations=20, dirichlet_eps=0.0)
    result = run_mcts(state, _small_net(), cfg, np.random.default_rng(0), add_noise=False)

    def walk(node: Node) -> None:
        for child in node.children.values():
            assert child.is_terminal or len(child.state.legal_moves_for(child.to_move)) > 1
            walk(child)

    walk(result.root)
    # Without collapsing, the same tree contains single-move nodes.
    cfg_raw = MCTSConfig(n_simulations=20, dirichlet_eps=0.0, collapse_forced_moves=False)
    raw = run_mcts(state, _small_net(), cfg_raw, np.random.default_rng(0), add_noise=False)
    forced = [
        ch for ch in raw.root.children.values()
        if not ch.is_terminal and len(ch.state.legal_moves_for(ch.to_move)) == 1
    ]
    assert forced


def test_select_action_temperature() -> None:
    rng = np.random.default_rng(0)
    pi = np.zeros(DECK_SIZE)
    pi[3] = 0.7
    pi[9] = 0.3
    assert select_action(pi, 0.0, rng) == 3
    draws = {select_action(pi, 1.0, rng) for _ in range(200)}
    assert draws == {3, 9}
    with pytest.raises(ValueError):
        MCTSConfig(opponent_mode="nope")
