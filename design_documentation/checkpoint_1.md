# Hearts-NN — checklist

Last updated: 2026-09-29
Status: **not started.** Hold rounds only (no passing, no match-to-100).

---

## What we are building

Two models, glued together at play time.

```
my hand + public history
        │
        ▼
   guess model          ← “how might the remaining cards be split?”
        │                  (refreshed every decision, never frozen)
        ▼
   sample possible deals
        │
        ▼
   for each deal: all-seeing player + search
        │             (may use hidden cards to score what happens next)
        ▼
   average the results
        │             (the card we play cannot depend on knowing which deal is true)
        ▼
   play one card
```

The all-seeing player is trained first, with every hand visible, so search is a normal perfect-information problem. The guess model is trained second, only from what a real player would have seen.

---

## Locked decisions

- **Inner search assumes strong play.** We evaluate “what’s bad for us if people play well,” including using the hidden deal inside the simulator. A good player should still beat weak players; we are not tuning only to beat other strong bots by failing to handle weak ones.
- **Guess model trains on hidden-info games.** Not on games where everyone could see all four hands. Those histories do not look like real Hearts.
- **Sampling is our algorithm.** The guess model does not emit a finished deal. We draw complete, legal assignments (right hand sizes, no card in two places, no suit for a seat that already showed out).

---

## Macro checklist

### 0. Scope

- [ ] Hold-round play only
- [ ] No passing
- [ ] No match score to 100
- [ ] No browser wiring until the loop below works

### 1. All-seeing player

A net that sees all four hands and a search that uses it.

- [x] Encode a position with every remaining card’s holder, not just “cards in my hand” (`hearts/perfect/encode.py`: 52 cards x 12 locations, egocentric)
- [x] Policy + value heads on that encoding (`hearts/perfect/model.py`: 52-card policy, 4-seat post-round value)
- [x] Search (tree search with the net) on a fully known deal (`hearts/perfect/mcts.py`: paranoid PUCT, no rollouts)
- [x] Train by self-play where all four seats see everything (`scripts/perfect_self_play.py`: populate -> train -> repeat)
- [x] Confirm it is clearly stronger than legal-random (ckpt_0039, 200 sims, 400 rounds: 2.13 pts/round vs 8.13 for random, gap +6.00 ± 0.52; lowest score in 67.5% of rounds — `artifacts/perfect/arena_iter39_1v3.json`)
- [ ] Confirm it is clearly stronger than the current imperfect-info net

### 2. Hidden-info games (data for the guesser)

Play real Hearts: each seat only sees its own hand plus public history. Inside search only, fill in guessed opponent hands.

- [ ] Generate games with **uniform legal** leftover shuffles (hand sizes correct) + the all-seeing evaluator
- [ ] Record, at every decision: what that player could observe, and where the remaining cards actually were
- [ ] Later: same pipeline, but leftovers drawn from the guess model instead of uniformly

### 3. Guess model

Given what a real player sees, output a distribution over where remaining cards might sit. Recompute from scratch each decision.

- [ ] Input: current public history + my hand (same kind of observation a human has)
- [ ] Output: scores/probabilities for remaining cards × other seats (not one fake deal)
- [ ] Train on the hidden-info traces from §2 (true remaining assignment is the target)
- [ ] Score it so honest uncertainty is allowed when the history does not pin a card
- [ ] Public facts must be sharp: cards already played, shown voids, hand sizes
- [ ] Beats “shuffle the leftovers at random,” especially after someone shows out of a suit

### 4. Deal sampler

Turn the guess model’s distribution into actual possible deals.

- [ ] Sequential (or equivalent) assignment of remaining cards to the three hidden hands
- [ ] Enforce: one card, one seat
- [ ] Enforce: each hidden hand has the right number of cards
- [ ] Enforce: a seat that showed out of a suit cannot be given that suit
- [ ] Reject or never propose impossible deals
- [ ] Draw several deals, weighted by the current guess

### 5. Play loop

This is the product.

- [ ] At each decision, rebuild the guess from the current observation (do not reuse an old sample)
- [ ] Sample possible deals with §4
- [ ] In each deal, run the all-seeing player + search to score each legal card
- [ ] Average those scores using sample weights
- [ ] Play the card that looks best on that average — one card, no dependence on which sample was “true”
- [ ] Confirm: guess-weighted search beats the same all-seeing player with uniform leftover shuffles, same number of samples

### 6. Iterate

- [ ] Fine-tune the all-seeing player on full positions that hidden-info play actually reaches (not only god-mode games)
- [ ] Replace uniform leftover shuffles in §2 with the trained sampler; retrain the guesser
- [ ] Repeat until scores stop moving

---

## Done when

- [ ] All-seeing player, given all four hands, wins by a clear round-score margin vs current bots
- [ ] Guesser is better than random legal shuffles, and nearly certain on facts the history already proves
- [ ] Sampler only produces legal deals
- [ ] Full play loop (guess → sample → search → average → one card) beats uniform sampling with the same search budget
- [ ] The card chosen at the table never uses information that seat would not have
