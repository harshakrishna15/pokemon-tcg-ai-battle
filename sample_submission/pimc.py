"""Determinized search (PIMC) v2 on top of the heuristic policy.

v1 rolled every determinization to terminal with a mirror-deck opponent and
scored win/loss.  That signal is high-variance (one 0/1 sample per rollout) and
slow (full games), so at a feasible budget it saw only a handful of noisy
samples per candidate -- no reliable edge over the pure heuristic -- and the
unbounded rollouts overran the per-move time budget.

v2 changes three things:

  1. Shallow rollout + board evaluation.  After committing a candidate action we
     let the heuristic finish our turn and play the opponent's next turn, then
     stop and score the *board*: prize differential is the dominant term, plus
     board HP, a powered-up active, board presence and deck health.  This is a
     low-variance continuous return and far cheaper than a full game, so we get
     many more samples per unit time.

  2. Paired determinizations (common random numbers).  Every candidate is
     evaluated on the *same* sampled hidden state inside each determinization,
     so the shared board cancels and only the action's effect remains -- the
     comparison is much less noisy for the same number of samples.  Only
     fully-completed determinizations are kept, so the pairing stays exact.

  3. A hard deadline enforced *inside* the rollout loop plus a depth cap, so a
     single decision can never overrun the per-move budget (v1's failure mode).

A trust region on top keeps the heuristic's top pick unless another candidate is
*clearly* better on average, so search noise can't drag us below the base
policy.

Opponent model: we do not know the opponent's deck, so the hidden opponent
cards are drawn from a coherent mirror of our own deck (this guarantees legal
Pokemon lines and at least one Basic for setup).  In self-play *against the
heuristic* the opponent literally is our deck, so this model is exact there; on
the live ladder it is just a plausible-threat proxy.

Everything is time-budgeted and exception-guarded: any failure yields None and
the caller falls back to the pure heuristic.
"""
import os
import random
import time
from collections import Counter


# ---- env-tunable knobs (so configs can be swept without editing code) -------
def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _envi(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


HORIZON_TURNS = _envi("PIMC_HORIZON", 2)     # roll to end of opp's next turn
MAX_DET = _envi("PIMC_MAXDET", 64)           # cap on determinizations
MAX_STEPS = _envi("PIMC_MAXSTEPS", 240)      # hard depth cap per rollout
SAFETY_S = _envf("PIMC_SAFETY", 0.02)        # reserve before the budget edge
MARGIN = _envf("PIMC_MARGIN", 16.0)          # trust region: deviate from the
#   heuristic only if clearly better.  16 (conservative) measured best in a
#   {4,8,16} sweep and is the safest choice on the live ladder, where the mirror
#   opponent model is wrong -- so override the base policy only when confident.
MIN_VISITS = _envi("PIMC_MINVISITS", 2)      # ...and seen on >=N determinizations

W = {
    "term": 1.0e6,     # terminal win/loss dominates any board score
    "prize": 100.0,    # prize differential -- the real win condition
    "hp": 0.04,        # board HP differential (attrition / staying power)
    "energy": 5.0,     # energy on my active (a powered-up attacker)
    "board": 8.0,      # Pokemon-count differential (board presence)
    "noactive": 60.0,  # having no active Pokemon is close to losing
    "deck": 6.0,       # deck-health term ...
    "deck_lo": 6,      # ... applied once deckCount drops to this or below
}
for _k in ("term", "prize", "hp", "energy", "board", "noactive", "deck"):
    W[_k] = _envf("PIMC_W_" + _k.upper(), W[_k])
W["deck_lo"] = _envi("PIMC_W_DECK_LO", W["deck_lo"])

_LAST_DECISION = {}     # populated by decide() for the offline probe / tuning


def _imports():
    from cg.api import search_begin, search_step, search_release, search_end
    return search_begin, search_step, search_release, search_end


def _in_play_ids(ps):
    ids = []
    pks = (list(ps.active) if ps.active else []) + list(ps.bench or [])
    for pk in pks:
        if not pk:
            continue
        ids.append(pk.id)
        for grp in (pk.energyCards, pk.tools, pk.preEvolution):
            for c in (grp or []):
                ids.append(c.id)
    return ids


def _determinize(obs, my_deck):
    """Sample the hidden information for search_begin:
    (your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active).

    Only *hidden* cards are supplied; the visible board (both players' in-play
    Pokemon, attached cards and discards) comes from `obs` itself."""
    st = obs.current
    yi = st.yourIndex
    me, opp = st.players[yi], st.players[1 - yi]

    # my hidden cards = my full 60 minus everything I can currently see
    rem = Counter(my_deck)
    for cid in ([c.id for c in (me.hand or [])]
                + [c.id for c in (me.discard or [])]
                + _in_play_ids(me)):
        rem[cid] -= 1
    remainder = [cid for cid, k in rem.items() for _ in range(max(0, k))]
    random.shuffle(remainder)
    pc = len(me.prize)
    your_prize = remainder[:pc]
    your_deck = remainder[pc:]
    if len(your_deck) < me.deckCount and my_deck:           # safety pad
        your_deck += [my_deck[0]] * (me.deckCount - len(your_deck))

    # opponent hidden cards: coherent mirror of our deck (legal lines + a Basic)
    need = opp.deckCount + (opp.handCount or 0) + len(opp.prize)
    src = list(my_deck)
    while len(src) < need:
        src += list(my_deck)
    random.shuffle(src)
    opp_deck = src[:opp.deckCount]
    opp_hand = src[opp.deckCount:opp.deckCount + (opp.handCount or 0)]
    opp_prize = src[opp.deckCount + (opp.handCount or 0):need]
    opp_active = []
    if opp.active and len(opp.active) > 0 and opp.active[0] is None:
        opp_active = [my_deck[0]] if my_deck else [119]     # face-down -> a Basic
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


# ----------------------------------------------------------- board evaluation ---
def _hp(ps):
    tot = 0
    for pk in (list(ps.active or []) + list(ps.bench or [])):
        if pk:
            tot += max(0, pk.hp or 0)
    return tot


def _count_pkmn(ps):
    return sum(1 for pk in (list(ps.active or []) + list(ps.bench or [])) if pk)


def _evaluate(cur, my_index):
    """Board value from my perspective; positive = good for me.  Prize
    differential dominates; the rest are positional tie-breakers."""
    if cur is None:
        return 0.0
    if cur.result != -1:
        if cur.result == my_index:
            return W["term"]
        if cur.result == 2:
            return 0.0
        return -W["term"]
    me = cur.players[my_index]
    opp = cur.players[1 - my_index]
    s = 0.0
    # prizes remaining: fewer of mine / more of theirs == I'm closer to winning
    s += (len(opp.prize) - len(me.prize)) * W["prize"]
    s += (_hp(me) - _hp(opp)) * W["hp"]
    s += (_count_pkmn(me) - _count_pkmn(opp)) * W["board"]
    a = me.active[0] if (me.active and me.active[0]) else None
    if a is None:
        s -= W["noactive"]
    else:
        s += min(len(a.energies or []), 4) * W["energy"]
    if not (opp.active and opp.active[0]):
        s += W["noactive"] * 0.5
    if me.deckCount is not None and me.deckCount <= W["deck_lo"]:
        s -= (W["deck_lo"] - me.deckCount) * W["deck"]
    if opp.deckCount is not None and opp.deckCount <= W["deck_lo"]:
        s += (W["deck_lo"] - opp.deckCount) * W["deck"] * 0.5
    return s


def _rollout(step, release, state, policy, my_index, start_turn, deadline):
    """Play the determinized clone with the heuristic for BOTH players until the
    horizon (end of the opponent's next turn), a terminal state, the depth cap,
    or the deadline; return the board evaluation of wherever we stopped."""
    chain = []
    cur = state.observation.current
    try:
        for _ in range(MAX_STEPS):
            obs = state.observation
            cur = obs.current
            if cur is not None and cur.result != -1:
                break
            if cur is not None and cur.turn is not None \
                    and cur.turn >= start_turn + HORIZON_TURNS:
                break
            if time.time() >= deadline:
                break
            sel = obs.select
            if sel is None:
                break
            try:
                pick = policy(obs)
            except Exception:
                pick = None
            if not pick and sel.minCount > 0:
                pick = list(range(min(sel.minCount, len(sel.option))))
            try:
                state = step(state.searchId, pick)
            except Exception:
                break
            chain.append(state.searchId)
            cur = state.observation.current
    finally:
        for sid in chain:                       # free this rollout's states
            try:
                release(sid)
            except Exception:
                pass
    return _evaluate(cur, my_index)


def decide(obs, my_deck, policy, budget_s=0.30, candidates=None,
           n_determinizations=MAX_DET):
    """Return the chosen MAIN option index, or None to defer to the heuristic."""
    sb, ss, sr, se = _imports()
    opts = obs.select.option
    n = len(opts)
    my_index = obs.current.yourIndex
    start_turn = obs.current.turn if obs.current.turn is not None else 0
    if candidates is None:
        candidates = list(range(n))
    candidates = [a for a in candidates if 0 <= a < n]
    if len(candidates) < 2:
        return None

    t0 = time.time()
    deadline = t0 + max(0.0, budget_s - SAFETY_S)
    total = {a: 0.0 for a in candidates}
    visits = {a: 0 for a in candidates}

    d = 0
    while d < n_determinizations and time.time() < deadline:
        try:
            args = _determinize(obs, my_deck)
            root = sb(obs, *args)
        except Exception:
            try:
                se()
            except Exception:
                pass
            d += 1
            continue
        # Evaluate every candidate on THIS determinization (paired / CRN).
        part = {}
        ok = True
        try:
            for a in candidates:
                if time.time() >= deadline:
                    ok = False
                    break
                try:
                    child = ss(root.searchId, [a])
                except Exception:
                    ok = False
                    break
                val = _rollout(ss, sr, child, policy, my_index,
                               start_turn, deadline)
                try:
                    sr(child.searchId)
                except Exception:
                    pass
                part[a] = val
        finally:
            try:
                se()
            except Exception:
                pass
        if ok and len(part) == len(candidates):     # keep only clean, paired runs
            for a in candidates:
                total[a] += part[a]
                visits[a] += 1
        d += 1

    # Trust region: keep the heuristic's top pick (candidates[0]) unless another
    # candidate is clearly better averaged over determinizations.  This prevents
    # search noise from dragging us below the base policy (v1's failure mode).
    scored = [(total[a] / visits[a], visits[a], a)
              for a in candidates if visits[a] > 0]
    base = candidates[0]
    base_mean = (total[base] / visits[base]) if visits[base] > 0 else None
    chosen = None
    if scored:
        best_mean, _bv, best = max(scored)
        chosen = best
        if best != base and base_mean is not None \
                and (visits[best] < MIN_VISITS or (best_mean - base_mean) < MARGIN):
            chosen = base
    # Always-on, cheap introspection for the offline probe / tuning (ignored at
    # runtime).  Records how many determinizations completed and what we picked.
    global _LAST_DECISION
    _LAST_DECISION = {
        "dets": d,
        "elapsed": time.time() - t0,
        "visits": dict(visits),
        "means": {a: (total[a] / visits[a] if visits[a] else None)
                  for a in candidates},
        "base": base,
        "chosen": chosen,
        "deviated": chosen is not None and chosen != base,
    }
    return chosen
