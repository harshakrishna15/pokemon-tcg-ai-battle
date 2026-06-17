"""Determinized search (PIMC / a tractable Information-Set MCTS) on top of the
heuristic policy.

For a MAIN decision we repeatedly:
  1. sample a *determinization* of the hidden state (my unseen deck+prizes from
     my known 60, plus a mirror-deck model of the opponent's hidden cards),
  2. clone the game at this exact decision with `search_begin`,
  3. for each candidate action, `search_step` it and roll the determinized game
     to terminal with the heuristic for BOTH players,
  4. score +1 win / +0.5 draw / 0 loss from our perspective.
Aggregated over determinizations, we play the action with the best win rate.

The engine only ever offers legal moves and is ~0.3 ms/step, so this stays cheap;
everything is time-budgeted and exception-guarded so a failure just yields None
and the caller falls back to the heuristic.
"""
import random
import time
from collections import Counter


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
    """Return (your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)."""
    st = obs.current
    yi = st.yourIndex
    me, opp = st.players[yi], st.players[1 - yi]

    # my hidden cards = my full 60 minus everything I can currently see
    rem = Counter(my_deck)
    for cid in ([c.id for c in (me.hand or [])]
                + [c.id for c in (me.discard or [])]
                + _in_play_ids(me)):
        rem[cid] -= 1
    remainder = [cid for cid, n in rem.items() for _ in range(max(0, n))]
    random.shuffle(remainder)
    pc = len(me.prize)
    your_prize = remainder[:pc]
    your_deck = remainder[pc:]
    if len(your_deck) < me.deckCount and my_deck:        # safety pad
        your_deck += [my_deck[0]] * (me.deckCount - len(your_deck))

    # opponent: coherent mirror-deck model (guarantees basics for legal rollouts)
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
        opp_active = [my_deck[0]] if my_deck else [119]  # face-down -> a basic
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


def _rollout(step, release, state, policy, max_steps=400):
    """Play the determinized clone to terminal; return winner index (0/1/2) or None."""
    chain = []
    res = None
    for _ in range(max_steps):
        obs = state.observation
        cur = obs.current
        if cur is not None and cur.result != -1:
            res = cur.result
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
    for sid in chain:                                    # free this rollout
        try:
            release(sid)
        except Exception:
            pass
    return res


def decide(obs, my_deck, policy, budget_s=0.30, candidates=None,
           n_determinizations=10):
    sb, ss, sr, se = _imports()
    opts = obs.select.option
    n = len(opts)
    my_index = obs.current.yourIndex
    if candidates is None:
        candidates = list(range(n))
    candidates = [a for a in candidates if 0 <= a < n]
    if len(candidates) < 2:
        return None

    wins = {a: 0.0 for a in candidates}
    visits = {a: 0 for a in candidates}
    t0 = time.time()
    d = 0
    while d < n_determinizations and time.time() - t0 < budget_s:
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
        try:
            for a in candidates:
                if time.time() - t0 >= budget_s and visits[a] > 0:
                    continue
                try:
                    child = ss(root.searchId, [a])
                except Exception:
                    continue
                res = _rollout(ss, sr, child, policy)
                try:
                    sr(child.searchId)
                except Exception:
                    pass
                if res is None:
                    continue
                wins[a] += 1.0 if res == my_index else (0.5 if res == 2 else 0.0)
                visits[a] += 1
        finally:
            try:
                se()
            except Exception:
                pass
        d += 1

    best, best_key = None, None
    for a in candidates:
        if visits[a] == 0:
            continue
        key = (wins[a] / visits[a], visits[a])
        if best_key is None or key > best_key:
            best_key, best = key, a
    return best
