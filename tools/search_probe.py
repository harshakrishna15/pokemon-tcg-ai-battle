"""Probe the cabt forward-search API to de-risk the PIMC / IS-MCTS layer.

Drives a real game to a mid-game MAIN decision, builds a determinization of the
hidden information, then exercises search_begin / search_step / search_release:
  - does the cloned root expose the same options as the real obs?
  - can we branch (try several root actions)?
  - can we roll a determinized game to terminal with the heuristic, and what
    result does it report?  how fast?
"""
import importlib.util
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "sample_submission")
sys.path.insert(0, SUB)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg.api import (to_observation_class, search_begin, search_step,  # noqa: E402
                    search_release, search_end)


def load_main():
    spec = importlib.util.spec_from_file_location("m", os.path.join(SUB, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = load_main()


def read_deck():
    rows = [r.strip() for r in open(os.path.join(SUB, "deck.csv")) if r.strip()]
    return [int(x) for x in rows[:60]]


def san(pick, sel):
    n = len(sel["option"]); mn, mx = sel["minCount"], sel["maxCount"]
    out, seen = [], set()
    for x in (pick or []):
        if isinstance(x, int) and 0 <= x < n and x not in seen:
            out.append(x); seen.add(x)
    if mx > 0:
        out = out[:mx]
    for i in range(n):
        if len(out) >= mn:
            break
        if i not in seen:
            out.append(i); seen.add(i)
    return out


def in_play_ids(ps):
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


def build_determinization(obs, my_deck):
    st = obs.current
    yi = st.yourIndex
    me, opp = st.players[yi], st.players[1 - yi]
    # my hidden cards = full deck multiset minus everything I can see
    rem = Counter(my_deck)
    for cid in [c.id for c in (me.hand or [])] + \
               [c.id for c in (me.discard or [])] + in_play_ids(me):
        rem[cid] -= 1
    remainder = []
    for cid, n in rem.items():
        remainder += [cid] * max(0, n)
    random.shuffle(remainder)
    pc = len(me.prize)
    your_prize = remainder[:pc]
    your_deck = remainder[pc:]
    # opponent: mirror-deck model (coherent, guarantees basics for rollouts)
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
        opp_active = [119]  # face-down -> predict a basic (Dreepy)
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


def rollout(state, my_index, max_steps=400):
    """Play the determinized clone to terminal with the heuristic for both sides."""
    steps = 0
    while steps < max_steps:
        obs = state.observation
        cur = obs.current
        if cur is not None and cur.result != -1:
            return cur.result, steps
        sel = obs.select
        if sel is None:
            return None, steps
        obs_dict = _to_dict(obs)
        try:
            pick = M.agent(obs_dict)
        except Exception:
            pick = None
        pick = san(pick, obs_dict["select"])
        state = search_step(state.searchId, pick)
        steps += 1
    return None, steps


def _to_dict(obs):
    """search_step gives an Observation dataclass; M.agent wants a dict-ish obs.
    Re-serialize minimally via the same path the engine uses is overkill, so we
    just hand the dataclass through a light shim that supports obs['select'] etc.
    """
    # Simplest correct approach: the heuristic only needs select + current; build
    # a dict view.  But dataclasses are not subscriptable -> use a tiny adapter.
    return obs  # placeholder; replaced after we see what search returns


def main():
    random.seed(7)
    deck = read_deck()
    obs, _ = battle_start(deck, list(deck))
    # advance to our ~3rd MAIN decision with the heuristic
    main_seen = 0
    target_obs_dict = None
    for _ in range(4000):
        cur = obs.get("current")
        if cur and cur.get("result", -1) != -1:
            break
        sel = obs.get("select")
        if sel is None:
            break
        yi = cur["yourIndex"]
        if yi == 0 and sel["context"] == 0:
            main_seen += 1
            if main_seen == 4:
                target_obs_dict = obs
                break
        obs = battle_select(san(M.agent(obs) if yi == 0 else _rand(sel), sel))
    if target_obs_dict is None:
        print("could not reach target MAIN obs")
        return
    sel = target_obs_dict["select"]
    print("real MAIN obs: options =", len(sel["option"]),
          "ctx =", sel["context"], "has search_begin_input =",
          target_obs_dict.get("search_begin_input") is not None)

    o = to_observation_class(target_obs_dict)
    yd, yp, od, op, oh, oa = build_determinization(o, deck)
    print(f"determinization: your_deck={len(yd)} your_prize={len(yp)} "
          f"opp_deck={len(od)} opp_hand={len(oh)} opp_prize={len(op)} opp_active={oa}")
    try:
        t0 = time.time()
        root = search_begin(o, yd, yp, od, op, oh, oa)
        dt = (time.time() - t0) * 1000
        print(f"search_begin OK in {dt:.1f}ms  searchId={root.searchId}")
        robs = root.observation
        print("root select options =",
              len(robs.select.option) if robs.select else None,
              "ctx =", robs.select.context if robs.select else None)
        # branch: step two different root actions
        nopt = len(robs.select.option)
        for a in (0, min(1, nopt - 1)):
            ch = search_step(root.searchId, [a])
            print(f"  step root with [{a}] -> searchId={ch.searchId} "
                  f"nextCtx={ch.observation.select.context if ch.observation.select else 'terminal'}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        try:
            search_end()
        except Exception:
            pass
    battle_finish()


def _rand(sel):
    n = len(sel["option"]); mn = sel["minCount"]; mx = sel["maxCount"]
    k = random.randint(mn, mx) if mx >= mn and mx > 0 else mn
    return random.sample(range(n), min(k, n)) if k > 0 else []


if __name__ == "__main__":
    main()
