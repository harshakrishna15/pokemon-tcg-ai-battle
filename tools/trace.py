"""Trace one heuristic(P0) vs random(P1) game: print our MAIN decisions + state
so we can see whether the Dragapult line comes online and attacks."""
import importlib.util
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "sample_submission")
sys.path.insert(0, SUB)
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg.api import all_card_data  # noqa: E402

NAME = {c.cardId: c.name for c in all_card_data()}


def load_agent():
    spec = importlib.util.spec_from_file_location("ag", os.path.join(SUB, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.agent


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


def rand_agent(obs):
    sel = obs["select"]
    n = len(sel["option"]); mn = sel["minCount"]; mx = sel["maxCount"]
    k = random.randint(mn, mx) if mx >= mn and mx > 0 else mn
    return random.sample(range(n), min(k, n)) if k > 0 else []


def pk_str(pk):
    if not pk:
        return "-"
    return f"{NAME.get(pk['id'], pk['id'])}(hp{pk['hp']}/{pk['maxHp']},e{len(pk.get('energies', []))})"


def main():
    random.seed(3)
    heur = load_agent()
    deck = read_deck()
    obs, _ = battle_start(deck, list(deck))
    OPT = {7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 12: "RETREAT",
           13: "ATTACK", 14: "END"}
    shown = 0
    for _ in range(4000):
        cur = obs.get("current")
        if cur and cur.get("result", -1) != -1:
            print("RESULT", cur["result"])
            break
        sel = obs.get("select")
        if sel is None:
            break
        yi = cur["yourIndex"]
        if yi == 0:
            pick = heur(obs)
        else:
            pick = rand_agent(obs)
        if yi == 0 and sel["context"] == 0 and shown < 70:   # our MAIN decisions
            me = cur["players"][0]; op = cur["players"][1]
            act = me["active"][0] if me["active"] else None
            oact = op["active"][0] if op["active"] else None
            bench = ",".join(NAME.get(b["id"], b["id"])[:5] for b in me["bench"])
            chosen = pick[0] if pick else None
            o = sel["option"][chosen] if chosen is not None else {}
            tag = OPT.get(o.get("type"), o.get("type"))
            print(f"T{cur['turn']:>2} deck{me['deckCount']:>2} h{me['handCount']} "
                  f"pz{len(me['prize'])}/{len(op['prize'])} | A:{pk_str(act)} "
                  f"B[{bench}] | opp:{pk_str(oact)} | -> {tag} ({len(sel['option'])}opt)")
            shown += 1
        obs = battle_select(san(pick, sel))
    # reason
    reason = None
    for lg in obs.get("logs", []):
        if lg.get("type") == 23:
            reason = lg.get("reason")
    print("reason:", reason)
    battle_finish()


if __name__ == "__main__":
    main()
