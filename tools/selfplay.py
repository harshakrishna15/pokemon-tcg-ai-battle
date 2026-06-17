"""Local self-play harness for hardening + measuring the agent.

Runs inside a Linux x86-64 container (the engine ships only libcg.so / cg.dll):

    docker run --rm -v "$PWD":/work -w /work python:3.11-slim \
        python tools/selfplay.py --games 40

Reports win/loss/draw of our agent vs a random baseline (sides swapped for
fairness), plus a mirror match to confirm no crashes / stalls and to measure
per-decision timing.
"""
import argparse
import importlib.util
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "sample_submission")
sys.path.insert(0, SUB)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def load_agent(name, pimc=False, budget=0.30):
    """Load sample_submission/main.py as an isolated module instance."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(SUB, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.PIMC_ENABLED = pimc
    mod.PIMC_BUDGET_S = budget
    mod._MY_DECK = read_deck()      # cwd here isn't the agent dir; inject the deck
    return mod.agent


def read_deck():
    rows = [r.strip() for r in open(os.path.join(SUB, "deck.csv")) if r.strip()]
    return [int(x) for x in rows[:60]]


def sanitize(pick, sel):
    n = len(sel["option"])
    mn, mx = sel["minCount"], sel["maxCount"]
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


def random_agent(obs):
    sel = obs.get("select")
    if sel is None:
        return read_deck()
    n = len(sel["option"])
    mx = sel["maxCount"] if sel["maxCount"] > 0 else 0
    mn = sel["minCount"]
    k = random.randint(mn, mx) if mx >= mn and mx > 0 else mn
    k = min(k, n)
    return random.sample(range(n), k) if k > 0 else []


def result_from_obs(obs):
    cur = obs.get("current")
    if cur and cur.get("result", -1) != -1:
        return cur["result"]
    for lg in obs.get("logs", []):
        if lg.get("type") == 23 and lg.get("result") is not None:
            return lg["result"]
    return None


def play_game(agents, deck0, deck1, step_cap=4000):
    """agents = {0: fn, 1: fn}. Returns (result, stats)."""
    stats = {"crashes": [0, 0], "calls": [0, 0], "max_ms": [0.0, 0.0],
             "steps": 0, "reason": None, "turn": 0}
    obs, _sd = battle_start(deck0, deck1)
    if obs is None:
        return None, stats
    res = None
    for _ in range(step_cap):
        for lg in obs.get("logs", []):
            if lg.get("type") == 23 and lg.get("reason") is not None:
                stats["reason"] = lg["reason"]
        if obs.get("current"):
            stats["turn"] = obs["current"].get("turn", stats["turn"])
        res = result_from_obs(obs)
        if res is not None:
            break
        sel = obs.get("select")
        if sel is None:
            break
        yi = obs["current"]["yourIndex"]
        t0 = time.time()
        try:
            pick = agents[yi](obs)
        except Exception:
            stats["crashes"][yi] += 1
            pick = None
        dt = (time.time() - t0) * 1000.0
        stats["calls"][yi] += 1
        stats["max_ms"][yi] = max(stats["max_ms"][yi], dt)
        obs = battle_select(sanitize(pick, sel))
        stats["steps"] += 1
    try:
        battle_finish()
    except Exception:
        pass
    return res, stats


def run(label, fn_a, fn_b, deck, n):
    """fn_a is the agent under test; play n games, swapping who is player 0."""
    w = l = d = err = 0
    crashes_a = 0
    max_ms_a = 0.0
    total_steps = 0
    rsn = {"win": {}, "loss": {}}
    turns = []
    names = {1: "prizes", 2: "deckout", 3: "no-active", 4: "effect"}
    for g in range(n):
        a_side = g % 2                       # alternate our side for fairness
        agents = {a_side: fn_a, 1 - a_side: fn_b}
        res, st = play_game(agents, deck, list(deck))
        total_steps += st["steps"]
        crashes_a += st["crashes"][a_side]
        max_ms_a = max(max_ms_a, st["max_ms"][a_side])
        turns.append(st["turn"])
        rname = names.get(st["reason"], str(st["reason"]))
        if res is None:
            err += 1
        elif res == 2:
            d += 1
        elif res == a_side:
            w += 1
            rsn["win"][rname] = rsn["win"].get(rname, 0) + 1
        else:
            l += 1
            rsn["loss"][rname] = rsn["loss"].get(rname, 0) + 1
    played = max(1, w + l + d)
    print(f"[{label}] games={n}  W={w} L={l} D={d} err={err}  "
          f"winrate={w / played:.1%}  our_crashes={crashes_a}  "
          f"our_max_decision={max_ms_a:.1f}ms  avg_turn={sum(turns) / n:.1f}")
    print(f"          win_reasons={rsn['win']}  loss_reasons={rsn['loss']}")
    return w, l, d, err, crashes_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--pimc", action="store_true", help="test PIMC vs heuristic")
    ap.add_argument("--pimc-games", type=int, default=12)
    ap.add_argument("--budget", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    deck = read_deck()
    print("deck size:", len(deck))
    if args.pimc:
        pa = load_agent("pimc_a", pimc=True, budget=args.budget)
        hb = load_agent("heur_b", pimc=False)
        run(f"PIMC(b={args.budget}s) vs heuristic", pa, hb, deck, args.pimc_games)
    else:
        heur = load_agent("agent_heur")
        heur2 = load_agent("agent_heur2")
        run("heuristic vs random", heur, random_agent, deck, args.games)
        run("heuristic mirror   ", heur, heur2, deck, max(10, args.games // 2))


if __name__ == "__main__":
    main()
