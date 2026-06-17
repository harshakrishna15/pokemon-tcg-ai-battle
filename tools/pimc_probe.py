"""Diagnose PIMC v2: at real mid-game MAIN decisions, how many determinizations
complete per budget, how often does search deviate from the heuristic, and what
do the per-candidate evals look like?  Answers "am I sample-starved?".

    docker run --rm -v "$PWD":/work -w /work python:3.11-slim \
        python tools/pimc_probe.py
"""
import importlib.util
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "sample_submission")
sys.path.insert(0, SUB)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SUB, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = load("m_main", "main.py")
P = load("m_pimc", "pimc.py")


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


def main():
    budgets = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                                  else ["0.15", "0.25", "0.50", "1.0"])]
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    random.seed(0)
    deck = read_deck()
    M._MY_DECK = deck
    M.PIMC_ENABLED = False        # advance the game with the pure heuristic
    M._load_cards()

    agg = {b: {"dets": [], "vis": [], "dev": 0, "ms": []} for b in budgets}
    samples = 0
    games = 0
    while samples < n_samples and games < 30:
        obs, _ = battle_start(deck, list(deck))
        games += 1
        for _ in range(4000):
            cur = obs.get("current")
            if cur and cur.get("result", -1) != -1:
                break
            sel = obs.get("select")
            if sel is None:
                break
            yi = cur["yourIndex"]
            if yi == 0 and sel["context"] == 0 and len(sel["option"]) >= 2:
                o = M.to_observation_class(obs)
                cand = M._pimc_candidates(o)
                if len(cand) >= 2:
                    for b in budgets:
                        P.decide(o, deck, M.choose, budget_s=b, candidates=cand)
                        ld = P._LAST_DECISION
                        agg[b]["dets"].append(ld["dets"])
                        vis = [v for v in ld["visits"].values()]
                        agg[b]["vis"].append(min(vis) if vis else 0)
                        agg[b]["dev"] += 1 if ld["deviated"] else 0
                        agg[b]["ms"].append(ld["elapsed"] * 1000.0)
                    samples += 1
                    if samples >= n_samples:
                        break
            obs = battle_select(san(M.agent(obs), sel))
        try:
            battle_finish()
        except Exception:
            pass

    print(f"samples (MAIN decisions w/ >=2 candidates) = {samples}  over {games} games")
    print(f"{'budget':>8} {'avg_dets':>9} {'min_visits/cand':>16} {'deviate%':>9} "
          f"{'avg_ms':>8} {'max_ms':>8}")
    for b in budgets:
        a = agg[b]
        n = max(1, len(a["dets"]))
        print(f"{b:>8.2f} {sum(a['dets'])/n:>9.1f} {sum(a['vis'])/n:>16.1f} "
              f"{100.0*a['dev']/n:>8.0f}% {sum(a['ms'])/n:>8.1f} {max(a['ms']):>8.1f}")


if __name__ == "__main__":
    main()
