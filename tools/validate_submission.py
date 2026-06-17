"""Validate a built submission the way Kaggle's validation episode does:
extract-only files, agent plays mirror games against itself, must not crash.

    python tools/validate_submission.py /path/to/extracted_bundle
"""
import os
import sys

bundle = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, bundle)
os.chdir(bundle)                      # so deck.csv + cg load like on Kaggle

import main                            # noqa: E402  (from the bundle only)
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


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


deck = main.read_deck_csv()
print("bundle:", bundle)
print("deck cards:", len(deck))
crashes = 0
games = 3
for g in range(games):
    obs, _ = battle_start(deck, list(deck))
    res = None
    for _ in range(5000):
        cur = obs.get("current")
        if cur and cur.get("result", -1) != -1:
            res = cur["result"]
            break
        sel = obs.get("select")
        if sel is None:
            break
        try:
            pick = main.agent(obs)
        except Exception as e:
            crashes += 1
            print("  AGENT CRASH:", repr(e))
            pick = None
        obs = battle_select(san(pick, sel))
    battle_finish()
    print(f"  game {g}: result={res}")
print("crashes:", crashes, "->", "PASS" if crashes == 0 else "FAIL")
sys.exit(1 if crashes else 0)
