"""Diagnostic harness: load the cabt engine, summarize the deck, and drive a
random mirror match to reveal observation/option shapes, per-move timing, and
which selection contexts actually occur. Run inside a Linux x86-64 container:

    docker run --rm -v "$PWD":/work -w /work python:3.11-slim python tools/inspect_engine.py
"""
import sys, os, json, time, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "sample_submission")
sys.path.insert(0, SUB)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import all_card_data, all_attack


def read_deck(path):
    lines = [l.strip() for l in open(path) if l.strip()]
    return [int(x) for x in lines[:60]]


deck = read_deck(os.path.join(SUB, "deck.csv"))
print("DECK size:", len(deck))

cards = all_card_data()
attacks = all_attack()
print("engine cards:", len(cards), "attacks:", len(attacks))
by_id = {c.cardId: c for c in cards}
atk_by_id = {a.attackId: a for a in attacks}

print("\n--- deck card summary ---")
seen = set()
for cid in deck:
    if cid in seen:
        continue
    seen.add(cid)
    c = by_id.get(cid)
    if not c:
        print(cid, "NOT FOUND IN ENGINE")
        continue
    atk_names = [atk_by_id[a].name for a in c.attacks if a in atk_by_id]
    print(f"{cid:>5}  {c.name:<28} type={c.cardType} hp={c.hp} "
          f"basic={c.basic} s1={c.stage1} s2={c.stage2} ex={c.ex} "
          f"atks={atk_names}")

print("\n=== start mirror battle ===")
obs, sd = battle_start(deck, list(deck))
print("errorPlayer:", sd.errorPlayer, "errorType:", sd.errorType)
if obs is None:
    print("battle_start returned None obs -> deck likely illegal for engine")
    sys.exit(0)
print("obs keys:", list(obs.keys()))

random.seed(1)
t0 = time.time()
steps = 0
ctx_hist = {}
max_opts = 0
slowest = 0.0
try:
    while obs is not None and steps < 8000:
        cur = obs.get("current")
        if cur and cur.get("result", -1) != -1:
            print("RESULT result-field:", cur.get("result"))
            break
        sel = obs.get("select")
        if sel is None:
            print(f"select=None at step {steps} (current present: {cur is not None})")
            break
        opts = sel["option"]
        mn, mx = sel["minCount"], sel["maxCount"]
        ctx = sel["context"]
        ctx_hist[ctx] = ctx_hist.get(ctx, 0) + 1
        max_opts = max(max_opts, len(opts))
        if steps < 10:
            yi = cur.get("yourIndex") if cur else None
            print(f"\n[{steps}] selType={sel['type']} ctx={ctx} min={mn} max={mx} "
                  f"nopt={len(opts)} yourIndex={yi} turn={cur.get('turn') if cur else None}")
            for o in opts[:5]:
                print("     opt:", json.dumps(o))
        if len(opts) == 0:
            pick = []
        else:
            k = min(mx, len(opts)) if mx > 0 else 0
            pick = random.sample(range(len(opts)), k) if k > 0 else []
            if len(pick) < mn:
                pick = list(range(min(mn, len(opts))))
        s = time.time()
        obs = battle_select(pick)
        slowest = max(slowest, time.time() - s)
        steps += 1
except Exception:
    import traceback
    traceback.print_exc()

print("\n=== summary ===")
print("steps:", steps, "wall:", round(time.time() - t0, 3), "s")
print("max options seen:", max_opts, "slowest battle_select:", round(slowest, 4), "s")
print("context histogram (SelectContext -> count):",
      dict(sorted(ctx_hist.items(), key=lambda kv: -kv[1])))
try:
    battle_finish()
except Exception as e:
    print("finish err:", e)