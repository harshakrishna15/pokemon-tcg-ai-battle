"""Pokémon TCG AI Battle agent (Dragapult ex / Dusknoir control).

Design notes
------------
* The engine calls ``agent(obs)`` for *every* micro-decision and only ever
  presents legal options, so the job is to (a) choose well among the offered
  options and (b) NEVER crash and NEVER stall (the Kaggle validation episode
  rejects agents that error or time out).
* Everything is wrapped so that any internal failure degrades to a guaranteed
  legal fallback move.  No network / external state is used (competition
  rule 2.12: no ingress/egress).
* Stage 1 (this file) is a hand-crafted heuristic policy.  It is also the
  rollout / leaf policy and safe fallback for the determinized search
  (PIMC / IS-MCTS) layer added on top later.
"""

import os
import random
import time

try:
    from cg.api import to_observation_class, all_card_data, all_attack
    _HAVE_CG = True
except Exception:  # pragma: no cover - cg should always be present at runtime
    _HAVE_CG = False

# ---------------------------------------------------------------- card ids ---
DREEPY, DRAKLOAK, DRAGAPULT = 119, 120, 121
DUSKULL, DUSCLOPS, DUSKNOIR = 131, 132, 133
BUDEW, FEZ, MEOWTH, MUNKI = 235, 140, 1071, 112

LILLIE, CRISPIN, BOSS, DAWN = 1227, 1198, 1182, 1231
ULTRA, POKEPAD, POFFIN, HAMMER, STRETCHER = 1121, 1152, 1086, 1120, 1097
STAMP, JUDGE, FAN, WATCHTOWER, JAMMING = 1080, 1213, 1161, 1256, 1246

# EnergyType ints: COLORLESS0 GRASS1 FIRE2 WATER3 LIGHTNING4 PSYCHIC5
#                  FIGHTING6 DARK7 METAL8
# CardType ints:   POKEMON0 ITEM1 TOOL2 SUPPORTER3 STADIUM4 BASIC_E5 SPECIAL_E6
# AreaType ints:   DECK1 HAND2 DISCARD3 ACTIVE4 BENCH5 PRIZE6 ... LOOKING12

# How good a Pokémon is as the thing we pour energy into / promote / protect.
ATTACKER_PRIO = {DRAGAPULT: 10, FEZ: 7, DUSKNOIR: 6, MUNKI: 5,
                 DRAKLOAK: 4, DREEPY: 3, MEOWTH: 2, DUSCLOPS: 2, DUSKULL: 1}

# Abilities that are pure value and engine-limited to once/turn -> safe to fire.
# (Drakloak Recon Directive, Fezandipiti Flip the Script.)  Everything else --
# notably Dusclops/Dusknoir "Cursed Blast" self-KO -- is skipped in Stage 1.
ABILITY_WHITELIST = {DRAKLOAK, FEZ}

TIME_BUDGET = 2.5  # soft per-decision cap (heuristic is far under this)

_CARD = {}
_ATK = {}


def _load_cards():
    if _CARD or not _HAVE_CG:
        return
    try:
        for c in all_card_data():
            _CARD[c.cardId] = c
        for a in all_attack():
            _ATK[a.attackId] = a
    except Exception:
        pass


def _cd(cid):
    return _CARD.get(cid)


# ------------------------------------------------------------------ deck io ---
def read_deck_csv():
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path, "r") as f:
        rows = f.read().split("\n")
    return [int(rows[i]) for i in range(60)]


# ------------------------------------------------------------- obs accessors ---
def _me(obs):
    st = obs.current
    return st.players[st.yourIndex]


def _opp(obs):
    st = obs.current
    return st.players[1 - st.yourIndex]


def _active(ps):
    if ps and ps.active and ps.active[0] is not None:
        return ps.active[0]
    return None


def _inplay_ids(ps):
    ids = set()
    a = _active(ps)
    if a:
        ids.add(a.id)
    for b in (ps.bench or []):
        if b:
            ids.add(b.id)
    return ids


def _opt_pokemon(obs, o):
    """Resolve the in-play Pokémon an option refers to (via area/index/player)."""
    st = obs.current
    pi = o.playerIndex if o.playerIndex is not None else st.yourIndex
    try:
        ps = st.players[pi]
    except Exception:
        return None
    if o.inPlayArea is not None:
        area, idx = o.inPlayArea, o.inPlayIndex
    else:
        area, idx = o.area, o.index
    if area == 4:
        return _active(ps)
    if area == 5 and idx is not None and 0 <= idx < len(ps.bench):
        return ps.bench[idx]
    return None


def _resolve_card(obs, o):
    """Best-effort cardId an option references (deck search / looking / field)."""
    if getattr(o, "cardId", None):
        return o.cardId
    sel = obs.select
    st = obs.current
    me = _me(obs)
    area, idx = o.area, o.index
    try:
        if area == 1 and sel.deck and idx is not None and idx < len(sel.deck):
            return sel.deck[idx].id
        if area == 12 and st.looking and idx is not None and idx < len(st.looking) \
                and st.looking[idx] is not None:
            return st.looking[idx].id
        if area == 2 and me.hand and idx is not None and idx < len(me.hand):
            return me.hand[idx].id
        if area == 3 and idx is not None and idx < len(me.discard):
            return me.discard[idx].id
        if area == 4:
            a = _active(me)
            return a.id if a else None
        if area == 5 and idx is not None and 0 <= idx < len(me.bench):
            return me.bench[idx].id
    except Exception:
        pass
    return None


def _energy_needs(cid):
    """Non-colorless energy types this Pokémon's attacks want."""
    c = _cd(cid)
    need = set()
    if not c:
        return need
    for aid in c.attacks:
        a = _ATK.get(aid)
        if a:
            for e in a.energies:
                if int(e) != 0:
                    need.add(int(e))
    return need


def _card_value(cid):
    """Rough 'how much I want to keep this' (used for discards, generic picks)."""
    c = _cd(cid)
    if not c:
        return 1.5
    ct = int(c.cardType)
    if ct == 0:
        if c.ex or getattr(c, "megaEx", False):
            return 6.0
        if c.stage2:
            return 5.0
        if c.stage1:
            return 4.0
        return 3.0
    if ct == 3:
        return 2.6          # supporter
    if ct == 6:
        return 2.4          # special energy
    if ct == 4:
        return 2.2          # stadium
    if ct == 5:
        return 2.0          # basic energy (scarce here -- keep above items)
    if ct == 2:
        return 1.7          # tool
    return 1.6              # item


def _best_damage(obs):
    """Approx best damage our active can currently deal (affordable attacks)."""
    a = _active(_me(obs))
    if not a:
        return 0
    c = _cd(a.id)
    if not c:
        return 0
    best = 0
    have = len(a.energies or [])
    for aid in c.attacks:
        atk = _ATK.get(aid)
        if atk and len(atk.energies) <= have and atk.damage > best:
            best = atk.damage
    return best


# ----------------------------------------------------------------- MAIN turn ---
def _evolve_score(obs, o):
    c = _cd(_resolve_card(obs, o))
    if c and c.stage2:
        return 110.0
    if c and c.stage1:
        return 104.0
    return 101.0


def _attach_score(obs, o):
    tgt = _opt_pokemon(obs, o)
    tcid = tgt.id if tgt else None
    prio = ATTACKER_PRIO.get(tcid, 1)
    ecid = _resolve_card(obs, o)
    ec = _cd(ecid)
    bonus = 0.0
    if ec and int(ec.cardType) in (5, 6):
        need = _energy_needs(tcid)
        if need and int(ec.energyType) in need:
            bonus = 4.0
    # Feed the ACTIVE Pokémon first so it can attack or pay to retreat -- this
    # is what stops us stranding energy on a benched attacker while the active
    # starves and we durdle into a deck-out.
    is_active = (o.inPlayArea == 4)
    return 78.0 + prio + bonus + (12.0 if is_active else 0.0)


def _in_hand(obs, cid):
    me = _me(obs)
    return bool(me.hand) and any(h.id == cid for h in me.hand)


def _best_inplay_attacker(obs):
    me = _me(obs)
    best, bp = None, -1
    for pk in [_active(me)] + list(me.bench or []):
        if pk and ATTACKER_PRIO.get(pk.id, 0) > bp:
            best, bp = pk, ATTACKER_PRIO.get(pk.id, 0)
    return best


def _have_attacker(obs):
    """Is a real attacker online or one evolution step away in hand?"""
    ids = _inplay_ids(_me(obs))
    if ids & {DRAGAPULT, FEZ, DUSKNOIR}:
        return True
    if DRAKLOAK in ids and _in_hand(obs, DRAGAPULT):
        return True
    if DUSCLOPS in ids and _in_hand(obs, DUSKNOIR):
        return True
    return False


def _attacker_wants_energy(obs):
    pk = _best_inplay_attacker(obs)
    if not pk:
        return True
    need = 1
    c = _cd(pk.id)
    if c:
        for aid in c.attacks:
            a = _ATK.get(aid)
            if a:
                need = max(need, len(a.energies))
    return len(pk.energies or []) < need


def _need_small_basics(obs):
    me = _me(obs)
    ids = [pk.id for pk in [_active(me)] + list(me.bench or []) if pk]
    return sum(1 for i in ids if i in (DREEPY, DUSKULL, BUDEW)) < 2


def _stretcher_score(obs):
    for c in (_me(obs).discard or []):
        if c.id in (DRAGAPULT, DRAKLOAK, DUSKNOIR, DUSCLOPS, DREEPY, DUSKULL,
                    FEZ, MUNKI):
            return 52.0
    return 16.0


def _play_score(obs, o):
    """Deck-count- and need-aware.  A control deck must not mill itself out:
    assemble the line early, then stop thinning and just attach + attack."""
    me = _me(obs)
    if not me.hand or o.index is None or not (0 <= o.index < len(me.hand)):
        return 30.0
    cid = me.hand[o.index].id
    c = _cd(cid)
    if not c:
        return 30.0
    ct = int(c.cardType)
    deck = me.deckCount or 0
    hand_n = me.handCount if me.handCount is not None else len(me.hand)
    bench_room = (me.benchMax or 5) - len(me.bench or [])
    have_atk = _have_attacker(obs)

    if ct == 0:                                   # play a basic to the bench
        if bench_room <= 0:
            return 5.0
        if cid in (DREEPY, DUSKULL):
            return 72.0
        if cid in (BUDEW, MUNKI, FEZ, MEOWTH):
            return 66.0
        return 60.0
    if ct == 3:                                   # supporter (1/turn)
        if cid == BOSS:
            return _boss_score(obs)
        if cid == CRISPIN:                        # energy accel for Dragapult
            return 75.0 if (_attacker_wants_energy(obs) and deck > 4) else 38.0
        if cid == DAWN:                           # fetch a whole evolution line
            return 80.0 if (not have_atk and deck > 8) else 12.0
        if cid == LILLIE:                         # hand refresh (draw 6/8)
            if deck < 6:
                return 10.0
            return 72.0 if hand_n <= 3 else 20.0
        if cid == JUDGE:                          # disrupt + refuel small hand
            return 55.0 if (hand_n <= 2 and deck > 6) else 10.0
        return 50.0
    if ct == 1:                                   # item
        if cid == POFFIN:
            return 76.0 if (bench_room > 0 and deck > 6
                            and _need_small_basics(obs)) else 12.0
        if cid == ULTRA:                          # -3 cards: only when needed
            return 64.0 if (not have_atk and deck > 8 and hand_n >= 3) else 14.0
        if cid == POKEPAD:                        # -1 card: cheap, fetch a piece
            return 58.0 if (not have_atk and deck > 3) else 20.0
        if cid == STRETCHER:
            return _stretcher_score(obs)
        if cid == HAMMER:                         # disrupt opp energy
            return 44.0
        if cid == STAMP:                          # ACE SPEC (offered only if legal)
            return 33.0
        return 40.0
    if ct == 4:                                   # stadium
        return 40.0 if obs.current.stadium else 22.0
    if ct == 2:                                   # tool -> onto attacker
        return 43.0
    return 30.0


def _boss_score(obs):
    """Gust only if it sets up a KO on a benched target we can reach."""
    opp = _opp(obs)
    dmg = _best_damage(obs)
    for b in (opp.bench or []):
        if b and 0 < b.hp <= dmg:
            return 88.0
    return -1e9


def _retreat_score(obs):
    me = _me(obs)
    a = _active(me)
    if not a:
        return -1e9
    if a.id in (DREEPY, DUSKULL, BUDEW):
        for b in (me.bench or []):
            if b and ATTACKER_PRIO.get(b.id, 0) >= 5 and (b.energies or []):
                return 49.0
    return -1e9


def _attack_score(obs, o):
    atk = _ATK.get(o.attackId)
    if not atk:
        return 28.0
    opp = _opp(obs)
    oa = _active(opp)
    ma = _active(_me(obs))
    eff = float(atk.damage)
    if oa and ma:
        oc, mc = _cd(oa.id), _cd(ma.id)
        if oc and mc and oc.weakness is not None and mc.energyType is not None \
                and int(oc.weakness) == int(mc.energyType):
            eff *= 2.0
    ko = oa is not None and 0 < oa.hp <= eff
    base = 55.0 if ko else 30.0
    return base + min(eff, 400.0) / 100.0


def _main_scores(obs):
    """Score every MAIN option; returns [(score, index)] sorted best-first."""
    sel = obs.select
    opts = sel.option
    # Can the active deal damage right now?  If not, and a real attacker waits
    # on the bench, retreating to promote it beats sitting and decking out.
    has_dmg_attack = any(
        int(o.type) == 13 and _ATK.get(o.attackId) and _ATK[o.attackId].damage > 0
        for o in opts)
    me = _me(obs)
    a = _active(me)
    benched_better = False
    if a:
        ap = ATTACKER_PRIO.get(a.id, 0)
        for b in (me.bench or []):
            if b and ATTACKER_PRIO.get(b.id, 0) >= 5 \
                    and ATTACKER_PRIO.get(b.id, 0) > ap:
                benched_better = True
                break
    scored = []
    for i, o in enumerate(opts):
        t = int(o.type)
        if t == 9:
            s = _evolve_score(obs, o)
        elif t == 10:
            s = 90.0 if _resolve_card(obs, o) in ABILITY_WHITELIST else -1e9
        elif t == 8:
            s = _attach_score(obs, o)
        elif t == 7:
            s = _play_score(obs, o)
        elif t == 12:                              # RETREAT (offered => affordable)
            s = 100.0 if (not has_dmg_attack and benched_better) else _retreat_score(obs)
        elif t == 13:
            s = _attack_score(obs, o)
        elif t == 14:
            s = 0.0
        else:
            s = -1e8
        scored.append((s, i))
    scored.sort(key=lambda x: -x[0])
    return scored


def _main_heuristic(obs):
    opts = obs.select.option
    st = obs.current
    if st.turnActionCount is not None and st.turnActionCount > 55:
        e = _find_type(opts, 14)
        if e is not None:
            return [e]
    scored = _main_scores(obs)
    if not scored or scored[0][0] <= -1e17:
        e = _find_type(opts, 14)
        return [e if e is not None else 0]
    return [scored[0][1]]


def _pimc_candidates(obs, k=3):
    """Top-k plausible MAIN actions for the search layer to compare."""
    return [i for s, i in _main_scores(obs) if s > -1e8][:k]


# --------------------------------------------------------- other contexts ----
def _find_type(opts, t):
    for i, o in enumerate(opts):
        if int(o.type) == t:
            return i
    return None


def _topk(scored, k):
    scored.sort(reverse=True)
    return [i for _, i in scored[:max(0, k)]]


def _setup_active(obs, opts):
    pref = {FEZ: 90, MEOWTH: 82, BUDEW: 70, MUNKI: 64, DREEPY: 30, DUSKULL: 28}
    scored = [(pref.get(_resolve_card(obs, o), 50), i) for i, o in enumerate(opts)]
    return [_topk(scored, 1)[0]] if scored else [0]


def _bench_prio(cid):
    pref = {DREEPY: 90, DUSKULL: 88, BUDEW: 80, MUNKI: 76, FEZ: 74, MEOWTH: 60}
    return pref.get(cid, 50)


def _setup_bench(obs, opts, sel):
    scored = [(_bench_prio(_resolve_card(obs, o)), i) for i, o in enumerate(opts)]
    chosen = _topk(scored, sel.maxCount)
    if len(chosen) < sel.minCount:
        chosen = _topk(scored, sel.minCount)
    return chosen


def _promote(obs, opts, sel):
    scored = []
    for i, o in enumerate(opts):
        pk = _opt_pokemon(obs, o) or None
        cid = _resolve_card(obs, o)
        s = ATTACKER_PRIO.get(cid, 1) * 2.0
        if pk and (pk.energies or []):
            s += 10.0
        if pk:
            s += pk.hp / 100.0
        scored.append((s, i))
    n = max(1, sel.minCount)
    return _topk(scored, n)


def _want_score(obs, cid):
    if cid is None:
        return 0.0
    c = _cd(cid)
    if not c:
        return 0.0
    have = _inplay_ids(_me(obs))
    if cid == DRAGAPULT and (DRAKLOAK in have or DREEPY in have):
        return 100.0
    if cid == DRAKLOAK and DREEPY in have:
        return 95.0
    if cid == DUSKNOIR and DUSCLOPS in have:
        return 86.0
    if cid == DUSCLOPS and DUSKULL in have:
        return 82.0
    if cid == DREEPY:
        return 70.0
    if cid == DUSKULL:
        return 60.0
    if cid in (FEZ, MUNKI, BUDEW, MEOWTH):
        return 56.0
    ct = int(c.cardType)
    if ct == 0:
        return 50.0
    if ct == 3:
        return 46.0
    if cid in (POFFIN, ULTRA):
        return 44.0
    if ct in (5, 6):
        return 38.0
    return 33.0


def _to_hand(obs, opts, sel):
    scored = [(_want_score(obs, _resolve_card(obs, o)), i) for i, o in enumerate(opts)]
    k = sel.maxCount if sel.maxCount > 0 else sel.minCount
    chosen = _topk(scored, k)
    if len(chosen) < sel.minCount:
        chosen = _topk(scored, sel.minCount)
    return chosen


def _discard(obs, opts, sel):
    # discard the *least* valuable cards, exactly as many as required
    scored = [(-_card_value(_resolve_card(obs, o)), i) for i, o in enumerate(opts)]
    k = sel.minCount
    if k <= 0:
        return []
    return _topk(scored, k)


def _damage(obs, opts, sel):
    # concentrate counters on the most KO-able / valuable opponent Pokémon
    scored = []
    for i, o in enumerate(opts):
        pk = _opt_pokemon(obs, o)
        hp = pk.hp if pk else 999
        cid = pk.id if pk else None
        scored.append(((1000 - hp) + ATTACKER_PRIO.get(cid, 1), i))
    n = sel.minCount if sel.minCount > 0 else 1
    n = min(n, sel.maxCount) if sel.maxCount > 0 else n
    return _topk(scored, max(1, n))


def _heal(obs, opts, sel):
    scored = []
    for i, o in enumerate(opts):
        pk = _opt_pokemon(obs, o)
        miss = (pk.maxHp - pk.hp) if pk else 0
        cid = pk.id if pk else None
        scored.append((miss + ATTACKER_PRIO.get(cid, 1), i))
    n = max(1, sel.minCount)
    return _topk(scored, n)


def _yesno(obs, opts, sel, ctx):
    yes = _find_type(opts, 1)
    no = _find_type(opts, 2)
    want_yes = True
    if ctx == 42:          # MULLIGAN -> keep our hand
        want_yes = False
    if want_yes and yes is not None:
        return [yes]
    if no is not None:
        return [no]
    return [0]


def _count(obs, opts, sel, ctx):
    nums = [(o.number if o.number is not None else 0, i) for i, o in enumerate(opts)]
    nums.sort()
    return [nums[-1][1]] if nums else [0]


def _generic(obs, opts, sel):
    if sel.minCount <= 0:
        return []
    scored = [(_card_value(_resolve_card(obs, o)), i) for i, o in enumerate(opts)]
    return _topk(scored, sel.minCount)


# ----------------------------------------------------------------- dispatch ---
def _decide(obs):
    sel = obs.select
    opts = sel.option
    if not opts:
        return []
    ctx = int(sel.context)
    st = int(sel.type)
    if ctx == 0:
        return _main_heuristic(obs)
    if ctx == 1:
        return _setup_active(obs, opts)
    if ctx == 2:
        return _setup_bench(obs, opts, sel)
    if ctx in (3, 4, 5, 6):
        return _promote(obs, opts, sel)
    if ctx == 7:
        return _to_hand(obs, opts, sel)
    if ctx in (8, 9, 26, 27, 29, 30, 32):
        return _discard(obs, opts, sel)
    if ctx in (13, 14, 15):
        return _damage(obs, opts, sel)
    if ctx in (16, 17):
        return _heal(obs, opts, sel)
    if st == 9:
        return _yesno(obs, opts, sel, ctx)
    if st == 8:
        return _count(obs, opts, sel, ctx)
    return _generic(obs, opts, sel)


# ------------------------------------------------------------------ fallback ---
def _sanitize(pick, sel):
    n = len(sel.option)
    mn, mx = sel.minCount, sel.maxCount
    out, seen = [], set()
    for x in (pick or []):
        if isinstance(x, int) and 0 <= x < n and x not in seen:
            out.append(x)
            seen.add(x)
    if mx > 0:
        out = out[:mx]
    if len(out) < mn:
        for i in range(n):
            if i not in seen:
                out.append(i)
                seen.add(i)
                if len(out) >= mn:
                    break
    return out


def _fallback(sel_dict):
    opts = sel_dict.get("option", [])
    n = len(opts)
    mn = sel_dict.get("minCount", 0)
    mx = sel_dict.get("maxCount", 0)
    for i, o in enumerate(opts):           # pass turn if we can
        if o.get("type") == 14 and mn <= 1 <= max(mx, 1):
            return [i]
    if mx == 0:
        return []
    k = mn if mn > 0 else 1
    k = min(k, n, mx if mx > 0 else n)
    return list(range(k))


# --------------------------------------------------------------------- entry ---
# PIMC (determinized search) is ON by default: it beats the pure heuristic
# ~55.7% over 560 self-play games (p~0.004) and is fully fallback-guarded, so
# the worst case degrades to the heuristic.  Set PIMC=0 to force the pure
# heuristic.  Budget is a per-MAIN-decision cap, far under the ~10-min/match
# time bank (~20s/match in practice); PIMC_BUDGET overrides it.
PIMC_ENABLED = os.environ.get("PIMC", "1") == "1"
PIMC_BUDGET_S = float(os.environ.get("PIMC_BUDGET", "0.50"))
_MY_DECK = None


def _my_deck():
    global _MY_DECK
    if _MY_DECK is None:
        try:
            _MY_DECK = read_deck_csv()
        except Exception:
            _MY_DECK = []
    return _MY_DECK


def _fallback_dc(select):
    """Guaranteed-legal pick straight from a SelectData dataclass."""
    opts = select.option
    n, mn, mx = len(opts), select.minCount, select.maxCount
    for i, o in enumerate(opts):
        if int(o.type) == 14 and mn <= 1 <= max(mx, 1):
            return [i]
    if mx == 0:
        return []
    k = min(mn if mn > 0 else 1, n, mx if mx > 0 else n)
    return list(range(k))


def choose(obs) -> list:
    """Pure-heuristic legal pick from an Observation dataclass; never raises.
    Used as the agent's base policy AND as the rollout policy inside search."""
    try:
        pick = _sanitize(_decide(obs), obs.select)
        if pick is not None and len(pick) >= obs.select.minCount:
            return pick
    except Exception:
        pass
    return _fallback_dc(obs.select)


def agent(obs_dict: dict) -> list:
    sel = obs_dict.get("select")
    if sel is None:
        return read_deck_csv()
    _load_cards()
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return _fallback(sel)
    # Determinized search (PIMC) only on MAIN decisions with a real choice;
    # rollouts use the pure heuristic `choose`, so there is no recursion.
    if PIMC_ENABLED and obs.select and int(obs.select.context) == 0 \
            and len(obs.select.option) >= 2:
        try:
            import pimc
            cand = _pimc_candidates(obs)
            if len(cand) >= 2:
                a = pimc.decide(obs, _my_deck(), choose,
                                budget_s=PIMC_BUDGET_S, candidates=cand)
                if a is not None:
                    return [a]
        except Exception:
            pass
    return choose(obs)
