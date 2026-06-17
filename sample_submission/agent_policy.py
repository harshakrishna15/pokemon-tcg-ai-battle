"""Deck-profile-driven action policy.

The policy avoids hardcoding one exact deck sequence. It infers basic roles from
deck.csv and card facts, then scores the legal options that the simulator sends.
This keeps the agent usable when the deck changes, while still allowing card
facts to improve the scoring when the simulator database is available.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from time import monotonic
from typing import Any

from agent_card_db import CardInfo, load_card_database


# Numeric constants mirror cg.api enums. Keeping them here avoids importing
# cg.api in local tests, because importing cg.api loads the native simulator.
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12

OPT_NUMBER = 0
OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_TOOL_CARD = 4
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14
OPT_SKILL = 15

SEL_CARD = 1
SEL_ATTACHED_CARD = 2
SEL_CARD_OR_ATTACHED_CARD = 3
SEL_ENERGY = 4
SEL_ATTACK = 6
SEL_COUNT = 8
SEL_YES_NO = 9

CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_TO_ACTIVE = 4
CTX_TO_BENCH = 5
CTX_TO_HAND = 7
CTX_DISCARD = 8
CTX_TO_DECK = 9
CTX_ATTACH_TO = 22
CTX_DRAW_COUNT = 38
CTX_IS_FIRST = 41
CTX_MULLIGAN = 42
CTX_ACTIVATE = 43

SEARCH_TIME_LIMIT_SECONDS = 0.12
SEARCH_MAX_ROOT_OPTIONS = 6
SEARCH_MAX_FORCED_STEPS = 5
SEARCH_OVERRIDE_MARGIN = 35.0

ENERGY_TYPE_BY_VALUE = {
    0: "colorless",
    1: "grass",
    2: "fire",
    3: "water",
    4: "lightning",
    5: "psychic",
    6: "fighting",
    7: "darkness",
    8: "metal",
}

ENERGY_SYMBOL_TO_TYPE = {
    "{g}": "grass",
    "{r}": "fire",
    "{w}": "water",
    "{l}": "lightning",
    "{p}": "psychic",
    "{f}": "fighting",
    "{d}": "darkness",
    "{m}": "metal",
    "{c}": "colorless",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def read_deck_csv() -> list[int]:
    # Kaggle runs with deck.csv in the agent root. Local tests may import this
    # module from the repository root, so also check beside this file.
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "deck.csv",
        os.path.join(here, "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
    ]
    for file_path in candidates:
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                return [int(line.strip()) for line in file if line.strip()][:60]
    raise FileNotFoundError("deck.csv")


@dataclass(frozen=True)
class DeckProfile:
    """Facts inferred from the submitted deck list."""

    deck_counts: Counter[int]
    main_energy_ids: tuple[int, ...]
    basic_energy_ids: tuple[int, ...]
    pokemon_ids: frozenset[int]
    basic_ids: frozenset[int]
    evolution_ids: frozenset[int]
    attacker_ids: tuple[int, ...]
    evolves_from_name: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class ThreatMap:
    """Visible next-turn pressure from the opponent's board."""

    damage_to_active: int = 0
    active_damage_to_active: int = 0
    bench_damage_to_active: int = 0
    max_source_prize_value: int = 1
    can_ko_active: bool = False


def _card_info(card_id: int, cards: dict[int, CardInfo]) -> CardInfo:
    return cards.get(card_id, CardInfo(card_id, f"Card {card_id}"))


def _pokemon_value(card_id: int, cards: dict[int, CardInfo], deck_counts: Counter[int]) -> float:
    # A compact attacker/setup heuristic: durability, rule-box strength, stage,
    # and deck commitment all increase priority.
    info = _card_info(card_id, cards)
    if not info.is_pokemon:
        return 0.0
    value = float(info.hp)
    value += deck_counts[card_id] * 15
    if info.ex:
        value += 35
    if info.mega_ex:
        value += 45
    if info.stage1:
        value += 25
    if info.stage2:
        value += 15
    if info.attack_ids:
        value += 20
    return value


def _starter_value(card_id: int, cards: dict[int, CardInfo], deck_counts: Counter[int]) -> float:
    # Opening Active should usually be a Basic, and single-prize Basics are less
    # punishing than starting a high-value ex when there is a choice.
    info = _card_info(card_id, cards)
    value = _pokemon_value(card_id, cards, deck_counts)
    if not info.basic:
        value -= 500
    if info.ex:
        value -= 90
    if any(targets for source, targets in _evolution_map(cards, deck_counts).items() if source == info.name):
        value -= 20
    return value


def _evolution_map(cards: dict[int, CardInfo], deck_counts: Counter[int]) -> dict[str, tuple[int, ...]]:
    result: dict[str, list[int]] = defaultdict(list)
    for card_id in deck_counts:
        info = _card_info(card_id, cards)
        if info.evolves_from:
            result[info.evolves_from].append(card_id)
    return {name: tuple(ids) for name, ids in result.items()}


@lru_cache(maxsize=16)
def _profile_for_deck(deck_tuple: tuple[int, ...]) -> DeckProfile:
    # Cache by full deck contents so changing deck.csv automatically gives a new
    # profile without needing deck-specific code changes.
    cards, _ = load_card_database()
    counts = Counter(deck_tuple)
    energy_ids = [cid for cid in counts if _card_info(cid, cards).kind == "basic_energy"]
    energy_ids.sort(key=lambda cid: counts[cid], reverse=True)

    pokemon_ids = frozenset(cid for cid in counts if _card_info(cid, cards).is_pokemon)
    basic_ids = frozenset(cid for cid in pokemon_ids if _card_info(cid, cards).basic)
    evolution_ids = frozenset(cid for cid in pokemon_ids if not _card_info(cid, cards).basic)
    attackers = sorted(
        pokemon_ids,
        key=lambda cid: _pokemon_value(cid, cards, counts),
        reverse=True,
    )

    return DeckProfile(
        deck_counts=counts,
        main_energy_ids=tuple(energy_ids[:2]),
        basic_energy_ids=tuple(energy_ids),
        pokemon_ids=pokemon_ids,
        basic_ids=basic_ids,
        evolution_ids=evolution_ids,
        attacker_ids=tuple(attackers),
        evolves_from_name=_evolution_map(cards, counts),
    )


def _profile(deck: list[int]) -> DeckProfile:
    return _profile_for_deck(tuple(deck))


def _state(obs: Any) -> Any:
    return _get(obs, "current")


def _select(obs: Any) -> Any:
    return _get(obs, "select")


def _your_index(obs: Any) -> int:
    return _int(_get(_state(obs), "yourIndex", 0), 0)


def _players(obs: Any) -> list[Any]:
    return _as_list(_get(_state(obs), "players", []))


def _player(obs: Any, player_index: int | None = None) -> Any:
    players = _players(obs)
    if not players:
        return None
    idx = _your_index(obs) if player_index is None else player_index
    if idx < 0 or idx >= len(players):
        idx = 0
    return players[idx]


def _card_id(card: Any) -> int | None:
    value = _get(card, "id", _get(card, "cardId"))
    if value is None:
        return None
    return _int(value)


def _cards_in_area(obs: Any, area: int | None, player_index: int | None = None) -> list[Any]:
    # Options reference cards by area plus index. This resolver works against
    # raw dict observations and dataclass observations.
    state = _state(obs)
    select = _select(obs)
    player = _player(obs, player_index)

    if area == AREA_HAND:
        return _as_list(_get(player, "hand", []))
    if area == AREA_ACTIVE:
        return [c for c in _as_list(_get(player, "active", [])) if c is not None]
    if area == AREA_BENCH:
        return _as_list(_get(player, "bench", []))
    if area == AREA_DISCARD:
        return _as_list(_get(player, "discard", []))
    if area == AREA_PRIZE:
        return [c for c in _as_list(_get(player, "prize", [])) if c is not None]
    if area == AREA_STADIUM:
        return _as_list(_get(state, "stadium", []))
    if area == AREA_LOOKING:
        return [c for c in _as_list(_get(state, "looking", [])) if c is not None]
    if area == AREA_DECK:
        return _as_list(_get(select, "deck", []))
    return []


def _resolve_area_card_id(obs: Any, area: int | None, index: int | None, player_index: int | None = None) -> int | None:
    cards = _cards_in_area(obs, area, player_index)
    if index is None or index < 0 or index >= len(cards):
        return None
    return _card_id(cards[index])


def _resolve_option_card_id(obs: Any, option: Any) -> int | None:
    direct = _get(option, "cardId")
    if direct is not None:
        return _int(direct)

    option_type = _int(_get(option, "type"))
    player_index = _get(option, "playerIndex", _your_index(obs))

    if option_type == OPT_PLAY:
        return _resolve_area_card_id(obs, AREA_HAND, _get(option, "index"), _your_index(obs))
    if option_type in {OPT_CARD, OPT_ABILITY, OPT_DISCARD}:
        return _resolve_area_card_id(obs, _get(option, "area"), _get(option, "index"), player_index)
    if option_type in {OPT_ATTACH, OPT_EVOLVE}:
        return _resolve_area_card_id(obs, _get(option, "area"), _get(option, "index"), player_index)
    return None


def _resolve_target_pokemon_id(obs: Any, option: Any) -> int | None:
    in_play_area = _get(option, "inPlayArea")
    in_play_index = _get(option, "inPlayIndex")
    if in_play_area is not None and in_play_index is not None:
        return _resolve_area_card_id(obs, in_play_area, in_play_index, _your_index(obs))
    return _resolve_area_card_id(obs, _get(option, "area"), _get(option, "index"), _get(option, "playerIndex", _your_index(obs)))


def _opponent_active_info(obs: Any, cards: dict[int, CardInfo]) -> CardInfo:
    opponent = _active_pokemon(obs, _opponent_index(obs))
    return _card_info(_card_id(opponent) or 0, cards)


def _matchup_bonus(obs: Any, attacker_id: int | None, cards: dict[int, CardInfo]) -> float:
    if attacker_id is None:
        return 0.0

    attacker = _card_info(attacker_id, cards)
    defender = _opponent_active_info(obs, cards)
    if not attacker.is_pokemon or not defender.is_pokemon or attacker.energy_type is None:
        return 0.0

    bonus = 0.0
    if defender.weakness == attacker.energy_type:
        bonus += 230
        if defender.ex:
            bonus += 90
    if defender.resistance == attacker.energy_type:
        bonus -= 80
    return bonus


def _attach_energy_bonus(
    obs: Any,
    energy_id: int | None,
    target_id: int | None,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
) -> float:
    energy = _card_info(energy_id or 0, cards)
    target = _card_info(target_id or 0, cards)
    if energy.kind != "basic_energy":
        return 0.0

    bonus = 35.0
    if energy_id in profile.main_energy_ids:
        bonus += 70
    if target.energy_type == energy.energy_type:
        bonus += 130
    bonus += _matchup_bonus(obs, target_id, cards)
    return bonus


def _your_hand_ids(obs: Any) -> list[int]:
    return [cid for cid in (_card_id(card) for card in _cards_in_area(obs, AREA_HAND, _your_index(obs))) if cid is not None]


def _your_field_ids(obs: Any) -> list[int]:
    ids = []
    for area in (AREA_ACTIVE, AREA_BENCH):
        ids.extend(cid for cid in (_card_id(card) for card in _cards_in_area(obs, area, _your_index(obs))) if cid is not None)
    return ids


def _has_evolution_source_in_play(obs: Any, profile: DeckProfile, cards: dict[int, CardInfo]) -> bool:
    field_ids = _your_field_ids(obs)
    field_names = {_card_info(cid, cards).name for cid in field_ids}
    return any(name in field_names for name in profile.evolves_from_name)


def _missing_evolution_piece(obs: Any, profile: DeckProfile) -> bool:
    field_ids = set(_your_field_ids(obs))
    hand_ids = set(_your_hand_ids(obs))
    visible_ids = field_ids | hand_ids
    return any(cid not in visible_ids for cid in profile.evolution_ids)


def _bench_space(obs: Any) -> int:
    player = _player(obs)
    bench = _as_list(_get(player, "bench", []))
    bench_max = _int(_get(player, "benchMax", 5), 5)
    return max(0, bench_max - len(bench))


def _estimated_your_deck_ids(
    obs: Any,
    deck: list[int] | None,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
) -> list[int]:
    if not deck:
        return []
    your_deck, _, _, _, _, _ = _predict_hidden_zones(obs, deck, profile, cards)
    return your_deck


def _visible_your_ids(obs: Any) -> set[int]:
    visible = set(_your_hand_ids(obs))
    visible.update(_your_field_ids(obs))
    visible.update(_card_id(card) for card in _cards_in_area(obs, AREA_DISCARD, _your_index(obs)))
    visible.update(_known_prize_ids(obs, _your_index(obs)))
    visible.discard(None)
    return visible


def _best_energy_target_need(
    obs: Any,
    energy_type: str | None,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> float:
    if energy_type is None:
        return 0.0

    best = 0.0
    for pokemon in _pokemon_in_play(obs, _your_index(obs)):
        card_id = _card_id(pokemon)
        info = _card_info(card_id or 0, cards)
        desired = _preferred_energy_count(card_id, cards, attacks)
        gap = max(0, desired - _attached_energy_count(pokemon))
        if gap <= 0:
            continue

        value = 130 + min(gap, 3) * 85
        value += _pokemon_value(card_id or 0, cards, profile.deck_counts) * 0.12
        if info.energy_type == energy_type:
            value += 150
        if pokemon is _your_active_pokemon(obs):
            value += 70
        best = max(best, value)

    return best


def _card_need_value(
    obs: Any,
    card_id: int,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> float:
    info = _card_info(card_id, cards)
    visible_ids = _visible_your_ids(obs)
    hand_ids = set(_your_hand_ids(obs))
    field_ids = set(_your_field_ids(obs))
    value = 0.0

    if info.kind == "basic_energy":
        value = _best_energy_target_need(obs, info.energy_type, profile, cards, attacks)
        if card_id in profile.main_energy_ids:
            value += 70
        return value

    if info.is_pokemon:
        pokemon_value = _pokemon_value(card_id, cards, profile.deck_counts)
        if card_id in profile.evolution_ids and card_id not in hand_ids and _has_evolution_source_in_play(obs, profile, cards):
            value = max(value, 680 + pokemon_value * 0.45 + _matchup_bonus(obs, card_id, cards))
        if info.basic and _bench_space(obs) > 0 and card_id not in field_ids:
            value = max(value, 380 + _starter_value(card_id, cards, profile.deck_counts) * 0.35)
        if card_id in profile.attacker_ids[:4] and card_id not in field_ids:
            value = max(value, 430 + pokemon_value * 0.35 + _matchup_bonus(obs, card_id, cards))
        if card_id in visible_ids:
            value *= 0.45
        return value

    roles = _role(card_id, info)
    hand_count = len(_your_hand_ids(obs))
    if "search" in roles and (_missing_evolution_piece(obs, profile) or _bench_space(obs) > 0):
        value = max(value, 230)
    if "draw" in roles and hand_count <= 4:
        value = max(value, 260 - hand_count * 25)
    if "energy_accel" in roles:
        active = _your_active_pokemon(obs)
        active_info = _card_info(_card_id(active) or 0, cards)
        value = max(value, _best_energy_target_need(obs, active_info.energy_type, profile, cards, attacks) * 0.45)

    return value


def _expected_draw_value(
    obs: Any,
    deck: list[int] | None,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
    draw_count: int,
) -> float:
    deck_ids = _estimated_your_deck_ids(obs, deck, profile, cards)
    deck_size = len(deck_ids)
    if deck_size <= 0 or draw_count <= 0:
        return 0.0

    draw_count = min(draw_count, deck_size)
    hit_rate = draw_count / deck_size
    expected = sum(_card_need_value(obs, card_id, profile, cards, attacks) * hit_rate for card_id in deck_ids)
    hand_count = len(_your_hand_ids(obs))
    if hand_count <= 2:
        expected += 150
    elif hand_count <= 4:
        expected += 80
    return min(expected, 620)


def _estimated_draw_count_for_card(info: CardInfo) -> int:
    text = f"{info.name} {info.effect_text}".lower()
    if "top six" in text or "six cards" in text:
        return 6
    if "shuffle your hand" in text:
        return 5
    if "draw cards" in text or "draw" in text:
        return 3
    return 0


def _search_target_ids_for_card(
    search_card: CardInfo,
    deck_ids: list[int],
    cards: dict[int, CardInfo],
) -> list[int]:
    text = f"{search_card.name} {search_card.effect_text}".lower()
    targets: list[int] = []
    for card_id in sorted(set(deck_ids)):
        info = _card_info(card_id, cards)
        if "basic energy" in text and info.kind == "basic_energy":
            targets.append(card_id)
        elif "mega" in text and info.mega_ex:
            targets.append(card_id)
        elif "basic pokemon with 70 hp or less" in text and info.basic and info.hp <= 70:
            targets.append(card_id)
        elif "pokemon" in text and info.is_pokemon:
            targets.append(card_id)
    return targets


def _expected_search_value(
    obs: Any,
    card_id: int,
    deck: list[int] | None,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> float:
    deck_ids = _estimated_your_deck_ids(obs, deck, profile, cards)
    if not deck_ids:
        return 0.0

    info = _card_info(card_id, cards)
    targets = _search_target_ids_for_card(info, deck_ids, cards)
    if not targets:
        return 0.0

    target_values = sorted(
        (_card_need_value(obs, target_id, profile, cards, attacks) for target_id in targets),
        reverse=True,
    )
    if not target_values:
        return 0.0

    text = f"{info.name} {info.effect_text}".lower()
    take = 2 if "up to 2" in text else 1
    if "bench" in text:
        take = min(take, _bench_space(obs))
        if take <= 0:
            return 0.0
    value = sum(target_values[: max(1, take)])
    if "discard two cards" in text:
        value -= 140
    return min(max(0.0, value), 760)


def _expected_energy_accel_value(
    obs: Any,
    card_id: int,
    deck: list[int] | None,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> float:
    deck_ids = _estimated_your_deck_ids(obs, deck, profile, cards)
    info = _card_info(card_id, cards)
    text = f"{info.name} {info.effect_text}".lower()
    energy_ids = [cid for cid in deck_ids if _card_info(cid, cards).kind == "basic_energy"]
    if not energy_ids:
        return 0.0

    best_need = 0.0
    for energy_id in set(energy_ids):
        energy = _card_info(energy_id, cards)
        best_need = max(best_need, _best_energy_target_need(obs, energy.energy_type, profile, cards, attacks))

    if best_need <= 0:
        return 0.0

    if "two different basic energy" in text:
        different_types = {_card_info(cid, cards).energy_type for cid in energy_ids}
        if len(different_types) < 2:
            return best_need * 0.25
        return min(620, 180 + best_need)

    if "top six" in text:
        deck_size = max(1, len(deck_ids))
        miss_rate = max(0.0, 1.0 - len(energy_ids) / deck_size) ** min(6, deck_size)
        return min(600, (1.0 - miss_rate) * (190 + best_need))

    return min(520, best_need * 0.75)


def _role(card_id: int, info: CardInfo) -> set[str]:
    # Keyword roles are deliberately broad. They make unknown trainer cards
    # useful without needing a custom branch for every card name.
    text = f"{info.name} {info.effect_text}".lower()
    roles: set[str] = set()
    if "search your deck" in text or "mega signal" in text or "ultra ball" in text:
        roles.add("search")
    if "draw" in text or "shuffle your hand" in text:
        roles.add("draw")
    if "attach" in text and "energy" in text:
        roles.add("energy_accel")
    if "opponent" in text and "benched" in text and "active" in text:
        roles.add("gust")
    if "switch your active" in text or info.name.lower() == "switch":
        roles.add("switch")
    if "more damage" in text or "maximum belt" in text:
        roles.add("damage_boost")
    return roles


def _score_play_card(
    obs: Any,
    card_id: int | None,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
    deck: list[int] | None = None,
) -> float:
    # Main-phase play scoring rewards setup first, then search/draw/energy
    # acceleration. Supporters get a small tax because only one can be used.
    if card_id is None:
        return 100
    info = _card_info(card_id, cards)

    if info.is_pokemon:
        if info.basic and _bench_space(obs) > 0:
            return 430 + _starter_value(card_id, cards, profile.deck_counts) * 0.25
        return 120 + _pokemon_value(card_id, cards, profile.deck_counts) * 0.1

    roles = _role(card_id, info)
    hand_count = len(_your_hand_ids(obs))
    score = 150.0

    if "energy_accel" in roles:
        score = max(score, 360 + _expected_energy_accel_value(obs, card_id, deck, profile, cards, attacks))
    if "search" in roles:
        search_value = _expected_search_value(obs, card_id, deck, profile, cards, attacks)
        score = max(score, 330 + search_value)
        if _has_evolution_source_in_play(obs, profile, cards):
            score += 60
    if "draw" in roles:
        draw_count = _estimated_draw_count_for_card(info)
        draw_value = _expected_draw_value(
            obs,
            deck,
            profile,
            cards,
            attacks,
            draw_count,
        )
        deck_count = _int(_get(_player(obs), "deckCount", 0), 0)
        draw_risk = 650 if draw_count > 0 and deck_count <= draw_count <= 6 else 0
        if hand_count <= 3:
            score = max(score, 420 + draw_value - draw_risk)
        elif hand_count <= 6:
            score = max(score, 300 + draw_value - draw_risk)
        else:
            score = max(score, 180 + draw_value * 0.55 - draw_risk)
    if "damage_boost" in roles:
        score = max(score, 520)
    if "gust" in roles:
        score = max(score, 210 + _gust_expected_value(obs, profile, cards, attacks))
    if "switch" in roles:
        score = max(score, 230)

    if info.kind == "supporter":
        score -= 25
    return score


def _score_card_pick(
    obs: Any,
    card_id: int | None,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks=None,
) -> float:
    if card_id is None:
        return 0
    info = _card_info(card_id, cards)
    context = _int(_get(_select(obs), "context"))

    if context == CTX_SETUP_ACTIVE:
        return _starter_value(card_id, cards, profile.deck_counts)
    if context in {CTX_SETUP_BENCH, CTX_TO_BENCH}:
        if info.basic:
            return 500 + _starter_value(card_id, cards, profile.deck_counts)
        return -100
    if context == CTX_ATTACH_TO:
        return _pokemon_value(card_id, cards, profile.deck_counts)

    if card_id in profile.evolution_ids and _has_evolution_source_in_play(obs, profile, cards):
        return 900 + _pokemon_value(card_id, cards, profile.deck_counts)
    if card_id in profile.attacker_ids[:3]:
        return 720 + _pokemon_value(card_id, cards, profile.deck_counts) + _matchup_bonus(obs, card_id, cards)
    if info.is_pokemon:
        return 500 + _pokemon_value(card_id, cards, profile.deck_counts) + _matchup_bonus(obs, card_id, cards)
    if "search" in _role(card_id, info):
        return 430
    if "draw" in _role(card_id, info):
        return 390
    if info.kind == "basic_energy" and attacks is not None:
        need = _best_energy_target_need(obs, info.energy_type, profile, cards, attacks)
        if need > 0:
            score = 160 + need
            if card_id in profile.main_energy_ids:
                score += 70
            return score
    if card_id in profile.main_energy_ids:
        return 250
    if card_id in profile.basic_energy_ids:
        return 120
    return 100


def _discard_penalty(obs: Any, card_id: int | None, profile: DeckProfile, cards: dict[int, CardInfo]) -> float:
    # Lower penalty means a better discard. Basic Energy is usually the safest
    # discard in this deck shape, while unique attackers and evolutions are kept.
    if card_id is None:
        return 100
    info = _card_info(card_id, cards)
    hand_counts = Counter(_your_hand_ids(obs))

    penalty = 100.0
    if info.kind == "basic_energy":
        penalty = 95 if profile.deck_counts[card_id] <= 3 else 5
    elif info.is_energy:
        penalty = 20
    elif info.is_pokemon:
        penalty = 700 + _pokemon_value(card_id, cards, profile.deck_counts)
        if hand_counts[card_id] > 1:
            penalty -= 180
    elif "draw" in _role(card_id, info):
        penalty = 260
    elif "search" in _role(card_id, info):
        penalty = 360
    elif info.is_trainer:
        penalty = 190

    if profile.deck_counts[card_id] <= 1:
        penalty += 200
    return penalty


def _score_option(
    obs: Any,
    option: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
    deck: list[int] | None = None,
) -> float:
    # Convert each legal option into a single comparable score. The simulator
    # still enforces legality; this only chooses among legal indexes.
    option_type = _int(_get(option, "type"))
    card_id = _resolve_option_card_id(obs, option)

    if option_type == OPT_EVOLVE:
        return 920 + _score_card_pick(obs, card_id, profile, cards, attacks)
    if option_type == OPT_ATTACH:
        target_id = _resolve_target_pokemon_id(obs, option)
        target = _resolve_target_pokemon(obs, option)
        energy_bonus = _attach_energy_bonus(obs, card_id, target_id, profile, cards)
        return (
            760
            + energy_bonus
            + _pokemon_value(target_id or 0, cards, profile.deck_counts) * 0.3
            + _attachment_progress_bonus(obs, card_id, target, profile, cards, attacks)
        )
    if option_type == OPT_PLAY:
        return _score_play_card(obs, card_id, profile, cards, attacks, deck)
    if option_type == OPT_ABILITY:
        return 610 + _pokemon_value(card_id or 0, cards, profile.deck_counts) * 0.15
    if option_type == OPT_ATTACK:
        attack_id = _int(_get(option, "attackId"), -1)
        estimated_damage, attack = _estimate_attack_damage_and_source(obs, option, profile, cards, attacks)
        effective_damage = _effective_attack_damage(obs, option, profile, cards, attacks)
        progress_bonus = 0
        state = _state(obs)
        if (
            _int(_get(state, "turnActionCount", 0)) > 0
            or bool(_get(state, "energyAttached", False))
            or bool(_get(state, "supporterPlayed", False))
            or bool(_get(state, "stadiumPlayed", False))
        ):
            progress_bonus = 650
        known_damage = attack.damage if attack else 0
        return 820 + progress_bonus + max(known_damage, estimated_damage, effective_damage, 80) + _attack_tactical_bonus(
            obs,
            option,
            profile,
            cards,
            attacks,
        )
    if option_type == OPT_RETREAT:
        return 210
    if option_type == OPT_CARD:
        if _int(_get(_select(obs), "context")) == CTX_ATTACH_TO:
            return _attachment_target_value(obs, _resolve_target_pokemon(obs, option), profile, cards, attacks)
        return _score_card_pick(obs, card_id, profile, cards, attacks)
    if option_type == OPT_NUMBER:
        return _int(_get(option, "number"), 0)
    if option_type == OPT_YES:
        return 20
    if option_type == OPT_NO:
        return 10
    if option_type == OPT_END:
        return -1000
    return 0


def _attached_energy_count(pokemon: Any) -> int:
    # The observation exposes both energy units and the cards that created them.
    # Use the larger count so special energy or transformed energy still matter.
    return max(
        len(_as_list(_get(pokemon, "energies", []))),
        len(_as_list(_get(pokemon, "energyCards", []))),
    )


def _your_active_pokemon(obs: Any) -> Any:
    active = _cards_in_area(obs, AREA_ACTIVE, _your_index(obs))
    return active[0] if active else None


def _resolve_target_pokemon(obs: Any, option: Any) -> Any:
    in_play_area = _get(option, "inPlayArea")
    in_play_index = _get(option, "inPlayIndex")
    if in_play_area is None or in_play_index is None:
        in_play_area = _get(option, "area")
        in_play_index = _get(option, "index")

    cards = _cards_in_area(obs, in_play_area, _your_index(obs))
    if not cards:
        return None
    idx = _int(in_play_index, -1)
    if idx < 0 or idx >= len(cards):
        return None
    return cards[idx]


def _is_active_target(obs: Any, option: Any) -> bool:
    return _get(option, "inPlayArea") == AREA_ACTIVE and _int(_get(option, "inPlayIndex"), -1) == 0


def _opponent_index(obs: Any, player_index: int | None = None) -> int:
    idx = _your_index(obs) if player_index is None else player_index
    return 1 - idx


def _active_pokemon(obs: Any, player_index: int) -> Any:
    active = _cards_in_area(obs, AREA_ACTIVE, player_index)
    return active[0] if active else None


def _remaining_hp(pokemon: Any, cards: dict[int, CardInfo]) -> int:
    card_id = _card_id(pokemon)
    info = _card_info(card_id or 0, cards)
    return _int(_get(pokemon, "hp", info.hp), info.hp)


def _prize_value(card_id: int | None, cards: dict[int, CardInfo]) -> int:
    info = _card_info(card_id or 0, cards)
    if info.mega_ex:
        return 3
    if info.ex:
        return 2
    return 1


def _count_basic_energy_in_area(obs: Any, area: int, player_index: int, energy_ids: tuple[int, ...]) -> int:
    return sum(1 for card in _cards_in_area(obs, area, player_index) if _card_id(card) in energy_ids)


def _deck_energy_density(obs: Any, profile: DeckProfile) -> float:
    player = _player(obs, _your_index(obs))
    deck_count = max(1, _int(_get(player, "deckCount", 0), 0))
    visible_energy = 0
    for area in (AREA_HAND, AREA_DISCARD, AREA_ACTIVE, AREA_BENCH, AREA_PRIZE):
        visible_energy += _count_basic_energy_in_area(obs, area, _your_index(obs), profile.basic_energy_ids)
    remaining_energy = max(0, sum(profile.deck_counts[cid] for cid in profile.basic_energy_ids) - visible_energy)
    return min(1.0, remaining_energy / deck_count)


def _energy_type_from_unit(value: Any, cards: dict[int, CardInfo]) -> str | None:
    card_id = _int(value, -1)
    info = _card_info(card_id, cards)
    if info.is_energy and info.energy_type:
        return info.energy_type
    return ENERGY_TYPE_BY_VALUE.get(card_id)


def _attached_energy_types(pokemon: Any, cards: dict[int, CardInfo]) -> list[str]:
    energy_cards = _as_list(_get(pokemon, "energyCards", []))
    if energy_cards:
        return [
            energy_type
            for energy_type in (_energy_type_from_unit(_card_id(card), cards) for card in energy_cards)
            if energy_type is not None
        ]

    return [
        energy_type
        for energy_type in (_energy_type_from_unit(value, cards) for value in _as_list(_get(pokemon, "energies", [])))
        if energy_type is not None
    ]


def _attached_energy_type_count(pokemon: Any, energy_type: str | None, cards: dict[int, CardInfo]) -> int:
    if energy_type is None:
        return 0
    return sum(1 for value in _attached_energy_types(pokemon, cards) if value == energy_type)


def _pokemon_in_play(obs: Any, player_index: int) -> list[Any]:
    return _cards_in_area(obs, AREA_ACTIVE, player_index) + _cards_in_area(obs, AREA_BENCH, player_index)


def _board_energy_type_count(obs: Any, player_index: int, energy_type: str | None, cards: dict[int, CardInfo]) -> int:
    if energy_type is None:
        return 0
    return sum(_attached_energy_type_count(pokemon, energy_type, cards) for pokemon in _pokemon_in_play(obs, player_index))


def _count_basic_energy_type_in_area(
    obs: Any,
    area: int,
    player_index: int,
    energy_type: str | None,
    cards: dict[int, CardInfo],
) -> int:
    if energy_type is None:
        return 0
    return sum(
        1
        for card in _cards_in_area(obs, area, player_index)
        if _card_info(_card_id(card) or 0, cards).kind == "basic_energy"
        and _card_info(_card_id(card) or 0, cards).energy_type == energy_type
    )


def _attack_text(attack: Any) -> str:
    if attack is None:
        return ""
    return (
        f"{_get(attack, 'name', '')} {_get(attack, 'text', '')}"
        .lower()
        .replace("’", "'")
        .replace("\xa0", " ")
        .replace("pokémon", "pokemon")
        .replace("{ex}", "ex")
    )


def _energy_type_from_attack_text(text: str, fallback: str | None) -> str | None:
    for symbol, energy_type in ENERGY_SYMBOL_TO_TYPE.items():
        if symbol in text:
            return energy_type
    return fallback


def _more_damage_values(text: str) -> list[int]:
    return [int(match.group(1)) for match in re.finditer(r"does\s+(\d+)\s+more damage", text)]


def _bench_count(obs: Any, player_index: int) -> int:
    return len(_cards_in_area(obs, AREA_BENCH, player_index))


def _estimate_damage_for_attacker(
    obs: Any,
    attacker: Any,
    defender: Any,
    attack: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    player_index: int,
    attacker_energy_override: int | None = None,
) -> int:
    attacker_id = _card_id(attacker)
    attacker_info = _card_info(attacker_id or 0, cards)
    attacker_energy = _attached_energy_count(attacker) if attacker_energy_override is None else attacker_energy_override
    defender_info = _card_info(_card_id(defender) or 0, cards)
    defender_energy = _attached_energy_count(defender)
    text = _attack_text(attack)
    damage = _int(_get(attack, "damage", 0), 0)
    attack_type = _energy_type_from_attack_text(text, attacker_info.energy_type)
    more_values = _more_damage_values(text)

    if "for each energy attached" in text and "opponent" not in text:
        multiplier = damage if damage > 0 else (more_values[0] if more_values else 40)
        damage = max(damage, multiplier * attacker_energy)

    if "for each" in text and "energy attached to all of your pokemon" in text:
        multiplier = damage if damage > 0 else (more_values[0] if more_values else 40)
        board_energy = _board_energy_type_count(obs, player_index, attack_type, cards)
        damage = max(damage, multiplier * board_energy)

    if "for each basic" in text and "energy card in your discard" in text:
        energy_type = _energy_type_from_attack_text(text, attack_type)
        discard_energy = _count_basic_energy_type_in_area(obs, AREA_DISCARD, player_index, energy_type, cards)
        multiplier = damage if damage > 0 else (more_values[0] if more_values else 20)
        damage = max(damage, multiplier * discard_energy)

    if "discard the top 6 cards" in text and "basic" in text and "energy" in text:
        expected_hits = int(round(6 * _deck_energy_density(obs, profile)))
        multiplier = damage if damage > 0 else 100
        damage = max(damage, multiplier * max(1, expected_hits))

    if "pokemon ex" in text and "more damage" in text and defender_info.ex and more_values:
        damage += max(more_values)

    if "already has any damage counters" in text and "more damage" in text and more_values:
        if defender is not None and _remaining_hp(defender, cards) < defender_info.hp:
            damage += max(more_values)

    if "for each energy attached to your opponent's active" in text:
        multiplier = more_values[0] if more_values else damage
        damage += multiplier * defender_energy

    if "for each of your opponent's benched" in text or "for each your opponent's benched" in text:
        multiplier = more_values[0] if more_values else damage
        damage += multiplier * _bench_count(obs, _opponent_index(obs, player_index))

    if "for each benched pokemon" in text and "both yours and your opponent" in text:
        multiplier = more_values[0] if more_values else damage
        damage += multiplier * (
            _bench_count(obs, player_index) + _bench_count(obs, _opponent_index(obs, player_index))
        )

    return max(0, damage)


def _estimate_damage_from_attack(
    obs: Any,
    attack: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    player_index: int,
) -> int:
    active = _active_pokemon(obs, player_index)
    opponent = _active_pokemon(obs, _opponent_index(obs, player_index))
    return _estimate_damage_for_attacker(obs, active, opponent, attack, profile, cards, player_index)


def _attack_is_usable(attack: Any, active_energy: int) -> bool:
    cost = _int(_get(attack, "energy_count", 0), 0)
    return cost <= active_energy


def _option_attack_for_active(active_info: CardInfo, option: Any, attacks) -> Any | None:
    attack_id = _int(_get(option, "attackId"), -1)
    if attack_id not in active_info.attack_ids:
        return None
    return attacks.get(attack_id)


def _estimate_attack_damage_and_source(
    obs: Any,
    option: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
    player_index: int | None = None,
) -> tuple[int, Any | None]:
    player_index = _your_index(obs) if player_index is None else player_index
    active = _active_pokemon(obs, player_index)
    active_id = _card_id(active)
    active_info = _card_info(active_id or 0, cards)
    active_energy = _attached_energy_count(active)
    attack = _option_attack_for_active(active_info, option, attacks)

    if attack is not None:
        return _estimate_damage_from_attack(obs, attack, profile, cards, player_index), attack

    candidates = [attacks[attack_id] for attack_id in active_info.attack_ids if attack_id in attacks]
    usable = [candidate for candidate in candidates if _attack_is_usable(candidate, active_energy)]
    candidates = usable or candidates
    if not candidates:
        return 0, None

    best_attack = max(
        candidates,
        key=lambda candidate: _estimate_damage_from_attack(obs, candidate, profile, cards, player_index),
    )
    return _estimate_damage_from_attack(obs, best_attack, profile, cards, player_index), best_attack


def _estimate_attack_damage(
    obs: Any,
    option: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
    player_index: int | None = None,
) -> int:
    damage, _ = _estimate_attack_damage_and_source(obs, option, profile, cards, attacks, player_index)
    return damage


def _attack_ignores_weakness(attack: Any) -> bool:
    text = _attack_text(attack)
    return "affected by weakness" in text and "damage" in text


def _apply_weakness_resistance(
    raw_damage: int,
    attack_type: str | None,
    defender: Any,
    cards: dict[int, CardInfo],
) -> int:
    if raw_damage <= 0 or attack_type is None or defender is None:
        return raw_damage

    defender_info = _card_info(_card_id(defender) or 0, cards)
    damage = raw_damage
    if defender_info.weakness == attack_type:
        damage *= 2
    if defender_info.resistance == attack_type:
        damage = max(0, damage - 30)
    return damage


def _effective_attack_damage(
    obs: Any,
    option: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
    player_index: int | None = None,
) -> int:
    player_index = _your_index(obs) if player_index is None else player_index
    raw_damage, attack = _estimate_attack_damage_and_source(obs, option, profile, cards, attacks, player_index)
    active = _active_pokemon(obs, player_index)
    active_id = _card_id(active)
    if _attack_ignores_weakness(attack):
        return raw_damage

    attack_type = _card_info(active_id or 0, cards).energy_type
    defender = _active_pokemon(obs, _opponent_index(obs, player_index))
    return _apply_weakness_resistance(raw_damage, attack_type, defender, cards)


def _effective_damage_for_attacker(
    obs: Any,
    attacker: Any,
    defender: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
    player_index: int,
    extra_energy: int = 0,
) -> int:
    attacker_id = _card_id(attacker)
    info = _card_info(attacker_id or 0, cards)
    if not info.is_pokemon:
        return 0

    energy_count = _attached_energy_count(attacker) + extra_energy
    best = 0
    for attack_id in info.attack_ids:
        attack = attacks.get(attack_id)
        if attack is None or _int(_get(attack, "energy_count", 0), 0) > energy_count:
            continue
        raw_damage = _estimate_damage_for_attacker(
            obs,
            attacker,
            defender,
            attack,
            profile,
            cards,
            player_index,
            attacker_energy_override=energy_count,
        )
        if _attack_ignores_weakness(attack):
            damage = raw_damage
        else:
            attack_type = _energy_type_from_attack_text(_attack_text(attack), info.energy_type)
            damage = _apply_weakness_resistance(raw_damage, attack_type, defender, cards)
        best = max(best, damage)

    return best


def _fallback_threat_damage(pokemon: Any, cards: dict[int, CardInfo], extra_energy: int) -> int:
    card_id = _card_id(pokemon)
    if card_id is None:
        return 0

    info = _card_info(card_id, cards)
    energy_count = _attached_energy_count(pokemon) + extra_energy
    damage = energy_count * 45
    if info.ex:
        damage += 35
    if info.stage1 or info.stage2:
        damage += 25
    return damage


def _opponent_threat_map(obs: Any, profile: DeckProfile, cards: dict[int, CardInfo], attacks) -> ThreatMap:
    your_active = _active_pokemon(obs, _your_index(obs))
    your_hp = _remaining_hp(your_active, cards)
    opponent_index = _opponent_index(obs)
    opponent_active = _active_pokemon(obs, opponent_index)

    active_damage = _effective_damage_for_attacker(
        obs,
        opponent_active,
        your_active,
        profile,
        cards,
        attacks,
        opponent_index,
        extra_energy=1,
    )
    if active_damage <= 0:
        active_damage = _fallback_threat_damage(opponent_active, cards, extra_energy=1)

    bench_damage = 0
    source_prize_value = _prize_value(_card_id(opponent_active), cards)
    for pokemon in _cards_in_area(obs, AREA_BENCH, opponent_index):
        damage = _effective_damage_for_attacker(
            obs,
            pokemon,
            your_active,
            profile,
            cards,
            attacks,
            opponent_index,
            extra_energy=1,
        )
        if damage <= 0:
            damage = int(_fallback_threat_damage(pokemon, cards, extra_energy=1) * 0.8)
        if damage > bench_damage:
            bench_damage = damage
            source_prize_value = _prize_value(_card_id(pokemon), cards)

    max_damage = max(active_damage, bench_damage)
    return ThreatMap(
        damage_to_active=max_damage,
        active_damage_to_active=active_damage,
        bench_damage_to_active=bench_damage,
        max_source_prize_value=source_prize_value,
        can_ko_active=max_damage >= your_hp > 0,
    )


def _gust_expected_value(obs: Any, profile: DeckProfile, cards: dict[int, CardInfo], attacks) -> float:
    attacker = _active_pokemon(obs, _your_index(obs))
    if attacker is None:
        return 0.0

    best = 0.0
    for target in _cards_in_area(obs, AREA_BENCH, _opponent_index(obs)):
        damage = _effective_damage_for_attacker(
            obs,
            attacker,
            target,
            profile,
            cards,
            attacks,
            _your_index(obs),
            extra_energy=0,
        )
        target_hp = _remaining_hp(target, cards)
        target_prizes = _prize_value(_card_id(target), cards)
        target_energy = _attached_energy_count(target)

        value = min(220, damage) * 0.55
        if damage >= target_hp > 0:
            value += 820 + target_prizes * 360
        elif target_energy >= 2:
            value += 160 + target_energy * 45
        if _card_info(_card_id(target) or 0, cards).ex:
            value += 90
        best = max(best, value)

    return best


def _estimate_visible_threat_damage(obs: Any, player_index: int, profile: DeckProfile, cards: dict[int, CardInfo], attacks) -> int:
    if player_index == _opponent_index(obs):
        return _opponent_threat_map(obs, profile, cards, attacks).damage_to_active

    active = _active_pokemon(obs, player_index)
    active_id = _card_id(active)
    if active_id is None:
        return 0

    info = _card_info(active_id, cards)
    energy_count = _attached_energy_count(active)
    best = 0
    for attack_id in info.attack_ids:
        attack = attacks.get(attack_id)
        if attack is None or _int(_get(attack, "energy_count", 0), 0) > energy_count + 1:
            continue
        pseudo_option = {"type": OPT_ATTACK, "attackId": attack_id}
        best = max(best, _effective_attack_damage(obs, pseudo_option, profile, cards, attacks, player_index))

    if best <= 0:
        # Unknown opposing deck: approximate from board investment so we still
        # respect a powered-up attacker instead of goldfishing blindly.
        best = energy_count * 45
        if info.ex:
            best += 35
        if info.stage1 or info.stage2:
            best += 25

    return best


def _attack_tactical_bonus(obs: Any, option: Any, profile: DeckProfile, cards: dict[int, CardInfo], attacks) -> float:
    raw_damage, attack = _estimate_attack_damage_and_source(obs, option, profile, cards, attacks)
    damage = _effective_attack_damage(obs, option, profile, cards, attacks)
    opponent = _active_pokemon(obs, _opponent_index(obs))
    opponent_id = _card_id(opponent)
    opponent_hp = _remaining_hp(opponent, cards)
    prize_value = _prize_value(opponent_id, cards)

    bonus = min(damage, 260) * 1.1
    if damage > raw_damage:
        bonus += min(220, damage - raw_damage) * 0.6
    if opponent is not None and damage >= opponent_hp > 0:
        bonus += 850 + prize_value * 360
        if _int(_get(_player(obs, _opponent_index(obs)), "handCount", 0), 0) <= 2:
            bonus += 120
    elif opponent_hp > 0:
        pressure_ratio = min(1.0, damage / opponent_hp) if damage > 0 else 0.0
        bonus += pressure_ratio * 260

    your_active = _active_pokemon(obs, _your_index(obs))
    your_hp = _remaining_hp(your_active, cards)
    threat = _estimate_visible_threat_damage(obs, _opponent_index(obs), profile, cards, attacks)
    if threat >= your_hp > 0:
        # When our Active is likely to fall, cash in damage/prizes now instead
        # of taking another setup action that may never pay off.
        bonus += 260 + _prize_value(_card_id(your_active), cards) * 70

    return bonus


def _any_attack_takes_prize(obs: Any, options: list[Any], profile: DeckProfile, cards: dict[int, CardInfo], attacks) -> bool:
    opponent = _active_pokemon(obs, _opponent_index(obs))
    opponent_hp = _remaining_hp(opponent, cards)
    if opponent is None or opponent_hp <= 0:
        return False
    return any(
        _int(_get(option, "type")) == OPT_ATTACK
        and _effective_attack_damage(obs, option, profile, cards, attacks) >= opponent_hp
        for option in options
    )


def _preferred_energy_count(card_id: int | None, cards: dict[int, CardInfo], attacks) -> int:
    if card_id is None:
        return 0

    info = _card_info(card_id, cards)
    if not info.is_pokemon:
        return 0

    desired = 1 if info.basic else 2
    attack_text = ""
    for attack_id in info.attack_ids:
        attack = attacks.get(attack_id)
        if attack is None:
            continue
        desired = max(desired, attack.energy_count)
        attack_text += f" {attack.name} {_get(attack, 'text', '')}"

    text = f"{info.name} {info.effect_text} {attack_text}".lower()
    if "for each energy attached" in text or "scales with attached energy" in text:
        desired = max(desired, 5)
    if "discard 2 energy from this" in text or "discard 2 energy from this pokemon" in text:
        desired = max(desired, 4)
    if "attach up to 2" in text and "energy" in text:
        desired = max(desired, 4)
    if info.ex:
        desired = max(desired, 3)
    if info.mega_ex:
        desired = max(desired, 3)

    return min(desired, 5)


def _energy_gap_for_active(obs: Any, cards: dict[int, CardInfo], attacks) -> tuple[Any, int, int, int]:
    active = _your_active_pokemon(obs)
    active_id = _card_id(active)
    desired = _preferred_energy_count(active_id, cards, attacks)
    current = _attached_energy_count(active)
    return active, current, desired, max(0, desired - current)


def _energy_cost_gap(cost_types: tuple[str, ...], attached_types: list[str]) -> int:
    if not cost_types:
        return 0

    remaining = Counter(attached_types)
    missing_specific = 0
    colorless_needed = 0
    for required in cost_types:
        if required == "colorless":
            colorless_needed += 1
        elif remaining[required] > 0:
            remaining[required] -= 1
        else:
            missing_specific += 1

    spare_attached = sum(remaining.values())
    return missing_specific + max(0, colorless_needed - spare_attached)


def _attack_cost_types(attack: Any) -> tuple[str, ...]:
    energy_types = tuple(_get(attack, "energy_types", ()) or ())
    if energy_types:
        return energy_types
    return ("colorless",) * _int(_get(attack, "energy_count", 0), 0)


def _evolution_targets_for_card(card_id: int | None, profile: DeckProfile, cards: dict[int, CardInfo]) -> tuple[int, ...]:
    info = _card_info(card_id or 0, cards)
    return profile.evolves_from_name.get(info.name, ())


def _preferred_energy_count_for_line(card_id: int | None, profile: DeckProfile, cards: dict[int, CardInfo], attacks) -> int:
    desired = _preferred_energy_count(card_id, cards, attacks)
    for evolution_id in _evolution_targets_for_card(card_id, profile, cards):
        desired = max(desired, _preferred_energy_count(evolution_id, cards, attacks))
    return desired


def _line_attacker_bonus(card_id: int | None, profile: DeckProfile, cards: dict[int, CardInfo]) -> float:
    if card_id is None:
        return 0.0

    bonus = 0.0
    if card_id in profile.attacker_ids[:1]:
        bonus = max(bonus, 280.0)
    elif card_id in profile.attacker_ids[:3]:
        bonus = max(bonus, 100.0)

    for evolution_id in _evolution_targets_for_card(card_id, profile, cards):
        if evolution_id in profile.attacker_ids[:1]:
            bonus = max(bonus, 260.0)
        elif evolution_id in profile.attacker_ids[:3]:
            bonus = max(bonus, 140.0)

    return bonus


def _attack_energy_progress_bonus(
    obs: Any,
    energy_type: str | None,
    target: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> float:
    if target is None or energy_type is None:
        return 0.0

    card_id = _card_id(target)
    info = _card_info(card_id or 0, cards)
    if not info.is_pokemon:
        return 0.0

    current_types = _attached_energy_types(target, cards)
    next_types = current_types + [energy_type]
    best = 0.0
    for attack_id in info.attack_ids:
        attack = attacks.get(attack_id)
        if attack is None:
            continue

        cost_types = _attack_cost_types(attack)
        before_gap = _energy_cost_gap(cost_types, current_types)
        after_gap = _energy_cost_gap(cost_types, next_types)
        if after_gap >= before_gap:
            continue

        damage = _effective_damage_for_attacker(
            obs,
            target,
            _active_pokemon(obs, _opponent_index(obs)),
            profile,
            cards,
            attacks,
            _your_index(obs),
            extra_energy=1,
        )
        if after_gap == 0:
            best = max(best, 360 + min(180, damage * 0.45))
        else:
            best = max(best, 160 + min(100, damage * 0.25))

    return best


def _attachment_progress_bonus(
    obs: Any,
    energy_id: int | None,
    target: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> float:
    energy = _card_info(energy_id or 0, cards)
    target_id = _card_id(target)
    if energy.kind != "basic_energy" or target_id is None:
        return 0.0

    current = _attached_energy_count(target)
    desired = _preferred_energy_count_for_line(target_id, profile, cards, attacks)
    gap = max(0, desired - current)
    progress = _attack_energy_progress_bonus(obs, energy.energy_type, target, profile, cards, attacks)
    line_bonus = _line_attacker_bonus(target_id, profile, cards)

    if gap <= 0:
        return min(progress, 80) - 180

    bonus = progress
    if line_bonus > 0:
        bonus += line_bonus + min(gap, 3) * 55
        if current == 0:
            bonus += 60

    target_info = _card_info(target_id, cards)
    if target_info.energy_type == energy.energy_type:
        bonus += 70
    elif progress <= 0 and energy_id not in profile.main_energy_ids:
        bonus -= 90

    return bonus


def _attachment_target_value(
    obs: Any,
    target: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> float:
    target_id = _card_id(target)
    if target_id is None:
        return 0.0

    current = _attached_energy_count(target)
    desired = _preferred_energy_count_for_line(target_id, profile, cards, attacks)
    gap = max(0, desired - current)
    if gap <= 0:
        return 120 + _pokemon_value(target_id, cards, profile.deck_counts) * 0.08

    candidate_types = {
        _card_info(energy_id, cards).energy_type
        for energy_id in profile.basic_energy_ids
        if _card_info(energy_id, cards).energy_type is not None
    }
    target_info = _card_info(target_id, cards)
    if target_info.energy_type is not None:
        candidate_types.add(target_info.energy_type)

    progress = max(
        (_attack_energy_progress_bonus(obs, energy_type, target, profile, cards, attacks) for energy_type in candidate_types),
        default=0.0,
    )
    line_bonus = _line_attacker_bonus(target_id, profile, cards)

    value = 260 + _pokemon_value(target_id, cards, profile.deck_counts) * 0.18 + min(gap, 3) * 70
    value += progress
    value += line_bonus
    if current == 0 and line_bonus > 0:
        value += 70

    if target is _your_active_pokemon(obs):
        value += 35

    return value


def _energy_accel_available(options: list[Any], obs: Any, cards: dict[int, CardInfo]) -> bool:
    for option in options:
        if _int(_get(option, "type")) != OPT_PLAY:
            continue
        card_id = _resolve_option_card_id(obs, option)
        if "energy_accel" in _role(card_id or 0, _card_info(card_id or 0, cards)):
            return True
    return False


def _boost_build_before_attack(
    obs: Any,
    options: list[Any],
    scored: list[tuple[int, float]],
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> list[tuple[int, float]]:
    attack_scores = [score for idx, score in scored if _int(_get(options[idx], "type")) == OPT_ATTACK]
    if not attack_scores:
        return scored

    active, active_energy, desired_energy, gap = _energy_gap_for_active(obs, cards, attacks)
    active_id = _card_id(active)
    if active_id is None:
        return scored
    if _any_attack_takes_prize(obs, options, profile, cards, attacks):
        return scored
    active_is_threatened = (
        _estimate_visible_threat_damage(obs, _opponent_index(obs), profile, cards, attacks)
        >= _remaining_hp(active, cards)
        > 0
    )
    attack_can_pressure = any(
        _int(_get(option, "type")) == OPT_ATTACK
        and _effective_attack_damage(obs, option, profile, cards, attacks) > 0
        for option in options
    )
    if active_is_threatened and attack_can_pressure:
        return scored

    best_attack_score = max(attack_scores)
    if gap <= 0:
        adjusted = []
        for idx, score in scored:
            option = options[idx]
            option_type = _int(_get(option, "type"))
            card_id = _resolve_option_card_id(obs, option)
            info = _card_info(card_id or 0, cards)
            is_extra_energy_play = option_type == OPT_PLAY and "energy_accel" in _role(card_id or 0, info)
            is_extra_attach = option_type == OPT_ATTACH and _is_active_target(obs, option)
            if is_extra_energy_play or is_extra_attach:
                adjusted.append((idx, min(score, best_attack_score - 140)))
            else:
                adjusted.append((idx, score))
        return adjusted

    state = _state(obs)
    energy_attached = bool(_get(state, "energyAttached", False))
    supporter_played = bool(_get(state, "supporterPlayed", False))
    energy_accel_open = _energy_accel_available(options, obs, cards)
    scaling_attacker = desired_energy >= 5

    adjusted: list[tuple[int, float]] = []
    for idx, score in scored:
        option = options[idx]
        option_type = _int(_get(option, "type"))
        boosted = score

        if option_type == OPT_PLAY:
            card_id = _resolve_option_card_id(obs, option)
            info = _card_info(card_id or 0, cards)
            roles = _role(card_id or 0, info)
            can_play_supporter = info.kind != "supporter" or not supporter_played
            if "energy_accel" in roles and can_play_supporter and gap >= 2:
                # Prefer acceleration before manual attachment when both can
                # still happen this turn; that creates the stack-then-attack line.
                offset = 180 + min(gap, 3) * 55
                if scaling_attacker:
                    offset += 120
                boosted = max(boosted, best_attack_score + offset)

        elif option_type == OPT_ATTACH and not energy_attached and _is_active_target(obs, option):
            card_id = _resolve_option_card_id(obs, option)
            energy = _card_info(card_id or 0, cards)
            active_info = _card_info(active_id, cards)
            if energy.kind == "basic_energy":
                offset = 105 + min(gap, 3) * 45
                if energy.energy_type == active_info.energy_type:
                    offset += 65
                if card_id in profile.main_energy_ids:
                    offset += 30
                if scaling_attacker:
                    offset += 75
                if energy_accel_open and gap >= 2 and not supporter_played:
                    offset -= 120
                boosted = max(boosted, best_attack_score + offset)

        adjusted.append((idx, boosted))

    return adjusted


def _best_attack_damage_from_options(
    obs: Any,
    options: list[Any],
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> int:
    best = 0
    for option in options:
        if _int(_get(option, "type")) == OPT_ATTACK:
            best = max(best, _effective_attack_damage(obs, option, profile, cards, attacks))
    return best


def _choose_tactical_guardrail_action(
    obs: Any,
    options: list[Any],
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> list[int] | None:
    opponent = _active_pokemon(obs, _opponent_index(obs))
    opponent_hp = _remaining_hp(opponent, cards)
    if opponent is None or opponent_hp <= 0:
        return None

    lethal_attacks: list[tuple[int, int, int]] = []
    for idx, option in enumerate(options):
        if _int(_get(option, "type")) != OPT_ATTACK:
            continue
        damage = _effective_attack_damage(obs, option, profile, cards, attacks)
        if damage >= opponent_hp:
            lethal_attacks.append((_prize_value(_card_id(opponent), cards), damage, idx))

    if lethal_attacks:
        _, _, best_idx = max(lethal_attacks)
        return [best_idx]

    return None


def _has_reasonable_retreat_pivot(obs: Any, threat: ThreatMap, cards: dict[int, CardInfo]) -> bool:
    active = _your_active_pokemon(obs)
    active_prizes = _prize_value(_card_id(active), cards)
    active_energy = _attached_energy_count(active)

    for pokemon in _cards_in_area(obs, AREA_BENCH, _your_index(obs)):
        pivot_prizes = _prize_value(_card_id(pokemon), cards)
        pivot_energy = _attached_energy_count(pokemon)
        pivot_hp = _remaining_hp(pokemon, cards)
        if pivot_prizes < active_prizes:
            return True
        if pivot_energy + 1 < active_energy and pivot_hp > threat.active_damage_to_active * 0.55:
            return True
        if pivot_hp > _remaining_hp(active, cards) + 60:
            return True
    return False


def _apply_tactical_score_adjustments(
    obs: Any,
    options: list[Any],
    scored: list[tuple[int, float]],
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
) -> list[tuple[int, float]]:
    active = _your_active_pokemon(obs)
    if active is None:
        return scored

    threat = _opponent_threat_map(obs, profile, cards, attacks)
    if not threat.can_ko_active:
        return scored

    best_attack_damage = _best_attack_damage_from_options(obs, options, profile, cards, attacks)
    opponent = _active_pokemon(obs, _opponent_index(obs))
    opponent_hp = _remaining_hp(opponent, cards)
    active_prizes = _prize_value(_card_id(active), cards)
    active_energy = _attached_energy_count(active)
    can_take_prize = _any_attack_takes_prize(obs, options, profile, cards, attacks)
    should_preserve_active = (
        not can_take_prize
        and _bench_count(obs, _your_index(obs)) > 0
        and _has_reasonable_retreat_pivot(obs, threat, cards)
        and (active_prizes >= 2 or active_energy >= 3)
    )

    adjusted: list[tuple[int, float]] = []
    for idx, score in scored:
        option = options[idx]
        option_type = _int(_get(option, "type"))
        boosted = score

        if option_type == OPT_RETREAT and should_preserve_active:
            pressure_gap = max(0, threat.damage_to_active - _remaining_hp(active, cards))
            boosted += 640 + active_prizes * 140 + active_energy * 55 + min(220, pressure_gap)

        if option_type == OPT_ATTACH and should_preserve_active and _is_active_target(obs, option):
            boosted -= 520 + active_prizes * 90 + active_energy * 35

        if option_type == OPT_PLAY and should_preserve_active:
            card_id = _resolve_option_card_id(obs, option)
            info = _card_info(card_id or 0, cards)
            if "energy_accel" in _role(card_id or 0, info) and best_attack_damage < opponent_hp * 0.55:
                boosted -= 220

        adjusted.append((idx, boosted))

    return adjusted


def _ids_from_pokemon(pokemon: Any) -> list[int]:
    if pokemon is None:
        return []

    ids = []
    card_id = _card_id(pokemon)
    if card_id is not None:
        ids.append(card_id)
    for field in ("energyCards", "tools", "preEvolution"):
        ids.extend(cid for cid in (_card_id(card) for card in _as_list(_get(pokemon, field, []))) if cid is not None)
    return ids


def _known_prize_ids(obs: Any, player_index: int) -> list[int]:
    player = _player(obs, player_index)
    return [cid for cid in (_card_id(card) for card in _as_list(_get(player, "prize", []))) if cid is not None]


def _known_non_hidden_ids(obs: Any, player_index: int, include_hand: bool) -> list[int]:
    # Search needs guesses for hidden zones. These IDs are already visible, so
    # remove them from the guessed hidden deck/prize/hand pools where possible.
    player = _player(obs, player_index)
    ids: list[int] = []

    if include_hand:
        ids.extend(cid for cid in (_card_id(card) for card in _as_list(_get(player, "hand", []))) if cid is not None)
    ids.extend(cid for cid in (_card_id(card) for card in _as_list(_get(player, "discard", []))) if cid is not None)

    for pokemon in _as_list(_get(player, "active", [])) + _as_list(_get(player, "bench", [])):
        ids.extend(_ids_from_pokemon(pokemon))

    state = _state(obs)
    for card in _as_list(_get(state, "stadium", [])):
        if _get(card, "playerIndex") == player_index:
            card_id = _card_id(card)
            if card_id is not None:
                ids.append(card_id)

    for card in _as_list(_get(state, "looking", [])):
        if _get(card, "playerIndex") == player_index:
            card_id = _card_id(card)
            if card_id is not None:
                ids.append(card_id)

    return ids


def _remove_known_cards(base_cards: list[int], known_ids: list[int]) -> list[int]:
    counts = Counter(base_cards)
    for card_id in known_ids:
        if counts[card_id] > 0:
            counts[card_id] -= 1

    remaining: list[int] = []
    for card_id in base_cards:
        if counts[card_id] > 0:
            remaining.append(card_id)
            counts[card_id] -= 1
    return remaining


def _hidden_card_priority(card_id: int, profile: DeckProfile, cards: dict[int, CardInfo]) -> float:
    info = _card_info(card_id, cards)
    if card_id in profile.evolution_ids:
        return 500 + _pokemon_value(card_id, cards, profile.deck_counts)
    if card_id in profile.attacker_ids[:4]:
        return 420 + _pokemon_value(card_id, cards, profile.deck_counts)
    if "search" in _role(card_id, info):
        return 360
    if "draw" in _role(card_id, info):
        return 320
    if card_id in profile.main_energy_ids:
        return 180
    if card_id in profile.basic_energy_ids:
        return 140
    return 80


def _ordered_hidden_pool(pool: list[int], profile: DeckProfile, cards: dict[int, CardInfo]) -> list[int]:
    # Stable ordering matters because the simulator will use this guess for
    # draws. Keep it deterministic and mildly optimistic, not random.
    return sorted(
        pool,
        key=lambda card_id: (_hidden_card_priority(card_id, profile, cards), -card_id),
        reverse=True,
    )


def _fit_hidden_zone(known_ids: list[int], pool: list[int], count: int, fallback: list[int]) -> tuple[list[int], list[int]]:
    result = known_ids[:count]
    remaining = list(pool)

    for card_id in result:
        if card_id in remaining:
            remaining.remove(card_id)

    for card_id in list(remaining):
        if len(result) >= count:
            break
        result.append(card_id)
        remaining.remove(card_id)

    fallback_idx = 0
    while len(result) < count and fallback:
        result.append(fallback[fallback_idx % len(fallback)])
        fallback_idx += 1

    return result, remaining


def _basic_fallback(profile: DeckProfile, deck: list[int], cards: dict[int, CardInfo]) -> list[int]:
    basics = sorted(
        profile.basic_ids,
        key=lambda card_id: _starter_value(card_id, cards, profile.deck_counts),
        reverse=True,
    )
    if basics:
        return [basics[0]]
    return deck[:1]


def _predict_hidden_zones(
    obs: Any,
    deck: list[int],
    profile: DeckProfile,
    cards: dict[int, CardInfo],
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    state = _state(obs)
    your_index = _your_index(obs)
    opponent_index = 1 - your_index
    your_player = _player(obs, your_index)
    opponent_player = _player(obs, opponent_index)

    your_deck_count = _int(_get(your_player, "deckCount", 0), 0)
    your_prize_count = len(_as_list(_get(your_player, "prize", [])))
    opponent_deck_count = _int(_get(opponent_player, "deckCount", 0), 0)
    opponent_prize_count = len(_as_list(_get(opponent_player, "prize", [])))
    opponent_hand_count = _int(_get(opponent_player, "handCount", 0), 0)

    fallback = deck or list(profile.deck_counts.elements())
    your_pool = _remove_known_cards(deck, _known_non_hidden_ids(obs, your_index, include_hand=True))
    your_pool = _ordered_hidden_pool(your_pool, profile, cards)
    your_prize, your_pool = _fit_hidden_zone(_known_prize_ids(obs, your_index), your_pool, your_prize_count, fallback)
    your_deck, _ = _fit_hidden_zone([], your_pool, your_deck_count, fallback)

    opponent_pool = _remove_known_cards(deck, _known_non_hidden_ids(obs, opponent_index, include_hand=False))
    opponent_pool = _ordered_hidden_pool(opponent_pool, profile, cards)
    opponent_prize, opponent_pool = _fit_hidden_zone(
        _known_prize_ids(obs, opponent_index),
        opponent_pool,
        opponent_prize_count,
        fallback,
    )
    opponent_hand, opponent_pool = _fit_hidden_zone([], opponent_pool, opponent_hand_count, fallback)
    opponent_deck, _ = _fit_hidden_zone([], opponent_pool, opponent_deck_count, fallback)

    opponent_active = []
    active = _as_list(_get(opponent_player, "active", []))
    if active and active[0] is None:
        opponent_active = _basic_fallback(profile, deck, cards)

    # Some early setup states can have a missing player array. Keep the search
    # call valid by returning correctly typed empty guesses in that case.
    if state is None:
        return [], [], [], [], [], []

    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active


def _pokemon_state_value(pokemon: Any, profile: DeckProfile, cards: dict[int, CardInfo]) -> float:
    card_id = _card_id(pokemon)
    if card_id is None:
        return 0.0

    info = _card_info(card_id, cards)
    current_hp = _int(_get(pokemon, "hp", info.hp), info.hp)
    max_hp = max(current_hp, _int(_get(pokemon, "maxHp", info.hp), info.hp))
    damage = max(0, max_hp - current_hp)

    value = _pokemon_value(card_id, cards, profile.deck_counts) * 0.55
    value += current_hp * 1.15
    value -= damage * 0.35
    value += _attached_energy_count(pokemon) * 48
    value += len(_as_list(_get(pokemon, "tools", []))) * 24
    if info.ex:
        value += 30
    if info.mega_ex:
        value += 45
    return value


def _player_state_value(obs: Any, player_index: int, profile: DeckProfile, cards: dict[int, CardInfo]) -> float:
    player = _player(obs, player_index)
    active = _as_list(_get(player, "active", []))
    bench = _as_list(_get(player, "bench", []))

    score = 0.0
    if not active:
        score -= 8000
    for pokemon in active:
        score += _pokemon_state_value(pokemon, profile, cards) * 1.25
    for pokemon in bench:
        score += _pokemon_state_value(pokemon, profile, cards) * 0.82

    hand_count = _int(_get(player, "handCount", len(_as_list(_get(player, "hand", [])))), 0)
    deck_count = _int(_get(player, "deckCount", 0), 0)
    score += min(hand_count, 9) * 18
    if deck_count <= 2:
        score -= (3 - deck_count) * 350

    if _get(player, "asleep", False) or _get(player, "paralyzed", False):
        score -= 160
    if _get(player, "confused", False):
        score -= 70
    if _get(player, "poisoned", False) or _get(player, "burned", False):
        score -= 45

    return score


def _evaluate_observation(
    obs: Any,
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
    deck: list[int] | None = None,
    root_your_index: int | None = None,
) -> float:
    state = _state(obs)
    if state is None:
        return 0.0

    your_index = _your_index(obs) if root_your_index is None else root_your_index
    opponent_index = 1 - your_index
    result = _int(_get(state, "result", -1), -1)
    if result == your_index:
        return 500000.0
    if result == opponent_index:
        return -500000.0
    if result == 2:
        return 0.0

    your_player = _player(obs, your_index)
    opponent_player = _player(obs, opponent_index)
    your_prizes = len(_as_list(_get(your_player, "prize", [])))
    opponent_prizes = len(_as_list(_get(opponent_player, "prize", [])))

    score = (opponent_prizes - your_prizes) * 950
    score += _player_state_value(obs, your_index, profile, cards)
    score -= _player_state_value(obs, opponent_index, profile, cards) * 0.92

    select = _select(obs)
    options = _as_list(_get(select, "option", []))
    if options and _int(_get(state, "yourIndex", your_index), your_index) == your_index:
        best_next = max((_score_option(obs, option, profile, cards, attacks, deck) for option in options), default=0)
        score += max(-200, min(best_next, 900)) * 0.16

    return score


def _rollout_forced_search_steps(api: Any, search_state: Any, deck: list[int], deadline: float) -> Any:
    current = search_state
    steps = 0

    while steps < SEARCH_MAX_FORCED_STEPS and monotonic() < deadline:
        obs = _get(current, "observation")
        state = _state(obs)
        select = _select(obs)
        if select is None or _int(_get(state, "result", -1), -1) != -1:
            break
        if _int(_get(select, "type")) == 0:
            break

        action = choose_action(obs, deck, use_search=False)
        if not action:
            break
        current = api.search_step(_get(current, "searchId"), action)
        steps += 1

    return current


def _choose_with_simulator_search(
    obs: Any,
    deck: list[int],
    profile: DeckProfile,
    cards: dict[int, CardInfo],
    attacks,
    ranked_scores: list[tuple[int, float]],
) -> list[int] | None:
    # The official search API is available in the Kaggle/Linux simulator but
    # not in local macOS tests. This wrapper is intentionally fail-closed.
    if _get(obs, "search_begin_input") is None:
        return None

    deadline = monotonic() + SEARCH_TIME_LIMIT_SECONDS
    options = _as_list(_get(_select(obs), "option", []))
    if not options:
        return None

    candidate_indexes: list[int] = []
    for idx, score in ranked_scores:
        if score > -500 and idx not in candidate_indexes:
            candidate_indexes.append(idx)
        if len(candidate_indexes) >= SEARCH_MAX_ROOT_OPTIONS:
            break

    for idx, option in enumerate(options):
        if _int(_get(option, "type")) == OPT_ATTACK and idx not in candidate_indexes:
            candidate_indexes.append(idx)

    if not candidate_indexes:
        return None

    try:
        from cg import api

        agent_observation = api.to_observation_class(obs) if isinstance(obs, dict) else obs
        hidden = _predict_hidden_zones(obs, deck, profile, cards)
        root = api.search_begin(agent_observation, *hidden, manual_coin=False)

        heuristic = dict(ranked_scores)
        best_idx = ranked_scores[0][0]
        best_value = -float("inf")
        heuristic_best_value = -float("inf")

        for idx in candidate_indexes:
            if monotonic() >= deadline:
                break
            child = api.search_step(_get(root, "searchId"), [idx])
            final_state = _rollout_forced_search_steps(api, child, deck, deadline)
            value = _evaluate_observation(
                _get(final_state, "observation"),
                profile,
                cards,
                attacks,
                deck=deck,
                root_your_index=_your_index(obs),
            )
            value += heuristic.get(idx, 0) * 0.08
            if idx == ranked_scores[0][0]:
                heuristic_best_value = value
            if value > best_value:
                best_value = value
                best_idx = idx

        api.search_end()

        if best_idx != ranked_scores[0][0] and best_value < heuristic_best_value + SEARCH_OVERRIDE_MARGIN:
            return None
        return [best_idx]
    except Exception:
        try:
            api.search_end()
        except Exception:
            pass
        return None


def _choose_yes_no(obs: Any, options: list[Any]) -> list[int]:
    # Default yes for proactive effects, but do not voluntarily mulligan.
    context = _int(_get(_select(obs), "context"))
    prefer_yes = context in {CTX_IS_FIRST, CTX_ACTIVATE}
    if context == CTX_MULLIGAN:
        prefer_yes = False

    preferred = OPT_YES if prefer_yes else OPT_NO
    fallback = OPT_NO if prefer_yes else OPT_YES
    for option_type in (preferred, fallback):
        for idx, option in enumerate(options):
            if _int(_get(option, "type")) == option_type:
                return [idx]
    return [0] if options else []


def _score_count_choice(obs: Any, option: Any) -> int:
    number = _int(_get(option, "number"), 0)
    context = _int(_get(_select(obs), "context"))
    if context != CTX_DRAW_COUNT:
        return number

    deck_count = _int(_get(_player(obs), "deckCount", 0), 0)
    if number > deck_count:
        return -10000 - number
    return number


def choose_action(obs: Any, deck: list[int] | None = None, use_search: bool = True) -> list[int]:
    """Return legal option indexes for the current observation."""

    if deck is None:
        deck = read_deck_csv()

    select = _select(obs)
    if select is None:
        return deck

    options = _as_list(_get(select, "option", []))
    if not options:
        return []

    min_count = _int(_get(select, "minCount", 1), 1)
    max_count = _int(_get(select, "maxCount", 1), 1)
    select_type = _int(_get(select, "type"))
    context = _int(_get(select, "context"))

    if select_type == SEL_YES_NO:
        return _choose_yes_no(obs, options)

    cards, attacks = load_card_database()
    profile = _profile(deck)

    if select_type == SEL_COUNT or context == CTX_DRAW_COUNT:
        # Count prompts usually ask how much of a beneficial effect to take.
        ranked = sorted(
            range(len(options)),
            key=lambda idx: (_score_count_choice(obs, options[idx]), -idx),
            reverse=True,
        )
        return ranked[: max(1, min_count)]

    if context == CTX_DISCARD:
        # Discard prompts are inverted: choose the lowest-value cards first.
        ranked = sorted(
            range(len(options)),
            key=lambda idx: (_discard_penalty(obs, _resolve_option_card_id(obs, options[idx]), profile, cards), idx),
        )
        take = max(min_count, min(max_count, min_count))
        return ranked[:take]

    ranked_scores = [
        (idx, _score_option(obs, option, profile, cards, attacks, deck))
        for idx, option in enumerate(options)
    ]
    if select_type == 0 and min_count == 1 and max_count == 1:
        ranked_scores = _boost_build_before_attack(obs, options, ranked_scores, profile, cards, attacks)
        ranked_scores = _apply_tactical_score_adjustments(obs, options, ranked_scores, profile, cards, attacks)
    ranked_scores.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    if select_type == 0 and min_count == 1 and max_count == 1:
        guardrail_action = _choose_tactical_guardrail_action(obs, options, profile, cards, attacks)
        if guardrail_action is not None:
            return guardrail_action

    if use_search and select_type == 0 and min_count == 1 and max_count == 1:
        searched = _choose_with_simulator_search(obs, deck, profile, cards, attacks, ranked_scores)
        if searched is not None:
            return searched

    selected = [idx for idx, score in ranked_scores if score > 0][:max_count]
    if len(selected) < min_count:
        selected = [idx for idx, _ in ranked_scores[:min_count]]

    return selected[:max_count]
