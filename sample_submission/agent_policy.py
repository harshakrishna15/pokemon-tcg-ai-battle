"""Deck-profile-driven action policy.

The policy avoids hardcoding one exact deck sequence. It infers basic roles from
deck.csv and card facts, then scores the legal options that the simulator sends.
This keeps the agent usable when the deck changes, while still allowing card
facts to improve the scoring when the simulator database is available.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
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
    pokemon_ids: frozenset[int]
    basic_ids: frozenset[int]
    evolution_ids: frozenset[int]
    attacker_ids: tuple[int, ...]
    evolves_from_name: dict[str, tuple[int, ...]]


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
    if card_id in {1145, 1121}:
        roles.add("search")
    if card_id in {1227}:
        roles.add("draw")
    if card_id in {1235}:
        roles.update({"draw", "energy_accel"})
    return roles


def _score_play_card(obs: Any, card_id: int | None, profile: DeckProfile, cards: dict[int, CardInfo]) -> float:
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
        score = max(score, 690)
    if "search" in roles:
        score = max(score, 720 if _missing_evolution_piece(obs, profile) else 540)
        if _has_evolution_source_in_play(obs, profile, cards):
            score += 60
    if "draw" in roles:
        if hand_count <= 3:
            score = max(score, 650)
        elif hand_count <= 6:
            score = max(score, 510)
        else:
            score = max(score, 260)
    if "damage_boost" in roles:
        score = max(score, 520)
    if "gust" in roles:
        score = max(score, 330)
    if "switch" in roles:
        score = max(score, 230)

    if info.kind == "supporter":
        score -= 25
    return score


def _score_card_pick(obs: Any, card_id: int | None, profile: DeckProfile, cards: dict[int, CardInfo]) -> float:
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
        return 720 + _pokemon_value(card_id, cards, profile.deck_counts)
    if info.is_pokemon:
        return 500 + _pokemon_value(card_id, cards, profile.deck_counts)
    if "search" in _role(card_id, info):
        return 430
    if "draw" in _role(card_id, info):
        return 390
    if card_id in profile.main_energy_ids:
        return 250
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
        penalty = 5
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


def _score_option(obs: Any, option: Any, profile: DeckProfile, cards: dict[int, CardInfo], attacks) -> float:
    # Convert each legal option into a single comparable score. The simulator
    # still enforces legality; this only chooses among legal indexes.
    option_type = _int(_get(option, "type"))
    card_id = _resolve_option_card_id(obs, option)

    if option_type == OPT_EVOLVE:
        return 920 + _score_card_pick(obs, card_id, profile, cards)
    if option_type == OPT_ATTACH:
        target_id = _resolve_target_pokemon_id(obs, option)
        energy_bonus = 90 if card_id in profile.main_energy_ids else 0
        return 760 + energy_bonus + _pokemon_value(target_id or 0, cards, profile.deck_counts) * 0.3
    if option_type == OPT_PLAY:
        return _score_play_card(obs, card_id, profile, cards)
    if option_type == OPT_ABILITY:
        return 610 + _pokemon_value(card_id or 0, cards, profile.deck_counts) * 0.15
    if option_type == OPT_ATTACK:
        attack_id = _int(_get(option, "attackId"), -1)
        attack = attacks.get(attack_id)
        return 470 + (attack.damage if attack else 80)
    if option_type == OPT_RETREAT:
        return 210
    if option_type == OPT_CARD:
        return _score_card_pick(obs, card_id, profile, cards)
    if option_type == OPT_NUMBER:
        return _int(_get(option, "number"), 0)
    if option_type == OPT_YES:
        return 20
    if option_type == OPT_NO:
        return 10
    if option_type == OPT_END:
        return -1000
    return 0


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


def choose_action(obs: Any, deck: list[int] | None = None) -> list[int]:
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
            key=lambda idx: _int(_get(options[idx], "number"), idx),
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
        (idx, _score_option(obs, option, profile, cards, attacks))
        for idx, option in enumerate(options)
    ]
    ranked_scores.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    selected = [idx for idx, score in ranked_scores if score > 0][:max_count]
    if len(selected) < min_count:
        selected = [idx for idx, _ in ranked_scores[:min_count]]

    return selected[:max_count]
