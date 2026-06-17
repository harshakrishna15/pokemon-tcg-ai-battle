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
        progress_bonus = 0
        state = _state(obs)
        if (
            _int(_get(state, "turnActionCount", 0)) > 0
            or bool(_get(state, "energyAttached", False))
            or bool(_get(state, "supporterPlayed", False))
            or bool(_get(state, "stadiumPlayed", False))
        ):
            progress_bonus = 650
        return 820 + progress_bonus + (attack.damage if attack else 80)
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

    # Static fallback for the submitted deck when the native simulator metadata
    # is unavailable locally.
    if card_id == 154:
        desired = max(desired, 5)
    elif card_id == 944:
        desired = max(desired, 4)
    elif card_id in {721, 723}:
        desired = max(desired, 3)

    return min(desired, 5)


def _energy_gap_for_active(obs: Any, cards: dict[int, CardInfo], attacks) -> tuple[Any, int, int, int]:
    active = _your_active_pokemon(obs)
    active_id = _card_id(active)
    desired = _preferred_energy_count(active_id, cards, attacks)
    current = _attached_energy_count(active)
    return active, current, desired, max(0, desired - current)


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
            if card_id in profile.main_energy_ids:
                offset = 130 + min(gap, 3) * 45
                if scaling_attacker:
                    offset += 75
                if energy_accel_open and gap >= 2 and not supporter_played:
                    offset -= 120
                boosted = max(boosted, best_attack_score + offset)

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
        best_next = max((_score_option(obs, option, profile, cards, attacks) for option in options), default=0)
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
    if select_type == 0 and min_count == 1 and max_count == 1:
        ranked_scores = _boost_build_before_attack(obs, options, ranked_scores, profile, cards, attacks)
    ranked_scores.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    if use_search and select_type == 0 and min_count == 1 and max_count == 1:
        searched = _choose_with_simulator_search(obs, deck, profile, cards, attacks, ranked_scores)
        if searched is not None:
            return searched

    selected = [idx for idx, score in ranked_scores if score > 0][:max_count]
    if len(selected) < min_count:
        selected = [idx for idx, _ in ranked_scores[:min_count]]

    return selected[:max_count]
