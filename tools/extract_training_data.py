#!/usr/bin/env python3
"""Extract option-level training data from local CG simulator games.

The output is a fixed-schema CSV, optionally gzip-compressed by using a
``.gz`` suffix. Each row represents one legal option at one decision point.
Labels are attached after the game finishes, so rows can be used for
option-ranking, imitation, or value-model experiments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "sample_submission"

sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUBMISSION))

import agent_policy as ap  # noqa: E402
from run_cg_match import POLICIES, read_deck  # noqa: E402
from cg import game  # noqa: E402


ENERGY_TYPE_CODE = {
    None: 0,
    "colorless": 1,
    "grass": 2,
    "fire": 3,
    "water": 4,
    "lightning": 5,
    "psychic": 6,
    "fighting": 7,
    "darkness": 8,
    "metal": 9,
    "dragon": 10,
    "rainbow": 11,
}

CARD_KIND_CODE = {
    "unknown": 0,
    "pokemon": 1,
    "item": 2,
    "tool": 3,
    "supporter": 4,
    "stadium": 5,
    "basic_energy": 6,
    "special_energy": 7,
}


FEATURE_COLUMNS = [
    "game",
    "step",
    "turn",
    "turn_action_count",
    "decision_player",
    "decision_policy_id",
    "opponent_policy_id",
    "option_index",
    "selected",
    "selected_rank",
    "selection_size",
    "select_type",
    "select_context",
    "select_min_count",
    "select_max_count",
    "option_type",
    "option_area",
    "option_index_in_area",
    "option_player_index",
    "option_in_play_area",
    "option_in_play_index",
    "option_number",
    "option_attack_id",
    "option_card_id",
    "option_card_kind",
    "option_card_energy_type",
    "option_card_is_pokemon",
    "option_card_is_energy",
    "option_card_is_trainer",
    "option_card_is_basic",
    "option_card_is_ex",
    "option_card_is_mega_ex",
    "role_search",
    "role_draw",
    "role_energy_accel",
    "role_gust",
    "role_switch",
    "role_damage_boost",
    "target_card_id",
    "target_card_kind",
    "target_card_energy_type",
    "target_hp",
    "target_energy_count",
    "target_prize_value",
    "target_is_active",
    "target_is_bench",
    "energy_matches_target",
    "heuristic_score",
    "estimated_attack_damage",
    "effective_attack_damage",
    "attack_takes_prize",
    "attachment_progress_bonus",
    "attachment_target_value",
    "your_prizes_remaining",
    "opp_prizes_remaining",
    "prize_delta",
    "your_deck_count",
    "opp_deck_count",
    "your_hand_count",
    "opp_hand_count",
    "your_bench_count",
    "opp_bench_count",
    "your_active_id",
    "your_active_hp",
    "your_active_energy",
    "your_active_prize_value",
    "opp_active_id",
    "opp_active_hp",
    "opp_active_energy",
    "opp_active_prize_value",
    "opponent_threat_damage",
    "opponent_can_ko_active",
    "final_winner",
    "outcome_for_player",
    "game_steps",
]


@dataclass(frozen=True)
class ExtractConfig:
    games: int
    max_steps: int
    seed: int
    p0: str
    p1: str
    seat_swap: bool
    deck0: str
    deck1: str
    output: str
    keep_unfinished: bool


def _bool(value: bool) -> int:
    return 1 if value else 0


def _energy_code(value: str | None) -> int:
    return ENERGY_TYPE_CODE.get(value, 0)


def _kind_code(value: str) -> int:
    return CARD_KIND_CODE.get(value, 0)


def _player_counts(obs: Any, player_index: int) -> tuple[int, int, int, int]:
    player = ap._player(obs, player_index)
    prizes = len(ap._as_list(ap._get(player, "prize", [])))
    deck_count = ap._int(ap._get(player, "deckCount", 0), 0)
    hand_count = ap._int(ap._get(player, "handCount", len(ap._as_list(ap._get(player, "hand", [])))), 0)
    bench_count = len(ap._as_list(ap._get(player, "bench", [])))
    return prizes, deck_count, hand_count, bench_count


def _active_summary(obs: Any, player_index: int, cards: dict[int, ap.CardInfo]) -> tuple[int, int, int, int]:
    active = ap._active_pokemon(obs, player_index)
    active_id = ap._card_id(active) or 0
    return (
        active_id,
        ap._remaining_hp(active, cards) if active is not None else 0,
        ap._attached_energy_count(active) if active is not None else 0,
        ap._prize_value(active_id, cards) if active is not None else 0,
    )


def _selected_rank(action: list[int], option_index: int) -> int:
    try:
        return action.index(option_index)
    except ValueError:
        return -1


def _result_for_player(final_winner: int, player_index: int) -> float:
    if final_winner == player_index:
        return 1.0
    if final_winner == 2:
        return 0.5
    if final_winner in {0, 1}:
        return 0.0
    return 0.5


def _option_damage_features(
    obs: Any,
    option: Any,
    profile: ap.DeckProfile,
    cards: dict[int, ap.CardInfo],
    attacks,
) -> tuple[int, int, int]:
    if ap._int(ap._get(option, "type")) != ap.OPT_ATTACK:
        return 0, 0, 0

    estimated_damage, _ = ap._estimate_attack_damage_and_source(obs, option, profile, cards, attacks)
    effective_damage = ap._effective_attack_damage(obs, option, profile, cards, attacks)
    opponent = ap._active_pokemon(obs, ap._opponent_index(obs))
    opponent_hp = ap._remaining_hp(opponent, cards)
    takes_prize = effective_damage >= opponent_hp > 0
    return int(estimated_damage), int(effective_damage), _bool(takes_prize)


def option_feature_row(
    *,
    game_index: int,
    step: int,
    obs: Any,
    deck: list[int],
    policy_id: int,
    opponent_policy_id: int,
    option_index: int,
    option: Any,
    action: list[int],
) -> dict[str, Any]:
    cards, attacks = ap.load_card_database()
    profile = ap._profile(deck)

    state = ap._state(obs)
    select = ap._select(obs)
    player_index = ap._your_index(obs)
    opponent_index = ap._opponent_index(obs)
    selected_rank = _selected_rank(action, option_index)

    option_type = ap._int(ap._get(option, "type"))
    option_card_id = ap._resolve_option_card_id(obs, option) or 0
    option_card = ap._card_info(option_card_id, cards)
    roles = ap._role(option_card_id, option_card)
    target = ap._resolve_target_pokemon(obs, option)
    target_id = ap._card_id(target) or ap._resolve_target_pokemon_id(obs, option) or 0
    target_card = ap._card_info(target_id, cards)

    heuristic_score = ap._score_option(obs, option, profile, cards, attacks, deck)
    estimated_damage, effective_damage, takes_prize = _option_damage_features(obs, option, profile, cards, attacks)
    threat = ap._opponent_threat_map(obs, profile, cards, attacks)

    attachment_progress = 0.0
    if option_type == ap.OPT_ATTACH:
        attachment_progress = ap._attachment_progress_bonus(obs, option_card_id, target, profile, cards, attacks)
    attachment_target_value = 0.0
    if option_type == ap.OPT_CARD and ap._int(ap._get(select, "context")) == ap.CTX_ATTACH_TO:
        attachment_target_value = ap._attachment_target_value(obs, target, profile, cards, attacks)

    your_prizes, your_deck_count, your_hand_count, your_bench_count = _player_counts(obs, player_index)
    opp_prizes, opp_deck_count, opp_hand_count, opp_bench_count = _player_counts(obs, opponent_index)
    your_active_id, your_active_hp, your_active_energy, your_active_prize_value = _active_summary(obs, player_index, cards)
    opp_active_id, opp_active_hp, opp_active_energy, opp_active_prize_value = _active_summary(obs, opponent_index, cards)

    target_energy = target_card.energy_type
    energy_matches_target = option_card.kind == "basic_energy" and option_card.energy_type == target_energy

    return {
        "game": game_index,
        "step": step,
        "turn": ap._int(ap._get(state, "turn", 0), 0),
        "turn_action_count": ap._int(ap._get(state, "turnActionCount", 0), 0),
        "decision_player": player_index,
        "decision_policy_id": policy_id,
        "opponent_policy_id": opponent_policy_id,
        "option_index": option_index,
        "selected": _bool(selected_rank >= 0),
        "selected_rank": selected_rank,
        "selection_size": len(action),
        "select_type": ap._int(ap._get(select, "type", 0), 0),
        "select_context": ap._int(ap._get(select, "context", 0), 0),
        "select_min_count": ap._int(ap._get(select, "minCount", 0), 0),
        "select_max_count": ap._int(ap._get(select, "maxCount", 0), 0),
        "option_type": option_type,
        "option_area": ap._int(ap._get(option, "area", 0), 0),
        "option_index_in_area": ap._int(ap._get(option, "index", -1), -1),
        "option_player_index": ap._int(ap._get(option, "playerIndex", player_index), player_index),
        "option_in_play_area": ap._int(ap._get(option, "inPlayArea", 0), 0),
        "option_in_play_index": ap._int(ap._get(option, "inPlayIndex", -1), -1),
        "option_number": ap._int(ap._get(option, "number", 0), 0),
        "option_attack_id": ap._int(ap._get(option, "attackId", 0), 0),
        "option_card_id": option_card_id,
        "option_card_kind": _kind_code(option_card.kind),
        "option_card_energy_type": _energy_code(option_card.energy_type),
        "option_card_is_pokemon": _bool(option_card.is_pokemon),
        "option_card_is_energy": _bool(option_card.is_energy),
        "option_card_is_trainer": _bool(option_card.is_trainer),
        "option_card_is_basic": _bool(option_card.basic),
        "option_card_is_ex": _bool(option_card.ex),
        "option_card_is_mega_ex": _bool(option_card.mega_ex),
        "role_search": _bool("search" in roles),
        "role_draw": _bool("draw" in roles),
        "role_energy_accel": _bool("energy_accel" in roles),
        "role_gust": _bool("gust" in roles),
        "role_switch": _bool("switch" in roles),
        "role_damage_boost": _bool("damage_boost" in roles),
        "target_card_id": target_id,
        "target_card_kind": _kind_code(target_card.kind),
        "target_card_energy_type": _energy_code(target_energy),
        "target_hp": ap._remaining_hp(target, cards) if target is not None else 0,
        "target_energy_count": ap._attached_energy_count(target) if target is not None else 0,
        "target_prize_value": ap._prize_value(target_id, cards) if target_id else 0,
        "target_is_active": _bool(ap._get(option, "inPlayArea") == ap.AREA_ACTIVE),
        "target_is_bench": _bool(ap._get(option, "inPlayArea") == ap.AREA_BENCH),
        "energy_matches_target": _bool(energy_matches_target),
        "heuristic_score": round(float(heuristic_score), 4),
        "estimated_attack_damage": estimated_damage,
        "effective_attack_damage": effective_damage,
        "attack_takes_prize": takes_prize,
        "attachment_progress_bonus": round(float(attachment_progress), 4),
        "attachment_target_value": round(float(attachment_target_value), 4),
        "your_prizes_remaining": your_prizes,
        "opp_prizes_remaining": opp_prizes,
        "prize_delta": opp_prizes - your_prizes,
        "your_deck_count": your_deck_count,
        "opp_deck_count": opp_deck_count,
        "your_hand_count": your_hand_count,
        "opp_hand_count": opp_hand_count,
        "your_bench_count": your_bench_count,
        "opp_bench_count": opp_bench_count,
        "your_active_id": your_active_id,
        "your_active_hp": your_active_hp,
        "your_active_energy": your_active_energy,
        "your_active_prize_value": your_active_prize_value,
        "opp_active_id": opp_active_id,
        "opp_active_hp": opp_active_hp,
        "opp_active_energy": opp_active_energy,
        "opp_active_prize_value": opp_active_prize_value,
        "opponent_threat_damage": int(threat.damage_to_active),
        "opponent_can_ko_active": _bool(threat.can_ko_active),
        "final_winner": -1,
        "outcome_for_player": 0.5,
        "game_steps": 0,
    }


def _open_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def _metadata_path(output: Path) -> Path:
    if output.suffix == ".gz":
        return output.with_suffix("").with_suffix(output.with_suffix("").suffix + ".meta.json")
    return output.with_suffix(output.suffix + ".meta.json")


def _write_metadata(path: Path, config: ExtractConfig, policy_names: list[str]) -> None:
    metadata = {
        "schema_version": 1,
        "row_grain": "one row per legal option at a decision point",
        "columns": FEATURE_COLUMNS,
        "label_columns": ["selected", "selected_rank", "final_winner", "outcome_for_player"],
        "policy_id_to_name": {idx: name for idx, name in enumerate(policy_names)},
        "energy_type_code": {str(key): value for key, value in ENERGY_TYPE_CODE.items()},
        "card_kind_code": CARD_KIND_CODE,
        "config": asdict(config),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _finalize_rows(rows: list[dict[str, Any]], final_winner: int, game_steps: int) -> None:
    for row in rows:
        row["final_winner"] = final_winner
        row["outcome_for_player"] = _result_for_player(final_winner, row["decision_player"])
        row["game_steps"] = game_steps


def _extract_game_rows(
    *,
    game_index: int,
    deck0: list[int],
    deck1: list[int],
    policy0_name: str,
    policy1_name: str,
    policy_ids: dict[str, int],
    rng: random.Random,
    max_steps: int,
) -> tuple[list[dict[str, Any]], int, int, str | None]:
    obs = None
    rows: list[dict[str, Any]] = []

    try:
        obs, start = game.battle_start(deck0, deck1)
        if obs is None:
            return [], -1, 0, f"battle_start failed: errorPlayer={start.errorPlayer}, errorType={start.errorType}"

        steps = 0
        while steps < max_steps:
            result = ap._int(ap._get(ap._state(obs), "result", -1), -1)
            if result != -1:
                _finalize_rows(rows, result, steps)
                return rows, result, steps, None

            player = ap._your_index(obs)
            policy_name = policy0_name if player == 0 else policy1_name
            opponent_policy_name = policy1_name if player == 0 else policy0_name
            policy = POLICIES[policy_name]
            deck = deck0 if player == 0 else deck1
            action = policy(obs, deck, rng)

            select = ap._select(obs)
            options = ap._as_list(ap._get(select, "option", []))
            for option_index, option in enumerate(options):
                rows.append(
                    option_feature_row(
                        game_index=game_index,
                        step=steps,
                        obs=obs,
                        deck=deck,
                        policy_id=policy_ids[policy_name],
                        opponent_policy_id=policy_ids[opponent_policy_name],
                        option_index=option_index,
                        option=option,
                        action=action,
                    )
                )

            try:
                obs = game.battle_select(action)
            except Exception as exc:
                winner = 1 - player
                _finalize_rows(rows, winner, steps)
                return rows, winner, steps, f"{type(exc).__name__}: player={player}, action={action}, message={exc}"
            steps += 1

        _finalize_rows(rows, -1, max_steps)
        return rows, -1, max_steps, "max steps reached"
    finally:
        if obs is not None:
            game.battle_finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract option-level training data from local simulator games.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--deck0", type=Path, default=SUBMISSION / "deck.csv")
    parser.add_argument("--deck1", type=Path, default=SUBMISSION / "deck.csv")
    parser.add_argument("--p0", choices=sorted(POLICIES), default="agent-no-search")
    parser.add_argument("--p1", choices=sorted(POLICIES), default="first")
    parser.add_argument("--seat-swap", action="store_true", help="Alternate p0/p1 policies and decks between seats.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "training_data" / "option_features.csv.gz")
    parser.add_argument("--keep-unfinished", action="store_true", help="Keep rows from max-step or errored games.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deck0 = read_deck(args.deck0)
    deck1 = read_deck(args.deck1)
    rng = random.Random(args.seed)
    policy_names = sorted(POLICIES)
    policy_ids = {name: idx for idx, name in enumerate(policy_names)}

    config = ExtractConfig(
        games=args.games,
        max_steps=args.max_steps,
        seed=args.seed,
        p0=args.p0,
        p1=args.p1,
        seat_swap=args.seat_swap,
        deck0=str(args.deck0),
        deck1=str(args.deck1),
        output=str(args.output),
        keep_unfinished=args.keep_unfinished,
    )

    metadata_path = _metadata_path(args.output)
    _write_metadata(metadata_path, config, policy_names)

    total_rows = 0
    finished_games = 0
    skipped_games = 0
    errors = 0
    wins0 = 0
    wins1 = 0
    draws = 0

    with _open_output(args.output) as file:
        writer = csv.DictWriter(file, fieldnames=FEATURE_COLUMNS, extrasaction="raise")
        writer.writeheader()

        for game_index in range(1, args.games + 1):
            if args.seat_swap and game_index % 2 == 0:
                deck0_for_game = deck1
                deck1_for_game = deck0
                policy0_for_game = args.p1
                policy1_for_game = args.p0
            else:
                deck0_for_game = deck0
                deck1_for_game = deck1
                policy0_for_game = args.p0
                policy1_for_game = args.p1

            rows, winner, steps, error = _extract_game_rows(
                game_index=game_index,
                deck0=deck0_for_game,
                deck1=deck1_for_game,
                policy0_name=policy0_for_game,
                policy1_name=policy1_for_game,
                policy_ids=policy_ids,
                rng=rng,
                max_steps=args.max_steps,
            )

            if winner == 0:
                wins0 += 1
            elif winner == 1:
                wins1 += 1
            elif winner == 2:
                draws += 1

            if error:
                errors += 1
            if winner not in {0, 1, 2} and not args.keep_unfinished:
                skipped_games += 1
            else:
                writer.writerows(rows)
                total_rows += len(rows)
                finished_games += 1

            if not args.quiet:
                status = f"game={game_index} winner={winner} steps={steps} rows={len(rows)}"
                if error:
                    status += f" error={error}"
                print(status)

    print(
        f"wrote {total_rows} rows from {finished_games} games to {args.output} "
        f"(p0_wins={wins0}, p1_wins={wins1}, draws={draws}, skipped={skipped_games}, errors={errors})"
    )
    print(f"wrote metadata: {metadata_path}")
    return 1 if errors and not args.keep_unfinished else 0


if __name__ == "__main__":
    raise SystemExit(main())
