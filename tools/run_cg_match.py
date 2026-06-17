#!/usr/bin/env python3
"""Run local matches with the bundled CG simulator.

The competition-provided simulator is already built as `sample_submission/cg/libcg.so`
on Linux and `sample_submission/cg/cg.dll` on Windows. This script is a thin
local harness around that folder, so you can test deck and agent changes without
submitting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TextIO


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "sample_submission"

sys.dont_write_bytecode = True
sys.path.insert(0, str(SUBMISSION))

from agent_policy import choose_action  # noqa: E402
from cg import game  # noqa: E402


Policy = Callable[[Any, list[int], random.Random], list[int]]


@dataclass
class MatchResult:
    game: int
    winner: int
    steps: int
    policy0: str
    policy1: str
    error: str | None = None
    replay_path: str | None = None


def read_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path} must contain exactly 60 card IDs, found {len(deck)}")
    return deck


def agent_policy(obs: Any, deck: list[int], rng: random.Random) -> list[int]:
    if obs.get("select") is None:
        return deck
    return choose_action(obs, deck)


def agent_no_search_policy(obs: Any, deck: list[int], rng: random.Random) -> list[int]:
    if obs.get("select") is None:
        return deck
    return choose_action(obs, deck, use_search=False)


def first_policy(obs: Any, deck: list[int], rng: random.Random) -> list[int]:
    select = obs.get("select")
    if select is None:
        return deck
    min_count = int(select.get("minCount", 1))
    options = list(select.get("option") or [])
    return list(range(min(min_count, len(options))))


def random_policy(obs: Any, deck: list[int], rng: random.Random) -> list[int]:
    select = obs.get("select")
    if select is None:
        return deck

    options = list(select.get("option") or [])
    if not options:
        return []

    min_count = int(select.get("minCount", 1))
    max_count = int(select.get("maxCount", 1))
    upper = min(max_count, len(options))
    lower = min(min_count, upper)
    return rng.sample(range(len(options)), rng.randint(lower, upper))


POLICIES: dict[str, Policy] = {
    "agent": agent_policy,
    "agent-no-search": agent_no_search_policy,
    "first": first_policy,
    "random": random_policy,
}


def summarize_option(option: Any) -> dict[str, Any]:
    fields = (
        "type",
        "area",
        "index",
        "playerIndex",
        "inPlayArea",
        "inPlayIndex",
        "attackId",
        "cardId",
        "number",
    )
    return {field: option[field] for field in fields if field in option}


def summarize_select(obs: Any) -> dict[str, Any]:
    select = obs.get("select") or {}
    options = list(select.get("option") or [])
    return {
        "type": select.get("type"),
        "context": select.get("context"),
        "minCount": select.get("minCount"),
        "maxCount": select.get("maxCount"),
        "options": [summarize_option(option) for option in options],
    }


def write_jsonl(file: TextIO | None, record: dict[str, Any]) -> None:
    if file is not None:
        file.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def save_replay(replay_dir: Path | None, game_index: int) -> str | None:
    if replay_dir is None:
        return None
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / f"game_{game_index:04d}.json"
    replay_path.write_text(game.visualize_data(), encoding="utf-8")
    return str(replay_path)


def run_game(
    game_index: int,
    deck0: list[int],
    deck1: list[int],
    policy0_name: str,
    policy1_name: str,
    rng: random.Random,
    max_steps: int,
    jsonl: TextIO | None,
    replay_dir: Path | None,
    trace: bool,
) -> MatchResult:
    obs = None
    try:
        obs, start = game.battle_start(deck0, deck1)
        if obs is None:
            return MatchResult(
                game_index,
                -1,
                0,
                policy0_name,
                policy1_name,
                error=f"battle_start failed: errorPlayer={start.errorPlayer}, errorType={start.errorType}",
            )

        write_jsonl(jsonl, {"event": "game_start", "game": game_index, "policy0": policy0_name, "policy1": policy1_name})
        steps = 0
        while steps < max_steps:
            result = int(obs["current"].get("result", -1))
            if result != -1:
                return MatchResult(
                    game_index,
                    result,
                    steps,
                    policy0_name,
                    policy1_name,
                    replay_path=save_replay(replay_dir, game_index),
                )

            player = int(obs["current"].get("yourIndex", 0))
            policy_name = policy0_name if player == 0 else policy1_name
            policy = POLICIES[policy_name]
            deck = deck0 if player == 0 else deck1
            action = policy(obs, deck, rng)
            record = {
                "event": "step",
                "game": game_index,
                "step": steps,
                "player": player,
                "policy": policy_name,
                "turn": obs["current"].get("turn"),
                "turnActionCount": obs["current"].get("turnActionCount"),
                "select": summarize_select(obs),
                "action": action,
            }
            write_jsonl(jsonl, record)
            if trace:
                select = record["select"]
                print(
                    f"game={game_index} step={steps} player={player} policy={policy_name} "
                    f"select={select['type']}/{select['context']} options={len(select['options'])} action={action}"
                )

            try:
                obs = game.battle_select(action)
            except Exception as exc:
                return MatchResult(
                    game_index,
                    1 - player,
                    steps,
                    policy0_name,
                    policy1_name,
                    error=f"{type(exc).__name__}: player={player}, action={action}, message={exc}",
                    replay_path=save_replay(replay_dir, game_index),
                )
            steps += 1

        return MatchResult(
            game_index,
            -1,
            max_steps,
            policy0_name,
            policy1_name,
            error="max steps reached",
            replay_path=save_replay(replay_dir, game_index),
        )
    finally:
        if obs is not None:
            game.battle_finish()


def summarize_results(results: list[MatchResult]) -> str:
    finished = [result for result in results if result.error is None]
    wins0 = sum(1 for result in finished if result.winner == 0)
    wins1 = sum(1 for result in finished if result.winner == 1)
    draws = sum(1 for result in finished if result.winner == 2)
    errors = [result for result in results if result.error and result.error != "max steps reached"]
    unfinished = sum(1 for result in results if result.winner == -1)
    decided = wins0 + wins1
    win_rate = wins0 / decided * 100 if decided else 0.0
    avg_steps = sum(result.steps for result in results) / len(results) if results else 0.0

    return "\n".join(
        [
            f"games: {len(results)}",
            f"player 0 wins: {wins0}",
            f"player 1 wins: {wins1}",
            f"draws: {draws}",
            f"unfinished: {unfinished}",
            f"errors: {len(errors)}",
            f"player 0 win rate: {win_rate:.1f}%",
            f"avg steps: {avg_steps:.1f}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local matches with sample_submission/cg.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--deck0", type=Path, default=SUBMISSION / "deck.csv")
    parser.add_argument("--deck1", type=Path, default=SUBMISSION / "deck.csv")
    parser.add_argument("--p0", choices=sorted(POLICIES), default="agent")
    parser.add_argument("--p1", choices=sorted(POLICIES), default="agent")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deck0 = read_deck(args.deck0)
    deck1 = read_deck(args.deck1)
    rng = random.Random(args.seed)
    results: list[MatchResult] = []

    jsonl_file = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    try:
        for game_index in range(1, args.games + 1):
            result = run_game(
                game_index,
                deck0,
                deck1,
                args.p0,
                args.p1,
                rng,
                args.max_steps,
                jsonl_file,
                args.replay_dir,
                args.trace,
            )
            results.append(result)
            write_jsonl(jsonl_file, {"event": "game_result", **asdict(result)})
            if not args.quiet:
                status = f"game {game_index}: winner={result.winner}, steps={result.steps}"
                if result.replay_path:
                    status += f", replay={result.replay_path}"
                if result.error:
                    status += f", error={result.error}"
                print(status)
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    print(summarize_results(results))
    return 1 if any(result.error for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
