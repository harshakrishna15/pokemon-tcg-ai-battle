#!/usr/bin/env python3
"""Run repeated CG experiments and aggregate deck stats."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO


sys.dont_write_bytecode = True

from run_cg_match import POLICIES, SUBMISSION, MatchResult, read_deck, run_game, write_jsonl


@dataclass
class ExperimentResult:
    matchup: str
    game: int
    seed: int
    subject_seat: int
    subject_policy: str
    opponent_policy: str
    winner: int
    steps: int
    subject_result: str
    error: str | None


@dataclass
class AggregateStats:
    label: str
    games: int
    wins: int
    losses: int
    draws: int
    unfinished: int
    errors: int
    win_rate: float
    non_draw_win_rate: float
    avg_steps: float
    median_steps: float
    min_steps: int
    max_steps: int


def subject_result(match: MatchResult, subject_seat: int) -> str:
    if match.error:
        return "unfinished" if match.winner == -1 else "error"
    if match.winner == 2:
        return "draw"
    if match.winner == subject_seat:
        return "win"
    if match.winner in {0, 1}:
        return "loss"
    return "unfinished"


def aggregate(label: str, results: list[ExperimentResult]) -> AggregateStats:
    wins = sum(1 for result in results if result.subject_result == "win")
    losses = sum(1 for result in results if result.subject_result == "loss")
    draws = sum(1 for result in results if result.subject_result == "draw")
    unfinished = sum(1 for result in results if result.subject_result == "unfinished")
    errors = sum(1 for result in results if result.subject_result == "error")
    decided = wins + losses + draws
    non_draw = wins + losses
    steps = [result.steps for result in results]

    return AggregateStats(
        label=label,
        games=len(results),
        wins=wins,
        losses=losses,
        draws=draws,
        unfinished=unfinished,
        errors=errors,
        win_rate=(wins / decided * 100) if decided else 0.0,
        non_draw_win_rate=(wins / non_draw * 100) if non_draw else 0.0,
        avg_steps=statistics.fmean(steps) if steps else 0.0,
        median_steps=statistics.median(steps) if steps else 0.0,
        min_steps=min(steps) if steps else 0,
        max_steps=max(steps) if steps else 0,
    )


def print_stats(stats: AggregateStats) -> None:
    print(f"\n{stats.label}")
    print("-" * len(stats.label))
    print(f"games: {stats.games}")
    print(f"wins: {stats.wins}")
    print(f"losses: {stats.losses}")
    print(f"draws: {stats.draws}")
    print(f"unfinished: {stats.unfinished}")
    print(f"errors: {stats.errors}")
    print(f"win rate: {stats.win_rate:.1f}%")
    print(f"non-draw win rate: {stats.non_draw_win_rate:.1f}%")
    print(f"avg steps: {stats.avg_steps:.1f}")
    print(f"median steps: {stats.median_steps:.1f}")
    print(f"step range: {stats.min_steps}-{stats.max_steps}")


def print_seed_summary(results: list[ExperimentResult]) -> None:
    seeds = sorted({result.seed for result in results})
    if len(seeds) <= 1:
        return

    seed_stats = [aggregate(f"seed {seed}", [result for result in results if result.seed == seed]) for seed in seeds]
    rates = [stats.non_draw_win_rate for stats in seed_stats]
    print("\nseed sweep")
    print("----------")
    for stats in seed_stats:
        print(
            f"{stats.label}: {stats.wins}-{stats.losses}-{stats.draws}, "
            f"non-draw win rate {stats.non_draw_win_rate:.1f}%, avg steps {stats.avg_steps:.1f}"
        )
    print(f"non-draw win rate range: {min(rates):.1f}%-{max(rates):.1f}%")
    print(f"non-draw win rate mean: {statistics.fmean(rates):.1f}%")
    print(f"non-draw win rate stdev: {statistics.stdev(rates):.1f}%" if len(rates) > 1 else "non-draw win rate stdev: 0.0%")


def write_csv(path: Path, results: list[ExperimentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(ExperimentResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def open_jsonl(path: Path | None) -> TextIO | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated CG experiments for deck evaluation.")
    parser.add_argument("--games", type=int, default=50, help="Games per opponent policy.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--deck", type=Path, default=SUBMISSION / "deck.csv", help="Deck under test.")
    parser.add_argument("--opponent-deck", type=Path, help="Opponent deck. Defaults to --deck.")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="agent", help="Policy for the deck under test.")
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=sorted(POLICIES),
        default=["random", "first", "agent-no-search"],
        help="Opponent policies to test against.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, help="Run a held-out seed sweep. Overrides --seed.")
    parser.add_argument("--seat-swap", action="store_true", help="Alternate the deck under test between player 0 and 1.")
    parser.add_argument("--jsonl", type=Path, help="Write detailed simulator step records.")
    parser.add_argument("--csv", type=Path, help="Write one result row per game.")
    parser.add_argument("--replay-dir", type=Path, help="Write CG visualizer JSON replays.")
    parser.add_argument("--quiet", action="store_true", help="Do not print one-line game progress.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject_deck = read_deck(args.deck)
    opponent_deck = read_deck(args.opponent_deck or args.deck)
    seeds = args.seeds if args.seeds else [args.seed]
    all_results: list[ExperimentResult] = []
    jsonl_file = open_jsonl(args.jsonl)
    global_game_index = 1

    try:
        for seed in seeds:
            rng = random.Random(seed)
            if len(seeds) > 1:
                print(f"\nseed {seed}")
                print("=" * (5 + len(str(seed))))

            for opponent_policy in args.opponents:
                matchup = f"{args.policy} vs {opponent_policy}"
                matchup_results: list[ExperimentResult] = []
                for local_game in range(1, args.games + 1):
                    subject_seat = 1 if args.seat_swap and local_game % 2 == 0 else 0
                    if subject_seat == 0:
                        deck0 = subject_deck
                        deck1 = opponent_deck
                        policy0 = args.policy
                        policy1 = opponent_policy
                    else:
                        deck0 = opponent_deck
                        deck1 = subject_deck
                        policy0 = opponent_policy
                        policy1 = args.policy

                    replay_dir = None
                    if args.replay_dir is not None:
                        replay_dir = args.replay_dir / f"seed_{seed}" / opponent_policy

                    match = run_game(
                        game_index=global_game_index,
                        deck0=deck0,
                        deck1=deck1,
                        policy0_name=policy0,
                        policy1_name=policy1,
                        rng=rng,
                        max_steps=args.max_steps,
                        jsonl=jsonl_file,
                        replay_dir=replay_dir,
                        trace=False,
                    )
                    result = ExperimentResult(
                        matchup=matchup,
                        game=global_game_index,
                        seed=seed,
                        subject_seat=subject_seat,
                        subject_policy=args.policy,
                        opponent_policy=opponent_policy,
                        winner=match.winner,
                        steps=match.steps,
                        subject_result=subject_result(match, subject_seat),
                        error=match.error,
                    )
                    matchup_results.append(result)
                    all_results.append(result)
                    write_jsonl(jsonl_file, {"event": "experiment_result", **asdict(result)})

                    if not args.quiet:
                        print(
                            f"game={global_game_index} seed={seed} matchup={matchup} seat={subject_seat} "
                            f"result={result.subject_result} winner={match.winner} steps={match.steps}"
                        )
                    global_game_index += 1

                print_stats(aggregate(matchup, matchup_results))

    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    print_stats(aggregate("overall", all_results))
    print_seed_summary(all_results)

    if args.csv is not None:
        write_csv(args.csv, all_results)
        print(f"\nwrote csv: {args.csv}")

    return 1 if any(result.subject_result in {"error", "unfinished"} for result in all_results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
