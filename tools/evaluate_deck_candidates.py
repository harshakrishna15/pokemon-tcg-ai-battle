#!/usr/bin/env python3
"""Evaluate generated deck candidates against a baseline deck."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

sys.dont_write_bytecode = True
sys.path.insert(0, str(TOOLS))

from run_cg_experiments import aggregate, subject_result  # noqa: E402
from run_cg_match import POLICIES, read_deck, run_game  # noqa: E402


@dataclass
class CandidateResult:
    rank: int
    placement: int
    player: str
    deck_name: str
    deck_file: str
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


@dataclass
class EvalGameResult:
    subject_result: str
    steps: int


def load_candidates(summary_path: Path, limit: int | None) -> list[dict[str, str]]:
    with summary_path.open(encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("deck_file")]
    if limit is not None:
        rows = rows[:limit]
    return rows


def evaluate_candidate(
    *,
    candidate: dict[str, str],
    baseline_deck: list[int],
    policy: str,
    opponent_policy: str,
    games: int,
    max_steps: int,
    seat_swap: bool,
    rng: random.Random,
    quiet: bool,
    global_start_index: int,
) -> tuple[CandidateResult, int]:
    candidate_deck = read_deck(Path(candidate["deck_file"]))
    results = []
    global_game_index = global_start_index

    for local_game in range(1, games + 1):
        subject_seat = 1 if seat_swap and local_game % 2 == 0 else 0
        if subject_seat == 0:
            deck0 = candidate_deck
            deck1 = baseline_deck
            policy0 = policy
            policy1 = opponent_policy
        else:
            deck0 = baseline_deck
            deck1 = candidate_deck
            policy0 = opponent_policy
            policy1 = policy

        match = run_game(
            game_index=global_game_index,
            deck0=deck0,
            deck1=deck1,
            policy0_name=policy0,
            policy1_name=policy1,
            rng=rng,
            max_steps=max_steps,
            jsonl=None,
            replay_dir=None,
            trace=False,
        )
        results.append(EvalGameResult(subject_result=subject_result(match, subject_seat), steps=match.steps))
        global_game_index += 1

    stats = aggregate(candidate["deck_name"], results)
    result = CandidateResult(
        rank=int(candidate["rank"]),
        placement=int(candidate["placement"]),
        player=candidate["player"],
        deck_name=candidate["deck_name"],
        deck_file=candidate["deck_file"],
        games=stats.games,
        wins=stats.wins,
        losses=stats.losses,
        draws=stats.draws,
        unfinished=stats.unfinished,
        errors=stats.errors,
        win_rate=stats.win_rate,
        non_draw_win_rate=stats.non_draw_win_rate,
        avg_steps=stats.avg_steps,
        median_steps=stats.median_steps,
        min_steps=stats.min_steps,
        max_steps=stats.max_steps,
    )

    if not quiet:
        print(
            f"{result.rank:03d} place {result.placement:03d} {result.deck_name}: "
            f"{result.wins}-{result.losses}-{result.draws}, "
            f"non-draw {result.non_draw_win_rate:.1f}%"
        )
    return result, global_game_index


def write_results(path: Path, results: list[CandidateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(CandidateResult.__dataclass_fields__))
        writer.writeheader()
        for result in sorted(results, key=lambda row: (-row.non_draw_win_rate, row.rank)):
            writer.writerow(asdict(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deck candidates against a baseline.")
    parser.add_argument("--summary", type=Path, default=ROOT / "deck_candidates/naic_2026/summary.csv")
    parser.add_argument("--baseline-deck", type=Path, default=ROOT / "sample_submission/deck.csv")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="agent")
    parser.add_argument("--opponent-policy", choices=sorted(POLICIES), default="agent")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seat-swap", action="store_true", default=True)
    parser.add_argument("--no-seat-swap", action="store_false", dest="seat_swap")
    parser.add_argument("--csv", type=Path, default=ROOT / "deck_candidates/naic_2026/evaluation.csv")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = load_candidates(args.summary, args.limit)
    baseline_deck = read_deck(args.baseline_deck)
    rng = random.Random(args.seed)
    results: list[CandidateResult] = []
    global_game_index = 1

    for candidate in candidates:
        result, global_game_index = evaluate_candidate(
            candidate=candidate,
            baseline_deck=baseline_deck,
            policy=args.policy,
            opponent_policy=args.opponent_policy,
            games=args.games,
            max_steps=args.max_steps,
            seat_swap=args.seat_swap,
            rng=rng,
            quiet=args.quiet,
            global_start_index=global_game_index,
        )
        results.append(result)

    write_results(args.csv, results)
    best = max(results, key=lambda row: (row.non_draw_win_rate, -row.rank), default=None)
    if best is not None:
        print(
            f"best: rank {best.rank} place {best.placement} {best.deck_name} "
            f"{best.wins}-{best.losses}-{best.draws}, non-draw {best.non_draw_win_rate:.1f}%"
        )
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
