#!/usr/bin/env python3
"""Build simulator deck candidates from scraped Limitless decklists.

The Limitless lists use set/collector-number print IDs. The simulator uses
integer card IDs from ``EN_Card_Data.csv``. This tool maps direct print matches
first, then safe reprints by name, and finally basic energy print names to the
simulator's basic energy IDs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CARD_DATA = ROOT / "EN_Card_Data.csv"
DECKLISTS = ROOT / "data/limitless/naic_2026/decklists.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "deck_candidates/naic_2026"

ENERGY_NAME_TO_ID = {
    "grassenergy": 1,
    "fireenergy": 2,
    "waterenergy": 3,
    "lightningenergy": 4,
    "psychicenergy": 5,
    "fightingenergy": 6,
    "darknessenergy": 7,
    "metalenergy": 8,
}

SUMMARY_COLUMNS = [
    "rank",
    "placement",
    "player",
    "deck_name",
    "points",
    "wins",
    "losses",
    "ties",
    "deck_file",
    "missing_cards",
    "ambiguous_cards",
    "mapped_cards",
    "set_number_cards",
    "name_fallback_cards",
    "energy_fallback_cards",
    "unique_card_ids",
    "missing_detail",
    "ambiguous_detail",
]


@dataclass
class CardIndex:
    by_print: dict[tuple[str, str], int]
    by_name: dict[str, set[int]]
    card_names: dict[int, str]


@dataclass
class ResolvedDeck:
    source: dict[str, Any]
    card_ids: list[int] = field(default_factory=list)
    method_counts: Counter[str] = field(default_factory=Counter)
    missing: Counter[str] = field(default_factory=Counter)
    ambiguous: Counter[str] = field(default_factory=Counter)

    @property
    def placement(self) -> int:
        return int(self.source["placement"])

    @property
    def missing_cards(self) -> int:
        return sum(self.missing.values())

    @property
    def ambiguous_cards(self) -> int:
        return sum(self.ambiguous.values())

    @property
    def mapped_cards(self) -> int:
        return len(self.card_ids)

    @property
    def is_complete(self) -> bool:
        return self.missing_cards == 0 and self.ambiguous_cards == 0 and self.mapped_cards == 60


def normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", ascii_value.casefold())


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "deck"


def load_card_index(path: Path) -> CardIndex:
    by_print: dict[tuple[str, str], int] = {}
    by_name: dict[str, set[int]] = defaultdict(set)
    card_names: dict[int, str] = {}

    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            card_id = int(row["Card ID"])
            by_print[(row["Expansion"], row["Collection No."])] = card_id
            by_name[normalize_name(row["Card Name"])].add(card_id)
            card_names[card_id] = row["Card Name"]

    return CardIndex(by_print=by_print, by_name=dict(by_name), card_names=card_names)


def load_decklists(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def resolve_card(card: dict[str, Any], index: CardIndex) -> tuple[int | None, str]:
    print_key = (card["set"], card["number"])
    if print_key in index.by_print:
        return index.by_print[print_key], "set_number"

    normalized = normalize_name(card["name"])
    if normalized in ENERGY_NAME_TO_ID:
        return ENERGY_NAME_TO_ID[normalized], "energy_fallback"

    name_matches = index.by_name.get(normalized, set())
    if len(name_matches) == 1:
        return next(iter(name_matches)), "name_fallback"
    if len(name_matches) > 1:
        return None, "ambiguous"
    return None, "missing"


def resolve_deck(deck: dict[str, Any], index: CardIndex) -> ResolvedDeck:
    resolved = ResolvedDeck(source=deck)
    for card in deck["cards"]:
        count = int(card["count"])
        card_id, method = resolve_card(card, index)
        label = f"{count} {card['name']} [{card['set']} {card['number']}]"
        if card_id is None:
            if method == "ambiguous":
                resolved.ambiguous[label] += count
            else:
                resolved.missing[label] += count
            resolved.method_counts[method] += count
            continue

        resolved.card_ids.extend([card_id] * count)
        resolved.method_counts[method] += count

    return resolved


def deck_sort_key(deck: ResolvedDeck) -> tuple[int, int, int]:
    return (deck.missing_cards, deck.ambiguous_cards, deck.placement)


def write_deck(path: Path, card_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{card_id}\n" for card_id in card_ids), encoding="utf-8")


def detail(counter: Counter[str]) -> str:
    return "; ".join(f"{value}x {key}" for key, value in counter.most_common())


def write_summary(path: Path, decks: list[ResolvedDeck], output_dir: Path, written_ranks: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for rank, deck in enumerate(decks, start=1):
            deck_dir = candidate_dir(output_dir, rank, deck)
            writer.writerow(
                {
                    "rank": rank,
                    "placement": deck.placement,
                    "player": deck.source["player"],
                    "deck_name": deck.source.get("deck_name") or "",
                    "points": deck.source.get("points") or "",
                    "wins": deck.source.get("wins") or "",
                    "losses": deck.source.get("losses") or "",
                    "ties": deck.source.get("ties") or "",
                    "deck_file": str(deck_dir / "deck.csv") if rank in written_ranks else "",
                    "missing_cards": deck.missing_cards,
                    "ambiguous_cards": deck.ambiguous_cards,
                    "mapped_cards": deck.mapped_cards,
                    "set_number_cards": deck.method_counts["set_number"],
                    "name_fallback_cards": deck.method_counts["name_fallback"],
                    "energy_fallback_cards": deck.method_counts["energy_fallback"],
                    "unique_card_ids": len(set(deck.card_ids)),
                    "missing_detail": detail(deck.missing),
                    "ambiguous_detail": detail(deck.ambiguous),
                }
            )


def candidate_dir(output_dir: Path, rank: int, deck: ResolvedDeck) -> Path:
    name = deck.source.get("deck_name") or "unknown"
    return output_dir / f"{rank:03d}_place_{deck.placement:03d}_{safe_slug(name)}"


def write_candidates(output_dir: Path, decks: list[ResolvedDeck], limit: int) -> set[int]:
    written_ranks: set[int] = set()
    for rank, deck in enumerate(decks, start=1):
        if len(written_ranks) >= limit:
            break
        if not deck.is_complete:
            continue

        deck_dir = candidate_dir(output_dir, rank, deck)
        write_deck(deck_dir / "deck.csv", deck.card_ids)
        metadata = {
            "rank": rank,
            "placement": deck.placement,
            "player": deck.source["player"],
            "deck_name": deck.source.get("deck_name"),
            "points": deck.source.get("points"),
            "wins": deck.source.get("wins"),
            "losses": deck.source.get("losses"),
            "ties": deck.source.get("ties"),
            "method_counts": dict(deck.method_counts),
            "unique_card_ids": len(set(deck.card_ids)),
            "source": {
                "tournament_id": deck.source.get("tournament_id"),
                "labs_event_id": deck.source.get("labs_event_id"),
            },
        }
        (deck_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written_ranks.add(rank)
    return written_ranks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simulator deck candidates from Limitless decklists.")
    parser.add_argument("--decklists", type=Path, default=DECKLISTS)
    parser.add_argument("--card-data", type=Path, default=CARD_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20, help="Number of complete candidate deck files to write.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = load_card_index(args.card_data)
    decks = [resolve_deck(deck, index) for deck in load_decklists(args.decklists)]
    decks.sort(key=deck_sort_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = sum(1 for deck in decks if deck.is_complete)
    written_ranks = write_candidates(args.output_dir, decks, args.limit)
    write_summary(args.output_dir / "summary.csv", decks, args.output_dir, written_ranks)

    print(f"complete decklists: {complete}/{len(decks)}")
    print(f"wrote {len(written_ranks)} candidate decks to {args.output_dir}")
    print(f"summary: {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
