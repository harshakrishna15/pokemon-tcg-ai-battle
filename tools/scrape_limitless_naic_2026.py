#!/usr/bin/env python3
"""Scrape public NAIC 2026 decklists and standings from Limitless.

The output is intentionally simple to load:

* ``standings.csv.gz`` has one row per player from Limitless Labs.
* ``decklist_cards.csv.gz`` has one row per card line in each published list.
* ``decklists.jsonl.gz`` has one JSON object per published 60-card list.
* ``metadata.json`` records source URLs, fetch time, and row counts.

Raw HTML is cached under ``raw/`` so repeated parses do not hit the site.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, TextIO
from urllib.request import Request, urlopen


TOURNAMENT_ID = "518"
LABS_EVENT_ID = "0070"
DECKLIST_URL = f"https://limitlesstcg.com/tournaments/{TOURNAMENT_ID}/decklists"
STANDINGS_URL = f"https://labs.limitlesstcg.com/{LABS_EVENT_ID}/standings"
USER_AGENT = "pokemon-tcg-ai-battle-data-tool/0.1"

STANDINGS_COLUMNS = [
    "placement",
    "player_id",
    "tp_id",
    "name",
    "country",
    "points",
    "wins",
    "losses",
    "ties",
    "opw",
    "opw2",
    "day2",
    "topcut",
    "dropped",
    "drop_round",
    "late",
    "dqed",
    "decklist",
    "deck_id",
    "deck_name",
    "icons",
]

CARD_COLUMNS = [
    "tournament_id",
    "labs_event_id",
    "placement",
    "player",
    "player_id",
    "tp_id",
    "country",
    "points",
    "wins",
    "losses",
    "ties",
    "deck_id",
    "deck_name",
    "icons",
    "card_category",
    "card_count",
    "card_name",
    "set",
    "number",
    "lang",
    "card_key",
    "is_basic_energy",
    "source_url",
]


@dataclass
class CardLine:
    category: str
    count: int
    name: str
    set: str
    number: str
    lang: str
    card_key: str
    is_basic_energy: bool


@dataclass
class Decklist:
    placement: int
    player: str
    cards: list[CardLine] = field(default_factory=list)

    @property
    def total_cards(self) -> int:
        return sum(card.count for card in self.cards)


class FetchedScriptParser(HTMLParser):
    """Extract SvelteKit fetched JSON script bodies keyed by data URL."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[tuple[str, str]] = []
        self._in_script = False
        self._script_url = ""
        self._script_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "script":
            return
        attrs_dict = dict(attrs)
        if attrs_dict.get("type") != "application/json":
            return
        data_url = attrs_dict.get("data-url")
        if not data_url:
            return
        self._in_script = True
        self._script_url = data_url
        self._script_chunks = []

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script:
            self.scripts.append((self._script_url, "".join(self._script_chunks)))
            self._in_script = False
            self._script_url = ""
            self._script_chunks = []


class DecklistParser(HTMLParser):
    """Parse the text decklist blocks from Limitless tournament pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.decklists: list[Decklist] = []
        self._current_deck: Decklist | None = None
        self._current_category = ""
        self._current_card: dict[str, Any] | None = None
        self._collecting: str | None = None
        self._text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        classes = set(attrs_dict.get("class", "").split())

        if tag == "div" and "decklist-toggle" in classes:
            self._collecting = "deck-title"
            self._text_chunks = []
            return

        if tag == "div" and "decklist-column-heading" in classes:
            self._collecting = "category"
            self._text_chunks = []
            return

        if tag == "div" and "decklist-card" in classes:
            self._current_card = {
                "category": self._current_category,
                "count": 0,
                "name": "",
                "set": attrs_dict.get("data-set", ""),
                "number": attrs_dict.get("data-number", ""),
                "lang": attrs_dict.get("data-lang", ""),
                "is_basic_energy": "data-basic-energy" in attrs_dict,
            }
            return

        if self._current_card is None or tag != "span":
            return

        if "card-count" in classes:
            self._collecting = "card-count"
            self._text_chunks = []
        elif "card-name" in classes:
            self._collecting = "card-name"
            self._text_chunks = []

    def handle_data(self, data: str) -> None:
        if self._collecting:
            self._text_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._collecting and (
            (self._collecting in {"card-count", "card-name"} and tag == "span")
            or (self._collecting in {"deck-title", "category"} and tag == "div")
        ):
            self._finish_text_field()
            return

        if tag == "div" and self._current_card is not None:
            self._finish_card()

    def _finish_text_field(self) -> None:
        text = _clean_text("".join(self._text_chunks))
        field_name = self._collecting
        self._collecting = None
        self._text_chunks = []

        if field_name == "deck-title":
            placement, player = _parse_deck_title(text)
            self._current_deck = Decklist(placement=placement, player=player)
            self.decklists.append(self._current_deck)
        elif field_name == "category":
            self._current_category = re.sub(r"\s*\(\d+\)\s*$", "", text)
        elif field_name == "card-count" and self._current_card is not None:
            self._current_card["count"] = int(text or 0)
        elif field_name == "card-name" and self._current_card is not None:
            self._current_card["name"] = text

    def _finish_card(self) -> None:
        if self._current_deck is None or self._current_card is None:
            self._current_card = None
            return

        card_set = self._current_card["set"]
        number = self._current_card["number"]
        card_key = f"{card_set}-{number}" if card_set and number else ""
        self._current_deck.cards.append(
            CardLine(
                category=self._current_card["category"],
                count=self._current_card["count"],
                name=self._current_card["name"],
                set=card_set,
                number=number,
                lang=self._current_card["lang"],
                card_key=card_key,
                is_basic_energy=bool(self._current_card["is_basic_energy"]),
            )
        )
        self._current_card = None


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _parse_deck_title(value: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)(?:st|nd|rd|th)\s+(.+)$", value)
    if not match:
        raise ValueError(f"Could not parse decklist title: {value!r}")
    return int(match.group(1)), match.group(2).strip()


def _normalized_name(value: str) -> str:
    return _clean_text(value).casefold()


def fetch_url(url: str, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def read_html(
    *,
    url: str,
    cache_path: Path,
    timeout: int,
    refresh: bool,
    local_path: Path | None,
    sleep_seconds: float,
) -> str:
    if local_path is not None:
        return local_path.read_text(encoding="utf-8")
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    html = fetch_url(url, timeout)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    return html


def parse_standings(html: str) -> list[dict[str, Any]]:
    parser = FetchedScriptParser()
    parser.feed(html)
    for data_url, body in parser.scripts:
        if "/labs/data/tcg/standings" not in data_url:
            continue
        outer = json.loads(body)
        payload = json.loads(outer["body"])
        if not payload.get("ok"):
            raise ValueError(f"Limitless Labs standings request failed: {payload!r}")
        return list(payload["message"])
    raise ValueError("Could not find embedded Limitless Labs standings data.")


def parse_decklists(html: str) -> list[Decklist]:
    parser = DecklistParser()
    parser.feed(html)
    return parser.decklists


def open_text(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    count = 0
    with open_text(path) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def standing_key(row: dict[str, Any]) -> tuple[int | None, str]:
    placement = row.get("placement")
    return (int(placement) if placement is not None else None, _normalized_name(str(row.get("name", ""))))


def deck_key(deck: Decklist) -> tuple[int, str]:
    return (deck.placement, _normalized_name(deck.player))


def resolve_standing(
    deck: Decklist,
    standings_by_key: dict[tuple[int | None, str], dict[str, Any]],
    standings_by_placement: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    standing = standings_by_key.get(deck_key(deck))
    if standing is not None:
        return standing
    return standings_by_placement.get(deck.placement, {})


def card_rows(
    decklists: list[Decklist],
    standings_by_key: dict[tuple[int | None, str], dict[str, Any]],
    standings_by_placement: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for deck in decklists:
        standing = resolve_standing(deck, standings_by_key, standings_by_placement)
        for card in deck.cards:
            rows.append(
                {
                    "tournament_id": TOURNAMENT_ID,
                    "labs_event_id": LABS_EVENT_ID,
                    "placement": deck.placement,
                    "player": deck.player,
                    "player_id": standing.get("player_id", ""),
                    "tp_id": standing.get("tp_id", ""),
                    "country": standing.get("country", ""),
                    "points": standing.get("points", ""),
                    "wins": standing.get("wins", ""),
                    "losses": standing.get("losses", ""),
                    "ties": standing.get("ties", ""),
                    "deck_id": standing.get("deck_id", ""),
                    "deck_name": standing.get("deck_name", ""),
                    "icons": standing.get("icons", ""),
                    "card_category": card.category,
                    "card_count": card.count,
                    "card_name": card.name,
                    "set": card.set,
                    "number": card.number,
                    "lang": card.lang,
                    "card_key": card.card_key,
                    "is_basic_energy": int(card.is_basic_energy),
                    "source_url": DECKLIST_URL,
                }
            )
    return rows


def deck_json_rows(
    decklists: list[Decklist],
    standings_by_key: dict[tuple[int | None, str], dict[str, Any]],
    standings_by_placement: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for deck in decklists:
        standing = resolve_standing(deck, standings_by_key, standings_by_placement)
        rows.append(
            {
                "tournament_id": TOURNAMENT_ID,
                "labs_event_id": LABS_EVENT_ID,
                "placement": deck.placement,
                "player": deck.player,
                "player_id": standing.get("player_id"),
                "tp_id": standing.get("tp_id"),
                "country": standing.get("country"),
                "points": standing.get("points"),
                "wins": standing.get("wins"),
                "losses": standing.get("losses"),
                "ties": standing.get("ties"),
                "deck_id": standing.get("deck_id"),
                "deck_name": standing.get("deck_name"),
                "icons": standing.get("icons"),
                "total_cards": deck.total_cards,
                "cards": [asdict(card) for card in deck.cards],
            }
        )
    return rows


def write_metadata(
    *,
    path: Path,
    standings: list[dict[str, Any]],
    decklists: list[Decklist],
    cards_count: int,
    unmatched_decks: list[Decklist],
    placement_fallback_decks: int,
    output_files: dict[str, str],
) -> None:
    totals: dict[str, int] = {}
    for deck in decklists:
        totals[str(deck.total_cards)] = totals.get(str(deck.total_cards), 0) + 1

    metadata = {
        "tournament": "NAIC 2026, New Orleans",
        "tournament_id": TOURNAMENT_ID,
        "labs_event_id": LABS_EVENT_ID,
        "sources": {
            "decklists": DECKLIST_URL,
            "standings": STANDINGS_URL,
        },
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "standings_rows": len(standings),
            "published_decklists": len(decklists),
            "decklist_card_rows": cards_count,
            "unmatched_decklists": len(unmatched_decks),
            "placement_fallback_decklists": placement_fallback_decks,
            "deck_total_cards": totals,
        },
        "output_files": output_files,
        "notes": [
            "Limitless publishes fewer full decklists than total tournament players.",
            "Standings come from the embedded Limitless Labs standings payload.",
            "Decklist card rows come from the Limitless tournament decklists page.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape public Limitless NAIC 2026 decklists.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/limitless/naic_2026"))
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh HTML even when raw cache files exist.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay before each network fetch.")
    parser.add_argument("--limit-decks", type=int, help="Keep only the first N decklists. Useful for parser tests.")
    parser.add_argument("--decklists-html", type=Path, help="Parse decklists from a local HTML file instead of fetching.")
    parser.add_argument("--standings-html", type=Path, help="Parse standings from a local HTML file instead of fetching.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.output_dir / "raw"
    decklists_html = read_html(
        url=DECKLIST_URL,
        cache_path=raw_dir / "decklists.html",
        timeout=args.timeout,
        refresh=args.refresh,
        local_path=args.decklists_html,
        sleep_seconds=args.sleep,
    )
    standings_html = read_html(
        url=STANDINGS_URL,
        cache_path=raw_dir / "standings.html",
        timeout=args.timeout,
        refresh=args.refresh,
        local_path=args.standings_html,
        sleep_seconds=args.sleep,
    )

    standings = parse_standings(standings_html)
    decklists = parse_decklists(decklists_html)
    if args.limit_decks is not None:
        decklists = decklists[: args.limit_decks]

    standings_by_key = {standing_key(row): row for row in standings}
    standings_by_placement = {
        int(row["placement"]): row for row in standings if row.get("placement") is not None
    }
    placement_fallbacks = sum(
        1
        for deck in decklists
        if deck_key(deck) not in standings_by_key and deck.placement in standings_by_placement
    )
    unmatched = [
        deck
        for deck in decklists
        if not resolve_standing(deck, standings_by_key, standings_by_placement)
    ]
    cards = card_rows(decklists, standings_by_key, standings_by_placement)

    standings_path = args.output_dir / "standings.csv.gz"
    cards_path = args.output_dir / "decklist_cards.csv.gz"
    jsonl_path = args.output_dir / "decklists.jsonl.gz"
    metadata_path = args.output_dir / "metadata.json"

    standings_count = write_csv(standings_path, standings, STANDINGS_COLUMNS)
    cards_count = write_csv(cards_path, cards, CARD_COLUMNS)
    deck_count = write_jsonl(jsonl_path, deck_json_rows(decklists, standings_by_key, standings_by_placement))
    write_metadata(
        path=metadata_path,
        standings=standings,
        decklists=decklists,
        cards_count=cards_count,
        unmatched_decks=unmatched,
        placement_fallback_decks=placement_fallbacks,
        output_files={
            "standings": str(standings_path),
            "decklist_cards": str(cards_path),
            "decklists_jsonl": str(jsonl_path),
            "metadata": str(metadata_path),
        },
    )

    print(f"wrote {standings_count} standings rows to {standings_path}")
    print(f"wrote {deck_count} decklists to {jsonl_path}")
    print(f"wrote {cards_count} card rows to {cards_path}")
    if unmatched:
        print(f"warning: {len(unmatched)} decklists did not match Labs standings metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
