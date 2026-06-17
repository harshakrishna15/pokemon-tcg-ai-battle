import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "sample_submission"
CARD_DATA = ROOT / "EN_Card_Data.csv"

REQUIRED_FILES = [
    SUBMISSION / "main.py",
    SUBMISSION / "deck.csv",
    SUBMISSION / "cg" / "__init__.py",
    SUBMISSION / "cg" / "api.py",
    SUBMISSION / "cg" / "game.py",
    SUBMISSION / "cg" / "sim.py",
    SUBMISSION / "cg" / "utils.py",
    SUBMISSION / "cg" / "libcg.so",
    SUBMISSION / "cg" / "cg.dll",
]

FORBIDDEN_NAMES = {"__pycache__", ".DS_Store"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN_CODE_PATTERNS = [
    "/Users/",
    "http://",
    "https://",
    "requests",
    "urllib",
    "socket",
    "subprocess",
    "EN_Card_Data.csv",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def check_required_files() -> None:
    for path in REQUIRED_FILES:
        if not path.exists():
            fail(f"missing required file: {path.relative_to(ROOT)}")


def check_forbidden_artifacts() -> None:
    for path in SUBMISSION.rglob("*"):
        if path.name in FORBIDDEN_NAMES:
            fail(f"forbidden generated artifact: {path.relative_to(ROOT)}")
        if path.suffix in FORBIDDEN_SUFFIXES:
            fail(f"forbidden bytecode artifact: {path.relative_to(ROOT)}")


def check_code_patterns() -> None:
    for path in SUBMISSION.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_CODE_PATTERNS:
            if pattern in text:
                fail(f"forbidden submission code pattern {pattern!r} in {path.relative_to(ROOT)}")
    main_text = (SUBMISSION / "main.py").read_text(encoding="utf-8")
    if "def agent(" not in main_text:
        fail("main.py does not define agent(...)")


def load_card_rows() -> dict[int, list[dict[str, str]]]:
    if not CARD_DATA.exists():
        fail("missing EN_Card_Data.csv for local validation")
    by_id: dict[int, list[dict[str, str]]] = defaultdict(list)
    with CARD_DATA.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            by_id[int(row["Card ID"])].append(row)
    return by_id


def check_deck() -> None:
    deck_path = SUBMISSION / "deck.csv"
    try:
        deck = [int(line.strip()) for line in deck_path.read_text().splitlines() if line.strip()]
    except ValueError as exc:
        fail(f"deck.csv contains a non-integer card id: {exc}")

    if len(deck) != 60:
        fail(f"deck.csv must contain exactly 60 cards, found {len(deck)}")

    by_id = load_card_rows()
    unknown = [card_id for card_id in deck if card_id not in by_id]
    if unknown:
        fail(f"deck.csv contains unknown card IDs: {unknown}")

    counts = Counter(deck)
    name_counts = Counter()
    basic_pokemon = 0
    ace_specs = 0
    pokemon_names = set()
    evolution_requirements = []

    for card_id, count in counts.items():
        row = by_id[card_id][0]
        card_type = row["Stage (Pokémon)/Type (Energy and Trainer)"]
        name = row["Card Name"]
        if card_type != "Basic Energy":
            name_counts[name] += count
        if card_type == "Basic Pokémon":
            basic_pokemon += count
        if "Pokémon" in card_type:
            pokemon_names.add(name)
        if row["Rule"] == "ACE SPEC":
            ace_specs += count
        if row["Previous stage"] and row["Previous stage"] != "n/a":
            evolution_requirements.append((name, row["Previous stage"]))

    over_four = {name: count for name, count in name_counts.items() if count > 4}
    if over_four:
        fail(f"deck has more than 4 copies of non-basic-energy cards: {over_four}")
    if ace_specs > 1:
        fail(f"deck has more than 1 ACE SPEC card: {ace_specs}")
    if basic_pokemon == 0:
        fail("deck has no Basic Pokémon")

    missing_sources = [
        f"{evolved} requires {source}"
        for evolved, source in evolution_requirements
        if source not in pokemon_names
    ]
    if missing_sources:
        fail(f"deck has missing evolution sources: {missing_sources}")


def main() -> None:
    check_required_files()
    check_forbidden_artifacts()
    check_code_patterns()
    check_deck()
    print("OK: submission structure and deck checks passed")


if __name__ == "__main__":
    main()

