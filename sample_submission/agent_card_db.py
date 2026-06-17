"""Card facts used by the generic policy.

The simulator can expose full card and attack metadata in Kaggle/Linux. Local
macOS tests cannot load the bundled Linux native library, so this module keeps a
small static fallback for the current deck and lets the official API overwrite
it whenever that API is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class AttackInfo:
    attack_id: int
    name: str = ""
    damage: int = 0
    energy_count: int = 0


@dataclass(frozen=True)
class CardInfo:
    card_id: int
    name: str
    kind: str = "unknown"
    hp: int = 0
    energy_type: str | None = None
    evolves_from: str | None = None
    basic: bool = False
    stage1: bool = False
    stage2: bool = False
    ex: bool = False
    mega_ex: bool = False
    ace_spec: bool = False
    effect_text: str = ""
    attack_ids: tuple[int, ...] = ()

    @property
    def is_pokemon(self) -> bool:
        return self.kind == "pokemon"

    @property
    def is_energy(self) -> bool:
        return self.kind in {"basic_energy", "special_energy"}

    @property
    def is_trainer(self) -> bool:
        return self.kind in {"item", "tool", "supporter", "stadium"}


# Fallback facts are intentionally small. They should cover the submitted deck
# and local tests; the full simulator database is preferred at runtime.
STATIC_CARDS: dict[int, CardInfo] = {
    1: CardInfo(1, "Basic Grass Energy", "basic_energy", energy_type="grass"),
    2: CardInfo(2, "Basic Fire Energy", "basic_energy", energy_type="fire"),
    3: CardInfo(3, "Basic Water Energy", "basic_energy", energy_type="water"),
    4: CardInfo(4, "Basic Lightning Energy", "basic_energy", energy_type="lightning"),
    5: CardInfo(5, "Basic Psychic Energy", "basic_energy", energy_type="psychic"),
    6: CardInfo(6, "Basic Fighting Energy", "basic_energy", energy_type="fighting"),
    7: CardInfo(7, "Basic Darkness Energy", "basic_energy", energy_type="darkness"),
    8: CardInfo(8, "Basic Metal Energy", "basic_energy", energy_type="metal"),
    154: CardInfo(
        154,
        "Lapras ex",
        "pokemon",
        hp=220,
        energy_type="water",
        basic=True,
        ex=True,
        effect_text="Basic water attacker that scales with attached Energy.",
    ),
    721: CardInfo(
        721,
        "Kyogre",
        "pokemon",
        hp=150,
        energy_type="water",
        basic=True,
        effect_text="Riptide uses Basic Water Energy in the discard pile.",
    ),
    722: CardInfo(722, "Snover", "pokemon", hp=90, energy_type="water", basic=True),
    723: CardInfo(
        723,
        "Mega Abomasnow ex",
        "pokemon",
        hp=350,
        energy_type="water",
        evolves_from="Snover",
        stage1=True,
        ex=True,
        mega_ex=True,
        effect_text="Hammer-lanche discards the top six cards and rewards Basic Water Energy density.",
    ),
    944: CardInfo(
        944,
        "Regice ex",
        "pokemon",
        hp=230,
        energy_type="water",
        basic=True,
        ex=True,
        effect_text="Regi Charge attaches Basic Water Energy from discard.",
    ),
    1121: CardInfo(
        1121,
        "Ultra Ball",
        "item",
        effect_text="Discard two cards. Search your deck for a Pokemon.",
    ),
    1145: CardInfo(
        1145,
        "Mega Signal",
        "item",
        effect_text="Search your deck for a Mega Evolution Pokemon ex.",
    ),
    1158: CardInfo(
        1158,
        "Maximum Belt",
        "tool",
        ace_spec=True,
        effect_text="Attached Pokemon does more damage to opposing Pokemon ex.",
    ),
    1182: CardInfo(
        1182,
        "Boss's Orders",
        "supporter",
        effect_text="Switch in one of your opponent's Benched Pokemon.",
    ),
    1227: CardInfo(
        1227,
        "Lillie's Determination",
        "supporter",
        effect_text="Shuffle your hand into your deck, then draw cards.",
    ),
    1235: CardInfo(
        1235,
        "Waitress",
        "supporter",
        effect_text="Look at the top six cards and attach a Basic Energy to one of your Pokemon.",
    ),
}


# These values mirror cg.api enums without importing cg.api at module import
# time. Importing cg.api loads the native simulator library.
CARD_TYPE_BY_VALUE = {
    0: "pokemon",
    1: "item",
    2: "tool",
    3: "supporter",
    4: "stadium",
    5: "basic_energy",
    6: "special_energy",
}

ENERGY_BY_VALUE = {
    0: "colorless",
    1: "grass",
    2: "fire",
    3: "water",
    4: "lightning",
    5: "psychic",
    6: "fighting",
    7: "darkness",
    8: "metal",
    9: "dragon",
    10: "rainbow",
    11: "team_rocket",
}


def _enum_value(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _text_from_skills(card) -> str:
    # Skill text is useful for generic role detection, but the official card
    # objects may append new fields over time. getattr keeps this tolerant.
    skills = getattr(card, "skills", None) or []
    parts = []
    for skill in skills:
        name = getattr(skill, "name", "")
        text = getattr(skill, "text", "")
        if name:
            parts.append(str(name))
        if text:
            parts.append(str(text))
    return " ".join(parts)


@lru_cache(maxsize=1)
def load_card_database() -> tuple[dict[int, CardInfo], dict[int, AttackInfo]]:
    """Return card and attack facts.

    In Kaggle/Linux this can use the official simulator API. On local macOS the
    bundled native library is not loadable, so we fall back to lightweight facts
    for the deck currently in this repository.
    """
    cards = dict(STATIC_CARDS)
    attacks: dict[int, AttackInfo] = {}

    try:
        from cg.api import all_attack, all_card_data

        # In the real submission environment this branch gives the policy a
        # fuller view of every card in the simulator, not just the fallback set.
        for attack in all_attack():
            attack_id = int(getattr(attack, "attackId"))
            energies = getattr(attack, "energies", None) or []
            damage = getattr(attack, "damage", 0) or 0
            attacks[attack_id] = AttackInfo(
                attack_id=attack_id,
                name=str(getattr(attack, "name", "")),
                damage=int(damage),
                energy_count=len(energies),
            )

        for card in all_card_data():
            card_id = int(getattr(card, "cardId"))
            card_type = CARD_TYPE_BY_VALUE.get(_enum_value(getattr(card, "cardType", None)), "unknown")
            energy_type = ENERGY_BY_VALUE.get(_enum_value(getattr(card, "energyType", None)))
            rule_text = ""
            if getattr(card, "ex", False):
                rule_text += " ex"
            if getattr(card, "megaEx", False):
                rule_text += " mega"
            if getattr(card, "aceSpec", False):
                rule_text += " ace"

            cards[card_id] = CardInfo(
                card_id=card_id,
                name=str(getattr(card, "name", "")),
                kind=card_type,
                hp=int(getattr(card, "hp", 0) or 0),
                energy_type=energy_type,
                evolves_from=getattr(card, "evolvesFrom", None),
                basic=bool(getattr(card, "basic", False)),
                stage1=bool(getattr(card, "stage1", False)),
                stage2=bool(getattr(card, "stage2", False)),
                ex=bool(getattr(card, "ex", False)),
                mega_ex=bool(getattr(card, "megaEx", False)),
                ace_spec=bool(getattr(card, "aceSpec", False)),
                effect_text=f"{rule_text} {_text_from_skills(card)}".strip(),
                attack_ids=tuple(int(v) for v in (getattr(card, "attacks", None) or [])),
            )
    except Exception:
        # Local development on macOS lands here because libcg.so is Linux-only.
        # The fallback is enough for unit tests and still keeps the submission
        # offline and self-contained.
        pass

    return cards, attacks
