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
    text: str = ""
    damage: int = 0
    energy_count: int = 0
    energy_types: tuple[str, ...] = ()


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
    weakness: str | None = None
    resistance: str | None = None

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
STATIC_ATTACKS: dict[int, AttackInfo] = {
    1541: AttackInfo(
        1541,
        "Power Splash",
        "This attack does 40 damage for each Energy attached to this Pokemon.",
        damage=40,
        energy_count=1,
        energy_types=("water",),
    ),
    7211: AttackInfo(
        7211,
        "Riptide",
        "This attack does 20 damage for each Basic {W} Energy card in your discard pile.",
        damage=20,
        energy_count=1,
        energy_types=("water",),
    ),
    7212: AttackInfo(
        7212,
        "Swirling Waves",
        "Discard 2 Energy from this Pokemon.",
        damage=130,
        energy_count=3,
        energy_types=("water", "water", "colorless"),
    ),
    7221: AttackInfo(7221, "Flop", "", damage=10, energy_count=1, energy_types=("water",)),
    7231: AttackInfo(
        7231,
        "Hammer-lanche",
        "Discard the top 6 cards of your deck. This attack does 100 damage for each Basic Energy discarded this way.",
        damage=100,
        energy_count=3,
        energy_types=("water", "water", "colorless"),
    ),
    7911: AttackInfo(
        7911,
        "Fighting Wings",
        "If your opponent's Active Pokemon is a Pokemon ex, this attack does 90 more damage.",
        damage=20,
        energy_count=1,
        energy_types=("fire",),
    ),
    9441: AttackInfo(
        9441,
        "Regi Charge",
        "Attach up to 2 Basic {W} Energy cards from your discard pile to this Pokemon.",
        damage=0,
        energy_count=1,
        energy_types=("colorless",),
    ),
    9442: AttackInfo(
        9442,
        "Ice Prison",
        "Discard 2 Energy from this Pokemon, and your opponent's Active Pokemon is now Paralyzed.",
        damage=140,
        energy_count=4,
        energy_types=("water", "colorless", "colorless", "colorless"),
    ),
    9571: AttackInfo(9571, "Slashing Claw", "", damage=40, energy_count=1, energy_types=("lightning",)),
    9572: AttackInfo(
        9572,
        "Hadron Spark",
        "If your opponent's Active Pokemon is a Pokemon ex, this attack does 120 more damage.",
        damage=120,
        energy_count=3,
        energy_types=("lightning", "lightning", "colorless"),
    ),
    10301: AttackInfo(10301, "Water Gun", "", damage=20, energy_count=1, energy_types=("water",)),
    10311: AttackInfo(
        10311,
        "Jetting Blow",
        "This attack also does 50 damage to 1 of your opponent's Benched Pokemon.",
        damage=120,
        energy_count=1,
        energy_types=("water",),
    ),
    10312: AttackInfo(
        10312,
        "Nebula Beam",
        "This attack's damage isn't affected by Weakness or Resistance, or by any effects on your opponent's Active Pokemon.",
        damage=210,
        energy_count=3,
        energy_types=("colorless", "colorless", "colorless"),
    ),
}

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
        attack_ids=(1541,),
        weakness="metal",
    ),
    96: CardInfo(
        96,
        "Teal Mask Ogerpon ex",
        "pokemon",
        hp=210,
        energy_type="grass",
        basic=True,
        ex=True,
        effect_text="Grass ex attacker that accelerates Grass Energy.",
        weakness="fire",
    ),
    721: CardInfo(
        721,
        "Kyogre",
        "pokemon",
        hp=150,
        energy_type="water",
        basic=True,
        effect_text="Riptide uses Basic Water Energy in the discard pile.",
        attack_ids=(7211, 7212),
        weakness="lightning",
    ),
    722: CardInfo(722, "Snover", "pokemon", hp=90, energy_type="water", basic=True, attack_ids=(7221,)),
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
        attack_ids=(7231,),
        weakness="metal",
    ),
    791: CardInfo(
        791,
        "Moltres",
        "pokemon",
        hp=120,
        energy_type="fire",
        basic=True,
        effect_text="Fighting Wings does extra damage if the opponent's Active Pokemon is ex.",
        attack_ids=(7911,),
        weakness="water",
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
        attack_ids=(9441, 9442),
        weakness="metal",
    ),
    957: CardInfo(
        957,
        "Miraidon ex",
        "pokemon",
        hp=220,
        energy_type="lightning",
        basic=True,
        ex=True,
        effect_text="Hadron Spark does extra damage if the opponent's Active Pokemon is ex.",
        attack_ids=(9571, 9572),
        weakness="fighting",
    ),
    1030: CardInfo(
        1030,
        "Staryu",
        "pokemon",
        hp=70,
        energy_type="water",
        basic=True,
        effect_text="Basic Water Pokemon that evolves into Mega Starmie ex.",
        attack_ids=(10301,),
        weakness="lightning",
    ),
    1031: CardInfo(
        1031,
        "Mega Starmie ex",
        "pokemon",
        hp=330,
        energy_type="water",
        evolves_from="Staryu",
        stage1=True,
        ex=True,
        mega_ex=True,
        effect_text="Jetting Blow attacks for one Water Energy and hits the Bench. Nebula Beam is a three-Energy attack.",
        attack_ids=(10311, 10312),
        weakness="lightning",
    ),
    1121: CardInfo(
        1121,
        "Ultra Ball",
        "item",
        effect_text="Discard two cards. Search your deck for a Pokemon.",
    ),
    1086: CardInfo(
        1086,
        "Buddy-Buddy Poffin",
        "item",
        effect_text="Search your deck for up to 2 Basic Pokemon with 70 HP or less and put them onto your Bench.",
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
    1088: CardInfo(
        1088,
        "Prime Catcher",
        "item",
        ace_spec=True,
        effect_text="Switch in one of your opponent's Benched Pokemon. If you do, switch your Active Pokemon.",
    ),
    1182: CardInfo(
        1182,
        "Boss's Orders",
        "supporter",
        effect_text="Switch in one of your opponent's Benched Pokemon.",
    ),
    1198: CardInfo(
        1198,
        "Crispin",
        "supporter",
        effect_text="Search your deck for two different Basic Energy cards. Put one into your hand and attach the other to one of your Pokemon.",
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
    attacks: dict[int, AttackInfo] = dict(STATIC_ATTACKS)

    try:
        from cg.api import all_attack, all_card_data

        # In the real submission environment this branch gives the policy a
        # fuller view of every card in the simulator, not just the fallback set.
        for attack in all_attack():
            attack_id = int(getattr(attack, "attackId"))
            energies = getattr(attack, "energies", None) or []
            damage = getattr(attack, "damage", 0) or 0
            energy_types = tuple(
                ENERGY_BY_VALUE.get(_enum_value(energy), "colorless")
                for energy in energies
            )
            attacks[attack_id] = AttackInfo(
                attack_id=attack_id,
                name=str(getattr(attack, "name", "")),
                text=str(getattr(attack, "text", "")),
                damage=int(damage),
                energy_count=len(energies),
                energy_types=energy_types,
            )

        for card in all_card_data():
            card_id = int(getattr(card, "cardId"))
            card_type = CARD_TYPE_BY_VALUE.get(_enum_value(getattr(card, "cardType", None)), "unknown")
            energy_type = ENERGY_BY_VALUE.get(_enum_value(getattr(card, "energyType", None)))
            weakness = ENERGY_BY_VALUE.get(_enum_value(getattr(card, "weakness", None)))
            resistance = ENERGY_BY_VALUE.get(_enum_value(getattr(card, "resistance", None)))
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
                weakness=weakness,
                resistance=resistance,
            )
    except Exception:
        # Local development on macOS lands here because libcg.so is Linux-only.
        # The fallback is enough for unit tests and still keeps the submission
        # offline and self-contained.
        pass

    return cards, attacks
