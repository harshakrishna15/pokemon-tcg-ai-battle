#!/usr/bin/env python3
"""Focused regression tests for heuristic decisions."""

from __future__ import annotations

import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION = os.path.join(ROOT, "sample_submission")
sys.dont_write_bytecode = True
sys.path.insert(0, SUBMISSION)

from agent_policy import (  # noqa: E402
    AREA_ACTIVE,
    AREA_HAND,
    OPT_ATTACH,
    OPT_ATTACK,
    OPT_PLAY,
    choose_action,
    read_deck_csv,
)


def card(card_id: int) -> dict:
    return {"id": card_id, "serial": card_id * 100, "playerIndex": 0}


def pokemon(card_id: int, hp: int, energy_count: int = 0, energy_id: int = 3) -> dict:
    return {
        "id": card_id,
        "serial": card_id * 100,
        "playerIndex": 0,
        "hp": hp,
        "maxHp": hp,
        "appearThisTurn": False,
        "energies": [energy_id] * energy_count,
        "energyCards": [card(energy_id) for _ in range(energy_count)],
        "tools": [],
        "preEvolution": [],
    }


def base_obs(select: dict) -> dict:
    return {
        "select": select,
        "logs": [],
        "current": {
            "turn": 3,
            "turnActionCount": 0,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [
                {
                    "active": [pokemon(154, 220, energy_count=4)],
                    "bench": [],
                    "benchMax": 5,
                    "deckCount": 45,
                    "discard": [],
                    "prize": [],
                    "handCount": 3,
                    "hand": [card(1235), card(3), card(1227)],
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
                {
                    "active": [pokemon(944, 160)],
                    "bench": [],
                    "benchMax": 5,
                    "deckCount": 45,
                    "discard": [],
                    "prize": [],
                    "handCount": 4,
                    "hand": None,
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
            ],
        },
    }


class HeuristicScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.deck = read_deck_csv()

    def test_attack_for_prize_beats_extra_energy_with_placeholder_attack_id(self) -> None:
        obs = base_obs(
            {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": OPT_PLAY, "index": 0},
                    {
                        "type": OPT_ATTACH,
                        "area": AREA_HAND,
                        "index": 1,
                        "playerIndex": 0,
                        "inPlayArea": AREA_ACTIVE,
                        "inPlayIndex": 0,
                    },
                    {"type": OPT_ATTACK, "attackId": 999},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        self.assertEqual(choose_action(obs, self.deck), [2])


if __name__ == "__main__":
    unittest.main()
