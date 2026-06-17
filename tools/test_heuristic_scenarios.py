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
    AREA_BENCH,
    AREA_DECK,
    AREA_HAND,
    CTX_ATTACH_TO,
    CTX_DRAW_COUNT,
    CTX_TO_HAND,
    OPT_ATTACH,
    OPT_ATTACK,
    OPT_CARD,
    OPT_NUMBER,
    OPT_PLAY,
    OPT_RETREAT,
    SEL_COUNT,
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

    def test_draw_count_does_not_choose_more_than_remaining_deck(self) -> None:
        obs = base_obs(
            {
                "type": SEL_COUNT,
                "context": CTX_DRAW_COUNT,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": OPT_NUMBER, "number": 1},
                    {"type": OPT_NUMBER, "number": 3},
                    {"type": OPT_NUMBER, "number": 5},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["deckCount"] = 3

        self.assertEqual(choose_action(obs, self.deck), [1])

    def test_search_ev_prefers_needed_mega_evolution_search(self) -> None:
        obs = base_obs(
            {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": OPT_PLAY, "index": 0},
                    {"type": OPT_PLAY, "index": 1},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["active"] = [pokemon(1030, 70, energy_count=1)]
        obs["current"]["players"][0]["hand"] = [card(1145), card(1227)]
        obs["current"]["players"][0]["handCount"] = 2

        self.assertEqual(choose_action(obs, self.deck, use_search=False), [0])

    def test_threat_guardrail_retreats_instead_of_feeding_doomed_active(self) -> None:
        obs = base_obs(
            {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {
                        "type": OPT_ATTACH,
                        "area": AREA_HAND,
                        "index": 0,
                        "playerIndex": 0,
                        "inPlayArea": AREA_ACTIVE,
                        "inPlayIndex": 0,
                    },
                    {"type": OPT_RETREAT},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["active"] = [pokemon(154, 200, energy_count=4)]
        obs["current"]["players"][0]["bench"] = [pokemon(1030, 70, energy_count=0)]
        obs["current"]["players"][0]["hand"] = [card(3)]
        obs["current"]["players"][0]["handCount"] = 1
        obs["current"]["players"][1]["active"] = [pokemon(957, 220, energy_count=2, energy_id=4)]

        self.assertEqual(choose_action(obs, self.deck, use_search=False), [1])

    def test_attachment_powers_nearly_ready_mega_starmie(self) -> None:
        obs = base_obs(
            {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {
                        "type": OPT_ATTACH,
                        "area": AREA_HAND,
                        "index": 0,
                        "playerIndex": 0,
                        "inPlayArea": AREA_ACTIVE,
                        "inPlayIndex": 0,
                    },
                    {
                        "type": OPT_ATTACH,
                        "area": AREA_HAND,
                        "index": 0,
                        "playerIndex": 0,
                        "inPlayArea": AREA_BENCH,
                        "inPlayIndex": 0,
                    },
                    {
                        "type": OPT_ATTACH,
                        "area": AREA_HAND,
                        "index": 0,
                        "playerIndex": 0,
                        "inPlayArea": AREA_BENCH,
                        "inPlayIndex": 1,
                    },
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["active"] = [pokemon(944, 230, energy_count=2)]
        obs["current"]["players"][0]["bench"] = [
            pokemon(1031, 330, energy_count=2),
            pokemon(154, 220, energy_count=2),
        ]
        obs["current"]["players"][0]["hand"] = [card(3)]

        self.assertEqual(choose_action(obs, self.deck, use_search=False), [1])

    def test_energy_pick_prefers_water_when_water_attacker_needs_it(self) -> None:
        obs = base_obs(
            {
                "type": 1,
                "context": CTX_TO_HAND,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": OPT_CARD, "area": AREA_DECK, "index": 0, "playerIndex": 0},
                    {"type": OPT_CARD, "area": AREA_DECK, "index": 1, "playerIndex": 0},
                ],
                "deck": [card(4), card(3)],
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["active"] = [pokemon(1031, 330, energy_count=0)]

        self.assertEqual(choose_action(obs, self.deck, use_search=False), [1])

    def test_effect_attachment_target_prefers_starmie_line_progress(self) -> None:
        obs = base_obs(
            {
                "type": 1,
                "context": CTX_ATTACH_TO,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": OPT_CARD, "area": AREA_ACTIVE, "index": 0, "playerIndex": 0},
                    {"type": OPT_CARD, "area": AREA_BENCH, "index": 0, "playerIndex": 0},
                    {"type": OPT_CARD, "area": AREA_BENCH, "index": 1, "playerIndex": 0},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["active"] = [pokemon(944, 230, energy_count=2)]
        obs["current"]["players"][0]["bench"] = [
            pokemon(1031, 330, energy_count=2),
            pokemon(154, 220, energy_count=2),
        ]

        self.assertEqual(choose_action(obs, self.deck, use_search=False), [1])


if __name__ == "__main__":
    unittest.main()
