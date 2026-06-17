import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION = os.path.join(ROOT, "sample_submission")
sys.path.insert(0, SUBMISSION)

from agent_policy import (  # noqa: E402
    AREA_ACTIVE,
    AREA_BENCH,
    AREA_DECK,
    AREA_HAND,
    CTX_ATTACH_TO,
    CTX_DISCARD,
    CTX_SETUP_ACTIVE,
    CTX_TO_HAND,
    OPT_ATTACH,
    OPT_ATTACK,
    OPT_CARD,
    OPT_END,
    OPT_EVOLVE,
    OPT_PLAY,
    SEL_CARD,
    choose_action,
    read_deck_csv,
)


def card(card_id):
    return {"id": card_id, "serial": card_id * 100, "playerIndex": 0}


def pokemon(card_id, hp=100, energy_count=0):
    return {
        "id": card_id,
        "serial": card_id * 100,
        "playerIndex": 0,
        "hp": hp,
        "maxHp": hp,
        "appearThisTurn": False,
        "energies": [3] * energy_count,
        "energyCards": [card(3) for _ in range(energy_count)],
        "tools": [],
        "preEvolution": [],
    }


def base_obs(select):
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
                    "active": [pokemon(721, 150)],
                    "bench": [pokemon(722, 90)],
                    "benchMax": 5,
                    "deckCount": 45,
                    "discard": [],
                    "prize": [],
                    "handCount": 4,
                    "hand": [card(723), card(3), card(1145), card(1227)],
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
                {
                    "active": [pokemon(154, 220)],
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


class AgentPolicyTests(unittest.TestCase):
    def setUp(self):
        self.deck = read_deck_csv()

    def test_initial_observation_returns_deck(self):
        self.assertEqual(choose_action({"select": None}, self.deck), self.deck)
        self.assertEqual(len(self.deck), 60)

    def test_setup_active_prefers_non_ex_basic_attacker(self):
        obs = base_obs(
            {
                "type": SEL_CARD,
                "context": CTX_SETUP_ACTIVE,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": OPT_CARD, "area": AREA_HAND, "index": 0, "playerIndex": 0},
                    {"type": OPT_CARD, "area": AREA_HAND, "index": 1, "playerIndex": 0},
                    {"type": OPT_CARD, "area": AREA_HAND, "index": 2, "playerIndex": 0},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["hand"] = [card(722), card(154), card(721)]
        self.assertEqual(choose_action(obs, self.deck), [2])

    def test_main_phase_prefers_evolution_before_attack(self):
        obs = base_obs(
            {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {
                        "type": OPT_EVOLVE,
                        "area": AREA_HAND,
                        "index": 0,
                        "playerIndex": 0,
                        "inPlayArea": AREA_BENCH,
                        "inPlayIndex": 0,
                    },
                    {"type": OPT_ATTACK, "attackId": 999},
                    {"type": OPT_END},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        self.assertEqual(choose_action(obs, self.deck), [0])

    def test_deck_search_takes_needed_evolution(self):
        obs = base_obs(
            {
                "type": SEL_CARD,
                "context": CTX_TO_HAND,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": OPT_CARD, "area": AREA_DECK, "index": 0, "playerIndex": 0},
                    {"type": OPT_CARD, "area": AREA_DECK, "index": 1, "playerIndex": 0},
                    {"type": OPT_CARD, "area": AREA_DECK, "index": 2, "playerIndex": 0},
                ],
                "deck": [card(3), card(723), card(1227)],
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["hand"] = [card(3), card(1145)]
        self.assertEqual(choose_action(obs, self.deck), [1])

    def test_discard_prefers_basic_energy(self):
        obs = base_obs(
            {
                "type": SEL_CARD,
                "context": CTX_DISCARD,
                "minCount": 2,
                "maxCount": 2,
                "option": [
                    {"type": OPT_CARD, "area": AREA_HAND, "index": 0, "playerIndex": 0},
                    {"type": OPT_CARD, "area": AREA_HAND, "index": 1, "playerIndex": 0},
                    {"type": OPT_CARD, "area": AREA_HAND, "index": 2, "playerIndex": 0},
                    {"type": OPT_CARD, "area": AREA_HAND, "index": 3, "playerIndex": 0},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["hand"] = [card(723), card(3), card(3), card(1227)]
        self.assertEqual(choose_action(obs, self.deck), [1, 2])

    def test_attach_targets_best_attacker(self):
        obs = base_obs(
            {
                "type": 0,
                "context": CTX_ATTACH_TO,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {
                        "type": OPT_ATTACH,
                        "area": AREA_HAND,
                        "index": 1,
                        "playerIndex": 0,
                        "inPlayArea": AREA_ACTIVE,
                        "inPlayIndex": 0,
                    },
                    {
                        "type": OPT_ATTACH,
                        "area": AREA_HAND,
                        "index": 1,
                        "playerIndex": 0,
                        "inPlayArea": AREA_BENCH,
                        "inPlayIndex": 0,
                    },
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["active"] = [pokemon(721, 150)]
        obs["current"]["players"][0]["bench"] = [pokemon(723, 350)]
        self.assertEqual(choose_action(obs, self.deck), [1])

    def test_main_phase_uses_search_before_end(self):
        obs = base_obs(
            {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": OPT_PLAY, "index": 2},
                    {"type": OPT_END},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        self.assertEqual(choose_action(obs, self.deck), [0])

    def test_attack_after_setup_progress_beats_more_energy_stacking(self):
        obs = base_obs(
            {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": OPT_PLAY, "index": 3},
                    {
                        "type": OPT_ATTACH,
                        "area": AREA_HAND,
                        "index": 1,
                        "playerIndex": 0,
                        "inPlayArea": AREA_ACTIVE,
                        "inPlayIndex": 0,
                    },
                    {"type": OPT_ATTACK, "attackId": 999},
                    {"type": OPT_END},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["turnActionCount"] = 2
        obs["current"]["energyAttached"] = True
        self.assertEqual(choose_action(obs, self.deck), [2])

    def test_attack_beats_draw_when_attack_is_already_available(self):
        obs = base_obs(
            {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": OPT_PLAY, "index": 3},
                    {"type": OPT_ATTACK, "attackId": 999},
                    {"type": OPT_END},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        self.assertEqual(choose_action(obs, self.deck), [1])

    def test_energy_acceleration_can_happen_before_attack_for_scaling_attacker(self):
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
        obs["current"]["players"][0]["active"] = [pokemon(154, 220, energy_count=1)]
        obs["current"]["players"][0]["hand"] = [card(1235), card(3), card(1227)]
        self.assertEqual(choose_action(obs, self.deck), [0])

    def test_manual_attach_can_happen_before_attack_for_scaling_attacker(self):
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
                    {"type": OPT_ATTACK, "attackId": 999},
                ],
                "deck": None,
                "contextCard": None,
                "effect": None,
            }
        )
        obs["current"]["players"][0]["active"] = [pokemon(154, 220, energy_count=2)]
        obs["current"]["players"][0]["hand"] = [card(3), card(1227)]
        self.assertEqual(choose_action(obs, self.deck), [0])

    def test_attack_wins_when_scaling_attacker_is_already_built(self):
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
        obs["current"]["players"][0]["active"] = [pokemon(154, 220, energy_count=5)]
        obs["current"]["players"][0]["hand"] = [card(1235), card(3), card(1227)]
        self.assertEqual(choose_action(obs, self.deck), [2])


if __name__ == "__main__":
    unittest.main()
