"""Kaggle agent entry point.

Keep this file small: Kaggle calls `agent(obs_dict)` directly, and helper
modules must live beside this file in the submission zip root.
"""

from agent_policy import choose_action, read_deck_csv


# Read the deck once at import time so every initial observation returns the
# same 60-card list without repeatedly touching the filesystem.
DECK = read_deck_csv()


def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    if obs_dict.get("select") is None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return DECK
    
    # All in-game choices are returned as indexes into obs.select.option.
    # The policy module keeps the scoring generic so deck.csv can change
    # without rewriting this entry point.
    return choose_action(obs_dict, DECK)
