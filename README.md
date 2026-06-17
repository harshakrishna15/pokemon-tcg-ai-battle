# Pokemon TCG AI Battle

Local tooling for iterating on the Kaggle Pokemon TCG AI Battle submission.

## Run Matches

```bash
python3 tools/run_cg_match.py --games 10 --p0 agent --p1 agent-no-search --quiet
```

For more stable comparisons, use seat-swapped multi-seed experiments:

```bash
python3 tools/run_cg_experiments.py \
  --games 40 \
  --policy agent \
  --opponents agent-no-search \
  --seat-swap \
  --seeds 1 2 3 4 5 \
  --quiet
```

## Extract Training Data

`tools/extract_training_data.py` writes one row per legal option at each
decision point. The default output is a gzip-compressed CSV plus a sidecar
metadata JSON file with the schema and run settings.

```bash
python3 tools/extract_training_data.py \
  --games 200 \
  --p0 agent-no-search \
  --p1 first \
  --seat-swap \
  --seed 1 \
  --output training_data/option_features.csv.gz \
  --quiet
```

Load it with pandas:

```python
import pandas as pd

df = pd.read_csv("training_data/option_features.csv.gz")
```

Useful label columns:

- `selected`: whether the policy selected this legal option.
- `selected_rank`: order within a multi-option selection, or `-1`.
- `outcome_for_player`: final result from the decision player's perspective.
- `final_winner`: simulator winner id, with `2` meaning draw.

## Scrape Limitless NAIC 2026 Decklists

`tools/scrape_limitless_naic_2026.py` fetches the public NAIC 2026
decklists from Limitless and standings metadata from Limitless Labs. It writes
compressed files that are easy to load without keeping the large raw HTML.

```bash
python3 tools/scrape_limitless_naic_2026.py \
  --output-dir data/limitless/naic_2026
```

Outputs:

- `standings.csv.gz`: one row per player in the Labs standings.
- `decklist_cards.csv.gz`: one row per card line in each published list.
- `decklists.jsonl.gz`: one JSON object per published 60-card deck.
- `metadata.json`: source URLs, fetch timestamp, and row counts.

## Build Deck Candidates

Use the scraped Limitless data to create simulator-compatible candidate decks.
The builder maps direct set/number matches first, then safe name reprints and
basic energy print variants.

```bash
python3 tools/build_deck_candidates.py --limit 250
```

Evaluate generated candidates against the current submission deck:

```bash
python3 tools/evaluate_deck_candidates.py \
  --games 10 \
  --limit 20 \
  --seed 23
```

Promising decks should then get a larger seat-swapped seed sweep before
replacing `sample_submission/deck.csv`.
