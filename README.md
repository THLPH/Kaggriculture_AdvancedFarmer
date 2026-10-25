# Kavya's Orchestrator

**Heuristic economic orchestration for Kaggriculture**

An autonomous agent developed for [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) as part of **Team Advanced Farmer**. The implementation coordinates crop production, livestock operations, market transactions, expansion decisions, and worker assignments within a single rule-based decision system.

## Approach

Rather than focusing exclusively on crop operations, the agent evaluates several competing uses of capital and labor throughout the season. Each turn, it reads the current observation, generates market decisions and farm tasks, and returns actions for the primary farmer and available hired hands.

### Economic planning

- **Crop selection:** Scores candidate crops using estimated production value, seed cost, growth time, current market inventory, and visible standing production across both farms.
- **Market operations:** Uses price-relative thresholds and limited-size selling batches, with more aggressive liquidation toward the end of the season.
- **Livestock:** Evaluates cows, geese, and sheep using heuristic estimates of expected revenue, maintenance costs, and remaining production time.
- **Land expansion:** Uses a payback-style condition to evaluate expansion against purchase cost and remaining time.
- **Spending controls:** Applies budget reserves and configurable limits when purchasing inputs or making investments.

### Task orchestration

The task generator considers planting, watering, harvesting, weeding, animal feeding and care, collecting produce and fertilizer, building structures, and moving goods between workers and the shed. Tasks have assigned priorities; each worker greedily selects an unreserved task, using Manhattan distance to break priority ties. Reservations reduce duplicate task selection within a turn.

This is **priority-based heuristic scheduling**, not global assignment optimization or a learned policy.

## Implementation

| File | Purpose |
| --- | --- |
| `main.py` | Standalone Kaggriculture agent exposing `agent(obs, config=None)` |
| `test.py` | Local match harness against Kaggle's built-in starter opponent |

The submission agent uses the Python standard library. For local evaluation, install `kaggle-environments` and run:

```bash
python test.py
```

The provided test harness runs three seeds from both player positions (six matches total), then reports terminal cash and margins where observations are available. Results from those runs have not been independently verified in this documentation.

## Current limitations

- Crop choice used for seed purchases is not consistently carried through to planting when different seeds are already held.
- Worker assignments are greedy and do not retain task targets between turns.
- Routine unloading of carried produce is narrowly scheduled; this may reduce inventory throughput.
- Economic projections use hand-tuned assumptions rather than fitted forecasts or demonstrated calibrated estimates.
- The agent catches unexpected exceptions and returns a safe `PASS` action, which can mask implementation errors during testing.

## Attribution

Developed as part of **Team Advanced Farmer**. The Kaggriculture environment and starter materials are provided by Kaggle.
