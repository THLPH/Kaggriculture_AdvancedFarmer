# Kavya's Orchestrator

**Market-aware economic planning and priority-based farm orchestration for Kaggriculture.**

Kavya's Orchestrator is an autonomous agent developed for Kaggle's [Kaggriculture competition](https://www.kaggle.com/competitions/kaggriculture) as part of **Team Advanced Farmer**. Unlike a crop-only baseline, the implementation combines agricultural production, livestock purchases, market transactions, land expansion, and worker task allocation within one deterministic policy.

The submission-facing `main.py` uses the Python standard library. Its decisions are based on explicit heuristics, not a trained reinforcement-learning policy or an exact global optimizer.

## Design objective

The orchestrator addresses a central economic question:

> How should a farm allocate its available money, production space, and worker turns across competing opportunities over a finite season?

The policy evaluates potential returns from crops, animals, and expansion, then schedules concrete farming tasks for the primary farmer and any hired hands. Its planning is recalculated from the current game observation on each turn.

## Decision pipeline

Each call to `agent(obs, config=None)` follows the same general sequence:

1. **Read state:** Parse money, current day and hour, owned land, workers, inventories, market prices, crops, structures, animals, and empty tiles.
2. **Plan market actions:** Decide what to sell, whether to hire, whether to expand, what seeds to purchase, whether to buy livestock, and whether additional wheat is needed for feed.
3. **Construct farm tasks:** Generate tasks for crop maintenance, construction, animal placement and care, fertilizer collection, pickups, and harvesting.
4. **Assign workers:** Select a currently unreserved task for each worker using fixed priorities and Manhattan distance as a tie-breaker.
5. **Return actions:** Produce the expected `farmer`, `hands`, and `market` action fields.

### 1. Market-aware crop planning

Candidate crops are evaluated using their estimated output, seed cost, growth duration, modeled future selling price, and a cap on concurrent plants. The code also estimates standing production visible on **both farms** to anticipate potential pressure on market inventory.

For each eligible crop, a score is computed in the form:

```text
projected_inventory = market_inventory + standing_weight * visible_production
                      + 0.5 * expected_yield
estimated_price     = average(projected_price_before, projected_price_after_yield)
crop_score          = (estimated_price * expected_yield - seed_cost)
                      / (growth_days + 1)
```

Strawberry and melon scores receive additional fixed multipliers, reflecting manually chosen strategy preferences. The highest-scoring eligible crop determines **seed purchasing**, subject to available plots and a cash reserve.

**Current limitation:** Planting does not always follow this selected crop. The task generator instead chooses the first crop with seeds remaining in inventory, so crop purchasing and actual planting may diverge.

### 2. Metered selling and inventory management

Rather than always selling all stored products, the agent compares current prices with configured base prices and selectively issues sales orders. Routine sales are capped at **three units for selected premium products** and **eight units for other products**. Larger inventories relax the selling threshold, and liquidation becomes more aggressive near the end of the season.

The agent also withholds a portion of stored wheat when animals are present, attempting to preserve feedstock instead of selling it all.

These rules are designed to balance liquidity against the possibility of receiving better prices later. They are heuristic policies, not experimentally validated optimal selling rules.

### 3. Livestock investment and recurring production

The agent considers **cows, geese, and sheep**. It estimates each animal's remaining production opportunity using its acquisition cost, first yield delay, recurring production interval, expected yield, current product price, fertilizer value, approximate feeding cost, and the number of animals already owned.

A simplified version of the implemented model is:

```text
estimated_daily_return = product_yield_per_day * current_product_price * saturation
                         + fertilizer_price * saturation
                         - wheat_price
estimated_roi          = estimated_daily_return * remaining_productive_days
                         / animal_purchase_cost
```

New purchases require the estimated ROI to exceed a configurable threshold and must satisfy animal-count limits, available cash, and reserve constraints. The agent also generates tasks for structures, transporting purchased animals, placement, feeding, collecting yields, care, and fertilizer retrieval.

The estimates are approximate and rely on hand-selected assumptions rather than calibrated production forecasts.

### 4. Land expansion and hiring

**Land expansion** uses a payback-style heuristic: the estimated benefit of additional production tiles over the remaining season is compared with the cost of unlocking more land. Expansion is also restricted by available cash and the number of open crop sites.

**Hiring** responds to a workload estimate based on plants, animals, weeds, and empty tiles. Hiring is considered early in each day, subject to available funds and the configured cap on hired hands.

Both mechanisms are rule-based capacity decisions. Neither solves a formal investment optimization problem.

### 5. Worker task orchestration

The agent generates a pool of actions covering:

- **Farm operations:** harvest, water, dig, and plant.
- **Animal operations:** feed, care, collect produce, and collect fertilizer.
- **Infrastructure:** build pastures and coops, then place purchased livestock.
- **Inventory handling:** pick up wheat or animals from the shed and drop carried goods in the defined conditions.

Each candidate task has a fixed **priority value** and a target position. For each worker, the scheduler selects the unreserved candidate minimizing:

```text
(priority, Manhattan_distance(worker_position, target_position))
```

Lower priority numbers take precedence; distance only breaks ties within the same priority. Workers are processed sequentially, and a task-reservation set discourages duplicate assignment within the turn. Movement uses a deterministic one-step grid action toward the selected target.

**Important distinction:** This is a **priority-based greedy scheduler**, not utility-maximizing global multi-worker matching. It does not maintain persistent worker targets across turns.

## Configuration and design constraints

The implementation exposes several constants near the beginning of `main.py` for experimental tuning:

| Parameter | Default | Purpose |
|---|---:|---|
| `MAX_ANIMALS` | `10` | Maximum planned livestock holdings |
| `ROI_MIN` | `1.5` | Minimum estimated livestock ROI |
| `MAX_HANDS` | `8` | Cap on requested hired hands |
| `SELL_THR` | `0.55` | Base price-relative selling threshold |
| `RESERVE` | `150` | Cash buffer for selected purchases |
| `TILE_VALUE` | `20.0` | Assumed value of an additional tile-day |
| `STAND_W` | `0.6` | Weight assigned to visible standing production |
| `LAND_GAIN` | `0.8` | Factor in the land-payback decision |

These are fixed code defaults. Although the constants are labeled with `V2_` names internally, `_k()` currently returns its default value rather than reading external configuration.

## Files

```text
Kavyas_Orchestrator/
├── main.py    # Standalone agent exposing agent(obs, config=None)
├── test.py    # Local evaluation against the built-in Kaggle starter opponent
└── README.md  # Implementation documentation
```

## Local evaluation

The agent itself requires only the Python standard library. The test harness uses `kaggle-environments`:

```bash
python -m pip install -U kaggle-environments
python test.py
```

Run this from the directory containing both `main.py` and `test.py`.

The included `test.py` defines a **720-turn** evaluation against the built-in `starter` opponent using seeds **31, 32, and 33**, with the agent playing from **both starting positions**. This produces up to six matches and prints terminal cash, score margin, total wins, average cash, and average margin for completed matches. It also checks an internal **60% win-rate gate**.

The gate is a development heuristic rather than proof of general performance. Results are not included here because the matches have not been independently verified. The test script should also be hardened for cases where every episode fails to produce a result.

## Known limitations and next experiments

- **Crop-policy consistency:** Apply the selected economic crop decision to actual planting, not only to seed purchasing.
- **Worker efficiency:** Evaluate joint assignment or continuity-aware scheduling to reduce unnecessary travel and repeated target changes.
- **Inventory throughput:** Improve routine unloading of carried products, since the current explicit `DROP` task for ordinary goods is only generated very late in the final day.
- **Model calibration:** Compare market forecasts and livestock ROI assumptions with realized in-game outcomes.
- **Failure visibility:** The submission wrapper falls back to `PASS` on exceptions; enable more explicit error reporting during development.
- **Benchmark coverage:** Compare additional seeds, both player positions, and stronger agents before making performance claims.

## Acknowledgments

Developed as part of **Team Advanced Farmer**. The simulation environment and competition starter resources are provided by Kaggle. This README documents the behavior implemented in the current agent source; it does not claim a verified ranking or competition result.
