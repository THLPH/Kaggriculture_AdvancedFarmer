# Farming Agent V1

A rule-based autonomous farming agent developed for the [Kaggriculture competition](https://www.kaggle.com/competitions/kaggriculture) as part of **Team Advanced Farmer**.

## Overview

This implementation uses a deterministic, rule-based strategy to automate basic farming operations in the Kaggriculture environment. The agent focuses on wheat cultivation within a fixed 5×5 farming area, combining crop management, grid navigation, and simple market interactions.

The approach prioritizes straightforward decision-making without relying on learning-based methods or complex optimization algorithms.

## Implementation

### Farming Strategy

The agent scans the top-left 5×5 section of the farm and identifies tasks based on the state of each tile.

| Task | Condition | Action |
|---|---|---|
| Harvest | Crop age is at least 2 days | `HARVEST` |
| Water | Crop is younger than 2 days and has not been watered today | `WATER` |
| Clear weeds | Tile contains weeds | `DIG` |
| Plant | Tile is empty and wheat seeds are available | `PLANT WHEAT` |

The agent selects the first eligible task encountered during its row-by-row scan. If the farmer is already at the target location, the action is executed immediately; otherwise, the farmer moves toward it.

### Navigation

Movement follows a deterministic Manhattan-style routing strategy.

The agent prioritizes horizontal movement before vertical movement, selecting one of four directional actions: `NORTH`, `SOUTH`, `EAST`, or `WEST`.

### Market Operations

The agent performs two basic market operations:

- **Seed purchasing:** Purchases one wheat seed when seed inventory falls below five and the available balance is at least 10.
- **Crop selling:** Sells all wheat currently stored in the shed.

Market operations are evaluated each turn alongside the farming decision.

## Agent Interface

The implementation exposes the `agent(obs)` entry point expected by Kaggle Environments.

It receives the current game observation and returns an action dictionary containing:

```python
{
    "farmer": ["ACTION"],
    "hands": [],
    "market": []
}
```

This version controls only the primary farmer and does not assign actions to hired workers.

## Scope and Limitations

The implementation intentionally uses a minimal decision-making strategy.

- Operates within a fixed 5×5 farming region.
- Supports wheat cultivation only.
- Does not implement worker hiring or multi-worker coordination.
- Does not maintain task memory between turns.
- Uses first-encountered task selection rather than global task optimization.
- Does not incorporate market price forecasting or adaptive crop selection.

## Running the Agent

The implementation is intended to be used as a Python agent within the Kaggriculture simulation environment.

The agent function can be imported as:

```python
from main import agent
```

For environment setup and simulation instructions, refer to the [official Kaggriculture documentation](https://www.kaggle.com/competitions/kaggriculture).

## Acknowledgments

Developed as part of **Team Advanced Farmer** using the Kaggriculture simulation environment provided by Kaggle.

For the broader project and other agent implementations, refer to the [Advanced Farmer repository](https://github.com/THLPH/Kaggriculture_AdvancedFarmer).

## Built With
- Python 3
- `kaggle-environments`
