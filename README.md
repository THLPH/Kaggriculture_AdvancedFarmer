# Kaggriculture Bot - Version 1

An autonomous agent built for Kaggle's Kaggriculture simulation competition.

## Strategy
- **Crops**: Focuses on planting, daily watering, and harvesting Wheat on the starting 5x5 quadrant.
- **Market**: Automatically sells harvested wheat from the shed and buys replacement seeds.
- **Pathfinding**: Simple Manhattan distance grid navigation targeting tasks by priority (Water > Harvest > Dig > Plant).

## Built With
- Python 3
- `kaggle-environments`
