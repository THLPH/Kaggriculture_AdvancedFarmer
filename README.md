# Advanced Farmer — Kaggriculture

**Autonomous Farming Agents · Worker Coordination · Economic Decision-Making**

**Advanced Farmer** is a collaborative project developed for the [Kaggriculture competition](https://www.kaggle.com/competitions/kaggriculture) on Kaggle. This repository brings together independently developed agents and supporting experiments to investigate different approaches to autonomous farm management.

## Project Overview

In Kaggriculture, autonomous agents make decisions about farming operations, labor, inventory, and market transactions over a fixed simulation horizon. Our work examines how different decision-making methods balance immediate operational needs against longer-term economic objectives.

Rather than presenting one unified algorithm, this project documents multiple approaches created by team members. Each implementation addresses a different part of the decision-making problem.

## The Kaggriculture Challenge

The competition takes place over a **30-day season (720 turns)**. Agents operate within a grid-based farming environment and must coordinate production, worker actions, and financial decisions while competing for the strongest final economic outcome.

For the full environment specification, action formats, and competition rules, refer to the [official competition documentation](https://www.kaggle.com/competitions/kaggriculture). The repository does not duplicate the provided starter guide.

## Agent Implementations

| Implementation | Approach | Primary focus |
| --- | --- | --- |
| **Farming Agent V1** | Rule-based | Wheat cultivation, deterministic task selection, grid navigation, and basic market transactions. |
| **Worker-Optimized Agent V2** | Utility-based worker scheduling | Global assignment of workers to candidate tasks using travel costs, urgency, continuity, and reservations. |
| **RL Lion** | Behavior cloning and PPO | CNN/MLP policy, masked multi-worker actions, curriculum learning, self-play, and parallel simulation. |
| **Kavya's Orchestrator** | Heuristic economic orchestration | Crop profitability estimation, livestock investment, land expansion, market decisions, and coordinated task execution. |

These agents are separate development efforts. Their names do not imply a verified performance ranking or a shared version lineage.

### Farming Agent V1

A compact baseline that scans a fixed 5×5 farming area, selects eligible wheat-farming tasks, and moves the farmer using deterministic grid navigation. It performs basic wheat-seed purchases and sells wheat held in the shed. This implementation provides a simple reference point for discussing more elaborate decisions, without claiming to be an optimized benchmark.


### Worker-Optimized Agent V2

This agent emphasizes **worker scheduling** rather than broad economic diversification. It scores candidate jobs using task value, movement cost, urgency, target continuity, and local opportunities. It then searches candidate worker–task assignments while preventing conflicting target reservations and excess simultaneous seed use. Its documented crop policy uses wheat and carrots.


### RL Lion

RL Lion investigates **learning-based farm management** using a behavior-cloning initialization followed by **Proximal Policy Optimization (PPO)** with Generalized Advantage Estimation (GAE). Its policy combines a convolutional encoder for the farm grid with a multilayer perceptron for global game-state features. The model has separate action heads for worker and market decisions, plus a value head; action masks constrain decisions to valid actions.

### Kavya's Orchestrator

An integrated heuristic agent exploring the relationship between **economic planning and farm operations**. It estimates crop profitability using available market information, evaluates livestock investments and land-expansion opportunities, manages purchases and sales, and creates operational tasks for available workers. Task assignment uses heuristic priorities and distance-aware movement rather than global assignment optimization.


## Research and Supporting Experiments

### Dimension Reduction

The project includes a separate dimension-reduction workstream. Its relationship to individual agent implementations is not yet established.


### Column-Based Worker Allocation

A proposed spatial allocation scheme partitions a farming region into columns and assigns workers to designated areas. The design aims to make responsibilities predictable and reduce conflicting assignments.


## Evaluation Methodology

Agent comparisons should use consistent episode seeds, starting positions, and opponents wherever possible. Tests should record match completion, final rewards, and win/loss outcomes to distinguish reliable execution from strategic performance. For RL Lion, evaluation should additionally distinguish behavior-cloning and PPO checkpoints and report the opponents used.

No leaderboard placement or relative performance claim is made without supporting test results.

## Technologies

- **Python** — Agent development, simulation, and evaluation.
- **Kaggle Environments** — Competition simulator and agent interface.
- **PyTorch and NumPy** — RL Lion's neural policy, training, and observation processing.
- **Python multiprocessing** — Optional parallel simulation for RL training.
- **Git and GitHub** — Version control, collaboration, and pull-request review.


## Contributors

Developed collaboratively by **Team Advanced Farmer**.


## Acknowledgments

The Kaggriculture competition, environment, and original starter resources are provided by **Kaggle**. The team's agent implementations and experiments are documented separately from those materials.

## References

- [Kaggriculture — Official Competition](https://www.kaggle.com/competitions/kaggriculture)
- [Kaggle Environments — Source Repository](https://github.com/Kaggle/kaggle-environments)
