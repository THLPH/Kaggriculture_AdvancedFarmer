# Advanced Farmers V2

Advanced Farmers V2 is the worker-efficiency optimization of the original
crops-first Kaggriculture baseline. V2 deliberately keeps V1's crop mix and
market policy largely unchanged so the experiment isolates a single question:

> Can we earn more by assigning and routing the same workers more intelligently?

The Kaggle-facing agent remains self-contained and standard-library only.

## V2 changes

V1 scheduled jobs using a fixed priority and then processed workers one at a
time. This is reliable, but worker order can create avoidable travel. V2 replaces
that assignment layer with a utility-based global matcher.

For worker `w` and task `t`, V2 uses an interpretable utility of the form

```text
utility(w, t)
  = task value
  - travel cost
  - action cost
  + urgency bonus
  + continuity bonus
  + local chain bonus
```

The exact coefficients are constants near the top of `main.py` so they can be
tuned later without changing the architecture.

### 1. Global multi-worker assignment

Each worker receives its best candidate tasks, then V2 searches the joint
assignment and maximizes total utility while preventing two workers from
reserving the same target. With the normal farmer + two hands, the search is
small enough to run in a few milliseconds.

This fixes the classic greedy failure mode where worker 0 takes the task nearest
to itself even though worker 1 needs that task much more.

### 2. Distance-aware utility

A high-value task can still beat a nearby low-value task, but travel is no longer
free. This makes the agent explicitly trade off urgency and reward against time
spent walking.

### 3. Deadline-aware watering

Endangered crops receive very high value. If an endangered crop cannot
physically be reached and watered before end-of-day, V2 does not waste a worker
chasing an impossible rescue.

Routine watering grows more valuable as the day gets later.

### 4. Task reservation

A target position can be assigned to at most one worker per turn. Planting also
reserves seeds by crop, so simultaneous workers do not exceed the available seed
inventory. V2 also enforces the remaining `TARGET_PLOTS` capacity while making
simultaneous planting decisions.

### 5. Continuity and task chaining

V2 remembers each worker's previous target position for the current episode and
gives a bonus for continuing useful work there. This discourages ping-pong.

It also naturally creates chains such as:

```text
HARVEST -> PLANT -> WATER
```

because the next useful job often appears on the same tile and therefore has
zero travel cost plus a continuity bonus.

### 6. Local work clustering

Tasks with other useful jobs nearby get a small chain bonus. This encourages a
worker to finish a productive local cluster instead of repeatedly crossing the
farm.

## What V2 intentionally does NOT change

To keep the V1-vs-V2 comparison interpretable, this version still uses:

- roughly 40% wheat / 60% carrot planning
- a target of 20 crop plots
- two inexpensive hired hands per day
- deterministic Manhattan routing
- V1-style seed replacement
- V1-style selling and late-season liquidation
- no animals
- no fertilizer optimization
- no land expansion
- no speculative market forecasting
- no opponent model

Those belong in later optimization stages.

## Files

```text
Advanced_Farmers_V2/
├── main.py                  # official submission source
├── submission.py            # identical convenience copy
├── submission.tar.gz        # Kaggle-ready archive; main.py at archive root
├── v1_reference.py          # exact V1 source used for A/B testing
├── benchmark_kaggle.py      # full-season V2 vs V1/starter/random benchmark
├── benchmark_synthetic.py   # no-dependency scheduler + latency benchmark
├── BENCHMARK_RESULTS.md
├── requirements-dev.txt
└── tests/
    └── test_agent.py
```

## Test locally

The unit and synthetic tests need only Python:

```bash
python -m unittest discover -s tests -v
python benchmark_synthetic.py
```

For full-season evaluation:

```bash
python -m pip install -r requirements-dev.txt
python benchmark_kaggle.py
```

A Kaggle Notebook already has an appropriate environment for this workflow, so
`benchmark_kaggle.py` is particularly convenient there.

## Submit

The safest upload is the included archive, which contains `main.py` at the root:

```bash
kaggle competitions submit -c kaggriculture \
  -f submission.tar.gz \
  -m "Advanced Farmers V2 - utility scheduler"
```

You can also submit the root-level `main.py` directly if the competition accepts
the single-file route in your current Kaggle setup.

## Recommended experiment

Keep V1 as the baseline and compare the same episode seeds:

```text
V1 vs starter
V2 vs starter
V2 vs V1
```

Record final reward, win/loss/tie, and whether both agents completed normally.
This gives a clean ablation for the scheduler change before moving on to V3's
economic optimization.
