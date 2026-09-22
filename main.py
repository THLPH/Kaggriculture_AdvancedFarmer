"""Advanced Farmers V2: worker-efficiency optimized Kaggriculture agent.

V2 intentionally keeps V1's crops-first economy and market policy so that the
main experiment is isolated: better worker scheduling.

Changes from V1:
- utility-based worker/task scoring instead of fixed-priority greedy assignment
- global multi-worker matching across the best task candidates
- per-turn task reservation plus plant-seed reservation
- urgency/deadline awareness for endangered crops and daily watering
- short-lived target memory to reduce worker ping-pong and encourage task chains
  such as HARVEST -> PLANT -> WATER on the same tile
- local-density/chain bonuses that prefer finishing work in productive clusters

The Kaggle-facing file is self-contained and uses only the Python standard
library.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


CROPS: Dict[str, Dict[str, int]] = {
    "WHEAT": {"seed": 10, "first": 2, "harvest": 4, "yield": 6, "base": 25},
    "CARROT": {"seed": 20, "first": 2, "harvest": 3, "yield": 4, "base": 35},
}
PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
DEFAULT_SEASON_DAYS = 30
TURNS_PER_DAY = 24
TARGET_PLOTS = 20
TARGET_HANDS = 2
SHED_SOFT_LIMIT = 70
SELL_FLOOR_RATIO = 0.60
MAX_CANDIDATES_PER_WORKER = 12

# V2 scheduler tuning. These are deliberately interpretable rather than learned.
TRAVEL_COST = 6.0
ACTION_COST = 4.0
CONTINUITY_BONUS = 18.0
MAX_LOCAL_CHAIN_BONUS = 12.0

Position = Tuple[int, int]
Action = List[Any]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read dict-like and attribute-like observations without raising."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class Worker:
    index: int
    position: Position
    inventory: Dict[str, int]


@dataclass(frozen=True)
class Task:
    """A single position-based farm job.

    base_value is the intrinsic value before worker-specific travel/continuity
    adjustments. owner is used for DROP jobs that belong to one worker's carried
    inventory. chain_value estimates the value of remaining at/near this tile for
    the next likely action.
    """

    position: Position
    action: Action
    name: str
    base_value: float
    owner: Optional[int] = None
    chain_value: float = 0.0
    must_finish_today: bool = False


class StateView:
    """Defensive parser for the public/private Kaggriculture observation."""

    def __init__(self, obs: Any):
        self.raw = obs
        self.player = int(_get(obs, "player", 0) or 0)
        self.step = int(_get(obs, "step", 0) or 0)
        self.day = int(_get(obs, "day", self.step // TURNS_PER_DAY) or 0)
        self.hour = int(_get(obs, "hour", self.step % TURNS_PER_DAY) or 0)

        farms = _get(obs, "farms", []) or []
        self.valid = bool(farms) and 0 <= self.player < len(farms)
        self.farm = farms[self.player] if self.valid else {}
        self.tiles = _get(self.farm, "tiles", []) or []
        self.board_size = len(self.tiles) if self.tiles else 10
        self.money = float(_get(self.farm, "money", 0) or 0)

        private = _get(obs, "private", {}) or {}
        self.shed = _as_dict(_get(private, "shed", {}) or {})
        self.seeds = _as_dict(_get(private, "seeds", {}) or {})
        inventories = _get(private, "inventories", []) or []

        positions = [_get(self.farm, "farmer", [0, 0])]
        positions.extend(_get(self.farm, "hands", []) or [])
        self.workers: List[Worker] = []
        for idx, raw_pos in enumerate(positions):
            try:
                pos = (int(raw_pos[0]), int(raw_pos[1]))
            except (TypeError, ValueError, IndexError):
                pos = (0, 0)
            inv = _as_dict(inventories[idx]) if idx < len(inventories) else {}
            self.workers.append(Worker(idx, pos, inv))

        market = _get(obs, "market", {}) or {}
        self.prices = _as_dict(_get(market, "prices", {}) or {})
        town = _get(obs, "town", {}) or {}
        self.shops = list(_get(town, "unlocked_shops", []) or [])
        self.hires_today = int(_get(self.farm, "hires_today", 0) or 0)

    @property
    def turns_left_today(self) -> int:
        return max(1, TURNS_PER_DAY - self.hour)

    def iter_tiles(self) -> Iterable[Tuple[int, int, Any]]:
        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                yield x, y, tile

    def tile_at(self, pos: Position) -> Any:
        x, y = pos
        if 0 <= y < len(self.tiles) and 0 <= x < len(self.tiles[y]):
            return self.tiles[y][x]
        return "LOCKED"

    def shed_access(self) -> List[Position]:
        half = self.board_size // 2
        return [
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        ]

    def plant_counts(self) -> Dict[str, int]:
        counts = {crop: 0 for crop in CROPS}
        for _, _, tile in self.iter_tiles():
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if crop in counts:
                    counts[crop] += 1
        return counts


class ManhattanRouter:
    """Deterministic one-step routing; locked tiles remain traversable."""

    MOVES = {
        (1, 0): ["EAST"],
        (-1, 0): ["WEST"],
        (0, 1): ["SOUTH"],
        (0, -1): ["NORTH"],
    }

    @staticmethod
    def distance(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, source: Position, target: Position) -> Action:
        sx, sy = source
        tx, ty = target
        # Horizontal-first remains deterministic. There are no blocking farm
        # obstacles for movement, and locked tiles are traversable.
        if sx < tx:
            return self.MOVES[(1, 0)]
        if sx > tx:
            return self.MOVES[(-1, 0)]
        if sy < ty:
            return self.MOVES[(0, 1)]
        if sy > ty:
            return self.MOVES[(0, -1)]
        return ["PASS"]


class CropPlanner:
    """V1 crop economy retained: stable 40% wheat / 60% carrot field."""

    @staticmethod
    def planned_crop(pos: Position) -> str:
        x, y = pos
        return "WHEAT" if (x + 2 * y) % 5 < 2 else "CARROT"

    @staticmethod
    def can_finish(crop: str, day: int) -> bool:
        return day + CROPS[crop]["harvest"] < DEFAULT_SEASON_DAYS

    def desired_seed_buys(self, state: StateView) -> Dict[str, int]:
        if state.day >= DEFAULT_SEASON_DAYS - 3:
            return {}
        counts = state.plant_counts()
        desired = {"WHEAT": 0, "CARROT": 0}
        for x, y, tile in state.iter_tiles():
            if tile is None or (isinstance(tile, dict) and tile.get("kind") == "WEED"):
                crop = self.planned_crop((x, y))
                if self.can_finish(crop, state.day):
                    desired[crop] += 1

        room = max(
            0,
            TARGET_PLOTS
            - sum(counts.values())
            - sum(int(state.seeds.get(c, 0) or 0) for c in CROPS),
        )
        buys: Dict[str, int] = {}
        for crop in ("CARROT", "WHEAT"):
            if room <= 0:
                break
            have = counts[crop] + int(state.seeds.get(crop, 0) or 0)
            quota = 12 if crop == "CARROT" else 8
            amount = min(room, desired[crop], max(0, quota - have))
            affordable = int(max(0, state.money - 200) // CROPS[crop]["seed"])
            amount = min(amount, affordable)
            if amount > 0:
                buys[crop] = amount
                room -= amount
        return buys


class MarketStrategy:
    """V1 market policy retained so V2 isolates worker scheduling changes."""

    def orders(self, state: StateView, seed_buys: Dict[str, int]) -> List[Action]:
        orders: List[Action] = []

        if state.hour <= 1 and state.hires_today < TARGET_HANDS:
            for _ in range(TARGET_HANDS - state.hires_today):
                orders.append(["HIRE"])

        shed_total = sum(max(0, int(v or 0)) for v in state.shed.values())
        force_sell = state.day >= 26 or shed_total >= SHED_SOFT_LIMIT
        for item in PRODUCTS:
            amount = int(state.shed.get(item, 0) or 0)
            if amount <= 0:
                continue
            price = float(state.prices.get(item, 0) or 0)
            base = CROPS[item]["base"] if item in CROPS else price
            if force_sell or price >= base * SELL_FLOOR_RATIO:
                orders.append(["SELL", item, amount])

        for crop in ("CARROT", "WHEAT"):
            amount = int(seed_buys.get(crop, 0) or 0)
            if amount > 0:
                orders.append(["BUY_SEED", crop, amount])

        return orders[:10]


class PlannerMemory:
    """Tiny per-player continuity memory; safe to reset when a new game starts."""

    def __init__(self):
        self.last_step: Dict[int, int] = {}
        self.targets: Dict[int, Dict[int, Position]] = {}

    def prepare(self, state: StateView) -> None:
        previous = self.last_step.get(state.player)
        if state.step == 0 or (previous is not None and state.step <= previous):
            self.targets[state.player] = {}
        self.last_step[state.player] = state.step
        self.targets.setdefault(state.player, {})

    def target_for(self, player: int, worker_index: int) -> Optional[Position]:
        return self.targets.get(player, {}).get(worker_index)

    def update(self, state: StateView, assignments: Dict[int, Task]) -> None:
        self.targets[state.player] = {
            worker_index: task.position
            for worker_index, task in assignments.items()
        }


class UtilityTaskScheduler:
    """Cost-aware global worker assignment with reservations and continuity."""

    def __init__(
        self,
        crop_planner: CropPlanner,
        router: ManhattanRouter,
        memory: PlannerMemory,
    ):
        self.crop_planner = crop_planner
        self.router = router
        self.memory = memory

    def _tasks(self, state: StateView) -> List[Task]:
        tasks: List[Task] = []
        plant_count = sum(state.plant_counts().values())

        for x, y, tile in state.iter_tiles():
            pos = (x, y)

            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if crop not in CROPS:
                    continue

                age = state.day - int(tile.get("planted_day", state.day))
                watered = bool(tile.get("watered_today", False))
                consecutive = int(tile.get("consecutive_unwatered", 0) or 0)
                units = int(tile.get("yield_units", 0) or 0)

                # Rescue watering gets the highest intrinsic value. A worker-task
                # pair still has to be reachable before end-of-day to be eligible.
                if not watered and consecutive >= 1:
                    tasks.append(
                        Task(
                            position=pos,
                            action=["WATER"],
                            name="rescue-water",
                            base_value=190.0 + 2.0 * state.hour,
                            chain_value=6.0,
                            must_finish_today=True,
                        )
                    )

                harvest_age = (
                    CROPS[crop]["first"]
                    if state.day >= 28
                    else CROPS[crop]["harvest"]
                )
                if units > 0 and age >= harvest_age:
                    overdue = max(0, age - CROPS[crop]["harvest"])
                    tasks.append(
                        Task(
                            position=pos,
                            action=["HARVEST"],
                            name="harvest",
                            base_value=112.0 + 9.0 * overdue + 3.0 * units,
                            # One-time harvest normally exposes an empty tile,
                            # making PLANT then WATER a natural local chain.
                            chain_value=14.0,
                        )
                    )

                # Do not duplicate a rescue-water task with an ordinary water task.
                if not watered and consecutive < 1:
                    tasks.append(
                        Task(
                            position=pos,
                            action=["WATER"],
                            name="water",
                            # Routine water rises in value as the day runs out.
                            base_value=78.0 + 2.2 * state.hour,
                            chain_value=5.0,
                            must_finish_today=False,
                        )
                    )

            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                tasks.append(
                    Task(
                        position=pos,
                        action=["DIG"],
                        name="dig",
                        base_value=45.0,
                        chain_value=10.0,
                    )
                )

            elif tile is None and plant_count < TARGET_PLOTS and state.hour <= 14:
                crop = self.crop_planner.planned_crop(pos)
                if (
                    int(state.seeds.get(crop, 0) or 0) > 0
                    and self.crop_planner.can_finish(crop, state.day)
                ):
                    tasks.append(
                        Task(
                            position=pos,
                            action=["PLANT", crop],
                            name=f"plant-{crop.lower()}",
                            # Planting is useful, but less urgent than care/harvest.
                            # Value tapers late in the planting window because a
                            # just-planted crop still needs same-day watering.
                            base_value=66.0 - 1.5 * state.hour,
                            chain_value=18.0,
                        )
                    )

        # Carried output has no bank value until it reaches the shed and is sold.
        # Each DROP task is owned by exactly one worker.
        for worker in state.workers:
            carried = sum(max(0, int(v or 0)) for v in worker.inventory.values())
            if carried and (carried >= 6 or state.hour >= 18 or state.day >= 28):
                nearest = min(
                    state.shed_access(),
                    key=lambda p: self.router.distance(worker.position, p),
                )
                tasks.append(
                    Task(
                        position=nearest,
                        action=["DROP"],
                        name=f"drop-{worker.index}",
                        base_value=(70.0 + 4.0 * carried + 2.5 * max(0, state.hour - 17)),
                        owner=worker.index,
                        chain_value=4.0,
                    )
                )

        return tasks

    def _local_chain_bonus(self, task: Task, tasks: List[Task]) -> float:
        nearby = 0
        for other in tasks:
            if other is task or other.position == task.position:
                continue
            d = self.router.distance(task.position, other.position)
            if d <= 2:
                nearby += 1
        return min(MAX_LOCAL_CHAIN_BONUS, 2.0 * nearby)

    def _utility(
        self,
        state: StateView,
        worker: Worker,
        task: Task,
        all_tasks: List[Task],
    ) -> float:
        if task.owner is not None and task.owner != worker.index:
            return float("-inf")

        distance = self.router.distance(worker.position, task.position)
        turns_to_complete = distance + 1  # travel steps + action on target

        # Rescue tasks that cannot physically be completed before refresh should
        # not consume a worker. Routine water may still be approached because a
        # single missed day is survivable, so only rescue uses the hard cutoff.
        if task.must_finish_today and turns_to_complete > state.turns_left_today:
            return float("-inf")

        utility = task.base_value
        utility -= TRAVEL_COST * distance
        utility -= ACTION_COST
        utility += task.chain_value
        utility += self._local_chain_bonus(task, all_tasks)

        previous_target = self.memory.target_for(state.player, worker.index)
        if previous_target == task.position:
            utility += CONTINUITY_BONUS

        # Being on the tile is especially valuable: the worker can perform useful
        # work immediately rather than spending this turn moving.
        if distance == 0:
            utility += 12.0

        # A water job becomes more pressing when little time remains relative to
        # travel distance. This is a soft deadline for normal water and a strong
        # extra bonus for rescue water.
        if task.action and task.action[0] == "WATER":
            slack = state.turns_left_today - turns_to_complete
            utility += max(0.0, 18.0 - 3.0 * slack)
            if task.name == "rescue-water":
                utility += max(0.0, 36.0 - 5.0 * slack)

        return utility

    def _candidate_indices(
        self,
        state: StateView,
        worker: Worker,
        tasks: List[Task],
    ) -> List[int]:
        scored: List[Tuple[float, int, int, int, str, int]] = []
        for idx, task in enumerate(tasks):
            utility = self._utility(state, worker, task, tasks)
            if utility == float("-inf"):
                continue
            scored.append(
                (
                    utility,
                    -self.router.distance(worker.position, task.position),
                    -task.position[1],
                    -task.position[0],
                    task.name,
                    idx,
                )
            )

        # Highest utility first. Stable spatial/name tie-breakers make V2 fully
        # deterministic for a fixed observation.
        scored.sort(reverse=True)
        return [row[-1] for row in scored[:MAX_CANDIDATES_PER_WORKER]]

    @staticmethod
    def _plant_crop(task: Task) -> Optional[str]:
        if task.action and task.action[0] == "PLANT" and len(task.action) >= 2:
            return str(task.action[1])
        return None

    def _best_assignment(
        self,
        state: StateView,
        tasks: List[Task],
    ) -> Dict[int, Task]:
        """Search globally across worker candidate sets.

        There are normally only three workers (farmer + two hands), so an exact
        recursive search over the top 12 candidates per worker is tiny while still
        avoiding the classic greedy-assignment failure mode.
        """
        workers = state.workers
        candidate_lists = [self._candidate_indices(state, w, tasks) for w in workers]
        utility_cache: Dict[Tuple[int, int], float] = {}
        for worker, candidates in zip(workers, candidate_lists):
            for task_idx in candidates:
                utility_cache[(worker.index, task_idx)] = self._utility(
                    state, worker, tasks[task_idx], tasks
                )

        best_score = 0.0
        best_signature: Tuple[int, ...] = tuple([-1] * len(workers))
        best_choice: Dict[int, int] = {}
        plant_room = max(0, TARGET_PLOTS - sum(state.plant_counts().values()))

        def search(
            i: int,
            score: float,
            chosen: Dict[int, int],
            reserved_positions: set,
            reserved_seed: Dict[str, int],
            reserved_plants: int,
        ) -> None:
            nonlocal best_score, best_signature, best_choice

            if i >= len(workers):
                signature = tuple(chosen.get(w.index, -1) for w in workers)
                # Deterministic tie-break: prefer lexicographically smaller task
                # indices after maximizing score.
                if score > best_score + 1e-9 or (
                    abs(score - best_score) <= 1e-9
                    and signature < best_signature
                ):
                    best_score = score
                    best_signature = signature
                    best_choice = dict(chosen)
                return

            worker = workers[i]

            # PASS is always legal and has utility 0.
            search(
                i + 1,
                score,
                chosen,
                reserved_positions,
                reserved_seed,
                reserved_plants,
            )

            for task_idx in candidate_lists[i]:
                task = tasks[task_idx]
                utility = utility_cache[(worker.index, task_idx)]
                if utility <= 0.0:
                    continue
                if task.position in reserved_positions:
                    continue

                crop = self._plant_crop(task)
                if crop is not None:
                    available = int(state.seeds.get(crop, 0) or 0)
                    if reserved_seed.get(crop, 0) >= available:
                        continue
                    if reserved_plants >= plant_room:
                        continue

                chosen[worker.index] = task_idx
                reserved_positions.add(task.position)
                if crop is not None:
                    reserved_seed[crop] = reserved_seed.get(crop, 0) + 1

                search(
                    i + 1,
                    score + utility,
                    chosen,
                    reserved_positions,
                    reserved_seed,
                    reserved_plants + (1 if crop is not None else 0),
                )

                if crop is not None:
                    reserved_seed[crop] -= 1
                reserved_positions.remove(task.position)
                del chosen[worker.index]

        search(0, 0.0, {}, set(), {crop: 0 for crop in CROPS}, 0)
        return {
            worker_idx: tasks[task_idx]
            for worker_idx, task_idx in best_choice.items()
        }

    def actions(self, state: StateView) -> List[Action]:
        if not state.workers:
            return []

        self.memory.prepare(state)
        tasks = self._tasks(state)
        assignments = self._best_assignment(state, tasks)
        actions: List[Action] = [["PASS"] for _ in state.workers]

        for worker in state.workers:
            task = assignments.get(worker.index)
            if task is None:
                continue
            if worker.position == task.position:
                actions[worker.index] = task.action
            else:
                actions[worker.index] = self.router.step(worker.position, task.position)

        self.memory.update(state, assignments)
        return actions


ROUTER = ManhattanRouter()
CROP_PLANNER = CropPlanner()
MARKET = MarketStrategy()
MEMORY = PlannerMemory()
SCHEDULER = UtilityTaskScheduler(CROP_PLANNER, ROUTER, MEMORY)


def agent(obs: Any) -> Dict[str, Any]:
    """Kaggle entry point. Always returns a legal, complete action structure."""
    try:
        state = StateView(obs)
        if not state.valid:
            return {"farmer": ["PASS"], "hands": [], "market": []}

        unit_actions = SCHEDULER.actions(state)
        if not unit_actions:
            unit_actions = [["PASS"]]

        seed_buys = CROP_PLANNER.desired_seed_buys(state)
        market_orders = MARKET.orders(state, seed_buys)
        return {
            "farmer": unit_actions[0],
            "hands": unit_actions[1:],
            "market": market_orders,
        }
    except Exception:
        # A single PASS is much cheaper than a competition crash.
        return {"farmer": ["PASS"], "hands": [], "market": []}
