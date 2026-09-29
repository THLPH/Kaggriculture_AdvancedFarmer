"""
mp_worker.py -- multiprocessing vectorized runner for kaggriculture PPO (v2.1 contract).

Public API (mirrors VecKag21 in the notebook):
    vec = MpVecKag(n_envs, cfg, opponent_pool, n_workers=4, base_seed=0, snapshot=None)
    S, G, FM, MM = vec.reset()                 # (N,U,24,10,10), (N,U,56), (N,U,27), (N,22)
    (S, G, FM, MM), rewards, dones = vec.step(f_actions (N,U), m_actions (N,))
    vec.final_moneys                           # completed-episode final monies
    vec.results                                # tag -> [final money] per opponent
    vec.set_pool(["starter", "random", "policy", ...])   # curriculum hook
    vec.set_sell_mode(auto_sell, wheat_reserve)          # curriculum hook
    vec.update_snapshot(make_snapshot(model))            # self-play hook
    vec.close()

opponent_pool entries: "starter" / "random" (built-ins) or "policy" (frozen
snapshot played greedily inside the worker; requires snapshot= at construction
or an update_snapshot() call before the policy opponent is first used).

NOTEBOOK USAGE (IMPORTANT): this file must exist on disk as an importable
module -- worker processes re-import it under multiprocessing 'spawn'.
Ship it next to the notebook and do:
    import sys; sys.path.insert(0, "<dir containing mp_worker.py>")
    import mp_worker
Explicit spawn context is used, which is safe under macOS/Jupyter AND Colab.
Any driver *script* needs the `if __name__ == "__main__"` guard.
"""

from __future__ import annotations

import importlib
import inspect
import math
import multiprocessing as mp
import traceback

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# ==== BEGIN ENV-CONTRACT SECTION (v2.1) =====================================
# ==== Must match the notebook's contract cells exactly.
# ============================================================================
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}

LAND_PRICES = [1000, 2000, 4000]
BOARD = 10
SHED_ADJACENT = {(4, 4), (5, 4), (4, 5), (5, 5)}
SHED_CAPACITY = 100
MAX_MARKET_ORDERS = 10
MAX_HANDS = 4
WHEAT_PICKUP_N = 4
FARM_HAND_COST_MULT = 1

SELLABLE = [p for p in PRODUCTS if p != "FERTILIZER"]

MARKET_I0 = 10000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0
MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def _shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    if func == "log10":  return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inventory, params=None):
    p = (params or MARKET_PARAMS)[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        amp = p["below_target"] * base / _shape(p["below_func"], T, T)
        price = base + amp * _shape(p["below_func"], I0 - inventory, T)
    else:
        amp = p["above_target"] * base / _shape(p["above_func"], T, T)
        price = base - amp * _shape(p["above_func"], inventory - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_cost(hires_today, mult=FARM_HAND_COST_MULT):
    return mult * _fib(hires_today)


FARMER_ACTIONS = (
    ["NORTH", "SOUTH", "EAST", "WEST", "PASS", "WATER", "HARVEST", "DIG"]
    + [f"PLANT_{c}" for c in CROPS]
    + ["FERTILIZE", "PICKUP_FERTILIZER"]
    + ["BUILD_COOP", "BUILD_PASTURE"]
    + ["PLACE_GOOSE", "PLACE_COW", "PLACE_SHEEP"]
    + ["FEED", "CARE", "COLLECT_FERTILIZER"]
    + ["PICKUP_WHEAT", "PICKUP_GOOSE", "PICKUP_COW", "PICKUP_SHEEP"]
)
MARKET_ACTIONS = (
    ["NOOP"]
    + [f"BUY_SEED_{c}" for c in CROPS]
    + ["BUY_FERTILIZER", "BUY_LAND"]
    + ["BUY_WHEAT"]
    + ["BUY_ANIMAL_GOOSE", "BUY_ANIMAL_COW", "BUY_ANIMAL_SHEEP"]
    + ["HIRE"]
    + [f"SELL_{p}" for p in PRODUCTS]
)
N_FARMER = len(FARMER_ACTIONS)     # 27
N_MARKET = len(MARKET_ACTIONS)     # 22
N_SPATIAL = 24
N_GLOBAL = 56


def _unit_pos(farm, unit):
    if unit == 0:
        return farm["farmer"]
    if 1 <= unit <= len(farm["hands"]):
        return farm["hands"][unit - 1]
    return None


def encode_obs2(obs, unit=0):
    player = obs["player"]
    farm = obs["farms"][player]
    opp = obs["farms"][1 - player]
    day = obs["day"]

    s = np.zeros((N_SPATIAL, BOARD, BOARD), dtype=np.float32)
    n_animals = 0
    for y in range(BOARD):
        for x in range(BOARD):
            t = farm["tiles"][y][x]
            if t is None:
                s[0, y, x] = 1.0
                continue
            if t == "LOCKED":
                s[1, y, x] = 1.0
                continue
            if not isinstance(t, dict):
                continue
            kind = t.get("kind")
            if kind == "WEED":
                s[2, y, x] = 1.0
            elif kind == "PLANT":
                s[3 + CROPS.index(t["crop"]), y, x] = 1.0
                s[8, y, x] = min((day - t["planted_day"]) / 16.0, 1.0)
                s[9, y, x] = 1.0 if t["watered_today"] else 0.0
                s[10, y, x] = min(t["consecutive_unwatered"] / 2.0, 1.0)
                s[11, y, x] = min(t["yield_units"] / 6.0, 1.0)
                s[12, y, x] = 1.0 if t.get("fertilized_until_day", -1) >= day else 0.0
            elif kind == "COOP":
                s[13, y, x] = 1.0
            elif kind == "PASTURE":
                s[14, y, x] = 1.0
            if "animal" in t:
                n_animals += 1
                s[15 + ANIMALS.index(t["animal"]), y, x] = 1.0
                s[18, y, x] = 1.0 if t["fed_today"] else 0.0
                s[19, y, x] = min(t["yield_units"] / 6.0, 1.0)
                s[20, y, x] = min(t["consecutive_unfed"] / 2.0, 1.0)
                s[21, y, x] = 1.0 if t["fertilizer_available"] else 0.0
                s[22, y, x] = 1.0 if t["cared_today"] else 0.0

    pos = _unit_pos(farm, unit) or farm["farmer"]
    ux, uy = pos[0], pos[1]
    s[23, uy, ux] = 1.0

    priv = obs["private"]
    shed = priv["shed"]
    invs = priv["inventories"]
    uinv = invs[unit] if unit < len(invs) else {}

    g = np.zeros(N_GLOBAL, dtype=np.float32)
    g[0] = farm["money"] / 5000.0
    g[1] = day / 30.0
    g[2] = obs["hour"] / 24.0
    g[3] = ux / (BOARD - 1)
    g[4] = uy / (BOARD - 1)
    for i, c in enumerate(CROPS):
        g[5 + i] = min(priv["seeds"].get(c, 0) / 10.0, 2.0)
    for i, p in enumerate(PRODUCTS):
        g[10 + i] = min(shed.get(p, 0) / 20.0, 2.0)
    for i, a in enumerate(ANIMALS):
        g[19 + i] = shed.get(a, 0) / 4.0
    g[22] = min(sum(uinv.values()) / 10.0, 2.0)
    g[23] = uinv.get("FERTILIZER", 0) / 5.0
    g[24] = uinv.get("WHEAT", 0) / 10.0
    for i, p in enumerate(PRODUCTS):
        g[25 + i] = obs["market"]["prices"][p] / 250.0
    for i, p in enumerate(PRODUCTS):
        g[34 + i] = (obs["market"]["inventory"][p] - 10000.0) / 2000.0
    g[43] = len(farm["unlocked_quadrants"]) / 4.0
    g[44] = opp["money"] / 5000.0
    g[45] = len(obs["town"]["unlocked_shops"]) / 8.0
    g[46] = 1.0 if (ux, uy) in SHED_ADJACENT else 0.0
    g[47] = min(farm.get("hires_today", 0) / 5.0, 1.0)
    g[48] = unit / 4.0
    g[49] = len(farm["hands"]) / 4.0
    g[50] = min(hire_cost(farm.get("hires_today", 0)) / 13.0, 1.0)
    g[51] = min(sum(shed.values()) / float(SHED_CAPACITY), 1.0)
    for i, a in enumerate(ANIMALS):
        g[52 + i] = min(uinv.get(a, 0) / 2.0, 1.0)
    g[55] = n_animals / 8.0
    return s, g


def action_masks2(obs, unit=0, auto_sell=True, wheat_reserve=0):
    player = obs["player"]
    farm = obs["farms"][player]
    money = farm["money"]
    seeds = obs["private"]["seeds"]
    shed = obs["private"]["shed"]
    invs = obs["private"]["inventories"]
    uinv = invs[unit] if unit < len(invs) else {}
    mkt = obs["market"]
    params = mkt.get("params") if isinstance(mkt, dict) else None
    shed_full = sum(shed.values()) >= SHED_CAPACITY
    day = obs["day"]

    pos = _unit_pos(farm, unit)
    fm = np.zeros(N_FARMER, dtype=bool)
    if pos is None:
        fm[FARMER_ACTIONS.index("PASS")] = True
    else:
        fx, fy = pos
        tile = farm["tiles"][fy][fx]
        fm[0] = fy > 0
        fm[1] = fy < BOARD - 1
        fm[2] = fx < BOARD - 1
        fm[3] = fx > 0
        fm[4] = True

        is_plant = isinstance(tile, dict) and tile.get("kind") == "PLANT"
        is_weed = isinstance(tile, dict) and tile.get("kind") == "WEED"
        is_animal = isinstance(tile, dict) and "animal" in tile
        is_structure = (isinstance(tile, dict)
                        and tile.get("kind") in ("COOP", "PASTURE"))
        shed_adj = (fx, fy) in SHED_ADJACENT

        fm[5] = is_plant and not tile["watered_today"]
        fm[6] = ((is_plant and tile["yield_units"] > 0
                  and day - tile["planted_day"] >= CROP_FIRST_YIELD[tile["crop"]])
                 or (is_animal and tile["yield_units"] > 0))
        fm[7] = (is_plant or is_weed
                 or (is_structure and not is_animal))
        for i, c in enumerate(CROPS):
            fm[8 + i] = (tile is None) and seeds.get(c, 0) > 0
        fm[13] = (is_plant and uinv.get("FERTILIZER", 0) > 0
                  and tile.get("fertilized_until_day", -1) < day)
        fm[14] = shed_adj and shed.get("FERTILIZER", 0) > 0
        fm[15] = tile is None
        fm[16] = tile is None
        for i, a in enumerate(ANIMALS):
            fm[17 + i] = (is_structure and not is_animal
                          and tile.get("kind") == ANIMAL_STRUCTURE[a]
                          and uinv.get(a, 0) >= 1)
        fm[20] = (is_animal and not tile["fed_today"]
                  and uinv.get("WHEAT", 0) >= 1)
        fm[21] = is_animal and not tile["cared_today"]
        fm[22] = is_animal and tile["fertilizer_available"]
        fm[23] = shed_adj and shed.get("WHEAT", 0) > 0
        for i, a in enumerate(ANIMALS):
            fm[24 + i] = shed_adj and shed.get(a, 0) > 0

    mm = np.zeros(N_MARKET, dtype=bool)
    mm[0] = True
    for i, c in enumerate(CROPS):
        mm[1 + i] = money >= SEED_COST[c] and seeds.get(c, 0) < 30
    mm[6] = (money >= market_price("FERTILIZER", mkt["inventory"]["FERTILIZER"] - 1, params)
             and not shed_full)
    n_quads = len(farm["unlocked_quadrants"])
    mm[7] = n_quads < 4 and money >= LAND_PRICES[n_quads - 1]
    mm[8] = (money >= market_price("WHEAT", mkt["inventory"]["WHEAT"] - 1, params)
             and not shed_full)
    for i, a in enumerate(ANIMALS):
        mm[9 + i] = money >= ANIMAL_COST[a] and not shed_full
    mm[12] = (len(farm["hands"]) < MAX_HANDS
              and money >= hire_cost(farm.get("hires_today", 0)))
    if not auto_sell:
        for i, p in enumerate(PRODUCTS):
            n = shed.get(p, 0) - (wheat_reserve if p == "WHEAT" else 0)
            mm[13 + i] = n > 0
    return fm, mm


def decode_farmer_action(f_idx):
    name = FARMER_ACTIONS[f_idx]
    if name.startswith("PLANT_"):
        return ["PLANT", name.split("_", 1)[1]]
    if name.startswith("PLACE_"):
        return ["PLACE", name.split("_", 1)[1]]
    if name == "PICKUP_FERTILIZER":
        return ["PICKUP", "FERTILIZER", 1]
    if name == "PICKUP_WHEAT":
        return ["PICKUP", "WHEAT", WHEAT_PICKUP_N]
    if name.startswith("PICKUP_"):
        return ["PICKUP", name.split("_", 1)[1], 1]
    return [name]


def decode_market_action(m_idx, obs, wheat_reserve=0):
    name = MARKET_ACTIONS[m_idx]
    shed = obs["private"]["shed"]
    if name == "NOOP":
        return None
    if name.startswith("BUY_SEED_"):
        return ["BUY_SEED", name.split("_", 2)[2], 1]
    if name == "BUY_FERTILIZER":
        return ["BUY_PRODUCT", "FERTILIZER", 1]
    if name == "BUY_WHEAT":
        return ["BUY_PRODUCT", "WHEAT", 1]
    if name == "BUY_LAND":
        return ["BUY_LAND"]
    if name.startswith("BUY_ANIMAL_"):
        return ["BUY_ANIMAL", name.split("_", 2)[2], 1]
    if name == "HIRE":
        return ["HIRE"]
    if name.startswith("SELL_"):
        p = name.split("_", 1)[1]
        n = int(shed.get(p, 0)) - (wheat_reserve if p == "WHEAT" else 0)
        return ["SELL", p, n] if n > 0 else None
    raise ValueError(f"unknown market action {m_idx} ({name})")


def build_action_dict2(f_idx, m_idx, hand_f_idxs, obs, auto_sell=True,
                       wheat_reserve=0):
    shed = obs["private"]["shed"]
    action = {
        "farmer": decode_farmer_action(f_idx),
        "hands": [decode_farmer_action(i) for i in (hand_f_idxs or [])],
        "market": [],
    }
    order = decode_market_action(m_idx, obs, wheat_reserve)
    if order is not None:
        action["market"].append(order)
    if auto_sell:
        for p in SELLABLE:
            if len(action["market"]) >= MAX_MARKET_ORDERS:
                break
            n = int(shed.get(p, 0)) - (wheat_reserve if p == "WHEAT" else 0)
            if n > 0:
                action["market"].append(["SELL", p, n])
    return action


def call_agent(fn, obs, config):
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = 2
    return fn(obs, config) if n >= 2 else fn(obs)


class FarmPolicy(nn.Module):
    def __init__(self, n_spatial=N_SPATIAL, n_global=N_GLOBAL,
                 n_farmer=N_FARMER, n_market=N_MARKET):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_spatial, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(nn.Linear(n_global, 64), nn.ReLU())
        torso_in = 64 * BOARD * BOARD + 64
        self.torso = nn.Sequential(nn.Linear(torso_in, 512), nn.ReLU())
        self.farmer_head = nn.Linear(512, n_farmer)
        self.market_head = nn.Linear(512, n_market)
        self.value_head = nn.Linear(512, 1)

    def forward(self, spatial, glob):
        z = torch.cat([self.conv(spatial), self.mlp(glob)], dim=1)
        z = self.torso(z)
        return self.farmer_head(z), self.market_head(z), self.value_head(z).squeeze(-1)


def masked_categorical(logits, mask):
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))
# ============================================================================
# ==== END ENV-CONTRACT SECTION ==============================================
# ============================================================================

from kaggle_environments import make  # noqa: E402

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _tt(arr):
    """numpy -> torch, robust to torch builds without numpy>=2 support."""
    try:
        return torch.from_numpy(np.ascontiguousarray(arr))
    except Exception:
        return torch.tensor(arr.tolist())


def make_snapshot(model, class_path="mp_worker.FarmPolicy"):
    return {"class_path": class_path,
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


# ------------------------------------------------------------------ worker --
class _Worker:
    def __init__(self, worker_id, m_envs, cfg, base_seed, snapshot):
        torch.set_num_threads(1)
        self.worker_id = worker_id
        self.m = m_envs
        self.cfg = cfg
        self.U = 1 + cfg.get("max_hands", MAX_HANDS)
        self.base_seed = base_seed
        self.auto_sell = True
        self.wheat_reserve = cfg.get("wheat_reserve", 0)
        self.envs = [None] * m_envs
        self.episode_count = [0] * m_envs
        self.prev_money = np.zeros(m_envs, dtype=np.float64)
        self.opps = [None] * m_envs
        self._agent_cache = {}
        self.policy = None
        if snapshot is not None:
            self.set_snapshot(snapshot)

    def set_snapshot(self, snapshot):
        class_path = snapshot.get("class_path", "mp_worker.FarmPolicy")
        mod_name, cls_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
        model = cls()
        model.load_state_dict(snapshot["state_dict"])
        model.eval()
        self.policy = model

    def set_sell_mode(self, auto_sell, wheat_reserve):
        self.auto_sell = auto_sell
        self.wheat_reserve = wheat_reserve

    @torch.no_grad()
    def _policy_action(self, obs):
        """Frozen-snapshot opponent: multi-unit greedy, plays either seat."""
        farm = obs["farms"][obs["player"]]
        n_hands = len(farm["hands"])
        S, G, FM = [], [], []
        for u in range(1 + n_hands):
            s, g = encode_obs2(obs, unit=u)
            fm, mm = action_masks2(obs, unit=u, auto_sell=self.auto_sell,
                                   wheat_reserve=self.wheat_reserve)
            S.append(s); G.append(g); FM.append(fm)
        f_logits, m_logits, _ = self.policy(_tt(np.stack(S)), _tt(np.stack(G)))
        f_logits = f_logits.masked_fill(~_tt(np.stack(FM)), -1e9)
        m_logits = m_logits[0].masked_fill(~_tt(mm), -1e9)
        fa = f_logits.argmax(-1).tolist()
        ma = int(m_logits.argmax(-1))
        return build_action_dict2(fa[0], ma, fa[1:], obs,
                                  auto_sell=self.auto_sell,
                                  wheat_reserve=self.wheat_reserve)

    def _opp_action(self, i, env):
        spec = self.opps[i]
        obs = env.state[1].observation
        try:
            if spec == "policy":
                if self.policy is None:
                    return PASS_ACTION
                a = self._policy_action(obs)
            else:
                if spec not in self._agent_cache:
                    self._agent_cache[spec] = env.agents[spec] if isinstance(spec, str) else spec
                a = call_agent(self._agent_cache[spec], obs, env.configuration)
            if not isinstance(a, dict):
                a = None
        except Exception:
            a = None
        return a if a is not None else PASS_ACTION

    def _make_env(self, i):
        conf = {"episodeSteps": self.cfg["episode_steps"],
                "seed": self.base_seed + 7919 * self.worker_id
                        + 1000 * self.episode_count[i] + 7 * i + 1}
        env = make("kaggriculture", configuration=conf)
        env.reset(num_agents=2)
        return env

    def _try_recreate(self, i, opp_spec):
        self.opps[i] = opp_spec
        try:
            self.envs[i] = self._make_env(i)
            self.episode_count[i] += 1
        except Exception:
            traceback.print_exc()
            self.envs[i] = None
        self.prev_money[i] = self.cfg["starting_money"]

    def reset(self, opp_specs):
        for i in range(self.m):
            self._try_recreate(i, opp_specs[i])
        return self._observe()

    def step(self, f_actions, m_actions, opp_specs):
        rewards = np.zeros(self.m, dtype=np.float32)
        dones = np.zeros(self.m, dtype=bool)
        finished = []
        for i in range(self.m):
            env = self.envs[i]
            if env is None:
                dones[i] = True
                self._try_recreate(i, opp_specs[i])
                continue
            try:
                obs0 = env.state[0].observation
                n_hands = len(obs0["farms"][obs0["player"]]["hands"])
                hand_idxs = [int(f_actions[i, u]) for u in range(1, 1 + n_hands)]
                a0 = build_action_dict2(int(f_actions[i, 0]), int(m_actions[i]),
                                        hand_idxs, obs0, auto_sell=self.auto_sell,
                                        wheat_reserve=self.wheat_reserve)
                a1 = self._opp_action(i, env)
                env.step([a0, a1])
                money = float(env.state[0].observation["farms"][0]["money"])
                rewards[i] = (money - self.prev_money[i]) * self.cfg["reward_scale"]
                self.prev_money[i] = money
                if env.state[0].status != "ACTIVE":
                    dones[i] = True
                    finished.append((i, money))
                    self._try_recreate(i, opp_specs[i])
            except Exception:
                traceback.print_exc()
                dones[i] = True
                finished.append((i, float(self.prev_money[i])))
                self._try_recreate(i, opp_specs[i])
        out = self._observe()
        out.update(rewards=rewards, dones=dones, finished=finished)
        return out

    def _observe(self):
        S = np.zeros((self.m, self.U, N_SPATIAL, BOARD, BOARD), dtype=np.float32)
        G = np.zeros((self.m, self.U, N_GLOBAL), dtype=np.float32)
        FM = np.zeros((self.m, self.U, N_FARMER), dtype=bool)
        MM = np.zeros((self.m, N_MARKET), dtype=bool)
        FM[:, :, FARMER_ACTIONS.index("PASS")] = True
        MM[:, 0] = True
        for i, env in enumerate(self.envs):
            if env is None:
                continue
            obs = env.state[0].observation
            n_hands = len(obs["farms"][obs["player"]]["hands"])
            _, mm0 = action_masks2(obs, unit=0, auto_sell=self.auto_sell,
                                   wheat_reserve=self.wheat_reserve)
            MM[i] = mm0
            for u in range(min(1 + n_hands, self.U)):
                s, g = encode_obs2(obs, unit=u)
                fm, _ = action_masks2(obs, unit=u, auto_sell=self.auto_sell,
                                      wheat_reserve=self.wheat_reserve)
                S[i, u], G[i, u], FM[i, u] = s, g, fm
        return dict(spatial=S, globals=G, fmask=FM, mmask=MM)


def worker_main(remote, worker_id, m_envs, cfg, base_seed, snapshot):
    try:
        w = _Worker(worker_id, m_envs, cfg, base_seed, snapshot)
        while True:
            cmd, *args = remote.recv()
            try:
                if cmd == "reset":
                    res = w.reset(*args)
                elif cmd == "step":
                    res = w.step(*args)
                elif cmd == "update_snapshot":
                    w.set_snapshot(*args)
                    res = None
                elif cmd == "set_sell_mode":
                    w.set_sell_mode(*args)
                    res = None
                elif cmd == "close":
                    remote.send(("ok", None))
                    break
                else:
                    raise ValueError(f"unknown command {cmd!r}")
                remote.send(("ok", res))
            except Exception:
                remote.send(("error", traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        try:
            remote.close()
        except Exception:
            pass


# -------------------------------------------------------------- main process --
class MpVecKag:
    """Drop-in multiprocessing replacement for VecKag21 (same public API)."""

    def __init__(self, n_envs, cfg, opponent_pool, n_workers=4, base_seed=0, snapshot=None):
        if "policy" in opponent_pool and snapshot is None:
            raise ValueError("opponent_pool contains 'policy' but no snapshot was given")
        self.n = n_envs
        self.U = 1 + cfg.get("max_hands", MAX_HANDS)
        self.cfg = cfg
        self.opponent_pool = list(opponent_pool)
        self.n_workers = max(1, min(n_workers, n_envs))
        self.final_moneys = []
        self.results = {}                       # opponent tag -> [final money]
        self._rng = np.random.RandomState(base_seed)
        self._closed = False

        counts = [n_envs // self.n_workers + (1 if w < n_envs % self.n_workers else 0)
                  for w in range(self.n_workers)]
        self.slices = []
        s = 0
        for c in counts:
            self.slices.append((s, c))
            s += c

        self.opps = [self._sample_opp() for _ in range(n_envs)]

        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(self.n_workers):
            parent_r, child_r = ctx.Pipe()
            p = ctx.Process(target=worker_main,
                            args=(child_r, w, counts[w], dict(cfg), base_seed, snapshot),
                            daemon=True)
            p.start()
            child_r.close()
            self.remotes.append(parent_r)
            self.procs.append(p)

    # -- curriculum hooks -------------------------------------------------------
    def set_pool(self, pool):
        """Replace the opponent pool (list of tags, sampled uniformly; repeat a
        tag to weight it, e.g. ['starter', 'policy', 'policy'] = 1/3 starter)."""
        self.opponent_pool = list(pool)

    def set_sell_mode(self, auto_sell, wheat_reserve):
        for r in self.remotes:
            r.send(("set_sell_mode", auto_sell, wheat_reserve))
        for r in self.remotes:
            self._recv(r)

    # -- helpers ---------------------------------------------------------------
    def _sample_opp(self):
        return self.opponent_pool[self._rng.randint(len(self.opponent_pool))]

    def _recv(self, r):
        tag, payload = r.recv()
        if tag == "error":
            raise RuntimeError("worker failed:\n" + payload)
        return payload

    @staticmethod
    def _to_torch(payloads):
        S = np.concatenate([p["spatial"] for p in payloads])
        G = np.concatenate([p["globals"] for p in payloads])
        FM = np.concatenate([p["fmask"] for p in payloads])
        MM = np.concatenate([p["mmask"] for p in payloads])
        return (_tt(S), _tt(G), _tt(FM), _tt(MM))

    # -- VecKag-compatible API ---------------------------------------------------
    def reset(self):
        for w, r in enumerate(self.remotes):
            s, c = self.slices[w]
            r.send(("reset", list(self.opps[s:s + c])))
        payloads = [self._recv(r) for r in self.remotes]
        self.opps = [self._sample_opp() for _ in range(self.n)]
        return self._to_torch(payloads)

    @staticmethod
    def _as_int_arr(x, two_d=False):
        if isinstance(x, torch.Tensor):
            arr = np.array(x.detach().cpu().tolist(), dtype=np.int64)
        else:
            arr = np.asarray(x, dtype=np.int64)
        return arr if two_d else arr.reshape(-1)

    def step(self, fActions, mActions):
        fa = self._as_int_arr(fActions, two_d=True)
        ma = self._as_int_arr(mActions)
        assert fa.shape[0] == self.n and ma.shape[0] == self.n
        for w, r in enumerate(self.remotes):
            s, c = self.slices[w]
            r.send(("step", fa[s:s + c], ma[s:s + c], list(self.opps[s:s + c])))
        payloads = [self._recv(r) for r in self.remotes]
        rewards = np.concatenate([p["rewards"] for p in payloads])
        dones = np.concatenate([p["dones"] for p in payloads])
        for w, p in enumerate(payloads):
            s, _ = self.slices[w]
            for local_i, money in p["finished"]:
                self.final_moneys.append(money)
                tag = self.opps[s + local_i]
                self.results.setdefault(tag, []).append(float(money))
                self.opps[s + local_i] = self._sample_opp()
        return self._to_torch(payloads), _tt(rewards), _tt(dones).to(torch.float32)

    def update_snapshot(self, snapshot):
        for r in self.remotes:
            r.send(("update_snapshot", snapshot))
        for r in self.remotes:
            self._recv(r)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for r in self.remotes:
            try:
                r.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for r in self.remotes:
            try:
                r.recv()
            except (EOFError, BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
        for p in self.procs:
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
