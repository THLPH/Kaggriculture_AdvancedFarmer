
import os
import math
import numpy as np
import torch
import torch.nn as nn

CROPS = ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON']
PRODUCTS = ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER']
ANIMALS = ['GOOSE', 'COW', 'SHEEP']
SEED_COST = {'WHEAT': 10, 'CARROT': 20, 'TOMATO': 50, 'STRAWBERRY': 100, 'MELON': 80}
CROP_FIRST_YIELD = {'WHEAT': 2, 'CARROT': 2, 'TOMATO': 8, 'STRAWBERRY': 10, 'MELON': 10}
ANIMAL_COST = {'GOOSE': 300, 'COW': 400, 'SHEEP': 500}
ANIMAL_STRUCTURE = {'GOOSE': 'COOP', 'COW': 'PASTURE', 'SHEEP': 'PASTURE'}
ANIMAL_PRODUCT = {'GOOSE': 'EGG', 'COW': 'MILK', 'SHEEP': 'WOOL'}
LAND_PRICES = [1000, 2000, 4000]
BOARD = 10
SHED_ADJACENT = {(4, 4), (5, 4), (5, 5), (4, 5)}
SHED_CAPACITY = 100
MAX_MARKET_ORDERS = 10
MAX_HANDS = 4
WHEAT_PICKUP_N = 4
FARM_HAND_COST_MULT = 1
SELLABLE = ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL']
MARKET_I0 = 10000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0
MARKET_PARAMS = {'WHEAT': {'base': 25, 'I0': 10000, 'T': 400, 'below_func': 'sqrt', 'below_target': 0.8, 'above_func': 'log', 'above_target': 0.2}, 'CARROT': {'base': 35, 'I0': 10000, 'T': 450, 'below_func': 'hinge', 'below_target': 1.0, 'above_func': 'sqrt', 'above_target': 0.7}, 'TOMATO': {'base': 60, 'I0': 10000, 'T': 200, 'below_func': 'hinge', 'below_target': 0.4, 'above_func': 'sqrt', 'above_target': 0.6}, 'STRAWBERRY': {'base': 120, 'I0': 10000, 'T': 100, 'below_func': 'sqrt', 'below_target': 0.7, 'above_func': 'linear', 'above_target': 1.6}, 'MELON': {'base': 250, 'I0': 10000, 'T': 300, 'below_func': 'log', 'below_target': 0.2, 'above_func': 'sq', 'above_target': 3.6}, 'EGG': {'base': 50, 'I0': 10000, 'T': 332, 'below_func': 'hinge', 'below_target': 0.4, 'above_func': 'log', 'above_target': 0.2}, 'MILK': {'base': 160, 'I0': 10000, 'T': 122, 'below_func': 'sqrt', 'below_target': 0.6, 'above_func': 'linear', 'above_target': 1.6}, 'WOOL': {'base': 200, 'I0': 10000, 'T': 105, 'below_func': 'log', 'below_target': 0.2, 'above_func': 'sq', 'above_target': 3.2}, 'FERTILIZER': {'base': 100, 'I0': 10000, 'T': 200, 'below_func': 'linear', 'below_target': 0.4, 'above_func': 'linear', 'above_target': 0.4}}
FARMER_ACTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'PASS', 'WATER', 'HARVEST', 'DIG', 'PLANT_WHEAT', 'PLANT_CARROT', 'PLANT_TOMATO', 'PLANT_STRAWBERRY', 'PLANT_MELON', 'FERTILIZE', 'PICKUP_FERTILIZER', 'BUILD_COOP', 'BUILD_PASTURE', 'PLACE_GOOSE', 'PLACE_COW', 'PLACE_SHEEP', 'FEED', 'CARE', 'COLLECT_FERTILIZER', 'PICKUP_WHEAT', 'PICKUP_GOOSE', 'PICKUP_COW', 'PICKUP_SHEEP']
MARKET_ACTIONS = ['NOOP', 'BUY_SEED_WHEAT', 'BUY_SEED_CARROT', 'BUY_SEED_TOMATO', 'BUY_SEED_STRAWBERRY', 'BUY_SEED_MELON', 'BUY_FERTILIZER', 'BUY_LAND', 'BUY_WHEAT', 'BUY_ANIMAL_GOOSE', 'BUY_ANIMAL_COW', 'BUY_ANIMAL_SHEEP', 'HIRE', 'SELL_WHEAT', 'SELL_CARROT', 'SELL_TOMATO', 'SELL_STRAWBERRY', 'SELL_MELON', 'SELL_EGG', 'SELL_MILK', 'SELL_WOOL', 'SELL_FERTILIZER']
N_FARMER = 27
N_MARKET = 22
N_SPATIAL = 24
N_GLOBAL = 56

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
    """Exact replica of the env's pricing (floored at PRICE_FLOOR)."""
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
    """_fib(0)=1, _fib(1)=1, _fib(2)=2, _fib(3)=3, _fib(4)=5 ..."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_cost(hires_today, mult=FARM_HAND_COST_MULT):
    return mult * _fib(hires_today)


def _unit_pos(farm, unit):
    if unit == 0:
        return farm["farmer"]
    if 1 <= unit <= len(farm["hands"]):
        return farm["hands"][unit - 1]
    return None


def encode_obs2(obs, unit=0):
    """Real kaggriculture obs -> (spatial [24,10,10] f32, globals [56] f32)
    from the perspective of `unit` (0 = main farmer, 1..K = hired hands)."""
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
            if "animal" in t:                       # occupied structure
                n_animals += 1
                s[15 + ANIMALS.index(t["animal"]), y, x] = 1.0
                s[18, y, x] = 1.0 if t["fed_today"] else 0.0
                s[19, y, x] = min(t["yield_units"] / 6.0, 1.0)
                s[20, y, x] = min(t["consecutive_unfed"] / 2.0, 1.0)
                s[21, y, x] = 1.0 if t["fertilizer_available"] else 0.0
                s[22, y, x] = 1.0 if t["cared_today"] else 0.0

    pos = _unit_pos(farm, unit) or farm["farmer"]
    ux, uy = pos[0], pos[1]
    s[23, uy, ux] = 1.0                             # unit marker

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
    # ---- v2.1 globals ----
    g[48] = unit / 4.0
    g[49] = len(farm["hands"]) / 4.0
    g[50] = min(hire_cost(farm.get("hires_today", 0)) / 13.0, 1.0)
    g[51] = min(sum(shed.values()) / float(SHED_CAPACITY), 1.0)
    for i, a in enumerate(ANIMALS):
        g[52 + i] = min(uinv.get(a, 0) / 2.0, 1.0)
    g[55] = n_animals / 8.0
    return s, g


def action_masks2(obs, unit=0, auto_sell=True, wheat_reserve=0):
    """-> (farmer_mask [27] bool, market_mask [22] bool) for `unit`.

    auto_sell=True  -> SELL_* masked OFF (curriculum phase 1: scripted selling,
                       build_action_dict2 appends the sells itself).
    auto_sell=False -> SELL_* live (phase 2: learned selling).
    wheat_reserve   -> SELL_WHEAT (and auto-sell) never touch the last
                       `wheat_reserve` WHEAT in the shed (animal feed).
    """
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
    if pos is None:                     # no such unit (e.g. hand not hired)
        fm[FARMER_ACTIONS.index("PASS")] = True
    else:
        fx, fy = pos
        tile = farm["tiles"][fy][fx]
        fm[0] = fy > 0
        fm[1] = fy < BOARD - 1
        fm[2] = fx < BOARD - 1
        fm[3] = fx > 0
        fm[4] = True                                                  # PASS

        is_plant = isinstance(tile, dict) and tile.get("kind") == "PLANT"
        is_weed = isinstance(tile, dict) and tile.get("kind") == "WEED"
        is_animal = isinstance(tile, dict) and "animal" in tile
        is_structure = (isinstance(tile, dict)
                        and tile.get("kind") in ("COOP", "PASTURE"))
        shed_adj = (fx, fy) in SHED_ADJACENT

        fm[5] = is_plant and not tile["watered_today"]                # WATER
        fm[6] = ((is_plant and tile["yield_units"] > 0
                  and day - tile["planted_day"] >= CROP_FIRST_YIELD[tile["crop"]])
                 or (is_animal and tile["yield_units"] > 0))          # HARVEST
        fm[7] = (is_plant or is_weed
                 or (is_structure and not is_animal))                 # DIG
        for i, c in enumerate(CROPS):                                 # PLANT_<crop>
            fm[8 + i] = (tile is None) and seeds.get(c, 0) > 0
        fm[13] = (is_plant and uinv.get("FERTILIZER", 0) > 0          # FERTILIZE
                  and tile.get("fertilized_until_day", -1) < day)
        fm[14] = shed_adj and shed.get("FERTILIZER", 0) > 0           # PICKUP_FERTILIZER
        # ---- v2.1 ----
        fm[15] = tile is None                                         # BUILD_COOP (free)
        fm[16] = tile is None                                         # BUILD_PASTURE (free)
        for i, a in enumerate(ANIMALS):                               # PLACE_<animal>
            fm[17 + i] = (is_structure and not is_animal
                          and tile.get("kind") == ANIMAL_STRUCTURE[a]
                          and uinv.get(a, 0) >= 1)
        fm[20] = (is_animal and not tile["fed_today"]                 # FEED
                  and uinv.get("WHEAT", 0) >= 1)
        fm[21] = is_animal and not tile["cared_today"]                # CARE
        fm[22] = is_animal and tile["fertilizer_available"]           # COLLECT_FERTILIZER
        fm[23] = shed_adj and shed.get("WHEAT", 0) > 0                # PICKUP_WHEAT
        for i, a in enumerate(ANIMALS):                               # PICKUP_<animal>
            fm[24 + i] = shed_adj and shed.get(a, 0) > 0

    mm = np.zeros(N_MARKET, dtype=bool)
    mm[0] = True                                                      # NOOP
    for i, c in enumerate(CROPS):                                     # BUY_SEED_<crop>
        mm[1 + i] = money >= SEED_COST[c] and seeds.get(c, 0) < 30
    mm[6] = (money >= market_price("FERTILIZER", mkt["inventory"]["FERTILIZER"] - 1, params)
             and not shed_full)                                       # BUY_FERTILIZER
    n_quads = len(farm["unlocked_quadrants"])
    mm[7] = n_quads < 4 and money >= LAND_PRICES[n_quads - 1]         # BUY_LAND
    # ---- v2.1 ----
    mm[8] = (money >= market_price("WHEAT", mkt["inventory"]["WHEAT"] - 1, params)
             and not shed_full)                                       # BUY_WHEAT
    for i, a in enumerate(ANIMALS):                                   # BUY_ANIMAL_*
        mm[9 + i] = money >= ANIMAL_COST[a] and not shed_full
    mm[12] = (len(farm["hands"]) < MAX_HANDS
              and money >= hire_cost(farm.get("hires_today", 0)))     # HIRE
    if not auto_sell:
        for i, p in enumerate(PRODUCTS):                              # SELL_<product>
            n = shed.get(p, 0) - (wheat_reserve if p == "WHEAT" else 0)
            mm[13 + i] = n > 0
    return fm, mm


def decode_farmer_action(f_idx):
    """Farmer-table index (also used per hand) -> env unit action list."""
    name = FARMER_ACTIONS[f_idx]
    if name.startswith("PLANT_"):
        return ["PLANT", name.split("_", 1)[1]]
    if name.startswith("PLACE_"):
        return ["PLACE", name.split("_", 1)[1]]
    if name == "PICKUP_FERTILIZER":
        return ["PICKUP", "FERTILIZER", 1]
    if name == "PICKUP_WHEAT":
        return ["PICKUP", "WHEAT", WHEAT_PICKUP_N]
    if name.startswith("PICKUP_"):               # PICKUP_GOOSE/COW/SHEEP
        return ["PICKUP", name.split("_", 1)[1], 1]
    return [name]


def decode_market_action(m_idx, obs, wheat_reserve=0):
    """Market-table index -> env market order list (or None for NOOP/no-op)."""
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
    """(farmer idx, market idx, per-hand farmer idxs, obs) -> env action dict.

    Market: the policy's order goes first; then, if auto_sell, one SELL order
    per shed product (all PRODUCTS except FERTILIZER — now including
    EGG/MILK/WOOL), capped at MAX_MARKET_ORDERS total orders. The last
    `wheat_reserve` WHEAT are never auto-sold (animal feed).
    """
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
        self.value_head  = nn.Linear(512, 1)

    def forward(self, spatial, glob):
        z = torch.cat([self.conv(spatial), self.mlp(glob)], dim=1)
        z = self.torso(z)
        return self.farmer_head(z), self.market_head(z), self.value_head(z).squeeze(-1)

WHEAT_RESERVE = 4
LIQUIDATION_DAY = 29

_model = None

def _load():
    global _model
    if _model is None:
        _model = FarmPolicy()
        candidates = ["model.pth", "/kaggle_simulations/agent/model.pth"]
        try:  # __file__ is undefined when kaggle-environments execs this source
            candidates.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth"))
        except NameError:
            pass
        for cand in candidates:
            if os.path.exists(cand):
                _model.load_state_dict(torch.load(cand, map_location="cpu"))
                break
        else:
            print("WARNING: model.pth not found; running an untrained net")
        _model.eval()
    return _model

def agent(obs, config):
    model = _load()
    auto_sell = obs["day"] >= LIQUIDATION_DAY      # learned timing, forced liquidation at the end
    farm = obs["farms"][obs["player"]]
    n_hands = len(farm["hands"])
    S, G, FM = [], [], []
    for u in range(1 + n_hands):
        s, g = encode_obs2(obs, unit=u)
        fm, mm = action_masks2(obs, unit=u, auto_sell=auto_sell, wheat_reserve=WHEAT_RESERVE)
        S.append(s); G.append(g); FM.append(fm)
    with torch.no_grad():
        f_logits, m_logits, _ = model(torch.tensor(np.stack(S)), torch.tensor(np.stack(G)))
        f_logits = f_logits.masked_fill(~torch.tensor(np.stack(FM)), -1e9)
        m_logits = m_logits[0].masked_fill(~torch.tensor(mm), -1e9)
        fa = f_logits.argmax(-1).tolist()
        ma = int(m_logits.argmax(-1))
    return build_action_dict2(fa[0], ma, fa[1:], obs, auto_sell=auto_sell,
                              wheat_reserve=WHEAT_RESERVE)
