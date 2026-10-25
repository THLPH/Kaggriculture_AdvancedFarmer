"""Kaggriculture agent v2 (submission file): livestock + metered selling + payback-based expansion.

Standard library only. Written from the game rules (README/engine), not from any
other agent's code.
"""
import math

BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
        "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
SEED = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
# units per harvest, days to harvest, max concurrent plants (market depth)
CROP = {"WHEAT": (4, 4, 12), "CARROT": (3, 3, 12), "MELON": (6, 10, 40),
        "TOMATO": (4, 11, 30), "STRAWBERRY": (4, 16, 40)}
MAXAGE = {"WHEAT": 4, "CARROT": 3, "MELON": 10}
ANIMAL = {  # cost, structure, first-yield day, days between yields, product, avg units per yield
    "COW": (400, "PASTURE", 8, 2, "MILK", 2.5),
    "GOOSE": (300, "COOP", 4, 1, "EGG", 1.5),
    "SHEEP": (500, "PASTURE", 6, 3, "WOOL", 3.0),
}
STRUCT_OF = {"COW": "BUILD_PASTURE", "SHEEP": "BUILD_PASTURE", "GOOSE": "BUILD_COOP"}
FIB = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89)
SHED_ACCESS = [(4, 4), (5, 4), (4, 5), (5, 5)]
PREMIUM = ("MELON", "MILK", "WOOL", "STRAWBERRY", "FERTILIZER")


def _k(name, default):
    return default  # fixed constants for submission; edit the defaults below to tune


MAX_ANIMALS = _k("V2_MAX_ANIMALS", 10)
ROI_MIN = _k("V2_ROI_MIN", 1.5)
TILE_VALUE = _k("V2_TILE_VALUE", 20.0)   # coins per tile-day for the land payback rule
ALLOW_SHEEP = _k("V2_ALLOW_SHEEP", 1)
ALLOW_GOOSE = _k("V2_ALLOW_GOOSE", 1)
MAX_HANDS = _k("V2_MAX_HANDS", 8)
SELL_THR = _k("V2_SELL_THR", 0.55)
RESERVE = _k("V2_RESERVE", 150)
STAND_W = _k("V2_STAND_W", 0.6)
LAND_GAIN = _k("V2_LAND_GAIN", 0.8)


# Exact market price function (engine constants): base, T, below (func, target), above (func, target)
MP = {
    "WHEAT": (25, 400, "sqrt", .8, "log", .2), "CARROT": (35, 450, "hinge", 1., "sqrt", .7),
    "TOMATO": (60, 200, "hinge", .4, "sqrt", .6), "STRAWBERRY": (120, 100, "sqrt", .7, "linear", 1.6),
    "MELON": (250, 300, "log", .2, "sq", 3.6), "EGG": (50, 332, "hinge", .4, "log", .2),
    "MILK": (160, 122, "sqrt", .6, "linear", 1.6), "WOOL": (200, 105, "log", .2, "sq", 3.2),
    "FERTILIZER": (100, 200, "linear", .4, "linear", .4),
}


def _shape(f, x, T):
    x = max(0.0, x)
    if f == "linear":
        return x
    if f == "sq":
        return x * x
    if f == "sqrt":
        return math.sqrt(x)
    if f == "log":
        return math.log(1.0 + x)
    u = x / T
    return u + 0.5 * max(0.0, u - 1.0) ** 2


def price_at(item, inv):
    base, T, bf, bt, af, at = MP[item]
    if inv < 10000:
        amp = bt * base / _shape(bf, T, T)
        p = base + amp * _shape(bf, 10000 - inv, T)
    else:
        amp = at * base / _shape(af, T, T)
        p = base - amp * _shape(af, inv - 10000, T)
    return max(1.0, p)


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step(pos, tgt):
    dx, dy = tgt[0] - pos[0], tgt[1] - pos[1]
    if dx == 0 and dy == 0:
        return ["PASS"]
    if abs(dx) >= abs(dy):
        return ["EAST" if dx > 0 else "WEST"]
    return ["SOUTH" if dy > 0 else "NORTH"]


def _quad(x, y):
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


def agent(obs, config=None):
    try:
        return _agent(obs)
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}


def _agent(obs):
    pid = int(obs.get("player", 0))
    me = obs["farms"][pid]
    priv = obs.get("private", {}) or {}
    shed = priv.get("shed", {}) or {}
    seeds = priv.get("seeds", {}) or {}
    invs = priv.get("inventories", []) or []
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}
    minv = market.get("inventory", {}) or {}
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    money = float(me.get("money", 0))
    remaining = 30.0 - day - hour / 24.0
    unlocked = set(me.get("unlocked_quadrants", ["NW"]))
    tiles = me.get("tiles", [])
    farmer = tuple(me.get("farmer", [4, 4]))
    hands = [tuple(h) for h in (me.get("hands", []) or [])]
    units = [farmer] + hands
    n_units = len(units)

    def px(item):
        try:
            return max(1, int(prices.get(item, BASE[item])))
        except (TypeError, ValueError):
            return BASE.get(item, 1)

    # ---- read the board ----
    plants, weeds, empty, structs = [], [], [], []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if t == "LOCKED" or _quad(x, y) not in unlocked:
                continue
            if t is None:
                empty.append((x, y))
            elif isinstance(t, dict):
                kd = t.get("kind")
                if kd == "PLANT":
                    plants.append((x, y, t))
                elif kd == "WEED":
                    weeds.append((x, y))
                elif kd in ("PASTURE", "COOP"):
                    structs.append((x, y, t))
    animals = [(x, y, t) for x, y, t in structs if t.get("animal")]
    free_struct = {"PASTURE": [], "COOP": []}
    for x, y, t in structs:
        if not t.get("animal"):
            free_struct[t["kind"]].append((x, y))
    owned = {a: 0 for a in ANIMAL}
    for _, _, t in animals:
        owned[t["animal"]] = owned.get(t["animal"], 0) + 1
    in_shed = {a: int(shed.get(a, 0) or 0) for a in ANIMAL}
    in_hand = {a: 0 for a in ANIMAL}
    for inv in invs:
        for a in ANIMAL:
            in_hand[a] += int((inv or {}).get(a, 0) or 0)
    pending = {a: in_shed[a] + in_hand[a] for a in ANIMAL}
    total_animals = sum(owned.values()) + sum(pending.values())
    n_structs = len(structs)

    center = (4.5, 4.5)
    empty.sort(key=lambda p: _dist(p, center))
    reserve_n = max(0, (min(MAX_ANIMALS, 8) if day < 16 else 0) - n_structs)
    struct_sites = empty[:max(reserve_n, 1)]
    crop_sites = [p for p in empty if p not in struct_sites]

    orders = []
    spend = 0.0

    # ---- market: metered selling ----
    last = remaining <= 2.0
    keep_wheat = (len(animals) + sum(pending.values())) * 3 + 2
    for item in ("MELON", "MILK", "WOOL", "FERTILIZER", "STRAWBERRY", "TOMATO", "EGG", "CARROT", "WHEAT"):
        qty = int(shed.get(item, 0) or 0)
        if item == "WHEAT":
            qty -= keep_wheat
        if qty <= 0 or len(orders) >= 4:
            continue
        ratio = px(item) / BASE[item]
        thr = SELL_THR * min(1.0, max(0.0, (remaining - 1.0) / 8.0))
        if qty > 25:
            thr *= 0.7
        if last or ratio >= thr:
            batch = qty if last else min(qty, 3 if item in PREMIUM else 8)
            orders.append(["SELL", item, batch])

    # ---- hire ----
    hires_today = int(me.get("hires_today", 0))
    workload = len(plants) + 3 * len(animals) + len(weeds) + min(len(empty), 30)
    want_hands = 0 if remaining < 3 else min(MAX_HANDS, (workload + 4) // 6)
    if hour <= 2 and hires_today < want_hands:
        for n in range(hires_today, min(want_hands, len(FIB))):
            c = FIB[n]
            if money - spend - c < 3:
                break
            orders.append(["HIRE"])
            spend += c

    # ---- buy land (payback rule) ----
    cost = {1: 1000, 2: 2000, 3: 4000}.get(len(unlocked), None)
    if cost and len(crop_sites) <= _k("V2_LAND_EMPTY", 30) and day >= 1:
        gain = TILE_VALUE * 0.6 * 25 * max(0.0, remaining - 6.0)
        if gain > LAND_GAIN * cost and money - spend - cost >= 1200:
            orders.append(["BUY_LAND"])
            spend += cost

    # ---- crop choice and seeds ----
    counts = {c: 0 for c in CROP}
    for _, _, t in plants:
        counts[t.get("crop")] = counts.get(t.get("crop"), 0) + 1
    # Standing production of every crop on BOTH farms (public) feeds the price forecast.
    standing = {c: 0.0 for c in CROP}
    for f in obs["farms"]:
        for row in f.get("tiles", []):
            for t in row:
                if isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") in standing:
                    standing[t["crop"]] += CROP[t["crop"]][0]
    best_crop, best_score = None, 0.0
    for c, (units_y, days_c, cap) in CROP.items():
        if counts[c] >= cap or remaining < days_c + 0.5:
            continue
        proj = float(minv.get(c, 10000)) + STAND_W * standing[c] + 0.5 * units_y
        p_est = 0.5 * (price_at(c, proj) + price_at(c, proj + units_y))
        score = (p_est * units_y - SEED[c]) / (days_c + 1.0)
        if c == "STRAWBERRY":
            score *= 1.6
        elif c == "MELON":
            score *= 0.6
        if score > best_score:
            best_crop, best_score = c, score
    if best_crop and crop_sites and len(orders) < 9:
        have = int(seeds.get(best_crop, 0) or 0)
        want = min(len(crop_sites), 30) - have
        n_buy = int(max(0, min(want, (money - spend - RESERVE) // SEED[best_crop])))
        if n_buy > 0:
            orders.append(["BUY_SEED", best_crop, n_buy])
            spend += n_buy * SEED[best_crop]

    # ---- buy animals by ROI ----
    if remaining > 7 and total_animals < MAX_ANIMALS and (day >= 2 or len(crop_sites) <= 2):
        best = None
        for a, (acost, st, first, iv, prod, upy) in ANIMAL.items():
            if (a == "SHEEP" and not ALLOW_SHEEP) or (a == "GOOSE" and not ALLOW_GOOSE):
                continue
            days = remaining - first - 1.0
            if days <= 0:
                continue
            sat = 1.0 - 0.04 * total_animals
            daily = (upy / iv) * px(prod) * sat + px("FERTILIZER") * sat - px("WHEAT")
            roi = daily * days / acost
            if a == "SHEEP" and px("WOOL") / BASE["WOOL"] < 1.1:
                roi *= 0.6      # sheep only when wool demand (shops) is running hot
            if best is None or roi > best[0]:
                best = (roi, a, acost)
        if best and best[0] >= ROI_MIN and pending[best[1]] == 0 and len(orders) < 9:
            if money - spend - best[2] >= RESERVE + 100:
                orders.append(["BUY_ANIMAL", best[1], 1])
                spend += best[2]

    # ---- wheat for feeding ----
    n_fed_mouths = len(animals) + sum(pending.values())
    wheat_have = int(shed.get("WHEAT", 0) or 0) + sum(int((i or {}).get("WHEAT", 0) or 0) for i in invs)
    if n_fed_mouths and wheat_have < n_fed_mouths * 2 and len(orders) < 9:
        need = n_fed_mouths * 3 - wheat_have
        if money - spend - need * px("WHEAT") * 1.5 >= 20:
            orders.append(["BUY_PRODUCT", "WHEAT", need])
            spend += need * px("WHEAT")

    # ---- tasks for units ----
    taken = set()
    unfed_total = sum(1 for _, _, t in animals if not t.get("fed_today"))
    fertile = {c: sum(1 for _, _, t in animals if t.get("fertilizer_available")) for c in ("x",)}
    struct_needed = {"PASTURE": 0, "COOP": 0}
    for a in ANIMAL:
        struct_needed[ANIMAL[a][1]] += pending[a]
    for k in struct_needed:
        struct_needed[k] -= len(free_struct[k])

    def task_list(ui, pos, inv):
        inv = inv or {}
        T = []
        goods = sum(v for k, v in inv.items() if k not in ANIMAL and k != "WHEAT" and v > 0)
        if goods and (day >= 29 and hour >= 16):
            for s in SHED_ACCESS:
                T.append((-1, s, ["DROP"], ("drop", ui)))
        held = next((a for a in ANIMAL if inv.get(a, 0) > 0), None)
        if held:
            for (x, y) in free_struct[ANIMAL[held][1]]:
                T.append((0.2, (x, y), ["PLACE", held, 1], ("place", x, y)))
        for a in ANIMAL:
            if in_shed[a] > 0 and not held and free_struct[ANIMAL[a][1]]:
                for s in SHED_ACCESS:
                    T.append((0.4, s, ["PICKUP", a, 1], ("pick", a)))
        for k, act in (("PASTURE", "BUILD_PASTURE"), ("COOP", "BUILD_COOP")):
            if struct_needed[k] > 0 and struct_sites:
                for s in struct_sites[:struct_needed[k]]:
                    T.append((0.3, s, [act], ("build", s)))
        for x, y, t in animals:
            if not t.get("fed_today"):
                if inv.get("WHEAT", 0) > 0:
                    T.append((0, (x, y), ["FEED"], ("feed", x, y)))
            if t.get("yield_units", 0) >= (1 if remaining < 2 else 3 if t["animal"] != "COW" else 2):
                T.append((0.6, (x, y), ["HARVEST"], ("harv", x, y)))
            if not t.get("cared_today"):
                T.append((2, (x, y), ["CARE"], ("care", x, y)))
            if t.get("fertilizer_available"):
                T.append((2.2, (x, y), ["COLLECT_FERTILIZER"], ("fert", x, y)))
        if unfed_total and inv.get("WHEAT", 0) <= 0 and int(shed.get("WHEAT", 0) or 0) > 0:
            for s in SHED_ACCESS:
                T.append((0.1, s, ["PICKUP", "WHEAT", min(6, unfed_total)], ("wheat", ui)))
        for x, y, t in plants:
            age = day - int(t.get("planted_day", day))
            crop = t.get("crop", "")
            ready = t.get("yield_units", 0) > 0 and (
                (crop in MAXAGE and age >= MAXAGE[crop]) or crop in ("TOMATO", "STRAWBERRY"))
            if ready:
                T.append((1, (x, y), ["HARVEST"], ("harv", x, y)))
            elif not t.get("watered_today") and remaining > 0.5:
                T.append((1.5, (x, y), ["WATER"], ("water", x, y)))
        for (x, y) in weeds:
            T.append((2.8 if day < 26 else 9, (x, y), ["DIG"], ("dig", x, y)))
        pl = next((c for c in CROP if int(seeds.get(c, 0) or 0) > 0), None)
        if pl and remaining > CROP[pl][1] * 0.6:
            for (x, y) in crop_sites[-12:][::-1] if False else crop_sites[:12]:
                T.append((3, (x, y), ["PLANT", pl], ("plant", x, y)))
        return T

    actions = []
    seeds_left = {c: int(seeds.get(c, 0) or 0) for c in CROP}
    for ui, pos in enumerate(units):
        inv = invs[ui] if ui < len(invs) else {}
        cand = [t for t in task_list(ui, pos, inv) if t[3] not in taken]
        if not cand:
            actions.append(["PASS"])
            continue
        prio, tgt, act, key = min(cand, key=lambda t: (t[0], _dist(pos, t[1])))
        taken.add(key)
        if act[0] == "PLANT":
            if seeds_left.get(act[1], 0) <= 0:
                actions.append(["PASS"])
                continue
            seeds_left[act[1]] -= 1
        if pos == tgt or (act[0] in ("PICKUP", "DROP") and tuple(pos) in SHED_ACCESS):
            actions.append(act)
        else:
            actions.append(_step(pos, tgt))

    return {"farmer": actions[0], "hands": actions[1:], "market": orders[:10]}
