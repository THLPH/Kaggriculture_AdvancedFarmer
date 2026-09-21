%%writefile main.py

def agent(obs):
    player = obs["player"]
    my_farm = obs["farms"][player]
    private = obs["private"]
    fx, fy = my_farm["farmer"]
    
    market_actions = []
    farmer_action = ["PASS"]
    
    # 1. Market: Buy seeds if low, sell harvested wheat in shed
    wheat_seeds = private["seeds"].get("WHEAT", 0)
    if wheat_seeds < 5 and my_farm["money"] >= 10:
        market_actions.append(["BUY_SEED", "WHEAT", 1])
        
    wheat_in_shed = private["shed"].get("WHEAT", 0)
    if wheat_in_shed > 0:
        market_actions.append(["SELL", "WHEAT", wheat_in_shed])
        
    # 2. Farm: Scan top-left 5x5 quadrant
    target_pos = None
    for y in range(5):
        for x in range(5):
            tile = my_farm["tiles"][y][x]
            
            # Plant
            if type(tile) is dict and tile.get("kind") == "PLANT":
                crop_age = obs["day"] - tile["planted_day"]
                if crop_age >= 2: # Harvest mature
                    if fx == x and fy == y:
                        return {"farmer": ["HARVEST"], "hands": [], "market": market_actions}
                    if not target_pos: target_pos = (x, y)
                elif not tile["watered_today"]: # Water thirsty
                    if fx == x and fy == y:
                        return {"farmer": ["WATER"], "hands": [], "market": market_actions}
                    if not target_pos: target_pos = (x, y)
            
            # Weed
            elif type(tile) is dict and tile.get("kind") == "WEED":
                if fx == x and fy == y:
                    return {"farmer": ["DIG"], "hands": [], "market": market_actions}
                if not target_pos: target_pos = (x, y)
                
            # Empty
            elif tile is None and wheat_seeds > 0:
                if fx == x and fy == y:
                    return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market_actions}
                if not target_pos: target_pos = (x, y)

    # 3. Move toward target
    if target_pos:
        tx, ty = target_pos
        if tx > fx:   farmer_action = ["EAST"]
        elif tx < fx: farmer_action = ["WEST"]
        elif ty > fy: farmer_action = ["SOUTH"]
        elif ty < fy: farmer_action = ["NORTH"]

    return {
        "farmer": farmer_action,
        "hands": [],
        "market": market_actions
    }
