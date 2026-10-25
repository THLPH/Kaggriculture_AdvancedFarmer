import importlib.util, os
from collections import defaultdict
from kaggle_environments import make

EPISODE_STEPS = 720          # use 120 for a quick smoke test
SEEDS = (31, 32, 33)         # each seed is played in both seats
GATE_OPPONENTS = ["starter"]   # Kaggle's built-in local starter agent
GATE_MIN_WIN_RATE = 0.6

def load_agent(path):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("kg_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return getattr(mod, "agent", None) or getattr(mod, "main")

OPP = {"starter": "starter"}

def bank(state, i):
    try:
        obs = state.observation if hasattr(state, "observation") else state["observation"]
        return float(obs["farms"][i]["money"])
    except Exception:
        return None

res = defaultdict(list)
for name, opp in OPP.items():
    for seed in SEEDS:
        for seat in (0, 1):
            pl = [opp, opp]; pl[seat] = load_agent("main.py")
            env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}, debug=True)
            try:
                env.run(pl); f = env.steps[-1]
                a, b = bank(f[seat], seat), bank(f[1 - seat], 1 - seat)
                if a is None or b is None: continue
                res[name].append((a, a - b))
                print(f"{name:24s} seed={seed} seat={seat} me=${a:,.0f} margin={a-b:,.0f}")
            except Exception as e:
                print(name, seed, seat, "ERROR", repr(e)[:80])

print("\nSUMMARY")
for name, rows in res.items():
    w = sum(m > 0 for _, m in rows)
    print(f"{name:24s} W={w}/{len(rows)}  avg_cash={sum(c for c,_ in rows)/len(rows):>9,.0f}  avg_margin={sum(m for _,m in rows)/len(rows):>10,.0f}")

ok = all(name in res and sum(m > 0 for _, m in res[name]) / len(res[name]) >= GATE_MIN_WIN_RATE for name in GATE_OPPONENTS)
print("\nVERDICT:", "SUBMIT (passes the gate)" if ok else "DO NOT SUBMIT YET (fails the gate vs " + ", ".join(GATE_OPPONENTS) + ")")
