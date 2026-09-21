from kaggle_environments import make

# 1. Run the test game
env = make("kaggriculture", configuration={"episodeSteps": 120}, debug=True)
env.run(["main.py", "starter"])

# 2. Print the results safely
final_turn = env.steps[-1]
print("--- MATCH FINISHED ---")
print(f"Your Bot (Player 0): ${final_turn[0].reward:,.2f} coins | Status: {final_turn[0].status}")
print(f"Starter (Player 1): ${final_turn[1].reward:,.2f} coins | Status: {final_turn[1].status}")
