import numpy as np
import copy
import matplotlib.pyplot as plt

transition_matrix= {
        "Paper": {"Paper": 2/3, "Rock": 1/3, "Scissors": 0/3},
        "Rock": {"Paper": 0/3,"Rock": 2/3,"Scissors": 1/3},
        "Scissors":{"Paper": 2/3, "Rock": 0/3,"Scissors": 1/3}
}

static_matrix = copy.deepcopy(transition_matrix)
learning_matrix = copy.deepcopy(transition_matrix)

states = ["Paper", "Rock", "Scissors"]

#static move vs learning move

def scores(static_move: str,learning_move:str) -> int:
    result = None

    if static_move == learning_move:
        result = 0
    elif static_move == "Rock" and learning_move == "Scissors" or \
         static_move == "Paper" and learning_move == "Rock" or \
         static_move == "Scissors" and learning_move == "Paper":
        result = 1
    else:
        result = -1


    return result

def static_player(my_last_move:str,tm:dict)->str:
    row = tm[my_last_move]

    moves = list(row.keys())
    probs = list(row.values())
    return np.random.choice(moves, p=probs)


BEATS = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
COUNTER = {"Scissors": "Rock", "Paper": "Scissors", "Rock": "Paper"}

def learning_player(my_last_move: str, op_last_move: str, reward: int, tm: dict) -> tuple[str, dict]:
    learning_rate = 0.1
    EPS = 1e-6

    row = tm[my_last_move]

    if reward > 0:
        row[my_last_move] *= (1 + learning_rate)
    elif reward < 0:
        counter = COUNTER[op_last_move]
        row[counter] *= (1 + learning_rate)

    total = sum(max(v, EPS) for v in row.values())
    for k in row:
        row[k] = max(row[k], EPS) / total

    moves = list(row.keys())
    probs = list(row.values())
    next_move = np.random.choice(moves, p=probs)

    return next_move, tm



static_move = learning_move = np.random.choice(states)


static_reward = 0
learning_reward = 0
round_reward=0

static_history = []
learning_history = []

for i in range(10000):
    static_move = static_player(static_move, static_matrix)
    learning_move, learning_matrix = learning_player(
        learning_move, static_move,
        round_reward,
        learning_matrix
    )

    score = scores(static_move, learning_move)
    round_reward = -score

    static_reward += score
    learning_reward += -score

    static_history.append(static_reward)
    learning_history.append(learning_reward)

#Plot
#print(static_history)
#print(learning_history)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Rock-Paper-Scissors: Static vs Learning Agent (Markov Chains)")

x = range(1, 10000 + 1)

# Left: cumulative reward
axes[0].plot(x, static_history,   label="Static",   color="blue", linewidth=1.2)
axes[0].plot(x, learning_history, label="Learning", color="red", linewidth=1.2)
axes[0].set_title("Cumulative Reward")
axes[0].set_xlabel("Game")
axes[0].set_ylabel("Total reward")
axes[0].legend()
axes[0].grid(True)

# Right: rolling average
window = 200
per_round_static   = [static_history[i] - (static_history[i-1] if i > 0 else 0)   for i in range(10000)]
per_round_learning = [learning_history[i] - (learning_history[i-1] if i > 0 else 0) for i in range(10000)]

static_roll   = np.convolve(per_round_static,   np.ones(window)/window, mode='valid')
learning_roll = np.convolve(per_round_learning, np.ones(window)/window, mode='valid')

axes[1].plot(static_roll,   label="Static",   color="blue")
axes[1].plot(learning_roll, label="Learning", color="red",)
axes[1].set_title(f"Rolling Avg Reward (window={window})")
axes[1].set_xlabel("Game")
axes[1].set_ylabel("Avg reward per game")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
