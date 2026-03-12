import numpy as np
import matplotlib.pyplot as plt
from bokeh.models.labeling import NoOverlap

transition_matrix = {
        "Paper": {"Paper": 2/3, "Rock": 1/3, "Scissors": 0/3},
        "Rock": {"Paper": 0/3,"Rock": 2/3,"Scissors": 1/3},
        "Scissors":{"Paper": 2/3, "Rock": 0/3,"Scissors": 1/3}
}

#copy of the transition_matrix for learinign
states = ["Paper", "Rock", "Scissors"]

#static move vs learning move

def scores(static_move: str,learning_move:str) -> int:
    result = None




    return result

static_move = np.random.choice(["Paper", "Rock", "Scissors"])
learning_move = np.random.choice(["Paper", "Rock", "Scissors"])
score = scores(static_move,learning_move)
#tm=transition_matrix[static_move]

def static_player(my_last_move:str,tm:dict)->str:
    return "my next move"

def learning_player(my_last_move:str, op_last_move:str,reward:int,tm:dict)->tuple[str,dict]:

    return "my_next_move",tm


static_move = learning_move = np.random.choice(states)
static_reward = learning_reward = 0
for i in range(10000):
    static_move=static_player(static_move,static_matrix)
    learning_move,learning_matrix=learning_player(learning_move,static_move,learning_player,learning_matrix)

    #2
    score = scores(static_move,learning_move)
    static_reward = score
    learning_reward = -score



  """  get current moves
    ebalutate reward
    sace reward history_image_key(
        
    )
    update learingn model"""

#p +=reward *p *(optinol Learning rate(0.9))