from battleship import play_game
from random_agent import RandomAgent
from qlearning_agent import QLearningAgent
from hit_close_agent import HitCloseAgent
import numpy as np
import matplotlib.pyplot as plt
BOATS = {
    "carrier" : {"size" : 5, "number" : 1},
    "battleship" : 4,
    "submarine" : 3,
    "cruiser" : 3,
    "destroyer" : 2
}

GAME_CONFIG = {
    "width" : 5,
    "height" : 5,
    "boats": {
    #"carrier" : {"size" : 5, "number" : 1},
    #"battleship" : {"size" : 4, "number" : 1},
    "cruiser" : {"size" : 3, "number" : 1},
    "submarine" : {"size" : 3, "number" : 1},
    "destroyer" : {"size" : 2, "number" : 1}
    }
}


q_agent = QLearningAgent()
NB_GAME_UPDATE = 50000
NB_GAME = 10000
moves_list = []

for i in range(NB_GAME_UPDATE):
    #print(i)
    nb_move = play_game(GAME_CONFIG,q_agent,is_update=True)
q_agent.epsilon = 1
for i in range(NB_GAME_UPDATE):
    nb_move = play_game(GAME_CONFIG,q_agent,is_update=False)
    moves_list.append(nb_move)
    
print(np.mean(moves_list), np.quantile(moves_list,q=[0.25,0.5,0.75]))