import random
import numpy as np
from typing import Dict
from board import Board
from boat import Boat
from agent import Agent
# Game configuration
GAME_CONFIG = {
    "width" : 5,
    "height" : 5,
    "boats": {
    "carrier" : {"size" : 2, "number" : 1},
    "battleship" : {"size" : 2, "number" : 1},
    "cruiser" : {"size" : 3, "number" : 1},
    #"submarine" : {"size" : 3, "number" : 1},
    #"destroyer" : {"size" : 5, "number" : 1}
    }
}



def place_boats(game_config):
    board = Board(game_config["width"], game_config["height"])
    for boat_name, boat_config in game_config["boats"].items():
        num_boats = boat_config["number"]
        boat_size = boat_config["size"]
        for i in range(num_boats):
            while True:
                row = random.randint(0, game_config["width"] - 1)
                col = random.randint(0, game_config["height"] - 1)
                orientation = random.choice(['horizontal', 'vertical'])
                if orientation == 'horizontal':
                    if col + boat_size > game_config["height"]:
                        continue
                    position = (row, col)
                else:
                    if row + boat_size > game_config["width"]:
                        continue
                    position = (row, col)
                if not board.grid[position[0]:position[0]+boat_size, position[1]:position[1]+1].any():
                    board.add_boat(boat_name, boat_size, position, orientation)
                    break
    return board

def play_game(game_config:Dict, agent:Agent, is_update:bool = True):
    board = place_boats(game_config)
    nb_move = 0
    while not board.is_game_over():
        nb_move += 1
        state = board
        action = agent.choose_action(state)
        row, col = action
        reward = board.fire_shot(row, col)
        next_state = board
        if is_update:
            agent.update(state, action, next_state, reward)
        #print(nb_move)
        #print(board.grid)
        #print(board.is_game_over())
    return nb_move  

