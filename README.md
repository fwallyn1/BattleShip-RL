
# BattleShip project

This project is an implementation of a battleship player using reinforcement learning (specifically Q-learning) to discover the locations of three different sized ships on a 5x5 square game board. The goal of this project is to create a player capable of being strong at this game by learning through experience to make optimal decisions to discover the boats in as few moves as possible.




## How Q-learning works

Q-learning is a reinforcement learning algorithm that allows an agent to learn to make optimal decisions in an environment by learning from experience. It uses an evaluation function called Q-table that stores Q-values for each possible state and action. The Q-value represents the estimate of the expected future reward for a given action in a given state.

Here are the steps of how Q-learning works:

Initialization of the Q-table: The Q-table is initialized with arbitrary values for each possible state and action.

Observation of the current state: The agent observes the current state of the environment, in our case, the location of the boats on the game board.

Choosing an action: The agent chooses an action to take based on the current policy, which can be based on random exploration or exploitation of the Q-table.

Performing the action: The agent performs the chosen action and observes the reward obtained for this action.

Q-table update: The agent updates the Q-value for the chosen state and action using the Q-table update formula:

Q(s, a) = Q(s, a) + α * (r + γ * maxQ(s', a') - Q(s, a))

where s is the current state, a is the chosen action, r is the reward obtained, s' is the next state, a' is the next action, α is the learning rate that controls the importance of new information over previous information, and γ is the discount factor that controls the importance of future rewards over immediate rewards.

Repetition: Steps 2-5 are repeated until the agent reaches a final state, in our case, finding all the boats on the game board.

Exploration-exploitation: To improve the agent's performance, an exploration-exploitation policy can be used, which consists of exploring new states with a given probability to discover new strategies and exploiting known states with a given probability to maximize rewards.

The formula for updating the Q-table is essential for the convergence of the algorithm. It updates the Q-value for a given state and action using the reward obtained for that action, the maximum Q-value for the next state, and the learning and discount rates. Updating the Q-table ensures that the Q-values converge to the optimal values for each possible state and action, allowing the agent to make optimal decisions in the environment.

In summary, Q-learning is a reinforcement learning algorithm that allows an agent to learn to make optimal decisions by learning from experience and updating the Q-table to estimate expected future rewards

## Running the Code
To run the code, simply run the play_game() function in the battleship.py file. This will start a new game of Battleship and run the AI player using Q-learning. You can change the number of games played and other settings by modifying the parameters in the play_game() function.

For the deep learning agent in a gym environment with nice rendering, run the `dqn_play.ipynb`. But first you need to install requirements `pip install -r requirements.txt`


## Results
Due to the number of possible state, classic Q learning strategy does not outperform the random strategy (23 hits by mean to win). Even when adding constraint in the possible action (Hit Close : hit a cell adjacent with a touched cell : 17 hits by mean to win). So we explore deep Q learning using gym env from openAI and Tensorflow. The gym env can be found in `battleship_env.py`. The rewards was 

```
{ 
'win': 100,
'missed': 0,
'touched': 1,
'repeat_missed': -1,
'repeat_touched': -0.5
}
```

The architecture of the agent can be found in `dqn_agent.py` :

This deep learning approach need a lot of computational power. Without GPU, we were not able to have good results. Moreover, it is very difficult to have a change in environment in gym. So we were not able to constraint the possible action. Even penalizing these actions, our model trained on few episodes have bad behaviour (hit multiple times the same cell). 
## Conclusion
In conclusion, this project was very difficult because the battleship isn't well adapted for RL due to the number of possible state. When using classic Q learning, we were not able to outperform random policy neither the hit close policy that constraint the possible action. The exploration of deep q learning wasn't succesfull due to computational constraint.
## Authors

- [@gpain1999](https://www.github.com/gpain1999)
- [@fwallyn1](https://www.github.com/fwallyn1)
- [@kayser7](https://www.github.com/kayser7)
