
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


## Results
After training the AI player using Q-learning, we can see a significant improvement in its performance compared to random guessing. When playing randomly, it can take an average of x guesses to sink all of the ships on a 5x5 board with 3 ships of sizes 3, 3, and 2. However, after training with Q-learning, the AI player is able to sink all of the ships in significantly fewer guesses, often in less than x guesses.

## Conclusion
In conclusion, this project demonstrates the power of Q-learning for developing intelligent agents that can learn from their experiences and gradually improve their performance over time. The AI player developed in this project is able to outperform random guessing and provide a challenging opponent for players of the game of Battleship.
## Authors

- [@gpain1999](https://www.github.com/gpain1999)
- [@fwallyn1](https://www.github.com/fwallyn1)
- [@kayser7](https://www.github.com/kayser7)
