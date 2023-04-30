from battleship_env import BattleshipEnv
from dqn_agent import DQNAgent

env = BattleshipEnv()
agent = DQNAgent(env)

EPISODES = 1000
for episode in range(EPISODES):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = int(agent.act(state))
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
    agent.replay()

    print(f"Episode: {episode}, Score: {score}, Epsilon: {agent.epsilon}")

agent.save("battleship-dqn.h5")