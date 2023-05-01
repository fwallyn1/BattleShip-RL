import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from collections import deque
import random


class DQNAgent:
    def __init__(self, env):
        # Environment
        self.env = env
        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.n

        # Hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01

        # Replay memory
        self.memory = deque(maxlen=2000)

        # DNN
        self.model = self._build_model()

    def _build_model(self) -> Model:
        model = Sequential()
        model.add(Conv2D(32, (2, 2), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.model.predict(np.expand_dims(state, axis=0),verbose=0)
            return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, min(self.batch_size,len(self.memory)))

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target += self.discount_factor * np.amax(self.model.predict(np.expand_dims(next_state, axis=0),verbose=0)[0])

            target_f = self.model.predict(np.expand_dims(state, axis=0),verbose=0)
            target_f[0][action] = target
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
