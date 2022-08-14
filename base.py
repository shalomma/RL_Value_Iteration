from abc import ABC, abstractmethod
import numpy as np


class BaseAlgorithm(ABC):
    def __init__(self, env, episodes):
        np.random.seed(42)
        self.env = env
        self.d = self.env.n_states * self.env.n_actions
        self.buffer = np.zeros((episodes * self.env.episode_len + 1, 4))
        self.Q = np.zeros((self.env.n_states + 1, self.env.n_actions))
        self.V = np.zeros(self.env.n_states)
        self.K = episodes
        self.rewards = []

    @abstractmethod
    def update(self, i):
        pass

    @abstractmethod
    def act(self, s):
        pass

    def update_buffers(self, i, s, a, r, s_):
        self.buffer[i, :] = [s, a, r, s_]

    def run(self):
        i = 0
        for k in range(self.K):
            self.env.reset()
            done = False
            while not done:
                s = self.env.state
                a = self.act(s)
                r, s_, done = self.env.advance(a)
                self.update_buffers(i, s, a, r, s_)
                i += 1
                self.rewards.append(r)
            self.update(i)