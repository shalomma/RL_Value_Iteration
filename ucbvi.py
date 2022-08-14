import numpy as np
from base import BaseAlgorithm


class UCBVI(BaseAlgorithm):
    def __init__(self, env, episodes):
        super().__init__(env, episodes)
        self.Nsas_ = np.zeros((self.env.n_states, self.env.n_actions, self.env.n_states))
        self.Nsa = np.ones((self.env.n_states, self.env.n_actions))
        self.P = np.zeros((self.env.n_states, self.env.n_actions, self.env.n_states))

    def update(self, i):
        for j in range(self.env.episode_len):
            d = self.buffer[i - j]
            s, a, r, s_ = int(d[0]), int(d[1]), d[2], int(d[3])
            self.Nsa[s, a] += 1
            self.Nsas_[s, a, s_] += 1
            self.update_p_estimation()
            self.value_iteration()

    def value_iteration(self, tol=1e-3):
        v = np.zeros(self.env.n_states)
        q = np.zeros((self.env.n_states + 1, self.env.n_actions))
        while True:
            delta = 0.0
            for s in range(self.env.n_states):
                for a in range(self.env.n_actions):
                    value = q[s, a]
                    q[s, a] = self.env.R[s, a] + np.dot(self.P[s, a, :], v)
                    delta = max(delta, abs(value - q[s, a]))
            v = q[:-1].max(axis=1)
            if delta < tol:
                break
        self.Q = q.copy()
        self.V = v.copy()

    def update_p_estimation(self):
        for s in range(self.env.n_states):
            for a in range(self.env.n_actions):
                for s_ in range(self.env.n_states):
                    self.P[s, a, s_] = self.Nsas_[s, a, s_] / self.Nsa[s, a]

    def act(self, s):
        return self.env.argmax(self.Q[s, :])