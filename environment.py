import numpy as np


class TabularMDP:
    def __init__(self, n_states, n_actions, episode_len):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.episode_len = episode_len

        self.timestep = 0
        self.state = 0

        # Now initialize R and P
        self.R = np.zeros((n_states, n_actions))
        self.P = np.zeros((n_states, n_actions, n_states))

    def reset(self):
        """Resets the Environment"""
        self.timestep = 0
        self.state = 0

    def advance(self, action):
        reward = self.R[self.state, action]
        new_state = np.random.choice(self.n_states, p=self.P[self.state, action])

        # Update the environment
        self.state = new_state
        self.timestep += 1

        episode_end = False
        if self.timestep == self.episode_len:
            episode_end = True
            self.reset()

        return reward, new_state, episode_end

    @staticmethod
    def argmax(b):
        return np.random.choice(np.where(b == b.max())[0])


class RiverSwim(TabularMDP):
    def __init__(self, n_states, episode_len):
        super().__init__(n_states=n_states, n_actions=2, episode_len=episode_len)
        self.set_values()

    def set_values(self):
        for s in range(self.n_states):
            self.P[s, 0, s] = 1.
            self.P[s, 1, min(self.n_states - 1, s + 1)] = 1.
            self.R[s, 1] = -.01 / (self.n_states - 1)

        self.R[self.n_states - 1, 1] = 1.
