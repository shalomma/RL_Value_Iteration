import numpy as np


class LSVIUCB(object):
    def __init__(self, env, episodes, clip=1.):
        np.random.seed(42)
        self.env = env
        self.clip = clip
        self.K = episodes
        self.d = self.env.n_states * self.env.n_actions
        self.lam = 1.0
        self.L = self.lam * np.identity(self.d)
        self.L_inv = (1 / self.lam) * np.identity(self.d)
        self.w = np.zeros(self.d)
        self.Q = np.zeros((self.env.n_states + 1, self.env.n_actions))
        self.features_state_action = self.create_features(self.env.n_states, self.env.n_actions)
        self.buffer = np.zeros((episodes * self.env.episode_len, 4))
        self.buffer_Q = np.zeros((episodes * self.env.episode_len, self.env.n_states + 1, self.env.n_actions))
        self.rewards = []
        self.betas = []
        self.sums = np.zeros(self.d)
        # self.p = delta
        # self.c = 1. / 100
        # self.m_2 = .001

    def create_features(self, n_states, n_actions):
        features = np.zeros((self.env.n_states, self.env.n_actions, self.d))
        i = 0
        for s in range(n_states):
            for a in range(n_actions):
                features[s, a, i] = 1
                i += 1
        return features

    def update(self, i):
        q = np.zeros((self.env.n_states + 1, self.env.n_actions))
        d = self.buffer[i]
        s, a, r, s_ = int(d[0]), int(d[1]), d[2], int(d[3])

        self.L = self.L + np.outer(self.features_state_action[s, a], self.features_state_action[s, a])
        self.L_inv = np.linalg.inv(self.L)
        self.sums = self.sums + self.features_state_action[s, a] * (self.env.R[s, a] + self.Q[s_, :].max())
        self.w = np.matmul(self.L_inv, self.sums)

        # beta_i = self.beta()
        beta_i = 5. / 10.
        self.betas.append(beta_i)
        for ss in range(self.env.n_states):
            for aa in range(self.env.n_actions):
                feature = self.features_state_action[ss, aa]
                m = np.sqrt(np.dot(np.dot(feature, self.L_inv), feature))
                bonus = beta_i * m
                # bonus = 0
                estimation = np.inner(self.w, feature)
                # Q[ss, aa] = min(estimation + bonus, self.env.episode_len)
                q[ss, aa] = min(estimation + bonus, self.clip)
        self.Q = q.copy()

    def act(self, s):
        return self.env.argmax(self.Q[s, :])
        # return 1
        # return np.argmax(self.Q[s, :]) if np.random.uniform() > 0.1 else np.random.randint(0, self.env.n_actions)

    def beta(self):
        # iota = np.log(2 * self.d * self.K * self.env.episode_len/self.p)
        # return self.c * self.d * self.env.episode_len * np.sqrt(iota)
        first = self.m_2 * np.sqrt(self.lam)
        second = np.sqrt(2 * np.log(1 / self.p) + np.log(np.linalg.det(self.L) / self.lam))
        return first + second

    def run(self):
        i = 0
        for k in range(self.K):
            self.env.reset()
            done = False
            while not done:
                s = self.env.state
                a = self.act(s)
                r, s_, done = self.env.advance(a)

                self.buffer[i, :] = [s, a, r, s_]
                self.buffer_Q[i, :, :] = self.Q.copy()
                self.update(i)
                i += 1
                self.rewards.append(r)


if __name__ == '__main__':
    from environment import RiverSwim

    env_ = RiverSwim(n_states=4, episode_len=4)
    algo = LSVIUCB(env=env_, episodes=1000)
    algo.run()
