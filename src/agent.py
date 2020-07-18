import numpy as np


class Agent:
    def __init__(
        self,
        strategy=0,
        epsilon=0.9,
        decay_rate=0.99,
        alpha=0.3,
        t=0.5,
        final_t=0.1,
        player_type=0,
        d_exp=False,
        exponential_decay=False,
    ):
        self.reward_history = []
        self.action_history = []
        self.probability_history = []
        self.R = 0
        self.N = 0
        self.epsilon = (
            epsilon if 0 < epsilon and epsilon < 1 else 1
        )  # exploration parameter
        self.alpha = (
            alpha if 0 < alpha and alpha < 1 else 0.6
        )  # learning rate parameter
        self.original_alpha = alpha
        self.t = t  # boltzmann exploration temperature parameter
        self.original_t = t
        self.lastAction = 0
        self.decay_rate = decay_rate
        self.strategy = strategy
        self.player_type = player_type
        self.Qhistory = []
        # self.Qvalue = np.random.uniform(0,1,2)
        self.Qvalue = np.zeros(2)
        self.probabilities = self.calcProbabilities()
        self.d_exp = d_exp
        self.exponential_decay = exponential_decay
        self.final_t = final_t

    def reset(self):
        # self.Qvalue = np.zeros(2)
        self.Qvalue = np.zeros(2)
        # self.Qvalue = np.random.uniform(0,1,2)
        self.probabilities = self.calcProbabilities()
        self.epsilon = 1
        self.alpha = self.original_alpha
        self.t = self.original_t
        self.N = 0
        self.rewardHistory = []
        self.actionHistory = []
        self.probability_history = []
        self.Qhistory = []

    def decreaseEpsylon(self):
        if self.d_exp > 0.05:
            if self.epsilon * self.decay_rate - 0.1 > 0.1:
                self.epsilon = self.epsilon * self.decay_rate + 0.1

    def decreaseT(self):
        if self.exponential_decay:
            if self.t * self.decay_rate > self.final_t:
                self.t = self.t * self.decay_rate
        else:
            if self.t - self.decay_rate > self.final_t:
                self.t = self.t - self.decay_rate

    """
    def calcProbabilities(self):
        return (np.exp(self.Qvalue/ self.t) / np.sum(np.exp(self.Qvalue/ self.t)))
    """

    def calcProbabilities(self):
        # Numerically stable softmax
        # https://stackoverflow.com/a/39558290
        z = self.Qvalue.reshape(1, 2) / self.t
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        div = e_x / div
        return div.flatten()

    def randomExploration(self):
        if np.random.ranf() < self.epsilon:
            action = np.random.choice(2)
            if self.d_exp:
                self.decreaseEpsylon()
        else:
            action = np.argmax(self.Qvalue)
        self.lastAction = action
        return self.lastAction

    def boltzmanExploration(self):
        if np.random.ranf() < self.epsilon:
            action = np.random.choice(a=[0, 1], p=self.calcProbabilities())
            if self.d_exp:
                self.decreaseEpsylon()
        else:
            action = np.argmax(self.Qvalue)
        self.lastAction = action
        return self.lastAction

    def noExploration(self):
        self.lastAction = np.random.choice(a=[0, 1], p=self.calcProbabilities())
        if self.d_exp:
            self.decreaseT()
        return self.lastAction

    def take_action(self):
        self.probabilities = self.calcProbabilities()
        self.probability_history.append(self.probabilities[0])
        self.Qhistory.append(self.Qvalue)
        if self.strategy == 0:
            return self.boltzmanExploration()
        elif self.strategy == 1:
            return self.randomExploration()
        elif self.strategy == 2:
            return self.noExploration()

    def incremental_avg_update(self, R, a):
        return self.alpha * (R - self.Qvalue[a])

    def updateQ(self, R, a):
        self.N += 1
        self.Qvalue[a] += self.incremental_avg_update(R, a)

    def get_reward_for_action(self, R):
        self.updateQ(R, self.lastAction)
        self.reward_history.append(R)
        self.action_history.append(self.lastAction)
