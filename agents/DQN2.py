from matplotlib import pyplot as plt
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from tensorflow.python.keras.optimizer_v2.adam import Adam
from model.model1 import build_model_relu_final1


class DqnAgent2:
    def __init__(self, env):
        self.history = None
        self.env = env
        """
            For the policy we will have Epsilon Gredy Policy as exploration strategy and Linear
            Annealed Policy to compute threshold and decay the epsilon with passing steps.
        
            
            We will start with epsilon value of 1 and will go no longer than 0.01. 
        """
        self.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=0.01,
                                           value_test=.05, nb_steps=10000)

        """
            Sequential Memory with maximum capacity of 5000 experience steps
        """
        self.memory = SequentialMemory(limit=5000, window_length=1)
        self.model = build_model_relu_final1(self.env.action_space.n)
        self.agent = DQNAgent(model=self.model, memory=self.memory, policy=self.policy,
                              nb_actions=self.env.action_space.n,
                              nb_steps_warmup=1000, target_model_update=1e-2, batch_size=64)

    def train(self):
        self.agent.compile(Adam(lr=1e-2), metrics=['mae'])
        self.history = self.agent.fit(self.env, nb_steps=10000, visualize=True, verbose=1)
        self.print_history()

    def print_history(self):
        keys = self.history.history.keys()
        for key in keys:
            plt.plot(self.history.history[key])
            plt.title('model {}'.format(key))
            plt.ylabel(key)
            plt.xlabel('epoch')
            plt.show()
