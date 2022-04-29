from matplotlib import pyplot as plt
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from tensorflow.python.keras.optimizer_v2.adam import Adam

from model.model1 import  build_model_relu_final1


class DqnAgent:
    def __init__(self, env):
        self.history = None
        self.env = env
        """
            We will use BoltzmannQPolicy as exploration policy
        """
        self.policy = BoltzmannQPolicy()
        """
            Keras-RL provides us with a class called rl.memory.SequentialMemory that provides a fast and efficient 
            data structure that we can store the agent’s experiences in: Here we are saving the memory for maximum 10000
            steps

        """
        self.memory = SequentialMemory(limit=10000, window_length=1)
        """
            Model building
        """
        self.model = build_model_relu_final1(self.env.action_space.n)
        self.agent = DQNAgent(model=self.model, memory=self.memory, policy=self.policy,
                              nb_actions=self.env.action_space.n,
                              nb_steps_warmup=2000, target_model_update=.2, batch_size=64)

    def train(self):
        self.agent.compile(Adam(lr=1e-2), metrics=['mae'])
        self.history = self.agent.fit(self.env, nb_steps=20000, visualize=True  , verbose=1)
        self.print_history()

    def print_history(self):
        keys = self.history.history.keys()
        for key in keys:
            plt.plot(self.history.history[key])
            plt.title('model {}'.format(key))
            plt.ylabel(key)
            plt.xlabel('epoch')
            plt.show()
