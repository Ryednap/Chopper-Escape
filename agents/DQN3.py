from matplotlib import pyplot as plt
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from tensorflow.python.keras.optimizer_v2.adam import Adam

from model.model1 import build_model_relu1

class DqnAgent3:
    def __init__(self, env):
        self.history = None
        self.env = env
        self.policy = BoltzmannQPolicy()
        self.memory = SequentialMemory(limit=5000, window_length=1)
        self.model = build_model_relu1(self.env.action_space.n)
        self.agent = DQNAgent(model=self.model, memory=self.memory, policy=self.policy,
                              nb_actions=self.env.action_space.n,
                              nb_steps_warmup=2000, target_model_update=1e-2, batch_size=200)

    def train(self):
        self.agent.compile(Adam(lr=1e-2), metrics=['mae'])
        self.history = self.agent.fit(self.env, nb_steps=10000, visualize=False, verbose=1)
        self.agent.save_weights("Model_0.h5")
        self.print_history()

    def print_history(self):
        keys = self.history.history.keys()
        for key in keys:
            plt.plot(self.history.history[key])
            plt.title('model {}'.format(key))
            plt.ylabel(key)
            plt.xlabel('epoch')
            plt.show()

    def load_weigths(self):
        self.agent.compile(Adam(lr=1e-2), metrics=['mae'])
        self.agent.load_weights('Model_0.h5')
        self.agent.test(self.env, visualize=True, nb_episodes=5, nb_max_episode_steps=1000, verbose=1)