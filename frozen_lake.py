import gym

from agents.QAgent import QAgent

env = gym.make('FrozenLake8x8-v0')
env.reset()
agent = QAgent(env)
agent.train()
agent.visualize_epsilon()
agent.visualize_reward()