import cv2

from GameEnvironment import GameEnvironment
from PIL import Image as im

from agents.DQN import DqnAgent
from agents.DQN2 import DqnAgent2

"""
    The reward for each step is 1, therefore, the episodic return counter is updated by 1 every episode. 
    If there is a collision, the reward is -10 and the episode terminates. 
    The fuel counter is reduced by 1 at every step
"""

env = GameEnvironment()
agent = DqnAgent(env)
agent.train()
env.close()