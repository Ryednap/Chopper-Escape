import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time

from GameEnvironment import GameEnvironment


env = GameEnvironment()
obs = env.reset()

while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    # Render the game
    env.render()

    if done:
        break

env.close()
