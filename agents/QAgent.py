import random

import numpy as np
from matplotlib import pyplot as plt


class QAgent:
    def __init__(self, env):
        self.env = env
        self.Q = self.build_q_table()

        ''''
            AGENT HYPER-PARAMETERS
        '''

        self.alpha = 0.7  # learning rate
        self.discount_factor = 0.618  # gamma value
        self.epsilon = 1  # current epsilon value
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay = 0.01  # decay to balance exploration and exploitation

        ''''
            TRAINING CONSTANTS
        '''
        self.train_episodes = 5000
        self.test_episodes = 100
        self.max_steps = 100
        self.training_rewards = []
        self.epsilons = []

    def build_q_table(self):
        Q = np.zeros((self.env.observation_space.n,
                      self.env.action_space.n))
        return Q

    def update_q_table(self, state, action, reward, new_state):
        self.Q[state, action] = self.Q[state, action] \
                                + self.alpha * (reward + self.discount_factor * np.argmax(self.Q[new_state:])) \
                                - self.Q[state, action]

    def decay_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-1 * self.decay * episode)

    def train(self):

        for episode in range(self.train_episodes):
            # Reset the environment each episode as per requirement
            state = self.env.reset()
            total_training_rewards = 0

            for step in range(self.max_steps):
                exploration_exploitation_tradeoff = random.uniform(0, 1)

                """
                    STEP 2:
                    Choosing exploration vs exploitation based on the current epsilon value.
                    If the tradeoff is larger than epsilon we do exploitation
                """

                if exploration_exploitation_tradeoff > self.epsilon:
                    action = np.argmax(self.Q[state:])
                else:
                    action = self.env.action_space.sample()

                """
                    STEP 3:
                    Once we have chosen the action we can now perform the same on our environment
                    to get observation and reward.
                    
                    Also render the environment
                """
                new_state, reward, done, info = self.env.step(action)
                self.env.render()
                """
                    STEP 4:
                    Update the Q-Table and change the state. If the environment returned the done = True
                    then we end the game. 
                    Otherwise we decay the epsilon and append the total reward and current epsilon in the list
                """

                self.update_q_table(state, action, reward, new_state)
                total_training_rewards += reward
                state = new_state
                if done:
                    break

                # decay the epsilon
                self.decay_epsilon(episode)

                # Add the total reward and epsilon in the list
                self.training_rewards.append(total_training_rewards)
                self.epsilons.append(self.epsilon)

                print("Training score over time: {}".format(sum(self.training_rewards) / self.train_episodes))

        return True

    def visualize_reward(self):
        x = range(self.train_episodes)
        plt.plot(x, self.training_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Training total reward')
        plt.title('Total rewards over all episodes in training')
        plt.show()

    def visualize_epsilon(self):
        plt.plot(self.epsilons)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title("Epsilon for episode")
        plt.show()
