import random

import cv2
import numpy as np
from abc import ABC
from gym import Env, spaces
from multipledispatch import dispatch as Override
from numpy import int64

from Entities import Chopper, Bird, Fuel

font = cv2.FONT_HERSHEY_DUPLEX

RESIZE = (84, 84)

def preprocess(state):
    return cv2.resize(state, RESIZE, interpolation=cv2.INTER_NEAREST)

def get_action_meanings():
    return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Do Nothing"}


def has_collided(elem1, elem2):
    x_col = False
    y_col = False

    elem1_x, elem1_y = elem1.get_position()
    elem2_x, elem2_y = elem2.get_position()

    if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
        x_col = True

    if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
        y_col = True

    if x_col and y_col:
        return True

    return False


class GameEnvironment(Env, ABC):
    def __init__(self):
        super(GameEnvironment, self).__init__()

        # Define a 2-D observation space
        self.chopper = None
        self.fuel_count = None
        self.bird_count = None
        self.ep_return = None
        self.fuel_left = None
        self.observation_shape = (600, 800, 3)
        """
            Our observation space is a continuous space of dimensions (600, 800, 3) 
            corresponding to an RGB pixel observation of the same size
        """
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(self.observation_shape),
                                            dtype=np.float16)

        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(6, )

        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.observation_shape) * 1

        # Define elements present inside the environment
        self.elements = []

        # Maximum fuel chopper can take at once
        self.max_fuel = 1000

        # Permissible area of helicopter to be
        self.y_min = int(self.observation_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int(self.observation_shape[0] * 0.9)
        self.x_max = self.observation_shape[1]

    def draw_elements_on_canvas(self):
        # Init the canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw the helicopter on canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x, y = elem.x, elem.y
            self.canvas[y: y + elem_shape[1], x:x + elem_shape[0]] = elem.icon

        text = 'Fuel Left: {} | Rewards: {}'.format(self.fuel_left, self.ep_return)

        # Put the info on canvas
        self.canvas = cv2.putText(self.canvas, text, (10, 20), font,
                                  0.8, (0, 0, 0), 1, cv2.LINE_AA)

    @Override()
    def reset(self):
        """
        This function resets the environment to its initial state,
        and returns the observation of the environment corresponding to the initial state.

        :parameter: self
        :return: numpy array representing observation space
        """

        # Reset the fuel consumed
        self.fuel_left = self.max_fuel

        # Reset the reward
        self.ep_return = 0

        # Number of birds
        self.bird_count = 0
        self.fuel_count = 0

        # Determine a place to initialise the chopper in.
        # We initialise our chopper randomly in an area in the top-left of our image.
        # This area is 5-10 percent of the canvas width and 15-20 percent of the canvas height
        x = random.randrange(int(self.observation_shape[0] * 0.05), int(self.observation_shape[0] * 0.10))
        y = random.randrange(int(self.observation_shape[1] * 0.15), int(self.observation_shape[1] * 0.20))

        # Initialise the chopper
        self.chopper = Chopper("chopper", self.x_max, self.x_min, self.y_max, self.y_min)
        self.chopper.set_position(x, y)

        # Initialise the elements
        self.elements = [self.chopper]

        # Reset the Canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # return the observation
        return preprocess(self.canvas)

    @Override()
    def render(self, mode="human"):
        """
        If you want to see a screenshot of the game as an image, rather than as a pop-up window then call
        this function. It may also be used to get the current frame image incase you are training convNets.

        :param mode: str representing the mode in which you want to recieve current env screenshot.
                    If "human" then only displays, but if ("rgb_array") returns the screen as numpy 2d array
        :return: None or numpy 2d array based on mode
        """
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas

    @Override()
    def close(self):
        cv2.destroyAllWindows()

    @Override()
    def step(self, action: int64):
        """
        This function takes an action as an input and applies it to the environment,
        which leads to the environment transitioning to a new state.

        In one transition step of the environment
        (i): We provide actions to the game that will control what our chopper(agent) does. We basically have 5 actions,
            which are move right, left, down, up, or do nothing, denoted by 0, 1, 2, 3, and 4, respectively.

        (ii): Birds spawn randomly from the right edge of the screen with a probability of 1% (1 out of 100 frames).
            The bird moves 5 coordinate points every frame to the left. If they hit the Chopper the game ends.
            Otherwise, they disappear from the game once they reach the left edge.

        (iii): Fuel tanks spawn randomly from the bottom edge of the screen with a probability of 1 % (1 out of 100 frames).
             The fuel moves 5 coordinates up every frame. If they hit the Chopper, the Chopper is fuelled to its full capacity.
             Otherwise, they disappear from the game once they reach the top edge.

        :param: action (int) representing the action chosen by the agent
        :return:
            (i) -  self.canvas (np.array) representing the observation of the state of environment
            (ii) - reward (int) representing the reward that agent gets from the environment after executing the
                   given action.
            (iii) - done (boolean) representing whether the episode has terminated. If true the simulation ends
            (iv) - info (list) representing the additional information associated with environment
        """

        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the fuel counter
        self.fuel_left -= 1

        # Reward for executing a step.
        reward = 0.01

        # apply the action to the chopper
        if action == 0:
            self.chopper.move(0, 5)
        elif action == 1:
            self.chopper.move(0, -5)
        elif action == 2:
            self.chopper.move(5, 0)
        elif action == 3:
            self.chopper.move(-5, 0)
        elif action == 4:
            self.chopper.move(0, 0)

        # Spawn a bird at the right edge with prob 0.01
        if random.random() < 0.01:
            # Spawn a bird
            spawned_bird = Bird("bird_{}".format(self.bird_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.bird_count += 1

            # Compute the x,y co-ordinates of the position from where the bird has to be spawned
            # Horizontally, the position is on the right edge and vertically, the height is randomly
            # sampled from the set of permissible values
            bird_x = self.x_max
            bird_y = random.randrange(self.y_min, self.y_max)
            spawned_bird.set_position(self.x_max, bird_y)

            # Append the spawned bird to the elements currently present in Env.
            self.elements.append(spawned_bird)

            # Spawn a fuel at the bottom edge with prob 0.01
        if random.random() < 0.01:
            # Spawn a fuel tank
            spawned_fuel = Fuel("fuel_{}".format(self.bird_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.fuel_count += 1

            # Compute the x,y co-ordinates of the position from where the fuel tank has to be spawned
            # Horizontally, the position is randomly chosen from the list of permissible values and
            # vertically, the position is on the bottom edge
            fuel_x = random.randrange(self.x_min, self.x_max)
            fuel_y = self.y_max
            spawned_fuel.set_position(fuel_x, fuel_y)

            # Append the spawned fuel tank to the elements currently present in the Env.
            self.elements.append(spawned_fuel)

            # For elements in the Ev
        for elem in self.elements:
            if isinstance(elem, Bird):
                # If the bird has reached the left edge, remove it from the Env
                if elem.get_position()[0] <= self.x_min:
                    self.elements.remove(elem)
                else:
                    # Move the bird left by 5 pts.
                    elem.move(-5, 0)

                # If the bird has collided.
                if has_collided(self.chopper, elem):
                    # Conclude the episode and remove the chopper from the Env.
                    done = True
                    reward = -10
                    self.elements.remove(self.chopper)

            if isinstance(elem, Fuel):
                # If the fuel tank has reached the top, remove it from the Env
                if elem.get_position()[1] <= self.y_min:
                    self.elements.remove(elem)
                else:
                    # Move the Tank up by 5 pts.
                    elem.move(0, -5)

                # If the fuel tank has collided with the chopper.
                if has_collided(self.chopper, elem):
                    # Remove the fuel tank from the env.
                    self.elements.remove(elem)

                    # Fill the fuel tank of the chopper to full.
                    self.fuel_left = self.max_fuel

        # Increment the episodic return
        self.ep_return += 0.01

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # If out of fuel, end the episode.
        if self.fuel_left == 0:
            done = True

        return preprocess(self.canvas), reward, done, {"fuel" : self.fuel_left}

    @Override()
    def step(self, action: int):

         # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the fuel counter
        self.fuel_left -= 1

        # Reward for executing a step.
        reward = 0.01

        # apply the action to the chopper
        if action == 0:
            self.chopper.move(0, 5)
        elif action == 1:
            self.chopper.move(0, -5)
        elif action == 2:
            self.chopper.move(5, 0)
        elif action == 3:
            self.chopper.move(-5, 0)
        elif action == 4:
            self.chopper.move(0, 0)

        # Spawn a bird at the right edge with prob 0.01
        if random.random() < 0.01:
            # Spawn a bird
            spawned_bird = Bird("bird_{}".format(self.bird_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.bird_count += 1

            # Compute the x,y co-ordinates of the position from where the bird has to be spawned
            # Horizontally, the position is on the right edge and vertically, the height is randomly
            # sampled from the set of permissible values
            bird_x = self.x_max
            bird_y = random.randrange(self.y_min, self.y_max)
            spawned_bird.set_position(self.x_max, bird_y)

            # Append the spawned bird to the elements currently present in Env.
            self.elements.append(spawned_bird)

            # Spawn a fuel at the bottom edge with prob 0.01
        if random.random() < 0.01:
            # Spawn a fuel tank
            spawned_fuel = Fuel("fuel_{}".format(self.bird_count), self.x_max, self.x_min, self.y_max, self.y_min)
            self.fuel_count += 1

            # Compute the x,y co-ordinates of the position from where the fuel tank has to be spawned
            # Horizontally, the position is randomly chosen from the list of permissible values and
            # vertically, the position is on the bottom edge
            fuel_x = random.randrange(self.x_min, self.x_max)
            fuel_y = self.y_max
            spawned_fuel.set_position(fuel_x, fuel_y)

            # Append the spawned fuel tank to the elements currently present in the Env.
            self.elements.append(spawned_fuel)

            # For elements in the Ev
        for elem in self.elements:
            if isinstance(elem, Bird):
                # If the bird has reached the left edge, remove it from the Env
                if elem.get_position()[0] <= self.x_min:
                    self.elements.remove(elem)
                else:
                    # Move the bird left by 5 pts.
                    elem.move(-5, 0)

                # If the bird has collided.
                if has_collided(self.chopper, elem):
                    # Conclude the episode and remove the chopper from the Env.
                    done = True
                    reward = -10
                    self.elements.remove(self.chopper)

            if isinstance(elem, Fuel):
                # If the fuel tank has reached the top, remove it from the Env
                if elem.get_position()[1] <= self.y_min:
                    self.elements.remove(elem)
                else:
                    # Move the Tank up by 5 pts.
                    elem.move(0, -5)

                # If the fuel tank has collided with the chopper.
                if has_collided(self.chopper, elem):
                    # Remove the fuel tank from the env.
                    self.elements.remove(elem)

                    # Fill the fuel tank of the chopper to full.
                    self.fuel_left = self.max_fuel

        # Increment the episodic return
        self.ep_return += 0.01

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # If out of fuel, end the episode.
        if self.fuel_left == 0:
            done = True

        return preprocess(self.canvas), reward, done, {"fuel": self.fuel_left}
