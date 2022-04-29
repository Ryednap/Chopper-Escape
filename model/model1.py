import cv2
from keras.layers.pooling import MaxPool2D
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Reshape
from tensorflow.python.layers.core import Dropout

"""""
    
    Reward
        +0.001 -100
    

    Training for 100000 steps ...
Interval 1 (0 steps performed)
/home/adleon/miniconda3/envs/ryednap/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:2325: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.
  warnings.warn('`Model.state_updates` will be removed in a future version. '
2022-04-29 01:17:07.835349: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2022-04-29 01:17:07.955530: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.7
10000/10000 [==============================] - 598s 60ms/step - reward: -0.1999 
20 episodes - episode_reward: -99.952 [-99.987, -99.856] - loss: 8.959 - mae: 0.219 - mean_q: 0.012 - fuel: 776.986

Interval 2 (10000 steps performed)
10000/10000 [==============================] - 809s 81ms/step - reward: -0.1999
22 episodes - episode_reward: -90.862 [-99.984, 0.100] - loss: 10.291 - mae: 1.589 - mean_q: -1.648 - fuel: 712.431

Interval 3 (20000 steps performed)
10000/10000 [==============================] - 824s 82ms/step - reward: -0.2099
22 episodes - episode_reward: -95.410 [-99.986, 0.100] - loss: 9.634 - mae: 2.620 - mean_q: -2.889 - fuel: 737.845

Interval 4 (30000 steps performed)
10000/10000 [==============================] - 829s 83ms/step - reward: -0.2199
22 episodes - episode_reward: -99.957 [-99.982, -99.890] - loss: 10.123 - mae: 4.196 - mean_q: -4.793 - fuel: 765.879

Interval 5 (40000 steps performed)
10000/10000 [==============================] - 826s 83ms/step - reward: -0.2299
25 episodes - episode_reward: -91.958 [-99.985, 0.100] - loss: 9.765 - mae: 4.833 - mean_q: -5.552 - fuel: 721.974

Interval 6 (50000 steps performed)
10000/10000 [==============================] - 831s 83ms/step - reward: -0.2299
24 episodes - episode_reward: -95.794 [-99.985, 0.100] - loss: 10.248 - mae: 4.431 - mean_q: -5.036 - fuel: 735.628

Interval 7 (60000 steps performed)
10000/10000 [==============================] - 822s 82ms/step - reward: -0.1699
18 episodes - episode_reward: -94.385 [-99.987, 0.144] - loss: 9.466 - mae: 4.278 - mean_q: -4.881 - fuel: 753.185

Interval 8 (70000 steps performed)
10000/10000 [==============================] - 826s 83ms/step - reward: -0.1799
20 episodes - episode_reward: -89.952 [-99.986, 0.181] - loss: 7.649 - mae: 3.958 - mean_q: -4.554 - fuel: 720.928

Interval 9 (80000 steps performed)
10000/10000 [==============================] - 828s 83ms/step - reward: -0.2599
26 episodes - episode_reward: -99.962 [-99.987, -99.885] - loss: 9.895 - mae: 4.463 - mean_q: -5.116 - fuel: 779.147

Interval 10 (90000 steps performed)
10000/10000 [==============================] - 845s 84ms/step - reward: -0.2199
done, took 8038.352 seconds


"""


def build_model_relu1(actions):
    model = Sequential()
    model.add(Reshape((80, 80, 3), input_shape=(1, 80, 80, 3)))
    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(80, 80, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    print(model.summary())
    return model


"""
    Final Model
    
"""

def build_model_relu_final1(actions):

    model = Sequential()
    model.add(Reshape((84, 84, 3), input_shape=(1, 84, 84, 3)))
    model.add(Conv2D(32, 8, strides=4, activation='relu', input_shape=(84, 84, 3,)))
    model.add(Conv2D(64, 4, strides=2, activation="relu"))
    model.add(Conv2D(64, 3, strides=1, activation="relu"))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    print(model.summary())
    return model
