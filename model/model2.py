from keras.layers.pooling import MaxPool2D
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Reshape

"""
    
10000/10000 [==============================] - 749s 75ms/step - reward: -0.0060
19 episodes - episode_reward: -3.191 [-8.560, 11.480] - loss: 0.094 - mae: 0.264 - mean_q: 0.334 - mean_eps: 0.460 - fuel: 706.171

Interval 2 (10000 steps performed)
10000/10000 [==============================] - 957s 96ms/step - reward: -1.0000e-05
12 episodes - episode_reward: -0.709 [-8.360, 10.000] - loss: 0.075 - mae: 0.490 - mean_q: 0.609 - mean_eps: 0.100 - fuel: 690.416

Interval 3 (20000 steps performed)
10000/10000 [==============================] - 962s 96ms/step - reward: -0.0060
19 episodes - episode_reward: -2.797 [-8.640, 10.000] - loss: 0.066 - mae: 0.586 - mean_q: 0.718 - mean_eps: 0.100 - fuel: 664.796

Interval 4 (30000 steps performed)
10000/10000 [==============================] - 964s 96ms/step - reward: -0.0060
18 episodes - episode_reward: -3.523 [-8.690, 17.000] - loss: 0.099 - mae: 0.625 - mean_q: 0.775 - mean_eps: 0.100 - fuel: 688.665

Interval 5 (40000 steps performed)
10000/10000 [==============================] - 961s 96ms/step - reward: -0.0060
18 episodes - episode_reward: -3.271 [-8.000, 10.000] - loss: 0.084 - mae: 0.453 - mean_q: 0.562 - mean_eps: 0.100 - fuel: 692.480

Interval 6 (50000 steps performed)
10000/10000 [==============================] - 964s 96ms/step - reward: -0.0040
15 episodes - episode_reward: -2.944 [-8.350, 14.350] - loss: 0.076 - mae: 0.485 - mean_q: 0.598 - mean_eps: 0.100 - fuel: 706.296

Interval 7 (60000 steps performed)
10000/10000 [==============================] - 996s 100ms/step - reward: -0.0080
20 episodes - episode_reward: -4.131 [-8.540, 10.000] - loss: 0.101 - mae: 0.436 - mean_q: 0.542 - mean_eps: 0.100 - fuel: 702.050

"""

def build_model_relu2(actions):
    model = Sequential()
    model.add(Reshape((43, 32, 3), input_shape=(1, 43, 32, 3)))
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(43,32, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    print(model.summary())
    return model