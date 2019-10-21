
#%%
#!pip install -e ./env/Sneks_master
#%% imports
#import envirement
import numpy as np
import gym
import sneks
from sneks.envs.snek import SingleSnek

from tqdm import tqdm

#import model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam

from keras import backend
backend.tensorflow_backend._get_available_gpus()

#import agent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

#%% prep envoriment
maxSnakeLife = 1#10000
hunger = 200
gameSize = 10
env = SingleSnek(obs_type='raw',#test 'raw'
                 n_food=6,
                 size=(gameSize, gameSize),
                 dynamic_step_limit=hunger,
                 step_limit=maxSnakeLife + 1,
                 render_zoom=50,
                 add_walls=False)
env.seed(123)
nb_acthions = env.action_space.n

#%% build model
model = Sequential()
model.add(Conv2D(kernel_size=3,# midden plus zijkant
                 input_shape=(gameSize,gameSize,1),# 1 voor het aantal channels
                 activation='relu',
                 filters=16))# filters elke kop 'directie; van 2 pixels 8 * 2 voor iets extra's?

model.add(Flatten())
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(nb_acthions))
model.add(Activation('linear'))

print(model.summary())

#%%
observation = env.reset()
for _ in tqdm(range(maxSnakeLife)):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)

  observation, reward, done, info = env.step(action)
  print(observation)
  print(observation.shape)
  if done:
    observation = env.reset()
env.close()

#%%
