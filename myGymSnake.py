
#%%
#!pip install -e ./env/Sneks_master
#!pip install --upgrade wandb
#!wandb login f244ffe13b0872010de2092f7a8fe61186506c10

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
from rl.core import Processor

#%%import online logging
import wandb
import os
from wandb.keras import WandbCallback
#wandb.init(project="simple_snake")

#%% prep envoriment
maxSnakeLife = 10000
hunger = 100
gameSize = 8

env = SingleSnek(obs_type='raw',
                 n_food=1,
                 size=(gameSize,gameSize),
                 dynamic_step_limit=hunger,
                 step_limit=hunger,
                 render_zoom=50,
                 add_walls=False)
env.seed(123)
nb_acthions = env.action_space.n
observationSize = gameSize*2-1

#%% build model
model = Sequential()

model.add(Conv2D(kernel_size=(3,3),# midden plus zijkant
                 input_shape=(1,observationSize,observationSize),# 1 voor het aantal channels
                 activation='relu',
                 data_format='channels_first',
                 filters=18))# filters elke kop 'directie; van 2 pixels 8 * 2 voor iets extra's?

model.add(Conv2D(kernel_size=(3,3),# midden plus zijkant
                 data_format='channels_first',
                 activation='relu',
                 filters=9))# filters elke kop 'directie; van 2 pixels 8 * 2 voor iets extra's?

model.add(Flatten())
#model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(nb_acthions))
model.add(Activation('linear'))

print(model.summary())

#%%build processor to for the input
class my_input_processor(Processor):
    def __init__(self):
        return 

    def process_observation(self, observation):
        headCoordinate = (-1, -1)
        for x in range(gameSize):
            for y in range(gameSize):
                if (observation[x][y] == 101):
                    headCoordinate = (x, y)
                    break
            if(headCoordinate != (-1, -1)):
                break
        offsets = (gameSize - headCoordinate[0]- 1,
                   gameSize - headCoordinate[1] - 1)
        newObservation = np.zeros((observationSize, observationSize), dtype=observation.dtype)
        for x in range(gameSize):
            for y in range(gameSize):
                newObservation[offsets[0] + x][offsets[1] + y] = observation[x][y]
        return newObservation

#%% initialize agent
step_limit = 10000
memory = SequentialMemory(limit=step_limit, window_length=1)
policy = BoltzmannQPolicy()
fileName = 'saved_Weights/dqn_simpleSnake_weights.h5f'
dqn = DQNAgent(
               processor=my_input_processor(),
               model=model, 
               nb_actions=nb_acthions,
               memory=memory,
               nb_steps_warmup=30,
               target_model_update=1e-2, #how often to update the target model. t_m_u<1 = slowly update the model
               policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#dqn.load_weights(fileName)
#%% train!
#dqn.fit(env, nb_steps=step_limit, visualize=True, verbose=1, callbacks=[WandbCallback()])
dqn.fit(env, nb_steps=step_limit, visualize=True, verbose=1)
env.render(close=True)
print("saving....")
# save
model.save(os.path.join(wandb.run.dir, "model.h5"))
dqn.save_weights(fileName, overwrite=True)
#%%
dqn.test(env, nb_episodes=5, visualize=True)
env.render(close=True)

