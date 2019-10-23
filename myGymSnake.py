
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
from keras.callbacks.callbacks import ModelCheckpoint

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
wandb.init(project="simple_snake")

#%% prep envoriment
maxSnakeLife = 10000
hunger = 100
gameSize = 8

env = SingleSnek(obs_type='raw',
                 n_food=8,
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
                 filters=32))# filters elke kop 'directie; van 2 pixels 8 * 2 voor iets extra's?

model.add(Conv2D(kernel_size=(3,3),# midden plus zijkant
                 data_format='channels_first',
                 activation='relu',
                 filters=16))# filters elke kop 'directie; van 2 pixels 8 * 2 voor iets extra's?

model.add(Conv2D(kernel_size=(3,3),# midden plus zijkant
                 data_format='channels_first',
                 activation='relu',
                 filters=4))# filters elke kop 'directie; van 2 pixels 8 * 2 voor iets extra's?

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(24))
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(Activation('relu'))

model.add(Dense(nb_acthions))
model.add(Activation('linear'))

print(model.summary())

#%%build processor to for the input
class my_input_processor(Processor):
    def __init__(self):
        self.last = (1,1)
        return 

    def process_observation(self, observation):
        # print('\n',observation, '\n')
        headCoordinate = (-1, -1)
        for x in range(len(observation)):
            for y in range(len(observation[x])):
                if (observation[x][y] == 101):
                    headCoordinate = (x, y)
                    break
            if(headCoordinate != (-1, -1)):
                break
        if(headCoordinate == (-1, -1)): #if head isnt found (probably when dead it processes this)
            headCoordinate = self.last # use last head position
        else:
            self.last = headCoordinate
        newObservation = np.zeros((observationSize, observationSize), dtype=observation.dtype)
        offsets = (gameSize - headCoordinate[0]- 1,
                   gameSize - headCoordinate[1] - 1)
        for x in range(len(observation)):
            for y in range(len(observation[x])):
                newObservation[offsets[0] + x][offsets[1] + y] = observation[x][y]
        # print('n',newObservation,'n')
        return newObservation

#%% to test processor
# env.reset()
# ob, _, _, _ = env.step(0)
# something = my_input_processor().process_observation(ob)

#%% load model
from keras.models import load_model
loadFromFile = True
if(loadFromFile):
    model.load_weights('model.h5')

#%% initialize agent
step_limit = 200000
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
Checkpoint = ModelCheckpoint(os.path.join(wandb.run.dir, "model.h5"), verbose=1, save_best_only=False, save_weights_only=True, period=500)
dqn.fit(env, nb_steps=step_limit, visualize=True, verbose=1, callbacks=[WandbCallback(), Checkpoint])
#dqn.fit(env, nb_steps=step_limit, visualize=True, verbose=1)
env.render(close=True)
print("saving....")
#%%
dqn.test(env, nb_episodes=5, visualize=True)
env.render(close=True)

