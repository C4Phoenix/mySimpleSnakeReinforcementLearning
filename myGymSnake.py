
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
wandb.init(project="simple_snake")

#%% prep envoriment
maxSnakeLife = 1#10000
hunger = 200
gameSize = 10
env = SingleSnek(obs_type='raw',#test 'raw'
                 n_food=96,
                 size=(gameSize, gameSize),
                 dynamic_step_limit=hunger,
                 step_limit=maxSnakeLife + 1,
                 render_zoom=50,
                 add_walls=False)
env.seed(123)
nb_acthions = env.action_space.n

#%% build model
model = Sequential()
model.add(Conv2D(kernel_size=(3,3),# midden plus zijkant
                 input_shape=(1,gameSize,gameSize),# 1 voor het aantal channels
                 activation='relu',
                 data_format='channels_first',
                 filters=16))# filters elke kop 'directie; van 2 pixels 8 * 2 voor iets extra's?

model.add(Conv2D(kernel_size=(3,3),# midden plus zijkant
                 data_format='channels_first',
                 activation='relu',
                 filters=16))# filters elke kop 'directie; van 2 pixels 8 * 2 voor iets extra's?

model.add(Flatten())
#model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(nb_acthions))
model.add(Activation('linear'))

print(model.summary())

# build processor to for the input
# class my_input_processor(Processor):
#   def __init__(self):
#     print("using my_input_processor")
  
#     def process_state_batch(self, batch):
#       returnList = list()
#       print(batch)
#       for state in batch:
#         returnList.append(state[0])
#       return returnList

#%% initialize agent
step_limit = 3000
memory = SequentialMemory(limit=step_limit, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(
               #processor=my_input_processor(),
               model=model, 
               nb_actions=nb_acthions,
               memory=memory,
               nb_steps_warmup=10,
               target_model_update=1e-2, #how often to update the target model. t_m_u<1 = slowly update the model
               policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#%% train!
dqn.fit(env, nb_steps=step_limit, visualize=True, verbose=1, callbacks=[WandbCallback()])
#dqn.fit(env, nb_steps=step_limit, visualize=True, verbose=1)

# save
model.save(os.path.join(wandb.run.dir, "model.h5"))
#dqn.save_weights('dqn_simpleSnake_weights.h5f', overwrite=True)
#%%
dqn.test(env, nb_episodes=5, visualize=True)
