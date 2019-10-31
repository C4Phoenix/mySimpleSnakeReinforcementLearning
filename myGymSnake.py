
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
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping

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

loadFromFile = False
testOnly = False
if (not testOnly):
    wandb.init(project="simple_snake")

#%% prep envoriment
maxSnakeLife = 10000
hunger = 100
gameSize = 8

env = SingleSnek(obs_type='raw',
                 n_food=30,
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

model.add(Conv2D(kernel_size=(2,2),
                 strides=(1,1),
                 input_shape=(1,observationSize,observationSize),
                 activation='relu',
                 data_format='channels_first',
                 filters=10))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(kernel_size=(2,2),# midden plus zijkant
                 data_format='channels_first',
                 activation='relu',
                 filters=20))# filters elke kop 'directie; van 2 pixels 8 * 2 voor iets extra's?

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(kernel_size=(2,2),# midden plus zijkant
                 data_format='channels_first',
                 activation='relu',
                 filters=30))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(24))
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

# model.add(Dense(16))
# model.add(Activation('relu'))

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
warmup_steps = 100
if(loadFromFile):
    warmup_steps = 20#needs at least 1 entry to start
    model.load_weights('model.h5')


#%% initialize agent
step_limit = 100000
memory = SequentialMemory(limit=10000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(
               processor=my_input_processor(),
               model=model, 
               nb_actions=nb_acthions,
               memory=memory,
               nb_steps_warmup=warmup_steps,
               enable_dueling_network=True,
               dueling_type='max',
               target_model_update=1e-2, #how often to update the target model. t_m_u<1 = slowly update the model
               policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#%% testLoaded model
if (testOnly and loadFromFile):
    dqn.test(env, nb_episodes=50, visualize=True)
    env.render(close=True)
    exit()

#%% train!
Checkpoint = ModelCheckpoint(os.path.join(wandb.run.dir, "model.h5"), verbose=0, save_best_only=True, save_weights_only=True, period=200)
earlyStopper = EarlyStopping(monitor='episode_reward', min_delta=0, patience=400, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
dqn.fit(env, nb_steps=step_limit, visualize=False, verbose=1, callbacks=[WandbCallback(), Checkpoint, earlyStopper])
#dqn.fit(env, nb_steps=step_limit, visualize=True, verbose=1)
env.render(close=True)
print("saving....")
dqn.save_weights(os.path.join(wandb.run.dir, "model_final.h5"))
#%%
dqn.test(env, nb_episodes=5, visualize=True)
env.render(close=True)
