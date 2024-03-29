#%%
#!pip install --upgrade wandb
!wandb login f244ffe13b0872010de2092f7a8fe61186506c10
#%%
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from keras import backend
backend.tensorflow_backend._get_available_gpus()

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import wandb
from wandb.keras import WandbCallback
#wandb.init(project="my-test-cartpoll-project")
#%%

ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(nb_actions))
model.add(Activation('linear'))

print(model.summary())
#%%

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
step_limit = 3000#50000
memory = SequentialMemory(limit=step_limit, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=step_limit, visualize=True, verbose=2, callbacks=[WandbCallback()])

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)

#%%
