
#%%
#!pip install -e ./env/Sneks_master
#%%
import gym
import sneks
from sneks.envs.snek import SingleSnek
import time
from tqdm import tqdm

import pandas as pd
import numpy as np

#%%
maxSnakeLife = 10000
hunger = 200
gameSize = 10
env = SingleSnek(obs_type='rgb',
                 n_food=96,
                 size=(gameSize, gameSize),
                 dynamic_step_limit=hunger,
                 step_limit=maxSnakeLife + 1,
                 render_zoom=50,
                 add_walls=False)
env.seed(123)
np.random.seed(123)
#%%
rewards = list()
totalreward = 0
move_counts = list()
move = dict()
move[0] = 0
move[1] = 0
move[2] = 0
move[3] = 0

action0 = list()
action1 = list()
action2 = list()
action3 = list()
move_count = 0

#%%
observation = env.reset()
for _ in tqdm(range(maxSnakeLife)):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)

  move_count += 1
  move[action] += 1

  observation, reward, done, info = env.step(action)
  totalreward += reward

  if done:
    rewards.append(totalreward)
    totalreward = 0
    move_counts.append(move_count)
    move_count = 0

    action0.append(move[0])    
    action1.append(move[1])
    action2.append(move[2])    
    action3.append(move[3])
    move.clear()
    move[0] = 0
    move[1] = 0
    move[2] = 0
    move[3] = 0

    observation = env.reset()
env.close()

data = {'reward': rewards,
        'move_counts': move_counts,
        'action0': action0,
        'action1': action1,
        'action2': action2,
        'action3': action3}
df = pd.DataFrame(data)
data.clear()
#%%
df.head()
#%%
df.describe()

#%%
