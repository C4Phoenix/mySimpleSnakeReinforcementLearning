
#%%
#!pip install -e ./env/Sneks_master

#%%
import gym
import sneks
from sneks.envs.snek import SingleSnek
import time
from tqdm import tqdm

#%%
env = SingleSnek(obs_type='rgb', n_food=1, size=(32,32), dynamic_step_limit=100, step_limit=100000,render_zoom=25, add_walls=True)

#%%
observation = env.reset()
for _ in tqdm(range(1000)):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()

#%%
