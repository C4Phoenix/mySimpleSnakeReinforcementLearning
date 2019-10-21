
#%%
#!pip install -e ./env/Sneks_master
#%% enviroment
import gym
import sneks
from sneks.envs.snek import SingleSnek
from tqdm import tqdm

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

#%%
observation = env.reset()
for _ in tqdm(range(maxSnakeLife)):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)

  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()

#%%
