from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from custom_spiking_nn import CustomActorCriticPolicy

from env_gym import RISGymEnvironment

import matplotlib.pyplot as plt
import numpy as np

set_random_seed(42)

# Define environment as a dummy vectorized environment
# with a single instance of the RISGymEnvironment
env = DummyVecEnv([lambda: RISGymEnvironment(num_receivers=2, ris_dims=[4, 4], abs_receiver_position_bounds=[4, 4])])

env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Instantiate model as PPO with MlpPolicy
# Gamma, gae_lambda set to 0 so the model only considers 
# the immediate reward for the current step
"""
model = PPO(CustomActorCriticPolicy, env, verbose=0,
            learning_rate=1e-3,
            gamma=0.0,
            gae_lambda=0.0,
            n_steps=256,
            batch_size=64,
            clip_range=0.2,
            ent_coef=0.005,
            n_epochs=3)
"""
            
model = PPO("MlpPolicy", env, verbose=0,
            learning_rate=1e-3,
            gamma=0.0,
            gae_lambda=0.0,
            n_steps=256,
            batch_size=64,
            clip_range=0.2,
            ent_coef=0.01,
            n_epochs=3)


env.envs[0].evaluate(model)
for i in range(100): 
    model.learn(total_timesteps=1000)
    env.envs[0].evaluate(model)


reward_history = np.array(env.get_attr("reward_history")[0])
data_rate_history = np.array(env.get_attr("data_rate_history")[0])  

plt.plot(reward_history)
plt.title("Reward History")
plt.savefig("reward_history.png")
plt.clf()

plt.plot(data_rate_history)
plt.title("Data Rate History")
plt.savefig("data_rate_history.png")
plt.clf()

window_size = 10
moving_avg_reward = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')

plt.plot(moving_avg_reward)
plt.title("Moving Average of Reward History")
plt.savefig("moving_avg_reward_history.png")
plt.clf()

moving_avg_data_rate = np.convolve(data_rate_history, np.ones(window_size)/window_size, mode='valid')

plt.plot(moving_avg_data_rate)
plt.title("Moving Average of Data Rate History")
plt.savefig("moving_avg_data_rate_history.png")
plt.clf()
