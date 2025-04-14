from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from custom_spiking_nn import CustomActorCriticPolicy

from env_gym import RISGymEnvironment

import matplotlib.pyplot as plt

set_random_seed(42)

# Define environment as a dummy vectorized environment
# with a single instance of the RISGymEnvironment
env = DummyVecEnv([lambda: RISGymEnvironment(num_receivers=1, ris_dims=[5, 5], abs_receiver_position_bounds=[4, 4])])

env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Instantiate model as PPO with MlpPolicy
# Gamma, gae_lambda set to 0 so the model only considers 
# the immediate reward for the current step
model = PPO(CustomActorCriticPolicy, env, verbose=0,
            learning_rate=5e-4,
            gamma=0.0,
            gae_lambda=0.0,
            n_steps=128,
            batch_size=32,
            clip_range=0.2,
            ent_coef=0.005,
            n_epochs=3)


model.learn(total_timesteps=30000)

reward_history = env.get_attr("reward_history")[0]
data_rate_history = env.get_attr("data_rate_history")[0]  

plt.plot(reward_history)
plt.title("Reward History")
plt.savefig("reward_history.png")
plt.clf()

plt.plot(data_rate_history)
plt.title("Data Rate History")
plt.savefig("data_rate_history.png")
plt.clf()
