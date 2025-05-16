from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from custom_spiking_nn import CustomActorCriticPolicy

from env_gym import RISGymEnvironment

import matplotlib.pyplot as plt
import numpy as np

set_random_seed(42)


training_mode = "association"               # phase or association training mode
num_receivers = 10                          # Only used for association mode, otherwise ignored
ris_dims = [4, 4]                           # Dimensions of the RIS
abs_receiver_position_bounds = [200, 200]   # Bounds for the absolute receiver positions


if training_mode == "phase":
    env = DummyVecEnv([lambda: RISGymEnvironment(num_receivers=1, mode=training_mode, ris_dims=ris_dims, abs_receiver_position_bounds=abs_receiver_position_bounds)])

    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    model = PPO(CustomActorCriticPolicy, env, verbose=0,
                learning_rate=1e-3,
                gamma=0.0,
                gae_lambda=0.0,
                n_steps=256,
                batch_size=64,
                clip_range=0.2,
                ent_coef=0.005,
                n_epochs=3,
                policy_kwargs=dict(mode=training_mode))


    model.learn(total_timesteps=80_000)
    model.save("phase_model")


elif training_mode == "association":

    model_p = PPO.load("phase_model")

    env = DummyVecEnv([lambda: RISGymEnvironment(num_receivers=num_receivers, mode=training_mode, ris_dims=ris_dims, abs_receiver_position_bounds=abs_receiver_position_bounds, phase_model=model_p)])

    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    model = PPO(CustomActorCriticPolicy, env, verbose=0,
                learning_rate=1e-3,
                gamma=0.0,
                gae_lambda=0.0,
                n_steps=256,
                batch_size=64,
                clip_range=0.2,
                ent_coef=0.005,
                n_epochs=3,
                policy_kwargs=dict(mode=training_mode))



    model.learn(total_timesteps=10_000)
    model.save("association_model")


# Plotting model performance metrics
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


# Smoother, moving average based plots

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


