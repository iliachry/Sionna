import os

# Enable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Not strictly necessary for one GPU

# Check if GPU is available
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU set up and using memory growth.")
    except RuntimeError as e:
        print("Error setting memory growth:", e)
else:
    print("No GPU detected. Using CPU.")

import numpy as np
import csv
import matplotlib.pyplot as plt

import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Suppress LibreSSL warning (optional)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

set_random_seed(seed = 111)


class SINROptimizationEnv(gym.Env):
    def __init__(self):

        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        self.action_mapping = np.array([-0.1, 0, 0.1])

        self.observation_space = spaces.Box(
            low=np.array([-30.0, -30.0, -30.0]),  # Minimum values for x, y, z
            high=np.array([30.0, 30.0, 30.0]),    # Maximum values for x, y, z
            shape=(3,),                           # 3-dimensional space
            dtype=np.float64
        )

        self.current_step = 0

        self.scene = load_scene(sionna.rt.scene.munich)
        self.scene.frequency = 2.4e9
        self.noise_power = 1e-7

        # Configure antenna arrays
        self.scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern="dipole",
                                        polarization="V")
        self.scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern="dipole",
                                        polarization="V")

        # Initialize devices
        self.uav = Transmitter(name="uav", position=[0.0, 0.0, 50.0])
        self.receivers = [Receiver(name=f"ue_{i}", position=[0.0, 0.0, 1.5]) for i in range(3)]
        self.scene.add(self.uav)
        for rx in self.receivers:
            self.scene.add(rx)

        self.sinr_history = []

        self.data_rate_history_means = []
        self.data_rate_history_per_step = []
        self.episode_data_rate = []

        self.episode_rewards = []
        self.reward_history = []


        self.steps = []
        self.action_history = []
        self.state_history = []


    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        
        self.current_step = 0

        self.uav.position = np.array([8.5, 21, 27], dtype=np.float32)
        self.receivers[0].position = [90,90,1.5]
        self.receivers[1].position = [50,85,1.5]
        self.receivers[2].position = [45,45,1.5]
        return self._get_state(), {}

    def _get_state(self):
        """Create state vector from current positions"""
        state = np.concatenate([
            self.uav.position,
            *[rx.position for rx in self.receivers]
        ])
        state = self.uav.position
        return tf.convert_to_tensor(state, dtype=tf.float32)

    def _move_receivers(self):
        """Random ground receiver movement"""
        for rx in self.receivers:
            new_pos = rx.position._numpy()
            new_pos[:2] += np.random.uniform(-2, 2, size=2)
            rx.position = new_pos

    def _calculate_sinr(self):
        """Compute SINR for all receivers"""
        fft_size = 1
        subcarrier_spacing = 15e3
        channels = []

        paths = self.scene.compute_paths(max_depth=2, num_samples=1e6)
        a, tau = paths.cir()

        # Define OFDM parameters
        freqs = subcarrier_frequencies(fft_size, subcarrier_spacing)

        # Compute the frequency-domain channel
        h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=False)

        # Extract channels for each receiver by expanding the corresponding dimension
        h_freq_0 = tf.expand_dims(h_freq[:, 0, :, :, :, :, :], axis=1)
        h_freq_1 = tf.expand_dims(h_freq[:, 1, :, :, :, :, :], axis=1)
        h_freq_2 = tf.expand_dims(h_freq[:, 2, :, :, :, :, :], axis=1)

        # Compute the channel gain as the average of the absolute squared channel coefficients
        channel_0 = tf.reduce_mean(tf.abs(h_freq_0) ** 2)
        channel_1 = tf.reduce_mean(tf.abs(h_freq_1) ** 2)
        channel_2 = tf.reduce_mean(tf.abs(h_freq_2) ** 2)
        channels = [channel_0, channel_1, channel_2]

        # channel_per_subcarrier_0 = tf.abs(h_freq_0) ** 2
        # channel_per_subcarrier_1 = tf.abs(h_freq_1) ** 2
        # channel_per_subcarrier_2 = tf.abs(h_freq_2) ** 2

        # Calculate SINR for each receiver
        epsilon = 1e-10  # Small constant to avoid log(0)
        sinrs = []
        total_data_rates = []
        for i in range(3):
            desired = tf.maximum(channels[i], epsilon)
            interference = sum(channels[:i] + channels[i+1:])
            
            sinr = desired / (self.noise_power + tf.maximum(interference, epsilon))
            
            sinr_db = 10 * tf.math.log(sinr + epsilon) / tf.math.log(10.0)
            
            total_data_rate = tf.math.log(1 + sinr) / tf.math.log(2.0)
            
            total_data_rates.append(total_data_rate.numpy())
            sinrs.append(sinr_db.numpy())

        #print(sinrs)
        return sinrs, total_data_rates

    def step(self, action):
        """Execute one timestep with UAV movement"""

        # Move UAV with position constraints
        new_pos = self.uav.position + self.action_mapping[action]
        new_pos = np.clip(new_pos, [0, 21, 25], [8.5, 30, 30])
        self.uav.position = new_pos

        self.current_step += 1

        sinr, total_data_rates = self._calculate_sinr()
        self.sinr_history.append(sinr)
        self.data_rate_history_per_step.append(total_data_rates)
        self.state_history.append(self.uav.position)
        self.action_history.append(self.action_mapping[action])
        self.steps.append(self.current_step)

        reward = sum(sinr)

        data_rate_sum = sum(total_data_rates)

        self.episode_data_rate.append(data_rate_sum)

        self.episode_rewards.append(reward)

        truncated = False
        terminated = False
        if self.current_step == 50:
            self.reward_history.append(np.mean(self.episode_rewards))
            self.data_rate_history_means.append(np.mean(self.episode_data_rate))
            print("Episode Average Reward:", np.mean(self.episode_rewards))
            self.episode_rewards = []
            self.episode_data_rate = []
            truncated = True

        return self._get_state(), reward, terminated, truncated, {"sinr": sinr}

def train_rl_agent():

    env = DummyVecEnv([lambda: SINROptimizationEnv()])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", env, verbose=0,
                learning_rate=5e-4,  # Reduce if learning is unstable
                gamma=0.99,  # Discount factor (lower if rewards are sparse)
                n_steps=128,  # Increase if observations are complex
                batch_size=32,  # Smaller batch size can help in small environments
                clip_range=0.2,  # Reduce if training is unstable
                ent_coef=0.005,
                n_epochs=3)

    model.learn(total_timesteps=15000)

    model.save("ppo_sinr_model")
    env.save("vec_normalize_env.pkl")

    reward_history = env.get_attr("reward_history")[0]  
    sinr_history = np.array(env.get_attr("sinr_history")[0])
    data_rate_plot = env.get_attr("data_rate_history_means")[0]

    data_rate_csv = np.array(env.get_attr("data_rate_history_per_step")[0])
    states = np.array(env.get_attr("state_history")[0])
    actions = np.array(env.get_attr("action_history")[0])
    steps = np.array(env.get_attr("steps")[0])


    window_size = 10
    moving_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')

    csv_filename = "data.csv"

    # Writing to CSV
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        
        # Writing Header
        header = ["Step", "State", "Action", "SINR1", "SINR2", "SINR3", "C1", "C2", "C3"]
        writer.writerow(header)
        
        # Writing Data
        for i in range(len(states)):  # Iterate over index
            row = [
                steps[i],
                states[i],
                actions[i],
                *sinr_history[i],  # Expands SINR list into separate columns
                *data_rate_csv[i]      # Expands C list into separate columns
            ]
            writer.writerow(row)

    print(f"CSV file '{csv_filename}' has been created successfully.")


    plt.figure(figsize=(20,5))
    plt.plot(reward_history, label="Episode Reward", color='blue')
    plt.plot(range(window_size-1, len(reward_history)), moving_avg, label=f"Moving Avg (Window={window_size})", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Training Progress - Rewards")
    plt.legend()
    plt.savefig("rewards_history.png")
    plt.clf()

    moving_avg = np.convolve(data_rate_plot, np.ones(window_size)/window_size, mode='valid')
    plt.figure(figsize=(20,5))
    plt.plot(data_rate_plot, label="Episode Data Rate", color='blue')
    plt.plot(range(window_size-1, len(data_rate_plot)), moving_avg, label=f"Moving Avg (Window={window_size})", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Average Data Rate")
    plt.title("Training Progress - Rewards")
    plt.legend()
    plt.savefig("data_rate_history.png")
    plt.clf()

    plt.figure(figsize=(10,5))
    for i in range(sinr_history.shape[1]):  # Iterate over timesteps
        plt.plot(sinr_history[:, i], label=f"Receiver {i+1}", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("SINR Value")
    plt.title("SINR Evolution Per Episode")
    plt.legend()
    plt.savefig("sinr_history.png")


    env.training = False  # Disable training mode
    env.norm_reward = False  # Don't normalize rewards during testing
    
    obs = env.reset()

    for _ in range(5):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()


if __name__ == "__main__":
    train_rl_agent()
