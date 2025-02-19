import sionna
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies

# Suppress LibreSSL warning (optional)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

class UAVRLAgent(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.policy_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='tanh')  # Normalized actions [-1, 1]
        ])
        self.optimizer = tf.keras.optimizers.legacy.Adam(0.001)
        self.action_std = 2.0  # Initial exploration noise

    def call(self, state):
        return self.policy_net(state) * 5.0  # Scale actions to ±5 meters

class SINROptimizationEnv:
    def __init__(self):
        self.scene = load_scene(sionna.rt.scene.munich)
        self.scene.frequency = 2.4e9
        self.noise_power = 1e-7
        
        # Configure antenna arrays
        self.scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern="tr38901",
                                        polarization="V")
        self.scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                        vertical_spacing=0.5,
                                        horizontal_spacing=0.5,
                                        pattern="dipole",
                                        polarization="V")
        
        # Initialize devices
        self.uav = Transmitter(name="uav", position=[8.5,21,27])
        self.scene.add(self.uav)
        # Create receivers
        rx_0 = Receiver(name="rx0",
                    position=[90,90,1.5],
                    orientation=[0,0,0])

        rx_1 = Receiver(name="rx1",
                    position=[50,85,1.5],
                    orientation=[0,0,0])

        rx_2 = Receiver(name="rx2",
                    position=[45,45,1.5],
                    orientation=[0,0,0])
        
        self.receivers = [rx_0, rx_1, rx_2]

        # Add receiver instances to scene
        for rx in self.receivers:
            self.scene.add(rx)
        
        self.sinr_history = []

    def reset(self):
        """Reset environment to initial state"""
        self.uav.position = np.array([8.5,21,27], dtype=np.float32)
        self.receivers[0].position = [90,90,1.5]
        self.receivers[1].position = [50,85,1.5]
        self.receivers[2].position = [45,45,1.5]

        return self._get_state()

    def _get_state(self):
        """Create state vector from current positions"""
        state = np.concatenate([
            self.uav.position,
            *[rx.position for rx in self.receivers]
        ])
        return tf.convert_to_tensor(state, dtype=tf.float32)

    def _move_receivers(self):
        """Random ground receiver movement"""
        for rx in self.receivers:
            new_pos = rx.position._numpy()
            new_pos[:2] += np.random.uniform(-2, 2, size=2)
            rx.position = new_pos

    def _calculate_sinr(self):
        """Compute SINR for all receivers"""
        fft_size = 48
        subcarrier_spacing = 15e3
        channels = []

        # Compute the channel paths and corresponding CIR parameters
        paths = self.scene.compute_paths(max_depth=3, num_samples=1e6)
        a, tau = paths.cir()

        # Define OFDM parameters
        freqs = subcarrier_frequencies(fft_size, subcarrier_spacing)

        # Compute the frequency-domain channel
        h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=True)

        # Extract channels for each receiver by expanding the corresponding dimension
        h_freq_0 = tf.expand_dims(h_freq[:, 0, :, :, :, :, :], axis=1)
        h_freq_1 = tf.expand_dims(h_freq[:, 1, :, :, :, :, :], axis=1)
        h_freq_2 = tf.expand_dims(h_freq[:, 2, :, :, :, :, :], axis=1)

        # Compute the channel gain as the average of the absolute squared channel coefficients
        channel_0 = tf.reduce_mean(tf.abs(h_freq_0) ** 2)
        channel_1 = tf.reduce_mean(tf.abs(h_freq_1) ** 2)
        channel_2 = tf.reduce_mean(tf.abs(h_freq_2) ** 2)
        channels = [channel_0, channel_1, channel_2]

        channel_per_subcarrier_0 = tf.abs(h_freq_0) ** 2
        channel_per_subcarrier_1 = tf.abs(h_freq_1) ** 2
        channel_per_subcarrier_2 = tf.abs(h_freq_2) ** 2

        print("Channel per subcarrier for receiver 0:", channel_per_subcarrier_0.numpy())
        print("Channel per subcarrier for receiver 1:", channel_per_subcarrier_1.numpy())
        print("Channel per subcarrier for receiver 2:", channel_per_subcarrier_2.numpy())
        print("-------------------------------------------")
        print(f"Receiver 0 h: {h_freq_0.numpy()}")
        print(f"Receiver 1 h: {h_freq_1.numpy()}")
        print(f"Receiver 2 h: {h_freq_2.numpy()}")
        print("-------------------------------------------")
        print(f"Receiver 0 ch: {channel_0.numpy():.2e}")
        print(f"Receiver 1 ch: {channel_1.numpy():.2e}")
        print(f"Receiver 2 ch: {channel_2.numpy():.2e}")
        print("-------------------------------------------")

        # Calculate SINR for each receiver
        epsilon = 1e-10  # Small constant to avoid division or log of zero
        sinrs = []
        for i in range(3):
            desired = tf.maximum(channels[i], epsilon)
            # Sum the gains of the other receivers as interference
            interference = sum([channels[j] for j in range(3) if j != i])
            interference = tf.maximum(interference, epsilon)
            
            sinr = desired / (self.noise_power)
            sinr_db = 10 * tf.math.log(sinr + epsilon) / tf.math.log(10.0)
            sinrs.append(sinr_db.numpy())
            
        return sinrs

    def step(self, action):
        """Execute one timestep with UAV movement"""
        # Convert action to numpy array
        action = action.numpy()
        
        # Move UAV with position constraints
        new_pos = self.uav.position + action
        new_pos = np.clip(new_pos, [-200, -200, 20], [200, 200, 100])
        self.uav.position = new_pos
        
        # Move receivers
        self._move_receivers()
        
        # Calculate SINR and store
        sinr = self._calculate_sinr()
        self.sinr_history.append(sinr)
        
        # Reward is sum of SINRs with altitude penalty
        reward = sum(sinr) - 0.1 * abs(action[2])
        
        return self._get_state(), reward, sinr

def train_rl_agent():
    env = SINROptimizationEnv()
    agent = UAVRLAgent()
    
    # Training parameters
    num_episodes = 5
    steps_per_episode = 15
    gamma = 0.99
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        states = []
        actions = []
        
        for _ in range(steps_per_episode):
            # Generate action with exploration noise
            action = agent(state[None, ...])[0]  # Add batch dimension
            noise = tf.random.normal(action.shape, stddev=agent.action_std)
            noisy_action = action + noise
            
            # Step environment
            next_state, reward, sinr = env.step(noisy_action)
            
            # Store experience
            states.append(state)
            actions.append(noisy_action)
            episode_rewards.append(reward)
            
            state = next_state
        
        # Update policy
        with tf.GradientTape() as tape:
            returns = []
            G = 0
            for r in reversed(episode_rewards):
                G = r + gamma * G
                returns.insert(0, G)
            
            returns = tf.convert_to_tensor(returns, dtype=tf.float32)
            returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)
            
            log_probs = []
            for s, a in zip(states, actions):
                mu = agent(s[None, ...])[0]
                dist = tf.random.normal(mu.shape, stddev=1.0)
                log_prob = -0.5 * tf.reduce_sum((a - mu)**2)  # Simple Gaussian policy
                log_probs.append(log_prob)
            
            policy_loss = -tf.reduce_mean(tf.stack(log_probs) * returns)
        
        grads = tape.gradient(policy_loss, agent.trainable_variables)
        agent.optimizer.apply_gradients(zip(grads, agent.trainable_variables))
        
        # Decay exploration noise
        agent.action_std *= 0.95
        
        print(f"Episode {episode+1}: Avg Reward {np.mean(episode_rewards):.1f} dB")

    # Plot results
    plt.figure(figsize=(10, 6))
    sinr_data = np.array(env.sinr_history[-steps_per_episode:])
    for i in range(3):
        plt.plot(sinr_data[:, i], label=f'Receiver {i+1}')
    
    plt.title("SINR Optimization with UAV RL")
    plt.xlabel("Time Step")
    plt.ylabel("SINR (dB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sinr_optimization_results.png')
    plt.show()

if __name__ == "__main__":
    train_rl_agent()
