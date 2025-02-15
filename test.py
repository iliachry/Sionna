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
        self.uav = Transmitter(name="uav", position=[0.0, 0.0, 50.0])
        self.receivers = [Receiver(name=f"ue_{i}", position=[0.0, 0.0, 1.5]) for i in range(3)]
        self.scene.add(self.uav)
        for rx in self.receivers:
            self.scene.add(rx)
        
        self.sinr_history = []

    def reset(self):
        """Reset environment to initial state"""
        self.uav.position = np.array([0.0, 0.0, 50.0], dtype=np.float32)
        for rx in self.receivers:
            rx.position = np.array([np.random.uniform(-100, 100), 
                                  np.random.uniform(-100, 100), 
                                  1.5], dtype=np.float32)
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
        

        paths = self.scene.compute_paths(max_depth=3, diffraction=True)
        a, tau = paths.cir()
        freqs = subcarrier_frequencies(fft_size, subcarrier_spacing)
        channel = cir_to_ofdm_channel(freqs, a, tau, normalize=True)
        channels.append(tf.reduce_mean(tf.abs(channel)**2))
        
        # Calculate SINR for each receiver
        epsilon = 1e-10  # Small constant to avoid log(0)
        sinrs = []
        for i in range(3):
            desired = tf.maximum(channels[i], epsilon)
            interference = sum(channels[:i] + channels[i+1:])
            sinr = desired / (self.noise_power + tf.maximum(interference, epsilon))
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
