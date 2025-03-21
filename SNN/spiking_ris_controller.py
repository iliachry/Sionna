import sionna as sn
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies
import tensorflow as tf
import numpy as np
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import torch
import matplotlib.pyplot as plt
import time
import random

# Set random seeds for reproducibility
sn.config.seed = 42
torch.manual_seed(42)


# Define the RIS controller as an SNN
class RISController(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 64, options_per_element = 4):
        super().__init__()
        
        self.spike_grad = surrogate.fast_sigmoid()   # Surrogate gradient for spiking
        self.options_per_element = options_per_element                 # Number of actions per RIS element
        self.lif = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)  # Leaky Integrate-and-Fire neuron
        self.T = 16
        self.input_size = input_size
        self.output_size = output_size
        
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.lif1 = self.lif

        # Output layers for each RIS element
        self.output_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, options_per_element) for _ in range(self.output_size)
        ])

        # LIF neurons for each RIS element
        self.lif_output = torch.nn.ModuleList([
            self.lif for _ in range(self.output_size)
        ])

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        output_mems = [lif.init_leaky() for lif in self.lif_output]
        
        spk_rec = [[] for _ in range(self.output_size)]

        # Generate spikes through SNN timesteps
        for _ in range(self.T):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            for i in range(self.output_size):
                spk_out, output_mems[i] = self.lif_output[i](self.output_layers[i](spk1), output_mems[i])
                spk_rec[i].append(spk_out)

        # Sum spikes per element across time steps 
        spike_counts = [torch.stack(spk, dim=0).sum(dim=0) for spk in spk_rec]

        return spike_counts
    
    def get_action(self, state):
        spike_counts = self.forward(state)

        log_probs = [torch.nn.functional.log_softmax(counts.squeeze(0), dim=-1) for counts in spike_counts]

        actions = []
        action_log_probs = []

        for i in range(self.output_size):
            probs = torch.exp(log_probs[i])
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            
            actions.append(action)
            action_log_probs.append(log_probs[i].gather(1, action.unsqueeze(1)))

        return actions, action_log_probs

class RISEnvironment():
    def __init__(self, num_receivers, ris_dims = [3, 3]):

        self.num_receivers = num_receivers

        self.scene = sn.rt.load_scene(sn.rt.scene.simple_street_canyon)
        self.scene.tx_array = sn.rt.PlanarArray(num_rows=8,
                          num_cols=2,
                          vertical_spacing=0.7,
                          horizontal_spacing=0.5,
                          pattern="tr38901",
                          polarization="VH")
        self.scene.rx_array = sn.rt.PlanarArray(num_rows=1,
                          num_cols=1,
                          vertical_spacing=0.5,
                          horizontal_spacing=0.5,
                          pattern="dipole",
                          polarization="cross")

        camera = sn.rt.Camera("Cam", [0, 0, 300], look_at=[0, 0, 0])
        self.scene.add(camera)

        tx = sn.rt.Transmitter(name="tx", position=[-32,10,32])
        self.scene.add(tx)

        self.rx_positions = self.generate_rx_positions()
        for i in range(self.num_receivers):
            rx = sn.rt.Receiver(name=f"rx{i}", position=self.rx_positions[i])
            self.scene.add(rx) 

        self.ris_position = [32, -9, 32]
        self.ris_look_at = [-5, 30, 17]
        self.ris_num_rows = ris_dims[0]
        self.ris_num_cols = ris_dims[1]

        cell_grid = sn.rt.CellGrid(self.ris_num_rows, self.ris_num_cols)
        self.ris = sn.rt.RIS(name="ris", position=self.ris_position, num_rows=self.ris_num_rows, num_cols=self.ris_num_cols, look_at=self.ris_look_at)
        self.ris.phase_profile = sn.rt.DiscretePhaseProfile(cell_grid=cell_grid, num_modes=1, values=tf.zeros
                                                            ([1, self.ris_num_rows, self.ris_num_cols]))
        
        self.scene.add(self.ris)

        self.scene.frequency = 2.4e9


    def position_is_blocked(self, x, y, building_bounds):
        for (x1, y1, x2, y2) in building_bounds:
            if x1 <= x < x2 and y2 <= y < y1:
                return True
        return False


    def generate_rx_positions(self):
        max_abs_x = 80
        max_abs_y = 50
        z = 5

        # Define building bounds
        # (x1, y1, x2, y2)
        # (x1, y1) - bottom-left 
        # (x2, y2) - top-right
        building_bounds = [
            (28, 45, 68, 5),
            (28, -5, 68, -45),
            (-20, 45, 20, 5),
            (-20, -5, 20, -45),
            (-68, 45, -28, 5),
            (-68, -5, -28, -45)
        ]

        rx_positions = []

        for i in range(self.num_receivers):
            random_x = random.uniform(-max_abs_x, max_abs_x)
            random_y = random.uniform(-max_abs_y, max_abs_y)
            while self.position_is_blocked(random_x, random_y, building_bounds):
                random_x = random.uniform(-max_abs_x, max_abs_x)
                random_y = random.uniform(-max_abs_y, max_abs_y)
            rx_positions.append([random_x, random_y, z])
        return rx_positions


    def update_rx_positions(self, rx_positions):
        for i in range(self.num_receivers):
            self.scene.get(f"rx{i}").position = rx_positions[i]


    def compute_channel_gains(self, a, tau, rx_using_ris, ris=False):
        fft_size = 1
        subcarrier_spacing = 15e3
        channels = []
        
        # Define OFDM parameters
        freqs = subcarrier_frequencies(fft_size, subcarrier_spacing)

        # Compute the frequency-domain channel
        h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=False)
        #sn.rt.Scene.render_to_file(self=scene, camera="Cam", paths=paths, filename="scene.png")

        for i in range(self.num_receivers):
            if rx_using_ris[i] == ris:  # Matches behavior in original function
                h_freq_i = h_freq[:, i, :, :, :, :, :]  # No need to expand dims
                channel_i = tf.reduce_mean(tf.abs(h_freq_i) ** 2)  # Compute gain
                channels.append(channel_i)

        return channels    


    def compute_data_rates(self, channels):
        epsilon = 1e-10
        total_data_rates = []
        noise_power = 1e-7

        for i in range(len(channels)):
            desired = tf.maximum(channels[i], epsilon)
            sinr = desired / (noise_power + epsilon)
            total_data_rate = tf.math.log(1 + sinr) / tf.math.log(2.0)
            total_data_rates.append(total_data_rate.numpy())
        return total_data_rates


    # Utility function to compute received power
    def calculate_reward(self, rx_using_ris):
        
        paths = self.scene.compute_paths(max_depth=2, los=True, reflection=True, ris=True)
        print(paths.a)
        nan_mask = tf.math.is_nan(paths.a)
        contains_nan = tf.reduce_any(nan_mask)
        print("Contains NaN:", contains_nan.numpy())
        wait = input("PRESS ENTER TO CONTINUE.")
        # Only paths that are from LOS and reflection
        a_no_ris, tau_no_ris = paths.cir(los=True, reflection=True, ris=False, num_paths=3)

        # Only paths that are from reflection and RIS
        a_ris_reflection, tau_ris_reflection = paths.cir(los=False, reflection=True, ris=True, num_paths=3)

        # Only paths that are from reflection
        a_only_reflection, tau_only_reflection = paths.cir(los=False, reflection=True, ris=False, num_paths=3)

        # We "subtract" reflection only paths from RIS + reflection paths
        # to get paths that are either directly from RIS or from RIS
        # and then reflected.
        # Necessary because RIS + reflection paths include the reflection
        # only paths and RIS only paths do not include reflections.

        # Mask to check whether two paths are the same.
        # We leave tau the same, as if a is zeroed, the delay
        # is irrelevant.
        mask = tf.math.reduce_all(tf.abs(a_ris_reflection - a_only_reflection) < 1e-6, axis=-1)
        mask = tf.expand_dims(mask, axis=-1) 
        a_ris = tf.where(mask, tf.zeros_like(a_ris_reflection), a_ris_reflection)
        tau_ris = tau_ris_reflection

        
        channels = self.compute_channel_gains(a_no_ris, tau_no_ris, rx_using_ris, ris=False)
        channels_ris = self.compute_channel_gains(a_ris, tau_ris, rx_using_ris, ris=True)


        total_data_rates = self.compute_data_rates(channels)
        total_data_rates_ris = self.compute_data_rates(channels_ris)

        return total_data_rates, total_data_rates_ris # Sum of squared magnitudes = received power

# Convert tensor of actions to phase values
def action_to_phase(action):
    action_array = np.array([a.item() for a in action])
    return (action_array / 4) * (2 * np.pi)


# Initialize the RIS controller and optimizer

num_receivers = 4
ris_dims = [3, 3]
input_size = num_receivers * 3 + num_receivers
output_size = ris_dims[0] * ris_dims[1]

net = RISController(input_size=input_size, output_size=output_size)
env = RISEnvironment(num_receivers=4, ris_dims=[3, 3])

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training loop
num_steps = 2000
batch_size = 32

log_prob_batch = []
action_batch = []
reward_batch = []
state_batch = []

# Mean rewards for plotting
mean_rewards = []

for step in range(num_steps):
    rx_pos = env.generate_rx_positions()
    env.update_rx_positions(rx_pos)

    # Generate dummy association matrix randomly
    rx_using_ris = [random.choice([0, 1]) for _ in range(num_receivers)]

    rx_pos_tensor = torch.tensor(rx_pos, dtype=torch.float32).flatten()
    rx_using_ris_tensor = torch.tensor(rx_using_ris, dtype=torch.float32).flatten()


    state = torch.cat((rx_pos_tensor, rx_using_ris_tensor)).unsqueeze(0)[None, :]

    action, log_probs_per_element = net.get_action(state)
    
    log_prob = torch.stack(log_probs_per_element).sum()

    phase_shift = action_to_phase(action)

    env.ris.phase_profile.values = tf.reshape(tf.convert_to_tensor(phase_shift), [1, ris_dims[0], ris_dims[1]])

    _, reward = env.calculate_reward(rx_using_ris)

    print(state)
    print(sum(reward))

    log_prob_batch.append(log_prob)
    action_batch.append(action)
    reward_batch.append(sum(reward))
    state_batch.append(state)

    # Update the policy every batch_size steps
    if (step + 1) % batch_size == 0:
        rewards = torch.tensor(reward_batch, dtype=torch.float32)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-6)
        loss = -torch.stack(log_prob_batch) * normalized_rewards
        loss = loss.mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1}, Loss: {loss}")
        print(f"Mean Reward: {mean_reward*100}")

        mean_rewards.append(mean_reward)
        
        log_prob_batch = []
        action_batch = []
        reward_batch = []
        state_batch = []
        

# Plot the mean rewards over time
plt.plot(mean_rewards)
plt.xlabel("Step")
plt.ylabel("Mean Reward")
plt.show()
plt.savefig("mean_rewards.png")