import sionna as sn
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies
import tensorflow as tf
import numpy as np
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)


# Define the RIS controller as an SNN
# Treats environment as a grid, uses CNN
# Architecture too deep, spikes don't make it through
# Not great option, leaving it for now just in case
class RISController(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 128, options_per_element = 4):
        super().__init__()
        
        self.spike_grad = surrogate.fast_sigmoid()   # Surrogate gradient for spiking
        self.options_per_element = options_per_element                 # Number of actions per RIS element
        self.lif = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)  # Leaky Integrate-and-Fire neuron
        self.T = 32
        self.input_size = input_size
        self.output_size = output_size
    

        self.conv1 = torch.nn.Conv2d(1, 2, 5)
        self.lif_conv1 = self.lif
        self.conv2 = torch.nn.Conv2d(2, 4, 5)
        self.lif_conv2 = self.lif
        self.conv3 = torch.nn.Conv2d(4, 6, 5)
        self.lif_conv3 = self.lif
        self.fc1 = torch.nn.Linear(300, 128)
        self.lif_fc1 = self.lif

        # Output layers for each RIS element
        self.output_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, options_per_element) for _ in range(self.output_size)
        ])

        # LIF neurons for each RIS element
        self.lif_output = torch.nn.ModuleList([
            self.lif for _ in range(self.output_size)
        ])

    def forward(self, x):
        mem1 = self.lif_conv1.init_leaky()
        mem2 = self.lif_conv2.init_leaky()
        mem3 = self.lif_conv3.init_leaky()
        mem4 = self.lif_fc1.init_leaky()
        output_mems = [lif.init_leaky() for lif in self.lif_output]
        
        spk_rec = [[] for _ in range(self.output_size)]

        # Generate spikes through SNN timesteps
        for _ in range(self.T):
            cur1 = F.max_pool2d(self.conv1(x), 3)
            print(cur1)
            spk1, mem1 = self.lif_conv1(cur1, mem1)
            print(torch.count_nonzero(spk1))
            wait = input("L1.")

            cur2 = F.max_pool2d(self.conv2(spk1), 2)
            print(cur2)
            spk2, mem2 = self.lif_conv2(cur2, mem2)
            print(torch.count_nonzero(spk2))
            wait = input("L2.")

            cur3 = F.max_pool2d(self.conv3(spk2), 2)
            print(cur3)
            spk3, mem3 = self.lif_conv3(cur3, mem3)
            print(torch.count_nonzero(spk3))
            wait = input("L3.")

            cur4 = self.fc1(spk3.view(-1, 300))
            print(cur4)
            print(torch.count_nonzero(cur4))
            spk4, mem4 = self.lif_fc1(cur4, mem4)
            print(torch.count_nonzero(spk4))
            wait = input("L4.")

            for i in range(self.output_size):
                spk_out, output_mems[i] = self.lif_output[i](self.output_layers[i](spk4), output_mems[i])
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
            action_log_probs.append(log_probs[i].gather(0, action))

        return actions, action_log_probs

class RISEnvironment():
    def __init__(self, num_receivers, ris_dims = [3, 3]):

        self.num_receivers = num_receivers
        self.active_receivers = num_receivers

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
        
        self.grid_dimensions = (80, 50)
        self.grid = self.create_grid()
        

        camera = sn.rt.Camera("Cam", [0, 0, 300], orientation=[np.pi/2, np.pi/2, 0])
        self.scene.add(camera)

        tx = sn.rt.Transmitter(name="tx", position=[-32,10,32])
        self.scene.add(tx)


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


    def update_rx_positions(self):
        z = 5
        x_min, x_max = -self.grid_dimensions[0], self.grid_dimensions[0]
        y_min, y_max = -self.grid_dimensions[1], self.grid_dimensions[1]

        for i in range(self.active_receivers):
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            grid_y, grid_x = self.world_to_grid(x, y)
            while self.grid[grid_y, grid_x] == -1 or self.grid[grid_y, grid_x] == 1:
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                grid_y, grid_x = self.world_to_grid(x, y)

            rx = sn.rt.Receiver(name=f"rx{i}", position=[x, y, z])
            self.scene.add(rx)
            self.grid[grid_y, grid_x] = 10.0


    def world_to_grid(self, x, y):
        return y + self.grid_dimensions[1], x + self.grid_dimensions[0]   


    def create_grid(self):
        x_min, x_max = -self.grid_dimensions[0], self.grid_dimensions[0]
        y_min, y_max = -self.grid_dimensions[1], self.grid_dimensions[1]

        grid_width = x_max - x_min + 1 
        grid_height = y_max - y_min + 1

        grid = torch.zeros((grid_height, grid_width), dtype=torch.float)
    

        invalid_positions = [
            (28, 50, 80, 5),
            (28, -5, 80, -50),
            (-20, 50, 20, 5),
            (-20, -5, 20, -50),
            (-80, 50, -28, 5),
            (-80, -5, -28, -50)
        ]

        for (x1, y1, x2, y2) in invalid_positions:
            y1_idx, x1_idx = self.world_to_grid(x1, y1)
            y2_idx, x2_idx = self.world_to_grid(x2, y2)

            grid[y2_idx:y1_idx+1, x1_idx:x2_idx+1] = -1.0

        return grid
    

    def clear_rx(self):
        for i in range(self.active_receivers):
            self.scene.remove(f"rx{i}")
        self.grid = self.create_grid()


    def remove_nans(self, tensor):
        if tensor.dtype in (tf.complex64, tf.complex128):
            # Complex tensor handling
            real_part = tf.math.real(tensor)
            imag_part = tf.math.imag(tensor)
            is_nan = tf.math.logical_or(
                tf.math.is_nan(real_part),
                tf.math.is_nan(imag_part)
            )
        else:
            # Float tensor handling
            is_nan = tf.math.is_nan(tensor)
        
        # Create zeros with the same type as input tensor
        zeros = tf.zeros_like(tensor)
        
        # Replace NaN values with zeros
        cleaned_tensor = tf.where(is_nan, zeros, tensor)

        return cleaned_tensor


    def compute_channel_gains(self, a, tau):
        fft_size = 1
        subcarrier_spacing = 15e3
        channels = []
        
        # Define OFDM parameters
        freqs = subcarrier_frequencies(fft_size, subcarrier_spacing)

        # Compute the frequency-domain channel
        h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=False)
        for i in range(self.active_receivers):
            h_freq_i = h_freq[:, i, :, :, :, :, :]
            channel_i = tf.reduce_mean(tf.abs(h_freq_i) ** 2)
            channels.append(channel_i)

        return channels    


    def compute_data_rates(self, channels):
        epsilon = 1e-11
        total_data_rates = []
        noise_power = 1e-7
        for i in range(len(channels)):
            desired = tf.maximum(channels[i], epsilon)
            sinr = desired / (noise_power + epsilon)
            total_data_rate = tf.math.log(1 + sinr) / tf.math.log(2.0)
            total_data_rates.append(total_data_rate.numpy())
        return total_data_rates


    # Utility function to compute received power
    def calculate_reward(self):
        
        paths = self.scene.compute_paths(max_depth=3, los=True, reflection=True, ris=True)

        # sn.rt.Scene.render_to_file(self=self.scene, camera="Cam", paths=paths, filename="scene.png")

        # Only paths that are from LOS and reflection
        a_no_ris, tau_no_ris = paths.cir(los=True, reflection=True, ris=False, num_paths=3)

        # Only paths that are from reflection and RIS
        a_ris_reflection, tau_ris_reflection = paths.cir(los=False, reflection=True, ris=True, num_paths=3)
        
        # Replace nan values with zeros. Only necessary for RIS
        # Nan values do not appear in other paths. Maybe Sionna bug?
        a_ris_reflection = self.remove_nans(a_ris_reflection)
        tau_ris_reflection = self.remove_nans(tau_ris_reflection)

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

        
        channels = self.compute_channel_gains(a_no_ris, tau_no_ris)
        channels_ris = self.compute_channel_gains(a_ris, tau_ris)


        total_data_rates = self.compute_data_rates(channels)
        total_data_rates_ris = self.compute_data_rates(channels_ris)

        return total_data_rates, total_data_rates_ris

# Convert tensor of actions to phase values
def action_to_phase(action):
    action_array = np.array([a.item() for a in action])
    return (action_array / 4) * (2 * np.pi)


# Initialize the RIS controller and optimizer

num_receivers = 3
ris_dims = [3, 3]
input_size = num_receivers * 3 + num_receivers
output_size = ris_dims[0] * ris_dims[1]



net = RISController(input_size=input_size, output_size=output_size)
env = RISEnvironment(num_receivers=num_receivers, ris_dims=[3, 3])
sn.rt.Scene.render_to_file(self=env.scene, camera="Cam", filename="scene.png")
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

"""
plt.figure(figsize=(8, 8))
plt.imshow(env.grid, cmap="gray", origin="upper")
plt.show()
"""

# Training loop
num_steps = 30001
batch_size = 128

log_prob_batch = []
action_batch = []
reward_batch = []
state_batch = []

# Mean rewards for plotting
mean_rewards = []

mean_test_rewards = []
test_rx_pos = []
test_rx_using_ris = []
current_test = 0

"""
for i in range(256):
    test_rx_pos.append(env.generate_rx_positions())
    test_rx_using_ris.append([1] + [random.choice([0, 1]) for _ in range(num_receivers - 1)])
    random.shuffle(test_rx_using_ris[i])
"""

for step in range(num_steps):

    env.active_receivers = random.randint(1, env.num_receivers)
    env.update_rx_positions()

    state = env.grid.unsqueeze(0)[None, :]

    action, log_probs_per_element = net.get_action(state)
    
    log_prob = torch.stack(log_probs_per_element).sum()

    phase_shift = action_to_phase(action)

    env.ris.phase_profile.values = tf.reshape(tf.convert_to_tensor(phase_shift), [1, ris_dims[0], ris_dims[1]])

    _, reward = env.calculate_reward()

    log_prob_batch.append(log_prob)
    action_batch.append(action)
    reward_batch.append(sum(reward)*100)
    state_batch.append(state)

    # Update the policy every batch_size steps
    if (step + 1) % batch_size == 0:
        rewards = torch.tensor(reward_batch, dtype=torch.float32)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-6)
        print(torch.stack(log_prob_batch))
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

    env.clear_rx()

"""
    if step % 3000 == 0:
        test_rewards = []
        for i in range(len(test_rx_pos)):
            env.update_rx_positions(test_rx_pos[i])
            state = torch.cat((torch.tensor(test_rx_pos[i], dtype=torch.float32).flatten(), torch.tensor(test_rx_using_ris[i], dtype=torch.float32).flatten())).unsqueeze(0)[None, :]
            action, _ = net.get_action(state)
            phase_shift = action_to_phase(action)
            env.ris.phase_profile.values = tf.reshape(tf.convert_to_tensor(phase_shift), [1, ris_dims[0], ris_dims[1]])
            _, reward = env.calculate_reward(test_rx_using_ris[i])
            test_rewards.append(sum(reward)*100)


        mean_test_rewards.append(np.mean(test_rewards))
        plt.plot(test_rewards)
        plt.savefig(f"test_{current_test}.png")
        plt.clf()
        torch.save(net.state_dict(), f"snn_model{current_test}.pth")
        current_test += 1
""" 

# Plot the mean rewards over time
plt.plot(mean_rewards)
plt.xlabel("Step")
plt.ylabel("Mean Reward")
plt.savefig("mean_rewards.png")
plt.clf()

plt.plot(mean_test_rewards)
plt.xlabel("Test")
plt.ylabel("Mean Reward")
plt.savefig("mean_test_rewards.png")
plt.clf()